"""VGGT-based RAE (Representation AutoEncoder) for multi-level feature reconstruction.

Mirrors the RAE_DA3 interface but uses the VGGT backbone (Aggregator) instead of DA3.
Key differences from RAE_DA3:
  - Feature dim: 2048 (vs 1536)
  - OUT_LAYERS: [4, 11, 17, 23] (vs [5, 7, 9, 11])
  - No CLS token; uses zeros as dummy CLS
  - Decoder: VGGT pretrained DPTHead for depth (no separate MAE decoder for RGB)
  - ImageNet normalization is handled internally by the aggregator
"""

import torch
import torch.nn as nn
from typing import Optional, List, Tuple, Dict, Union
from math import sqrt

from .encoders import ARCHS
from .decoders import GeneralDecoder_Variable
from transformers import AutoConfig
from utils.config_utils import parse_encoder_size


class RAE_VGGT(nn.Module):
    """
    RAE using VGGT encoder for multi-level feature extraction.

    Provides the same unified interface as RAE_DA3:
    - Training: encode(mode='single') for single-level feature extraction
    - Stats computation: encode(mode='all') for all 4 levels
    - Validation: decode() for depth reconstruction
    - Propagation: propagate_features() for shallow→deep feature propagation

    Args:
        encoder_pretrained_path: HuggingFace path to VGGT model
        level: Feature level (0,1,2,3 or -1,-2,-3,-4)
        encoder_input_size: Input image size (default: 518)
        encoder_type: Encoder class name (default: 'VGGTEncoder')
        noise_tau: Noise level for training
        reshape_to_2d: Whether to reshape to (B,C,H,W)
        normalization_stat_path: Path to normalization statistics
        mae_weight: Unused for VGGT (kept for interface compatibility)
    """

    OUT_LAYERS = [4, 11, 17, 23]
    PATCH_SIZE = 14

    def __init__(
        self,
        encoder_pretrained_path: str = 'facebook/VGGT-1B',
        level: int = -1,
        encoder_input_size: Union[int, List[int], Tuple[int, int]] = 518,
        encoder_type: str = 'VGGTEncoder',
        noise_tau: float = 0.0,
        reshape_to_2d: bool = True,
        normalization_stat_path: Optional[str] = None,
        special_stat_path: Optional[str] = None,
        num_special_tokens: int = 0,
        eps: float = 1e-5,
        mae_weight: Optional[str] = None,
        # Legacy/unused params (kept for config compatibility)
        dpt_decoder_path: Optional[str] = None,
        da3_weights_path: Optional[str] = None,
        dpt_model_type: Optional[str] = None,
        decoder_config_path: Optional[str] = None,
        decoder_patch_size: int = 14,
        pretrained_decoder_path: Optional[str] = None,
    ):
        super().__init__()

        self.level = level
        self.encoder_h, self.encoder_w = parse_encoder_size(encoder_input_size)
        self.encoder_input_size = encoder_input_size
        self.reshape_to_2d = reshape_to_2d
        self.noise_tau = noise_tau
        self.eps = eps

        # Initialize encoder
        encoder_cls = ARCHS[encoder_type]
        self.encoder = encoder_cls(
            pretrained_path=encoder_pretrained_path,
            level=level,
        )

        self.encoder_patch_size = self.encoder.patch_size
        self.latent_dim = self.encoder.hidden_size  # 2048

        # Calculate patch counts
        h_patches = self.encoder_h // self.encoder_patch_size
        w_patches = self.encoder_w // self.encoder_patch_size
        self.num_patches = h_patches * w_patches
        self.num_patches_per_side = (h_patches, w_patches)

        # ImageNet normalization constants
        # Note: VGGT aggregator handles normalization internally, but we need
        # these for un-normalizing images before passing to the aggregator
        # (since prepare_data applies ImageNet norm externally)
        self.register_buffer('encoder_mean', torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1))
        self.register_buffer('encoder_std', torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1))

        # VGGT uses its own DPT head for depth (extracted during encoder init)
        self.depth_head = self.encoder.depth_head  # DPTHead from VGGT pretrained

        # MAE decoder for RGB reconstruction (4-level concat → 8192-dim → RGB)
        self.mae_decoder = None
        self._mae_hidden_size = self.latent_dim * 4  # 2048 * 4 = 8192

        if mae_weight is not None:
            print(f"[RAE_VGGT] Initializing MAE decoder for RGB (hidden_size={self._mae_hidden_size})")
            # Use existing ViTXL_ config as base, override hidden_size for 4-level concat
            _decoder_cfg_path = decoder_config_path or 'configs/decoder/ViTXL'
            dec_config = AutoConfig.from_pretrained(_decoder_cfg_path)
            dec_config.hidden_size = self._mae_hidden_size
            dec_config.patch_size = decoder_patch_size
            # MAE decoder image_size must match training resolution (504, not encoder_input_size)
            _mae_image_size = 504  # Wooseok's decoder trained on 504x504
            dec_config.image_size = _mae_image_size
            self.mae_decoder = GeneralDecoder_Variable(dec_config, base_image_size=(_mae_image_size, _mae_image_size))

            # Load weights — checkpoint has 'decoder.' prefix keys
            print(f"[RAE_VGGT] Loading MAE decoder weights from {mae_weight}")
            ckpt = torch.load(mae_weight, map_location='cpu')
            if 'ema' in ckpt:
                full_sd = ckpt['ema']
            elif 'model' in ckpt:
                full_sd = ckpt['model']
            elif 'state_dict' in ckpt:
                full_sd = ckpt['state_dict']
            else:
                full_sd = ckpt
            # Extract only decoder.* keys and strip prefix
            dec_sd = {}
            for k, v in full_sd.items():
                if k.startswith('decoder.'):
                    dec_sd[k[len('decoder.'):]] = v
            load_info = self.mae_decoder.load_state_dict(dec_sd, strict=False)
            print(f"[RAE_VGGT] MAE decoder loaded: missing={len(load_info.missing_keys)}, unexpected={len(load_info.unexpected_keys)}")
            if load_info.missing_keys:
                print(f"  Missing: {load_info.missing_keys[:5]}...")
            self.mae_decoder.eval()
            for p in self.mae_decoder.parameters():
                p.requires_grad = False

        # DPT decoder placeholder for interface compatibility
        self.rae_cl_decoder = None  # VGGT uses self.depth_head instead

        # Special tokens config
        self.num_special_tokens = num_special_tokens

        # Load normalization statistics
        self._init_normalization(normalization_stat_path)
        self._init_special_normalization(special_stat_path)

    def _init_normalization(self, stat_path: Optional[str]):
        """Load normalization statistics."""
        if stat_path is not None:
            stats = torch.load(stat_path, map_location='cpu')
            self.register_buffer('latent_mean', stats.get('mean', None))
            self.register_buffer('latent_var', stats.get('var', None))
            self.do_normalization = True
            print(f"[RAE_VGGT] Loaded normalization stats from {stat_path}")
        else:
            self.latent_mean = None
            self.latent_var = None
            self.do_normalization = False

    def _init_special_normalization(self, stat_path: Optional[str]):
        """Load special token normalization statistics."""
        if stat_path is not None:
            stats = torch.load(stat_path, map_location='cpu')
            self.register_buffer('special_mean', stats.get('mean', None))
            self.register_buffer('special_var', stats.get('var', None))
            self.do_special_normalization = True
            print(f"[RAE_VGGT] Loaded special token stats from {stat_path}")
        else:
            self.special_mean = None
            self.special_var = None
            self.do_special_normalization = False

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _prepare_input(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        """Prepare input tensor: handle 4D/5D."""
        if x.ndim == 4:
            x = x.unsqueeze(1)
        b, v, c, h, w = x.shape
        return x, b, v

    def _normalize(
        self,
        z: torch.Tensor,
        cls: torch.Tensor = None,
        dim: int = 2048,
    ):
        """raw → latent_norm: zero-mean, unit-var normalization using per-level stats."""
        if not self.do_normalization:
            return (z, cls) if cls is not None else z

        mean = self.latent_mean.to(z.device, z.dtype).reshape(-1)
        var = self.latent_var.to(z.device, z.dtype).reshape(-1)
        std = torch.sqrt(var + self.eps)

        C = dim
        if mean.numel() != C:
            raise ValueError(f"latent_mean len={mean.numel()} != dim={C}")

        def bcast(x, stat):
            ax = [i for i, s in enumerate(x.shape) if s == C]
            if len(ax) != 1:
                raise ValueError(f"Cannot uniquely infer channel axis for C={C} in shape={tuple(x.shape)}")
            shape = [1] * x.dim()
            shape[ax[0]] = C
            return stat.view(*shape)

        z = (z - bcast(z, mean)) / bcast(z, std)

        if cls is not None:
            cls = (cls - bcast(cls, mean)) / bcast(cls, std)
            return z, cls

        return z

    def _denormalize(
        self,
        z: torch.Tensor,
        cls: torch.Tensor = None,
        dim: int = 2048,
    ):
        """latent_norm → raw: reverse latent statistics normalization."""
        if not self.do_normalization:
            return (z, cls) if cls is not None else z

        mean = self.latent_mean.to(z.device, z.dtype).reshape(-1)
        var = self.latent_var.to(z.device, z.dtype).reshape(-1)
        std = torch.sqrt(var + self.eps)

        C = dim
        if mean.numel() != C:
            raise ValueError(f"latent_mean len={mean.numel()} != dim={C}")

        def bcast(x, stat):
            ax = [i for i, s in enumerate(x.shape) if s == C]
            if len(ax) != 1:
                raise ValueError(f"Cannot uniquely infer channel axis for C={C} in shape={tuple(x.shape)}")
            shape = [1] * x.dim()
            shape[ax[0]] = C
            return stat.view(*shape)

        z = z * bcast(z, std) + bcast(z, mean)

        if cls is not None:
            cls = cls * bcast(cls, std) + bcast(cls, mean)
            return z, cls

        return z

    def _separate_cls(self, z: torch.Tensor, grid_size: Optional[Tuple[int, int]] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Separate CLS token from patch features.

        VGGT has no CLS token — special tokens are already stripped by VGGTEncoder.
        Returns (patches, None) always.
        """
        return z, None

    def _reshape_to_2d(self, z: torch.Tensor, grid_size: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        """Reshape (B*V, N, C) -> (B*V, C, H, W)"""
        bv, n, c = z.shape

        if grid_size is not None:
            h, w = grid_size
        elif n == self.num_patches:
            h, w = self.num_patches_per_side
        else:
            h = w = int(sqrt(n))

        return z.transpose(1, 2).reshape(bv, c, h, w)

    def _reshape_to_seq(self, z: torch.Tensor) -> torch.Tensor:
        """Reshape (B*V, C, H, W) -> (B*V, N, C)"""
        bv, c, h, w = z.shape
        return z.reshape(bv, c, h * w).transpose(1, 2)

    def _undo_imagenet_norm(self, x: torch.Tensor) -> torch.Tensor:
        """Undo ImageNet normalization: convert back to [0,1] range.

        VGGT's aggregator applies its own normalization internally,
        so we need to first undo the external ImageNet norm applied by prepare_data.
        """
        if x.ndim == 5:
            b, v, c, h, w = x.shape
            x = x.reshape(b * v, c, h, w)
            x = x * self.encoder_std + self.encoder_mean
            x = x.reshape(b, v, c, h, w)
        else:
            x = x * self.encoder_std + self.encoder_mean
        return x

    # =========================================================================
    # Pack / Unpack (special tokens + patches → sequence format)
    # =========================================================================

    def pack_with_special(self, patches_2d: torch.Tensor, special_tokens: torch.Tensor) -> torch.Tensor:
        """Pack patches and special tokens into sequence format.

        Args:
            patches_2d: (BV, C, h, w) latent-normalized patches
            special_tokens: (BV, K, C) special-normalized special tokens

        Returns:
            (BV, C, K+N, 1) packed tensor
        """
        BV, C, h, w = patches_2d.shape
        patches_flat = patches_2d.reshape(BV, C, h * w)         # (BV, C, N)
        special_t = special_tokens.permute(0, 2, 1)             # (BV, C, K)
        packed = torch.cat([special_t, patches_flat], dim=2)    # (BV, C, K+N)
        return packed.unsqueeze(-1)                              # (BV, C, K+N, 1)

    def unpack_special(self, packed: torch.Tensor, grid_size: Optional[Tuple[int, int]] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Unpack sequence format into patches and special tokens.

        Args:
            packed: (BV, C, K+N, 1) packed tensor
            grid_size: (h, w) patch grid size. If None, assumes square.

        Returns:
            patches_2d: (BV, C, h, w)
            special: (BV, K, C)
        """
        K = self.num_special_tokens
        x = packed.squeeze(-1)                                   # (BV, C, K+N)
        special = x[:, :, :K].permute(0, 2, 1)                  # (BV, K, C)
        patches = x[:, :, K:]                                    # (BV, C, N)
        N = patches.shape[2]
        if grid_size is not None:
            h, w = grid_size
        elif hasattr(self, '_last_grid_size'):
            h, w = self._last_grid_size
        else:
            h = w = int(round(N ** 0.5))
        if h * w != N:
            raise ValueError(f"Cannot unpack {N} patches into grid ({h}x{w}={h*w})")
        patches_2d = patches.reshape(x.shape[0], -1, h, w)      # (BV, C, h, w)
        return patches_2d, special

    def _normalize_special(self, special: torch.Tensor) -> torch.Tensor:
        """Normalize special tokens: raw → special_norm (zero-mean, unit-var).

        Args:
            special: (BV, K, C) raw special tokens

        Returns:
            (BV, K, C) normalized
        """
        if not self.do_special_normalization:
            return special
        mean = self.special_mean.to(special.device, special.dtype).reshape(-1)  # (C,)
        std = torch.sqrt(self.special_var.to(special.device, special.dtype).reshape(-1) + self.eps)
        return (special - mean) / std

    def _denormalize_special(self, special: torch.Tensor) -> torch.Tensor:
        """Denormalize special tokens: special_norm → raw.

        Args:
            special: (BV, K, C) normalized special tokens

        Returns:
            (BV, K, C) raw
        """
        if not self.do_special_normalization:
            return special
        mean = self.special_mean.to(special.device, special.dtype).reshape(-1)
        std = torch.sqrt(self.special_var.to(special.device, special.dtype).reshape(-1) + self.eps)
        return special * std + mean

    # =========================================================================
    # Main Encoding Methods
    # =========================================================================

    def encode(
        self,
        x: torch.Tensor,
        mode: str = 'single',
        return_cls: bool = False,
        level: int = None,
    ) -> Union[torch.Tensor, Dict[int, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        """
        Unified encoding interface.

        Input images are expected to be ImageNet-normalized (as applied by prepare_data).
        We undo that normalization before passing to the VGGT aggregator, which
        applies its own internal normalization.

        Args:
            x: Input images (ImageNet-normalized). 4D (B,C,H,W) or 5D (B,V,C,H,W)
            mode: 'single' or 'all'
            return_cls: If True, also return CLS token (dummy zeros for VGGT)
            level: Override self.level for mode='single'

        Returns: [latent_norm]
            mode='single':
              - num_special_tokens > 0: (B*V, C, K+N, 1) packed sequence [latent_norm]
              - num_special_tokens == 0: (B*V, C, H, W) [latent_norm]
            mode='all': Dict[level_idx → (B*V, N, C)] raw features
        """
        x, b, v = self._prepare_input(x)

        # Undo external ImageNet norm → [0,1] for VGGT
        x = self._undo_imagenet_norm(x)

        if mode == 'all':
            with torch.no_grad():
                return self.encoder.forward(x, mode='all')

        # mode == 'single'
        use_special = self.num_special_tokens > 0

        if use_special:
            z, special = self.encoder.forward(x, mode='single', level=level,
                                               return_special_tokens=True)
        else:
            z = self.encoder.forward(x, mode='single', level=level)
            special = None

        # z: (B*S, N, 2048) — no CLS token
        if z.ndim == 4:
            b_out, v_out, n, c_out = z.shape
            z = z.reshape(b_out * v_out, n, c_out)

        # Add noise during training
        if self.training and self.noise_tau > 0:
            noise_sigma = self.noise_tau * torch.rand(
                (z.size(0),) + (1,) * (z.dim() - 1), device=z.device
            )
            z = z + noise_sigma * torch.randn_like(z)

        # Separate CLS (no-op for VGGT) and reshape
        cls = None
        if self.reshape_to_2d:
            h_p = x.shape[-2] // self.PATCH_SIZE
            w_p = x.shape[-1] // self.PATCH_SIZE
            self._last_grid_size = (h_p, w_p)
            z, cls = self._separate_cls(z, grid_size=(h_p, w_p))
            z = self._reshape_to_2d(z, grid_size=(h_p, w_p))

        # Normalize patches
        if cls is not None:
            z, cls = self._normalize(z, cls)
        else:
            z = self._normalize(z)

        # Pack with special tokens if enabled
        if use_special and special is not None:
            special = self._normalize_special(special)
            z = self.pack_with_special(z, special)
            # z is now (BV, C, K+N, 1)

        if return_cls:
            # Return dummy CLS for interface compatibility
            if cls is None:
                bv = z.shape[0]
                cls = torch.zeros(bv, self.latent_dim, device=z.device, dtype=z.dtype)
            return z, cls
        return z

    def encode_with_cls(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convenience alias for encode(return_cls=True)."""
        return self.encode(x, return_cls=True)

    def encode_all_levels(self, x: torch.Tensor) -> Dict[int, torch.Tensor]:
        """Convenience alias for encode(mode='all')."""
        return self.encode(x, mode='all')

    # =========================================================================
    # Decoding Methods
    # =========================================================================

    def decode(
        self,
        feats: List[Tuple[torch.Tensor, ...]],
        H: int = 518,
        W: int = 518,
        **kwargs,
    ) -> dict:
        """
        Decode multi-level features to RGB + Depth.

        RGB: MAE decoder (4-level concat → 8192-dim → RGB) if available
        Depth: VGGT DPT head (per-level features → depth map)

        Args:
            feats: List of (patches, cls_dummy) tuples, one per level (L0..L3).
                   patches: (B, V, N, 2048) — raw features at that level
                   cls_dummy: (B, V, 2048) — ignored (VGGT has no CLS)
            H, W: Output image size

        Returns:
            dict with 'rgb' (B*V, 3, H, W) or None, 'depth' (B*V, 1, H, W), 'depth_conf'
        """
        B, V = feats[0][0].shape[0], feats[0][0].shape[1]

        # ===== RGB via MAE decoder =====
        rgb = None
        if self.mae_decoder is not None:
            # Concatenate all 4 level features: (B, V, N, 2048) × 4 → (B*V, N, 8192)
            # Strip special tokens if present (different levels may have different N)
            K = self.num_special_tokens
            z_list = []
            for i in range(len(feats)):
                patches = feats[i][0]  # (B, V, N_i, C)
                if K > 0 and patches.shape[2] > feats[-1][0].shape[2]:
                    patches = patches[:, :, K:]  # strip special tokens
                z_list.append(patches)
            z_cat = torch.cat(z_list, dim=-1)  # (B, V, N, 8192)
            BV = B * V
            N = z_cat.shape[2]
            z_flat = z_cat.reshape(BV, N, self._mae_hidden_size)

            with torch.no_grad():
                output = self.mae_decoder(z_flat, input_size=(H, W), drop_cls_token=False).logits
                rgb = self.mae_decoder.unpatchify(output, (H, W))  # (B*V, 3, H, W)
                # Denormalize: ImageNet-normalized → [0, 1]
                rgb = rgb * self.encoder_std + self.encoder_mean

        # ===== Depth via DPT head =====
        depth = None
        depth_conf = None
        if self.depth_head is not None:
            aggregated_tokens_list = self._build_aggregated_tokens_list(feats)
            dummy_images = torch.zeros(B, V, 3, H, W, device=feats[0][0].device, dtype=feats[0][0].dtype)
            psi = self.encoder.aggregator.patch_start_idx  # 5

            with torch.no_grad():
                with torch.autocast(device_type=feats[0][0].device.type, enabled=False):
                    depth, depth_conf = self.depth_head(
                        aggregated_tokens_list,
                        images=dummy_images,
                        patch_start_idx=psi,
                    )

            # DPT head returns depth: (B, S, H, W, 1), depth_conf: (B, S, H, W)
            # Reshape to match DA3 convention: (B*V, 1, H, W)
            if depth.ndim == 5 and depth.shape[-1] == 1:
                depth = depth.squeeze(-1)  # (B, S, H, W)
            if depth.ndim == 4 and depth.shape[0] == B:
                H_d, W_d = depth.shape[-2:]
                depth = depth.reshape(B * V, H_d, W_d).unsqueeze(1)
            if depth_conf is not None:
                if depth_conf.ndim == 4 and depth_conf.shape[0] == B:
                    H_d, W_d = depth_conf.shape[-2:]
                    depth_conf = depth_conf.reshape(B * V, H_d, W_d).unsqueeze(1)

        return {
            'rgb': rgb,
            'depth': depth,
            'depth_conf': depth_conf,
        }

    def _build_aggregated_tokens_list(
        self,
        feats: List[Tuple[torch.Tensor, ...]],
    ) -> List[torch.Tensor]:
        """
        Convert propagated features to the 24-element list expected by VGGT DPTHead.

        DPTHead only reads from intermediate_layer_idx = [4, 11, 17, 23], so we only
        need valid data at those positions. The rest are filled with zeros.

        Each tensor in the list: (B, S, psi + P, 2048)
        where psi = 5 (camera + register tokens)

        Args:
            feats: List of (patches, cls_dummy) per level.
                   patches: (B, V, N, 2048) at each OUT_LAYERS position.
        """
        psi = self.encoder.aggregator.patch_start_idx  # 5
        K = self.num_special_tokens
        device = feats[0][0].device
        dtype = feats[0][0].dtype

        # Determine pure patch count (strip special tokens if present)
        # Use the last level as reference (propagated levels never have special tokens)
        B, V, _, C = feats[0][0].shape
        N_pure = feats[-1][0].shape[2]  # patches only, no special tokens

        # Create dummy special tokens (zeros)
        special = torch.zeros(B, V, psi, C, device=device, dtype=dtype)

        # Build 24-element list
        aggregated_tokens_list = []
        feat_map = {}
        for lvl_idx, layer_idx in enumerate(self.OUT_LAYERS):
            if lvl_idx < len(feats):
                patches = feats[lvl_idx][0]
                # Strip special tokens if this level has them
                if K > 0 and patches.shape[2] > N_pure:
                    patches = patches[:, :, K:]
                feat_map[layer_idx] = patches

        for i in range(24):
            if i in feat_map:
                patches = feat_map[i]  # (B, V, N_pure, 2048)
                tokens = torch.cat([special, patches], dim=2)  # (B, V, psi+N, 2048)
            else:
                tokens = torch.zeros(B, V, psi + N_pure, C, device=device, dtype=dtype)
            aggregated_tokens_list.append(tokens)

        return aggregated_tokens_list

    # =========================================================================
    # Feature Replacement
    # =========================================================================

    def replace_level_features(
        self,
        gt_feats: Dict[int, torch.Tensor],
        generated_latent: torch.Tensor,
        level: int = None,
        view_mask: torch.Tensor = None,
        batch_size: int = None,
        total_view: int = None,
    ) -> Dict[int, torch.Tensor]:
        """Replace features at a specific level with generated latent."""
        if level is None:
            level = self.level

        feats = {k: v.clone() for k, v in gt_feats.items()}
        old_feat = feats[level]  # (B*V, N, C)

        if generated_latent.dim() == 4:  # (B*V, C, H, W)
            bv, c, h, w = generated_latent.shape
            new_feat = generated_latent.reshape(bv, c, h * w).transpose(1, 2)
        elif generated_latent.dim() == 3:
            new_feat = generated_latent
        else:
            new_feat = generated_latent

        # No CLS token handling needed for VGGT

        if view_mask is not None:
            if view_mask.dim() == 2:
                view_mask = view_mask.reshape(-1)
            mask_expanded = view_mask.reshape(-1, 1, 1).expand_as(new_feat)
            result_feat = torch.where(mask_expanded, new_feat, old_feat)
        else:
            result_feat = new_feat

        feats[level] = result_feat
        return feats

    # =========================================================================
    # Feature Propagation
    # =========================================================================

    def propagate_features(
        self,
        features: torch.Tensor,
        from_level: int,
        total_view: int = 1,
        cls_token: torch.Tensor = None,
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Propagate shallow features to deeper levels via VGGT aggregator forward.

        Norm flow: latent_norm → _denormalize → raw → aggregator forward → raw patches

        Args:
            features: [latent_norm] Shallow level features.
                      Packed format: (B*V, C, K+N, 1) with special tokens
                      Spatial format: (B*V, C, H, W) without special tokens
            from_level: Starting level (0,1,2,3)
            total_view: Number of views for multi-view reshape
            cls_token: Ignored for VGGT (no CLS token)

        Returns:
            List of (patches_4d, cls_3d) tuples for each level >= from_level
            patches_4d: (B, V, N, C) — raw features at each level
            cls_3d: (B, V, C) — dummy zeros
        """
        layer_idx = self.OUT_LAYERS[from_level]

        # Detect packed format (BV, C, K+N, 1) vs spatial (BV, C, h, w)
        is_packed = (self.num_special_tokens > 0 and
                     features.ndim == 4 and features.shape[3] == 1)

        if is_packed:
            patches_2d, special_norm = self.unpack_special(features)
            # patches_2d: (BV, C, h, w) [latent_norm]
            # special_norm: (BV, K, C) [special_norm]

            b_tot, c, h, w = patches_2d.shape
            B = b_tot // total_view
            V = total_view

            # Denormalize patches
            features_seq = self._reshape_to_seq(patches_2d)  # (BV, N, C)
            features_4d = features_seq.reshape(B, V, -1, c)
            features_4d = self._denormalize(features_4d)
            features_flat = features_4d.reshape(B * V, -1, c)

            # Denormalize special tokens
            special_raw = self._denormalize_special(special_norm)  # (BV, K, C)
            # Reshape to (B, V, K, C) for encoder
            K = self.num_special_tokens
            cached_special = special_raw.reshape(B, V, K, c)

            # Run encoder propagation with cached special tokens
            raw_results = self.encoder.forward(
                features_flat,
                mode='from_layer',
                start_layer_idx=layer_idx,
                total_view=total_view,
                grid_size=(h, w),
                cached_special_tokens=cached_special,
            )
        else:
            b_tot, c, h, w = features.shape
            B = b_tot // total_view
            V = total_view

            # Convert to sequence format: (B*V, N, C)
            features_seq = self._reshape_to_seq(features)

            # Reshape and denormalize
            features_4d = features_seq.reshape(B, V, -1, c)
            features_4d = self._denormalize(features_4d)

            # Reshape back to (B*V, N, C) for encoder
            features_flat = features_4d.reshape(B * V, -1, c)

            # Run encoder propagation
            raw_results = self.encoder.forward(
                features_flat,
                mode='from_layer',
                start_layer_idx=layer_idx,
                total_view=total_view,
                grid_size=(h, w),
            )

        # Reshape outputs: (cls_dummy, patches) → (patches_4d, cls_3d)
        final_results = []
        for (cls_dummy, patches) in raw_results:
            b_tot_out = cls_dummy.shape[0]
            batch_size = b_tot_out // total_view
            n, c_out = patches.shape[1], patches.shape[2]

            patches_4d = patches.reshape(batch_size, total_view, n, c_out)
            cls_3d = cls_dummy.reshape(batch_size, total_view, -1)

            final_results.append((patches_4d, cls_3d))

        return final_results
