"""VGGT (Visual Geometry Grounded Transformer) encoder for multi-level feature extraction.

Wraps the VGGT Aggregator to provide the same 3-mode interface as DA3EncoderDirect:
  - 'single': Extract features at a single level (for training)
  - 'all': Extract features at ALL 4 levels in one pass (for stats computation)
  - 'from_layer': Continue forward from intermediate features (for propagation)
"""

import torch
import torch.nn as nn
from typing import Optional, List, Tuple, Dict, Union
from torch.utils.checkpoint import checkpoint

from . import register_encoder

# VGGT imports (requires vggt on PYTHONPATH)
import sys
# sys.path should be set via PYTHONPATH environment variable
from vggt.models.vggt import VGGT
from vggt.models.aggregator import slice_expand_and_flatten


@register_encoder()
class VGGTEncoder(nn.Module):
    """
    VGGT encoder that uses the Aggregator for multi-level feature extraction.

    VGGT's Aggregator alternates frame-level and global (cross-frame) attention
    over 24 block pairs, producing 24 intermediate outputs. Each output is
    a concatenation of frame + global features (2 × embed_dim = 2048).

    Feature hierarchy (aligned with DPT head):
        Level 0: Block pair  4 (shallowest)
        Level 1: Block pair 11
        Level 2: Block pair 17
        Level 3: Block pair 23 (deepest)

    Special tokens (excluded from patch features):
        Position 0: Camera token (1 per frame)
        Positions 1-4: Register tokens (4 per frame)
        patch_start_idx = 5

    Args:
        pretrained_path: HuggingFace path for VGGT.from_pretrained()
        level: Default feature level to extract (0,1,2,3)
    """

    OUT_LAYERS = [4, 11, 17, 23]  # Aggregator block-pair indices
    PATCH_SIZE = 14
    hidden_size = 2048   # 2 × embed_dim (1024 frame + 1024 global)
    patch_size = 14

    def __init__(
        self,
        pretrained_path: str = 'facebook/VGGT-1B',
        level: int = -1,
    ):
        super().__init__()

        assert level in [-1, -2, -3, -4, 0, 1, 2, 3], \
            f"Level must be 0,1,2,3 or -1,-2,-3,-4, got {level}"

        self.level = level

        # Load full VGGT, extract aggregator, discard heads
        print(f"[VGGTEncoder] Loading VGGT from {pretrained_path} ...")
        vggt_model = VGGT.from_pretrained(pretrained_path)
        self.aggregator = vggt_model.aggregator

        # Store depth head for later use in RAE_VGGT.decode()
        self.depth_head = vggt_model.depth_head

        # Freeze everything
        self.aggregator.eval()
        for param in self.aggregator.parameters():
            param.requires_grad = False
        if self.depth_head is not None:
            self.depth_head.eval()
            for param in self.depth_head.parameters():
                param.requires_grad = False

        # Discard other heads to save memory
        del vggt_model
        print(f"[VGGTEncoder] Loaded. patch_start_idx={self.aggregator.patch_start_idx}")

    # ------------------------------------------------------------------
    # Unified forward
    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        mode: str = 'single',
        level: int = None,
        start_layer_idx: int = None,
        total_view: int = 1,
        grid_size: Tuple[int, int] = None,
        return_special_tokens: bool = False,
        cached_special_tokens: torch.Tensor = None,
    ) -> Union[torch.Tensor, Dict[int, torch.Tensor], List[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Unified forward pass with multiple modes.

        Args:
            x: Input tensor.
               - 'single'/'all': Images [0,1] range. (B,C,H,W) or (B,S,C,H,W)
               - 'from_layer': Intermediate patch features (B*S,N,2048) or (B,S,N,2048)
            mode: 'single' | 'all' | 'from_layer'
            level: Override self.level for 'single' mode
            start_layer_idx: Required for 'from_layer' — which OUT_LAYERS index produced x
            total_view: Number of views (S) for reshaping in 'from_layer'
            grid_size: (h_patches, w_patches) for 'from_layer'
            return_special_tokens: If True, also return special tokens (camera + register).
                - 'single': returns (patches, special) where special is (B*S, psi, 2048)
                - 'all': returns Dict[level_idx → (patches, special)]
            cached_special_tokens: (B, S, psi, 2048) for 'from_layer' mode

        Returns:
            'single': (B*S, N, 2048) patch tokens only (no special tokens)
                      or ((B*S, N, 2048), (B*S, psi, 2048)) if return_special_tokens
            'all': Dict[level_idx → (B*S, N, 2048)]
                   or Dict[level_idx → ((B*S, N, 2048), (B*S, psi, 2048))] if return_special_tokens
            'from_layer': List[(cls_dummy, patches)] for each OUT_LAYER >= start
        """
        if mode == 'single':
            return self._forward_single(x, level, return_special_tokens=return_special_tokens)
        elif mode == 'all':
            return self._forward_all(x, return_special_tokens=return_special_tokens)
        elif mode == 'from_layer':
            if start_layer_idx is None:
                raise ValueError("start_layer_idx required for 'from_layer' mode")
            return self._forward_from_layer(
                x, start_layer_idx, total_view,
                grid_size=grid_size,
                cached_special_tokens=cached_special_tokens,
            )
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'single', 'all', or 'from_layer'")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _prepare_input(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        """Ensure input is 5D (B,S,C,H,W) and return B,S."""
        if x.ndim == 4:
            x = x.unsqueeze(1)
        B, S = x.shape[:2]
        return x, B, S

    def _run_aggregator(self, images: torch.Tensor) -> Tuple[List[torch.Tensor], int]:
        """
        Run the full VGGT aggregator.

        Args:
            images: (B, S, 3, H, W) in [0,1]

        Returns:
            output_list: List of 24 tensors, each (B, S, patch_start_idx+P, 2048)
            patch_start_idx: 5
        """
        return self.aggregator(images)

    def _forward_single(self, x: torch.Tensor, level: int = None,
                         return_special_tokens: bool = False):
        """Extract features at a single level. Returns (B*S, N, 2048).

        Also caches the special token states at the target layer as
        `self._cached_special_tokens`, which `_forward_from_layer` uses
        to avoid divergence from the full forward pass.

        If return_special_tokens=True, returns (patches, special) where:
            patches: (B*S, N, 2048)
            special: (B*S, psi, 2048)
        """
        x, B, S = self._prepare_input(x)

        level_idx = level if level is not None else self.level
        target_layer = self.OUT_LAYERS[level_idx]

        with torch.no_grad():
            output_list, psi = self._run_aggregator(x)

        # output_list[target_layer]: (B, S, psi+P, 2048)
        full_feats = output_list[target_layer]  # (B, S, psi+P, 2048)

        # Cache special token states (positions 0..psi-1) at this layer
        # These contain the evolved camera/register tokens after attention blocks.
        # Shape: (B, S, psi, 2048) — the global half (last 1024) is the current state.
        self._cached_special_tokens = full_feats[:, :, :psi].detach()

        feats = full_feats[:, :, psi:]  # strip special tokens → (B, S, P, 2048)
        feats = feats.reshape(B * S, -1, feats.shape[-1])  # (B*S, P, 2048)

        if return_special_tokens:
            special = full_feats[:, :, :psi]  # (B, S, psi, 2048)
            special_flat = special.reshape(B * S, psi, -1)  # (B*S, psi, 2048)
            return feats, special_flat

        return feats

    def _forward_all(self, x: torch.Tensor,
                     return_special_tokens: bool = False) -> Dict[int, torch.Tensor]:
        """Extract features at all 4 levels.

        Returns:
            Dict[level_idx → (B*S, N, 2048)]
            or Dict[level_idx → ((B*S, N, 2048), (B*S, psi, 2048))] if return_special_tokens
        """
        x, B, S = self._prepare_input(x)

        with torch.no_grad():
            output_list, psi = self._run_aggregator(x)

        level_results = {}
        for lvl, layer_idx in enumerate(self.OUT_LAYERS):
            full = output_list[layer_idx]  # (B, S, psi+P, 2048)
            feats = full[:, :, psi:]  # (B, S, P, 2048)
            feats = feats.reshape(B * S, -1, feats.shape[-1])  # (B*S, P, 2048)

            if return_special_tokens:
                special = full[:, :, :psi]  # (B, S, psi, 2048)
                special_flat = special.reshape(B * S, psi, -1)  # (B*S, psi, 2048)
                level_results[lvl] = (feats, special_flat)
            else:
                level_results[lvl] = feats

        return level_results

    def _forward_from_layer(
        self,
        x: torch.Tensor,
        start_layer_idx: int,
        total_view: int = 1,
        grid_size: Tuple[int, int] = None,
        cached_special_tokens: torch.Tensor = None,
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Continue forward from intermediate features through remaining aggregator blocks.

        This resumes the alternating frame/global attention from the block *after*
        start_layer_idx, collecting outputs at each subsequent OUT_LAYER.

        Args:
            x: Patch features (B*S, N, 2048) or (B, S, N, 2048) — no special tokens.
               These are the concatenated [frame, global] features from the given layer.
            start_layer_idx: Which OUT_LAYERS *block-pair index* produced x (e.g. 4,11,17,23).
            total_view: S
            grid_size: (h_patches, w_patches)
            cached_special_tokens: (B, S, psi, 2048) — evolved special token states from
                the encoding pass. If None, falls back to self._cached_special_tokens,
                then to initial learned parameters (which causes divergence).

        Returns:
            List of (cls_dummy, patches) tuples for each OUT_LAYER >= start_layer_idx.
            cls_dummy: (B*S, 2048) zeros (VGGT has no CLS token)
            patches: (B*S, N, 2048) features at that layer
        """
        # Reshape to (B, S, N, 2048)
        if x.ndim == 3:
            B_total, N, C = x.shape
            S = total_view
            B = B_total // S
            x = x.view(B, S, N, C)
        else:
            B, S, N, C = x.shape

        embed_dim = C // 2  # 1024

        # Infer spatial layout
        if grid_size is not None:
            h_pat, w_pat = grid_size
        else:
            side = int(N ** 0.5)
            if side * side != N:
                raise ValueError(f"Number of patches {N} is not a perfect square. Provide grid_size.")
            h_pat = w_pat = side

        H = h_pat * self.PATCH_SIZE
        W = w_pat * self.PATCH_SIZE
        psi = self.aggregator.patch_start_idx  # 5

        # Split into frame and global halves
        frame_tokens = x[..., :embed_dim]   # (B, S, N, 1024)
        global_tokens = x[..., embed_dim:]  # (B, S, N, 1024)

        # Resolve special tokens: prefer cached (evolved) > self._cached > learned init
        if cached_special_tokens is not None:
            # Use explicitly provided cached special tokens (2048-dim → take global half)
            special_tokens = cached_special_tokens[..., embed_dim:]  # (B, S, psi, 1024)
        elif hasattr(self, '_cached_special_tokens') and self._cached_special_tokens is not None:
            # Use auto-cached from last _forward_single call
            special_tokens = self._cached_special_tokens.to(x.device, x.dtype)[..., embed_dim:]
        else:
            # Fallback: initial learned parameters (will cause divergence at deeper layers)
            camera_tok = slice_expand_and_flatten(self.aggregator.camera_token, B, S)
            register_tok = slice_expand_and_flatten(self.aggregator.register_token, B, S)
            special_tokens = torch.cat([camera_tok, register_tok], dim=1)
            special_tokens = special_tokens.view(B, S, psi, embed_dim)

        P = psi + N  # total tokens per frame

        # Prepare RoPE positions
        pos = None
        if self.aggregator.rope is not None:
            pos = self.aggregator.position_getter(
                B * S, h_pat, w_pat, device=x.device
            )
            if psi > 0:
                pos = pos + 1
                pos_special = torch.zeros(B * S, psi, 2, device=x.device, dtype=pos.dtype)
                pos = torch.cat([pos_special, pos], dim=1)

        # Collect the start layer output first
        results = {}
        if start_layer_idx in self.OUT_LAYERS:
            # Reconstruct the original 2048-dim output from the input
            feats_combined = torch.cat([frame_tokens, global_tokens], dim=-1)  # (B, S, N, 2048)
            patches_flat = feats_combined.reshape(B * S, N, C)
            cls_dummy = torch.zeros(B * S, C, device=x.device, dtype=x.dtype)
            results[start_layer_idx] = (cls_dummy, patches_flat)

        # Resume from block after start_layer_idx
        start_block = start_layer_idx + 1

        # CRITICAL: VGGT aggregator uses a SINGLE token stream.
        # At each block pair i:
        #   frame_blocks[i] operates on tokens → frame_intermediate
        #   global_blocks[i] operates on the SAME (now modified) tokens → global_intermediate
        #   output = cat(frame_intermediate, global_intermediate)
        #
        # The global_intermediate is the most recent state (after both blocks),
        # so we use the global half of the input as the current token state.
        tokens = torch.cat([special_tokens, global_tokens], dim=2)  # (B, S, P, 1024)

        # Pre-compute global-shaped pos
        g_pos = pos.view(B, S * P, 2) if pos is not None else None

        for block_idx in range(start_block, self.aggregator.depth):
            # Frame attention: per-frame (B*S, P, embed_dim)
            tokens_frame = tokens.reshape(B * S, P, embed_dim)

            if self.aggregator.training:
                tokens_frame = checkpoint(
                    self.aggregator.frame_blocks[block_idx], tokens_frame, pos,
                    use_reentrant=self.aggregator.use_reentrant
                )
            else:
                tokens_frame = self.aggregator.frame_blocks[block_idx](tokens_frame, pos=pos)

            # Save frame intermediate (after frame attention only)
            frame_intermediate = tokens_frame.view(B, S, P, embed_dim)

            # Global attention: cross-frame (B, S*P, embed_dim)
            # Operates on the SAME tokens that frame_blocks just modified
            tokens_global = tokens_frame.view(B, S * P, embed_dim)

            if self.aggregator.training:
                tokens_global = checkpoint(
                    self.aggregator.global_blocks[block_idx], tokens_global, g_pos,
                    use_reentrant=self.aggregator.use_reentrant
                )
            else:
                tokens_global = self.aggregator.global_blocks[block_idx](tokens_global, pos=g_pos)

            # Update tokens for next iteration (global output is the latest state)
            tokens = tokens_global.view(B, S, P, embed_dim)

            # Collect at OUT_LAYERS: cat(frame_intermediate, global_intermediate)
            if block_idx in self.OUT_LAYERS:
                global_intermediate = tokens  # (B, S, P, embed_dim)
                combined = torch.cat([
                    frame_intermediate[:, :, psi:],   # strip special tokens
                    global_intermediate[:, :, psi:],
                ], dim=-1)  # (B, S, N, 2048)
                patches_flat = combined.reshape(B * S, N, C)
                cls_dummy = torch.zeros(B * S, C, device=x.device, dtype=x.dtype)
                results[block_idx] = (cls_dummy, patches_flat)

        # Return ordered by OUT_LAYERS
        return [results[layer] for layer in self.OUT_LAYERS if layer in results]

    # ------------------------------------------------------------------
    # Convenience aliases
    # ------------------------------------------------------------------
    def forward_all_levels(self, x: torch.Tensor) -> Dict[int, torch.Tensor]:
        """Alias for forward(mode='all')."""
        return self.forward(x, mode='all')

    def forward_from_layer(
        self,
        x: torch.Tensor,
        start_layer_idx: int,
        total_view: int = 1,
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Alias for forward(mode='from_layer')."""
        return self.forward(
            x, mode='from_layer',
            start_layer_idx=start_layer_idx,
            total_view=total_view,
        )
