#!/usr/bin/env python3
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
GLD Novel View Synthesis Evaluation Script

Config-driven evaluation with level-specific metrics:
- Level 0: PSNR, SSIM, LPIPS (decoded RGB) + MSE (latent)
- Levels 1-3: MSE only (latent space)

Usage:
    python src/eval_nvs_gld.py \
        --eval_config configs/eval/dl3dv.yaml \
        --checkpoint path/to/checkpoint.pt \
        --difficulty medium \
        --output_dir results/nvs_eval
"""

import os
import sys
import argparse
import json
import numpy as np

# Fix xformers compatibility with older PyTorch
import torch.backends.cuda
if not hasattr(torch.backends.cuda, 'is_flash_attention_available'):
    torch.backends.cuda.is_flash_attention_available = lambda: False

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from omegaconf import OmegaConf
from tqdm import tqdm
from datetime import datetime
from einops import rearrange

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from stage1 import RAE_DA3
from stage2.models import Stage2ModelProtocol
from stage2.transport.transport import Transport, Sampler, ModelType, WeightType, PathType
from utils.model_utils import instantiate_from_config
from utils.config_utils import init_config_defaults
from utils.camera.camera import get_camera_embedding
from utils.metrics import compute_psnr, compute_ssim, compute_lpips, compute_mse, compute_abs_rel, compute_depth_rmse, compute_delta


def set_seed(seed):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)


def create_transport(
    path_type='Linear',
    prediction="velocity",
    loss_weight=None,
    train_eps=None,
    sample_eps=None,
    time_dist_type="uniform",
    time_dist_shift=1.0,
):
    """Create Transport object."""
    if prediction == "noise":
        model_type = ModelType.NOISE
    elif prediction == "score":
        model_type = ModelType.SCORE
    else:
        model_type = ModelType.VELOCITY

    if loss_weight == "velocity":
        loss_type = WeightType.VELOCITY
    elif loss_weight == "likelihood":
        loss_type = WeightType.LIKELIHOOD
    else:
        loss_type = WeightType.NONE

    path_choice = {
        "Linear": PathType.LINEAR,
        "GVP": PathType.GVP,
        "VP": PathType.VP,
    }
    path_type = path_choice[path_type]

    if path_type in [PathType.VP]:
        train_eps = 1e-5 if train_eps is None else train_eps
        sample_eps = 1e-3 if sample_eps is None else sample_eps
    elif path_type in [PathType.GVP, PathType.LINEAR] and model_type != ModelType.VELOCITY:
        train_eps = 1e-3 if train_eps is None else train_eps
        sample_eps = 1e-3 if sample_eps is None else sample_eps
    else:
        train_eps = 0
        sample_eps = 0
    
    return Transport(
        model_type=model_type,
        path_type=path_type,
        loss_type=loss_type,
        train_eps=train_eps,
        sample_eps=sample_eps,
        time_dist_type=time_dist_type,
        time_dist_shift=time_dist_shift,
    )


def save_visualization(gt_images, pred_images, cond_num, save_path):
    """
    Save visualization: GT row on top, prediction row on bottom.
    
    Args:
        gt_images: (V, 3, H, W) ground truth in [0, 1]
        pred_images: (V, 3, H, W) predictions in [0, 1]
        cond_num: number of conditioning views
        save_path: path to save visualization
    """
    import cv2
    V, _, H, W = gt_images.shape
    
    # Convert to numpy (H, W, 3)
    gt_np = gt_images.permute(0, 2, 3, 1).cpu().numpy()
    pred_np = pred_images.permute(0, 2, 3, 1).cpu().numpy()
    
    # Stack horizontally
    gt_row = np.concatenate([gt_np[i] for i in range(V)], axis=1)
    pred_row = np.concatenate([pred_np[i] for i in range(V)], axis=1)
    
    # Stack vertically
    vis = np.concatenate([gt_row, pred_row], axis=0)
    
    # Save
    vis_bgr = (np.clip(vis, 0, 1) * 255).astype(np.uint8)[..., ::-1]
    cv2.imwrite(save_path, vis_bgr)


def save_depth_visualization(gt_depth, pred_depth, cond_num, save_path):
    """
    Save depth visualization: Pseudo-GT row on top, prediction row on bottom.
    Uses viridis colormap for visualization.
    
    Args:
        gt_depth: (V, 1, H, W) or (V, H, W) pseudo-GT depth
        pred_depth: (V, 1, H, W) or (V, H, W) predicted depth
        cond_num: number of conditioning views
        save_path: path to save visualization
    """
    import cv2
    import matplotlib.pyplot as plt
    
    # Handle shape - squeeze channel dim if present
    if gt_depth.dim() == 4 and gt_depth.shape[1] == 1:
        gt_depth = gt_depth.squeeze(1)
    if pred_depth.dim() == 4 and pred_depth.shape[1] == 1:
        pred_depth = pred_depth.squeeze(1)
    
    V, H, W = gt_depth.shape
    
    # Convert to numpy
    gt_np = gt_depth.cpu().numpy()
    pred_np = pred_depth.cpu().numpy()
    
    # Normalize depth values globally for consistent coloring
    all_depth = np.concatenate([gt_np.flatten(), pred_np.flatten()])
    vmin = np.percentile(all_depth[all_depth > 0], 2) if (all_depth > 0).any() else 0
    vmax = np.percentile(all_depth[all_depth > 0], 98) if (all_depth > 0).any() else 1
    
    # Apply colormap
    cmap = plt.get_cmap('viridis')
    
    gt_colored = []
    pred_colored = []
    for i in range(V):
        # Normalize
        gt_norm = (gt_np[i] - vmin) / (vmax - vmin + 1e-6)
        pred_norm = (pred_np[i] - vmin) / (vmax - vmin + 1e-6)
        gt_norm = np.clip(gt_norm, 0, 1)
        pred_norm = np.clip(pred_norm, 0, 1)
        
        # Apply colormap (returns RGBA, take RGB)
        gt_colored.append(cmap(gt_norm)[..., :3])
        pred_colored.append(cmap(pred_norm)[..., :3])
    
    # Stack horizontally
    gt_row = np.concatenate(gt_colored, axis=1)
    pred_row = np.concatenate(pred_colored, axis=1)
    
    # Stack vertically
    vis = np.concatenate([gt_row, pred_row], axis=0)
    
    # Save (RGB to BGR for cv2)
    vis_bgr = (np.clip(vis, 0, 1) * 255).astype(np.uint8)[..., ::-1]
    cv2.imwrite(save_path, vis_bgr)


def get_cascade_features(
    rae,
    cascade_model,
    sampler,
    source_features,      # [latent_norm] L1 features, shape (B*V, C, h, w)
    source_stat_path,     # L1 normalization stats
    target_stat_path,     # L0 normalization stats
    batch,
    device,
    total_view,
    cond_num,
    camera_mode='plucker',
    use_prope=False,
    cfg_scale=None,
    use_camera_drop=True,
    cfg_uncond_mode='keep',
    noise_tau=0.0,
    prope_image_size=None,
    eval_mode='cascade',
    ray_pose_mode='c2w',
):
    """
    Feature Flow: L1 → L0 via diffusion sampling.

    Args:
        source_features: [latent_norm] L1 features (BV, C, h, w)
    Returns:
        [latent_norm] L0 features (BV, C, h, w)
    """
    from utils.camera.camera import get_camera_embedding
    
    if eval_mode == 'cascade':
        print(f"[Feature Flow] L1 → L0 (flow), noise_tau={noise_tau}")
    else:
        print(f"[Feature Flow] L1 → L0 (propagation/pure-noise)")
        
    cascade_model.eval()
    rae.eval()
    
    # Extract batch data
    if all([key in batch for key in ['gt_inp', 'fxfycxcy', 'c2w']]):
        images = batch['gt_inp'].to(device)
        intrinsic = batch['fxfycxcy'].to(device)
        extrinsic = batch['c2w'].to(device)
    elif all([key in batch for key in ['image', 'c2w', 'intrinsic']]):
        images = batch['image'].to(device)
        intrinsic = batch['intrinsic'].to(device)
        extrinsic = batch['c2w'].to(device)
    else:
        raise ValueError("Batch must contain required keys")
    
    B, V, C, H, W = images.shape
    images_norm = (images - rae.encoder_mean[None]) / rae.encoder_std[None]
    
    # Get L0 ref features for conditioning
    original_mean = rae.latent_mean
    original_var = rae.latent_var
    original_do_norm = rae.do_normalization
    
    target_stats = torch.load(target_stat_path, map_location=device)
    rae.latent_mean = target_stats.get('mean', None)
    rae.latent_var = target_stats.get('var', None)
    if rae.latent_mean is not None:
        rae.latent_mean = rae.latent_mean.to(device)
    if rae.latent_var is not None:
        rae.latent_var = rae.latent_var.to(device)
    rae.do_normalization = True
    rae.level = 0
    
    with torch.no_grad():
        latents_ref_l0 = rae.encode(images_norm[:, :cond_num], level=0)
    
    rae.latent_mean = original_mean
    rae.latent_var = original_var
    rae.do_normalization = original_do_norm
    
    # Prepare Camera Embedding
    if extrinsic.shape[-2:] == (4, 4):
        extrinsic = extrinsic[..., :3, :4]
    
    if intrinsic.shape[-1] == 4:
        fx, fy, cx, cy = intrinsic.unbind(dim=-1)
        zeros = torch.zeros_like(fx)
        ones = torch.ones_like(fx)
        intrinsic_mat = torch.stack([
            torch.stack((fx, zeros, cx), dim=-1),
            torch.stack((zeros, fy, cy), dim=-1),
            torch.stack((zeros, zeros, ones), dim=-1),
        ], dim=-2)
    else:
        intrinsic_mat = intrinsic
    
    extri_ = rearrange(extrinsic, "b v c1 c2 -> (b v) c1 c2")
    intri_ = rearrange(intrinsic_mat, "b v c1 c2 -> (b v) c1 c2")
    
    if use_prope:
        camera_embedding, scale = get_camera_embedding(intri_, extri_, B, V, H, W, mode=camera_mode, ray_pose_mode=ray_pose_mode, return_scale=True)
    else:
        camera_embedding = get_camera_embedding(intri_, extri_, B, V, H, W, mode=camera_mode, ray_pose_mode=ray_pose_mode, return_scale=False)
    
    camera_embedding = rearrange(camera_embedding, "b v c h w -> (b v) c h w")
    
    random_masks = torch.ones((B, V, 1, H, W), device=device, dtype=source_features.dtype)
    random_masks[:, :cond_num] = 0
    random_masks = random_masks.reshape(B * V, 1, H, W)
    camera_embedding = torch.cat([random_masks, camera_embedding], dim=1)
    
    # Prepare Concat Mode Input
    _, C_lat, h_lat, w_lat = source_features.shape
    
    if eval_mode == 'cascade':
        # Mode: Noise-to-Feat (Generation from pure noise)
        sample_xT = torch.randn(B, V, C_lat, h_lat, w_lat, device=device, dtype=source_features.dtype)
    else:
        # Mode: Feat-to-Feat (Transformation from L1 + noise)
        source_5d = source_features.reshape(B, V, C_lat, h_lat, w_lat)
        if noise_tau > 0:
            noise_std = torch.abs(torch.randn(1, device=device) * noise_tau).item()
        else:
            noise_std = 0.0
        noise = torch.randn_like(source_5d) * noise_std
        sample_xT = source_5d + noise
    
    latents_ref_5d = latents_ref_l0.reshape(B, cond_num, C_lat, h_lat, w_lat)
    cond_channel = torch.zeros(B, V, C_lat, h_lat, w_lat, device=device, dtype=source_features.dtype)
    cond_channel[:, :cond_num] = latents_ref_5d
    
    # [condition | xT]
    sample_input = torch.cat([cond_channel, sample_xT], dim=2)
    sample_input_flat = sample_input.reshape(B * V, C_lat * 2, h_lat, w_lat)
    
    model_kwargs = dict(
        camera_embedding=camera_embedding,
        total_view=total_view,
        cond_num=cond_num,
        is_concat_mode=True,
        ref_cond=cond_channel.reshape(B * V, C_lat, h_lat, w_lat),
        source_condition=source_features if eval_mode == 'cascade' else None,  # Pass L1 features only for l1_as_cond mode
        prope_image_size=prope_image_size if prope_image_size is not None else (H, W),
    )
    
    if use_prope:
        with torch.no_grad():
            Ks = intrinsic_mat
            c2w = extrinsic
            last_row = torch.zeros(c2w.shape[:-2] + (1, 4), device=device)
            last_row[..., 3] = 1.0
            c2w_4x4 = torch.cat([c2w, last_row], dim=-2)
            w2c = torch.linalg.inv(c2w_4x4)
            
            if scale is not None:
                w2c = w2c.clone()
                w2c[..., :3, 3] = w2c[..., :3, 3] * scale
            
            model_kwargs['viewmats'] = w2c
            model_kwargs['Ks'] = Ks
    
    if cfg_scale is not None and cfg_scale > 1.0:
        model_kwargs['cfg_scale'] = cfg_scale
        model_kwargs['use_camera_drop'] = use_camera_drop  # NEW
        model_kwargs['uncond_mode'] = cfg_uncond_mode  # NEW
    
    with torch.no_grad():
        samples = sampler(sample_input_flat, cascade_model, **model_kwargs)[-1]
    
    samples = samples[:, C_lat:]
    print(f"[Feature Flow] Output L0 features: mean={samples.mean():.4f}, std={samples.std():.4f}")
    
    return samples


def get_propagated_features(
    rae,
    source_features,   # [latent_norm] L1 features, shape (B*V, C, h, w)
    stat_path,         # L1 normalization stats path
    batch,
    device,
    total_view,
    cond_num,
):
    """
    Propagate L1 features to L2, L3 using DA3 backbone's forward propagation.

    Args:
        source_features: [latent_norm] L1 features (BV, C, h, w)
    Returns:
        ⚠️ [layer_norm] dict {level_idx: (BV, C, h, w)} — NOT raw!
        Returned features have DINOv2 LayerNorm already applied.
        Do NOT pass to apply_da3_norm() again.
    """
    print(f"[Propagation] L1 → L2, L3")
    rae.eval()
    
    if 'image' in batch:
        images = batch['image'].to(device)
    elif 'gt_inp' in batch:
        images = batch['gt_inp'].to(device)
    else:
        raise ValueError("batch must contain 'image' or 'gt_inp'")
    
    B, V, C, H, W = images.shape
    images_norm = (images - rae.encoder_mean[None]) / rae.encoder_std[None]
    
    rae.level = 1
    rae._init_normalization(stat_path=stat_path)
    
    with torch.no_grad():
        ref_images_norm = images_norm[:, :cond_num].reshape(B * cond_num, C, H, W)
        _, ref_gt_cls = rae.encode(ref_images_norm, return_cls=True, mode='single', level=1)

    merged_cls = ref_gt_cls.reshape(B, cond_num, -1)[:, :1].expand(-1, V, -1).reshape(B * V, -1)
    # merged_cls = None
    with torch.no_grad():
        propagated_feats = rae.propagate_features(
            source_features,
            from_level=1,
            total_view=total_view,
            cls_token=merged_cls,
        )
    
    _, _, h_lat, w_lat = source_features.shape
    
    result = {}
    for i, (patches, cls_token) in enumerate(propagated_feats):
        level_idx = 1 + i
        B_p, V_p, N, C_feat = patches.shape
        feat_4d = patches.permute(0, 1, 3, 2)
        feat_5d = feat_4d.reshape(B_p, V_p, C_feat, h_lat, w_lat)
        feat_flat = feat_5d.reshape(B_p * V_p, C_feat, h_lat, w_lat)
        result[level_idx] = feat_flat
        print(f"[Propagation] Level {level_idx}: mean={feat_flat.mean():.4f}, std={feat_flat.std():.4f}")
    
    return result


def main(args):
    # stat_path will be set dynamically after loading model config (DA3 vs VGGT)
    stat_path = None  # placeholder, set below

    level = args.level if hasattr(args, 'level') else 0
    # Load eval config
    eval_cfg = OmegaConf.load(args.eval_config)
    print(f"Loaded eval config: {args.eval_config}")
    
    # Apply difficulty settings
    difficulty = args.difficulty or eval_cfg.get('default_difficulty', 'medium')
    if 'difficulties' in eval_cfg and difficulty in eval_cfg.difficulties:
        diff_cfg = eval_cfg.difficulties[difficulty]
        print(f"Using difficulty: {difficulty} - {diff_cfg.get('description', '')}")
    else:
        diff_cfg = {}
        print(f"Warning: Difficulty '{difficulty}' not found, using defaults")
    
    # Extract settings
    dataset_cfg = eval_cfg.dataset
    sampling_cfg = eval_cfg.sampling
    image_cfg = eval_cfg.image
    if args.cond_num is not None:
        sampling_cfg.cond_num=args.cond_num
    # Load model config (Base config for sampler/transport/RAE)
    target_cfg_path = args.model_config
    if target_cfg_path is None:
        # Fallback to level-specific config if global config is not provided
        if args.eval_mode == 'independent':
             # Use the config for the target level
             attr_name = f"model_config_level{level}"
             target_cfg_path = getattr(args, attr_name, None)
             if target_cfg_path is None:
                 raise ValueError(f"Independent mode (Level {level}) requires --model_config_level{level}")
        else:
             # For pipeline modes (cascade/cascade), use Level 1 config as base
             # (Since pipeline starts with L1 diffusion)
             target_cfg_path = args.model_config_level1
             if target_cfg_path is None:
                 # Fallback to Level 0 if Level 1 is not provided (unlikely in pipeline but safe fallback)
                 target_cfg_path = args.model_config_level0
             
             if target_cfg_path is None:
                 raise ValueError(f"{args.eval_mode} requires --model_config_level1 (as base config)")
                 
        print(f"No global --model_config provided. Using {target_cfg_path} as base config.")

    model_cfg  = OmegaConf.load(target_cfg_path)
    
    # Set seed
    seed = sampling_cfg.get('seed', 42)
    set_seed(seed)
    
    # Device
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f'level{level}_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    vis_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    
    # Save eval config to output
    OmegaConf.save(eval_cfg, os.path.join(output_dir, 'eval_config.yaml'))
    
    rae_config = model_cfg.stage_1
    stage2_config = model_cfg.stage_2
    misc_config = model_cfg.get('misc', {})

    # Determine stat_path based on RAE type (DA3 vs VGGT)
    rae_norm_path = rae_config.get('params', {}).get('normalization_stat_path', '')
    is_vggt = 'vggt' in str(rae_norm_path).lower() or 'rae_vggt' in str(rae_config.get('target', ''))
    backbone = "vggt" if is_vggt else "da3"
    stat_path = [f"model_stats/{backbone}/normalization_stats_level{i}.pt" for i in range(4)]
    if is_vggt:
        special_stat_path = [f"model_stats/vggt/special_stats_level{i}.pt" for i in range(4)]
    else:
        special_stat_path = [None, None, None, None]
    print(f"Using {backbone.upper()} normalization stats")

    if 'params' not in rae_config:
        rae_config.params = OmegaConf.create({})
    rae_config.params.level = level
    
    # CRITICAL: Initialize config defaults (sets cam_patch_size=14 for DA3)
    init_config_defaults(rae_config, stage2_config, misc_config, patch_size=14, is_da3=True)

    # Initialize RAE
    print("Loading RAE_DA3...")
    rae = instantiate_from_config(rae_config).to(device)
    rae.eval()
    
    # Initialize diffusion model
    print("Loading diffusion model...")
    # Helper to instantiate from config (NEW vs OLD determined by YAML)
    def instantiate_model(base_config, model_config_path=None):
        if model_config_path:
            inst_config = OmegaConf.load(model_config_path).stage_2
        else:
            inst_config = base_config
            
        config_copy = OmegaConf.create(OmegaConf.to_container(inst_config, resolve=True))
        
        # Determine mode from YAML (Default to "old" if not specified)
        arch_mode = config_copy.params.get("architecture_mode", "old")

        # If "old" mode is detected, switch to the legacy model file
        if arch_mode == "old":
            config_copy.target = "stage2.models.DDT_old.DiTwDDTHead"
            # Remove modern parameters that legacy DDT doesn't support
            config_copy.params.pop('architecture_mode', None)
            config_copy.params.pop('cfg_mode', None)
        # Else: New mode uses the default target (DDT.py)
                
        return instantiate_from_config(config_copy).to(device)

    # Initialize diffusion models (only load what's needed based on eval_mode)
    print("Loading diffusion model...")

    if args.eval_mode in ['cascade']:
        _required_levels = {1}
    else:  # independent
        _required_levels = set(range(level + 1))

    print(f"  eval_mode={args.eval_mode}: loading DiT for levels {sorted(_required_levels)}")
    model0 = instantiate_model(stage2_config, args.model_config_level0) if 0 in _required_levels else None
    model1 = instantiate_model(stage2_config, args.model_config_level1) if 1 in _required_levels else None
    model2 = instantiate_model(stage2_config, args.model_config_level2) if 2 in _required_levels else None
    model3 = instantiate_model(stage2_config, args.model_config_level3) if 3 in _required_levels else None

    # Load checkpoint
    # In cascade mode, only level1 checkpoint is needed (level0 comes from cascade model)
    if args.use_cascade:
        print(f"Loading checkpoints for Feature Flow mode...")
        if args.checkpoint_level1:
            ckpt_level1 = torch.load(args.checkpoint_level1, map_location="cpu")
            state_dict_level1 = ckpt_level1.get('ema', ckpt_level1.get('model', ckpt_level1))
            state_dict_level1 = {k.replace('module.', ''): v for k, v in state_dict_level1.items()}
            model1.load_state_dict(state_dict_level1, strict=False)
            print(f"Level 1 checkpoint loaded: {args.checkpoint_level1}")
        else:
            raise ValueError("--use_cascade requires --checkpoint_level1")

    # Load checkpoints individually (skip levels not loaded)
    for lvl in range(4):
        target_model = [model0, model1, model2, model3][lvl]
        if target_model is None:
            continue  # Model not loaded for this eval_mode
        ckpt_attr = f"checkpoint_level{lvl}"
        ckpt_path = getattr(args, ckpt_attr, None)
        if ckpt_path:
            print(f"Loading checkpoint for level {lvl}: {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location="cpu")
            sd = ckpt.get('ema', ckpt.get('model', ckpt))
            sd = {k.replace('module.', ''): v for k, v in sd.items()}

            res = target_model.load_state_dict(sd, strict=False)
            print(f"Level {lvl} load: missing={len(res.missing_keys)}, unexpected={len(res.unexpected_keys)}")
            if res.missing_keys: print(f"  Missing (first 5): {res.missing_keys[:5]}")
            if res.unexpected_keys: print(f"  Unexpected (first 5): {res.unexpected_keys[:5]}")
        else:
            print(f"Warning: No checkpoint provided for level {lvl}, using uninitialized weights.")
    for _m in [model0, model1, model2, model3]:
        if _m is not None:
            _m.eval()
    
    # Load mapping model (L1 → L0) if in pipeline mode
    cascade_model = None
    if args.eval_mode in ['cascade']:
        if args.checkpoint_cascade is None:
            raise ValueError(f"{args.eval_mode} requires --checkpoint_cascade to be specified")
        
        print(f"\n[{args.eval_mode.upper()} Pipeline ENABLED]")
        print(f"Loading cascade model from: {args.checkpoint_cascade}")
        
        print(f"Loading cascade model from: {args.checkpoint_cascade}")
        
        cascade_model = instantiate_model(stage2_config, args.model_config_cascade)
        ckpt_ff = torch.load(args.checkpoint_cascade, map_location="cpu")
        state_dict_ff = ckpt_ff.get('ema', ckpt_ff.get('model', ckpt_ff))
        state_dict_ff = {k.replace('module.', ''): v for k, v in state_dict_ff.items()}
        cascade_model.load_state_dict(state_dict_ff, strict=False)
        cascade_model.eval()
        print("cascade model loaded successfully.")
    
    # Create transport and sampler
    transport_cfg = OmegaConf.to_container(model_cfg.get('transport', {}), resolve=True)
    
    # Calculate time_dist_shift to match training
    # Training uses: shift_dim = C * H * W * total_view, time_dist_shift = sqrt(shift_dim / 4096)
    # This is critical for correct ODE integration!
    misc_cfg = OmegaConf.to_container(model_cfg.get('misc', {}), resolve=True) if model_cfg.get('misc') else {}
    
    # Get latent dimensions from RAE (after first inference, but we can compute from config)
    # DA3: patch_size=14, so for 504x336 image -> 36x24 patches, C=1536
    # For now, use model params or defaults
    stage1_cfg = model_cfg.get('stage_1', {})
    encoder_input_size = stage1_cfg.get('params', {}).get('encoder_input_size', 504)
    patch_size = 14
    
    # Use encoder_input_size from config for shift calculation
    if isinstance(encoder_input_size, (list, tuple)):
        override_input_size = encoder_input_size[0]
    else:
        override_input_size = int(encoder_input_size)
    h_lat = override_input_size // patch_size
    w_lat = override_input_size // patch_size
    
    # Read in_channels from stage_2 config (1536 for DA3, 2048 for VGGT, 768 for DINO)
    stage2_params = model_cfg.get('stage_2', {}).get('params', {})
    in_channels = stage2_params.get('in_channels', 1536)
    latent_size = (in_channels, h_lat, w_lat)
    
    import math
    shift_dim = misc_cfg.get("time_dist_shift_dim", math.prod(latent_size) * sampling_cfg.num_views)
    
        # shift_dim = args.shift_dim
    shift_base = misc_cfg.get("time_dist_shift_base", 4096)
    time_dist_shift = math.sqrt(shift_dim / shift_base)
    if args.shift_dim is not None:
        time_dist_shift = args.shift_dim
    print(f"[Eval] time_dist_shift = sqrt({shift_dim}/{shift_base}) = {time_dist_shift:.4f}")
    
    transport = create_transport(**transport_cfg.get('params', {}), time_dist_shift=time_dist_shift)
    
    sampler_cfg = OmegaConf.to_container(model_cfg.get('sampler', {}), resolve=True)
    sampler = Sampler(transport)
    
    if sampler_cfg.get('mode', 'ODE').upper() == 'ODE':
        sample_fn = sampler.sample_ode(**sampler_cfg.get('params', {}))
    else:
        sample_fn = sampler.sample_sde(**sampler_cfg.get('params', {}))
    
    # Create dataloader
    print(f"Loading dataset: {dataset_cfg.name}")
    
    # Import dataset based on name
    if dataset_cfg.name.startswith('cut3r_'):
        import sys
        from datasets.da3_nvs_dataset import create_nvs_dataloader_from_config
        ogl = model_cfg.get("dataset", {}).get("opengl_to_opencv", True)
        if "opengl_to_opencv" not in eval_cfg.dataset:
            OmegaConf.update(eval_cfg, "dataset.opengl_to_opencv", ogl)
        dataloader = create_nvs_dataloader_from_config(
            eval_cfg=eval_cfg,
            difficulty=difficulty,
            max_samples=args.max_samples,
            batch_size=1,
            num_workers=args.num_workers,
        )
    else:
        # Fallback to video dataset
        from video.video_load import DA3VideoDataset_Pose
        
        image_size = image_cfg.get('size', 518)
        if isinstance(image_size, (list, tuple)):
            process_res = image_size
        else:
            process_res = int(image_size)
        
        dataset = DA3VideoDataset_Pose(
            video_path=dataset_cfg.root,
            pose_path=dataset_cfg.get('pose_path', dataset_cfg.root + '_poses'),
            num_views=sampling_cfg.num_views,
            cond_num=sampling_cfg.cond_num,
            ref_view_sampling=sampling_cfg.ref_view_sampling,
            process_res=process_res,
            mode='test',
        )
        
        if args.max_samples:
            dataset = Subset(dataset, list(range(min(args.max_samples, len(dataset)))))
        
        dataloader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=args.num_workers,
        )
    
    
    # Get config values
    num_views = sampling_cfg.num_views
    cond_num = sampling_cfg.cond_num
    ref_view_sampling = sampling_cfg.ref_view_sampling
    camera_mode = model_cfg.get('dataset', {}).get('camera_mode', 'plucker')
    ray_pose_mode = model_cfg.get('dataset', {}).get('ray_pose_mode', 'c2w')
    is_concat_mode = stage2_config.get('params', {}).get('is_concat_mode', False)
    use_prope = stage2_config.get('params', {}).get('use_prope', False)

    print(f"Evaluating... (views={num_views}, cond={cond_num}, camera_mode={camera_mode}, ray_pose={ray_pose_mode}, concat_mode={is_concat_mode})")
    
    # Read guidance config from model config
    validation_cfg = model_cfg.get('validation', {})
    cfg_cfg = validation_cfg.get('guidance', {})
    cfg_scale = cfg_cfg.get('scale', None)

    
    if args.cfg_scale_ar is not None:
        cfg_scale_auto = args.cfg_scale_ar
    else:
        cfg_scale_auto = cfg_scale
    
    if args.cfg_scale is not None:
        cfg_scale = args.cfg_scale
    
    # NEW: Read CFG camera drop and unconditional modes
    use_camera_drop = cfg_cfg.get('use_camera_drop', True)  # Default: True (backward compat)
    cfg_l1_uncond_mode = cfg_cfg.get('cfg_l1_uncond_mode', 'keep')
    cfg_cascade_uncond_mode = cfg_cfg.get('cfg_cascade_uncond_mode', 'keep')

    # Override from command line if provided
    if hasattr(args, 'use_camera_drop') and args.use_camera_drop is not None:
        use_camera_drop = args.use_camera_drop
    if hasattr(args, 'cfg_l1_uncond_mode') and args.cfg_l1_uncond_mode is not None:
        cfg_l1_uncond_mode = args.cfg_l1_uncond_mode
    if hasattr(args, 'cfg_cascade_uncond_mode') and args.cfg_cascade_uncond_mode is not None:
        cfg_cascade_uncond_mode = args.cfg_cascade_uncond_mode

    # NEW: Level-specific camera drop (overrides general use_camera_drop if provided)
    use_camera_drop_l1 = use_camera_drop  # Default to general setting
    use_camera_drop_cascade = use_camera_drop  # Default to general setting

    if hasattr(args, 'use_camera_drop_l1') and args.use_camera_drop_l1 is not None:
        use_camera_drop_l1 = args.use_camera_drop_l1
    if hasattr(args, 'use_camera_drop_cascade') and args.use_camera_drop_cascade is not None:
        use_camera_drop_cascade = args.use_camera_drop_cascade

    print(f"CFG Settings: L1_camera_drop={use_camera_drop_l1}, L1_mode={cfg_l1_uncond_mode}, Cascade_camera_drop={use_camera_drop_cascade}, Cascade_mode={cfg_cascade_uncond_mode}")


    pag_cfg = validation_cfg.get('pag', {})
    pag_scale = pag_cfg.get('scale', None)
    pag_layer_idx = pag_cfg.get('layer_idx', None)


    if cfg_cfg.method == 'cfg':
        print(f"Using CFG guidance: scale={cfg_scale}")
    if pag_scale is not None:
        print(f"Using PAG guidance: scale={pag_scale}, layer_idx={pag_layer_idx}")

    # ========== Parse Attention Visualization Parameters ==========
    attention_config = {'enable_attention_vis': False}

    if args.visualize_attention:
        print("\n" + "="*60)
        print("ATTENTION VISUALIZATION ENABLED")
        print("="*60)

        # Parse timesteps
        if args.attention_timesteps is not None:
            timesteps = [int(t.strip()) for t in args.attention_timesteps.split(',')]
        else:
            # Default: first, middle, last
            num_steps = sampler_cfg.get('params', {}).get('num_steps', 50)
            timesteps = [0, num_steps // 2, num_steps - 1]

        # Parse layers
        if args.attention_layers is not None:
            layer_ids = [int(l.strip()) for l in args.attention_layers.split(',')]
        else:
            # Default: every 2nd layer for both encoder and decoder
            # Determine total layers from model
            total_depth = len(model1.blocks) if hasattr(model1, 'blocks') else 34
            layer_ids = list(range(0, total_depth, 2))

        # Parse query positions
        query_positions = None
        if args.attention_query_positions is not None:
            query_positions = []
            for pair in args.attention_query_positions.split(':'):
                h, w = pair.split(',')
                query_positions.append((int(h.strip()), int(w.strip())))

        attention_config = {
            'enable_attention_vis': True,
            'attention_timesteps': timesteps,
            'attention_layer_ids': layer_ids,
            'query_view_idx': args.attention_query_view,
            'query_positions': query_positions,
        }

        print(f"  Timesteps: {timesteps}")
        print(f"  Layers: {layer_ids[:5]}{'...' if len(layer_ids) > 5 else ''}")
        print(f"  Query view: {args.attention_query_view}")
        print(f"  Query positions: {'default (center+corners)' if query_positions is None else query_positions}")
        print("="*60 + "\n")
    
    # if level > 0 and not args.feature_only:
    #     print(f"WARNING: Level {level} > 0 requires feature-level validation. Enabling --feature_only automatically.")
    #     args.feature_only = True
    
    # Choose validation function based on --feature_only flag
    # if args.feature_only:
    #     print(f"Using feature-level validation (MSE + CosSim)")
    #     from utils.da3_feature_validation import validate_da3_features
        
    #     stats = validate_da3_features(
    #         rae=rae,
    #         model=model,
    #         transport=transport,
    #         sampler=sample_fn,
    #         loader=dataloader,
    #         device=device,
    #         total_view=num_views,
    #         cond_num=cond_num,
    #         val_num_batches=args.max_samples,
    #         use_prope=use_prope,
    #         rank=0,
    #         world_size=1,
    #         prope_image_size=None,
    #         predict_cls=False,
    #         joint_ode=False,
    #         ref_view_sampling=ref_view_sampling,
    #         camera_mode=camera_mode,
    #         is_concat_mode=is_concat_mode,
    #         pag_scale=pag_scale,
    #         pag_layer_idx=pag_layer_idx,
    #         cfg_scale=cfg_scale,
    #     )
    # RGB-level validation: PSNR/SSIM/LPIPS on decoded images (with propagation)
    print(f"Using RGB-level validation (PSNR/SSIM/LPIPS)")
    from utils.da3_validation_metric import get_denoised_features, decode_into_images
    
    # Initialize metrics accumulator
    per_sample_metrics = []
    
    # Determine total steps
    total_steps = len(dataloader)
    if args.max_samples is not None:
        total_steps = min(total_steps, args.max_samples)
    
    pbar = tqdm(enumerate(dataloader), total=total_steps, desc="RGB Validation")
    for i, batch in pbar:
        if args.max_samples is not None and i >= args.max_samples:
            break

        # Handle CUT3R batch format
        if isinstance(batch, list) and isinstance(batch[0], dict):
            from video.cut3r_adapter import convert_cut3r_batch
            batch = convert_cut3r_batch(batch, cond_num, ref_view_sampling)

        # calculate GT featues for all layers
        print("\nGetting GT features for all layers...")
        images = batch['image'].to(device)  # (B, V, C, H, W)
        B, V, C, H, W = images.shape

        feat_gt_norm = {}    # [latent_norm] for propagation
        feat_gt_denorm = {}  # [raw] for apply_da3_norm → layer_norm
        for lvl in range(4):
            if not os.path.exists(stat_path[lvl]):
                print(f"[WARN] Skipping GT extraction for level {lvl}: {stat_path[lvl]} not found")
                continue
            rae.level = lvl
            rae._init_normalization(stat_path=stat_path[lvl])
            if is_vggt and hasattr(rae, '_init_special_normalization'):
                rae._init_special_normalization(stat_path=special_stat_path[lvl])
            images_norm = (images - rae.encoder_mean[None]) / rae.encoder_std[None]
            images_flat = images_norm.reshape(B * V, C, H, W)
            with torch.no_grad():
                gt_feat, _ = rae.encode(images_flat, return_cls=True, mode='single', level=lvl)
                feat_gt_norm[lvl] = gt_feat              # [latent_norm]

                # For VGGT packed format: unpack → denormalize separately → spatial format
                if is_vggt and hasattr(rae, 'unpack_special') and gt_feat.ndim == 4 and gt_feat.shape[3] == 1:
                    patches_2d, special_norm = rae.unpack_special(gt_feat)
                    # Denormalize patches (latent_norm → raw)
                    patches_raw = rae._denormalize(patches_2d)
                    feat_gt_denorm[lvl] = patches_raw  # (BV, C, h, w) spatial
                else:
                    feat_gt_denorm[lvl] = rae._denormalize(gt_feat)  # [raw]

        feat = {}              # [latent_norm] per level
        feat_denormalized = {} # [raw] per level
        # ==================== PIPELINE MODES (FEATURE FLOW / PROPAGATION) ====================
        if args.eval_mode in ['cascade']:
            print("\n" + "="*60)
            print(f"PIPELINE MODE: {args.eval_mode.upper()}")
            print("L1 Diffusion → L1→L0 (DDT) → L1→L2,L3 (DA3 Propagate)")
            print("="*60)
            
            # Step 1: L1 Diffusion (noise → L1)
            print("\n[Step 1/3] L1 Diffusion: noise → L1...")
            rae.level = 1
            rae._init_normalization(stat_path=stat_path[1])

            # Use attention-enabled version if visualization is enabled
            if attention_config['enable_attention_vis']:
                from utils.attention_integration import get_denoised_features_with_attention

                # Create attention save directory for this sample
                attn_save_dir = os.path.join(vis_dir, f'attention_sample_{i:04d}')
                os.makedirs(attn_save_dir, exist_ok=True)

                # Prepare attention config without 'enable_attention_vis' key (already in attention_config)
                attn_kwargs = {k: v for k, v in attention_config.items() if k != 'enable_attention_vis'}
                attn_kwargs['attention_save_dir'] = attn_save_dir

                feat[1] = get_denoised_features_with_attention(
                    rae=rae,
                    model=model1,
                    transport=transport,
                    sampler=sample_fn,
                    batch=batch,
                    device=device,
                    total_view=num_views,
                    cond_num=cond_num,
                    use_prope=use_prope,
                    prope_image_size=(H, W),
                    cfg_scale=cfg_scale,
                    use_camera_drop=use_camera_drop_l1,
                    cfg_uncond_mode=cfg_l1_uncond_mode,
                    camera_mode=camera_mode,
                    is_concat_mode=is_concat_mode,
                    ray_pose_mode=ray_pose_mode,
                    pag_scale=pag_scale,
                    pag_layer_idx=pag_layer_idx,
                    stat_path=stat_path,
                    # Feature flow params for decoding
                    cascade_model=cascade_model,
                    cfg_scale_ar=cfg_scale_auto,
                    cfg_cascade_uncond_mode=cfg_cascade_uncond_mode,
                    cascade_noise_tau=args.cascade_noise_tau,
                    eval_mode=args.eval_mode,
                    # Attention visualization params
                    enable_attention_vis=True,
                    **attn_kwargs,
                )
                
            else:
                feat[1] = get_denoised_features(
                    rae=rae,
                    model=model1,
                    transport=transport,
                    sampler=sample_fn,
                    loader=dataloader,
                    device=device,
                    total_view=num_views,
                    cond_num=cond_num,
                    val_num_batches=args.max_samples,
                    use_prope=use_prope,
                    rank=0,
                    world_size=1,
                    prope_image_size=(H, W),
                    predict_cls=False,
                    joint_ode=False,
                    ref_view_sampling=ref_view_sampling,
                    camera_mode=camera_mode,
                    is_concat_mode=is_concat_mode,
                    ray_pose_mode=ray_pose_mode,
                    pag_scale=pag_scale,
                    pag_layer_idx=pag_layer_idx,
                    cfg_scale=cfg_scale,
                    use_camera_drop=use_camera_drop_l1,  # NEW
                    cfg_uncond_mode=cfg_l1_uncond_mode,  # NEW
                    batch=batch,
                )
            feat_denormalized[1] = rae._denormalize(feat[1])  # [latent_norm] → [raw]
            print(f"L1 features: mean={feat[1].mean():.4f}, std={feat[1].std():.4f}")
            # Step 2: L1 → L0 transformation
            print(f"\n[Step 2/2] L1 → L0 mapping ({args.eval_mode})...")
            feat[0] = get_cascade_features(  # returns [latent_norm]
                rae=rae,
                cascade_model=cascade_model,
                sampler=sample_fn,
                source_features=feat[1],
                source_stat_path=stat_path[1],
                target_stat_path=stat_path[0],
                batch=batch,
                device=device,
                total_view=num_views,
                cond_num=cond_num,
                camera_mode=camera_mode,
                ray_pose_mode=ray_pose_mode,
                use_prope=use_prope,
                cfg_scale=cfg_scale_auto,
                use_camera_drop=use_camera_drop_cascade,  # NEW
                cfg_uncond_mode=cfg_cascade_uncond_mode,  # NEW
                noise_tau=args.cascade_noise_tau,
                prope_image_size=(H, W),
                eval_mode=args.eval_mode,
            )
            # feat[0] = None
            rae.level = 0
            rae._init_normalization(stat_path=stat_path[0])
            feat_denormalized[0] = rae._denormalize(feat[0])  # [latent_norm] → [raw]
            # feat_denormalized[0] = torch.zeros_like(feat_denormalized[1])
            print(f"L0 features: mean={feat[0].mean():.4f}, std={feat[0].std():.4f}")

            # L2/L3 propagation is handled inside decode_into_images(level=1):
            #   feat[1] [latent_norm] → propagate_features → [layer_norm] L1,L2,L3
            #   feat_denormalized[0,1] [raw] → apply_da3_norm → [layer_norm] L0,L1
            # All 4 levels become [layer_norm] → rae.decode() → RGB + Depth
            print(f"Pipeline {args.eval_mode} completed. L2/L3 will be propagated during decode.\n")
            ##################################
            level=-1  # Keep -1 to skip independent mode blocks below
            ###################################
        # ==================== INDEPENDENT MODE ====================
        elif args.eval_mode == 'independent' and level >= 0:
            print("\nGetting denoised features for level 0...")
            rae.level = 0
            rae._init_normalization(stat_path=stat_path[0])
            if is_vggt and hasattr(rae, '_init_special_normalization'):
                rae._init_special_normalization(stat_path=special_stat_path[0])
            feat[0] = get_denoised_features(
                rae=rae,
                model=model0,
                transport=transport,
                sampler=sample_fn,
                loader=dataloader,
                device=device,
                total_view=num_views,
                cond_num=cond_num,
                val_num_batches=args.max_samples,
                use_prope=use_prope,
                rank=0,
                world_size=1,
                prope_image_size=(H, W),
                predict_cls=False,
                joint_ode=False,
                ref_view_sampling=ref_view_sampling,
                camera_mode=camera_mode,
                is_concat_mode=is_concat_mode,
                    ray_pose_mode=ray_pose_mode,
                pag_scale=pag_scale,
                pag_layer_idx=pag_layer_idx,
                cfg_scale=cfg_scale,
                batch=batch,
            )
            # For VGGT packed format: unpack before denormalize
            if is_vggt and hasattr(rae, 'unpack_special') and feat[0].ndim == 4 and feat[0].shape[3] == 1:
                patches_2d, _ = rae.unpack_special(feat[0])
                feat_denormalized[0] = rae._denormalize(patches_2d)
            else:
                feat_denormalized[0] = rae._denormalize(feat[0])  # [latent_norm] → [raw]
        if level >= 1: # level 1,2,3 need level 1 features
            print("\nGetting denoised features for level 1...")
            rae.level = 1
            rae._init_normalization(stat_path=stat_path[1])

            # Use attention-enabled version if visualization is enabled
            if attention_config['enable_attention_vis']:
                from utils.attention_integration import get_denoised_features_with_attention

                # Create attention save directory for this sample
                attn_save_dir = os.path.join(vis_dir, f'attention_sample_{i:04d}')
                os.makedirs(attn_save_dir, exist_ok=True)

                # Prepare attention config without 'enable_attention_vis' key (already in attention_config)
                attn_kwargs = {k: v for k, v in attention_config.items() if k != 'enable_attention_vis'}
                attn_kwargs['attention_save_dir'] = attn_save_dir

                feat[1] = get_denoised_features_with_attention(
                    rae=rae,
                    model=model1,
                    transport=transport,
                    sampler=sample_fn,
                    batch=batch,
                    device=device,
                    total_view=num_views,
                    cond_num=cond_num,
                    use_prope=use_prope,
                    prope_image_size=(H, W),
                    cfg_scale=cfg_scale,
                    use_camera_drop=use_camera_drop_l1,
                    cfg_uncond_mode=cfg_l1_uncond_mode,
                    camera_mode=camera_mode,
                    is_concat_mode=is_concat_mode,
                    ray_pose_mode=ray_pose_mode,
                    pag_scale=pag_scale,
                    pag_layer_idx=pag_layer_idx,
                    stat_path=stat_path,
                    # Feature flow params for decoding
                    cascade_model=cascade_model,
                    cfg_scale_ar=cfg_scale_auto,
                    cfg_cascade_uncond_mode=cfg_cascade_uncond_mode,
                    cascade_noise_tau=args.cascade_noise_tau,
                    eval_mode=args.eval_mode,
                    # Attention visualization params
                    enable_attention_vis=True,
                    **attn_kwargs,
                )
            else:
                feat[1] = get_denoised_features(
                    rae=rae,
                    model=model1,
                    transport=transport,
                    sampler=sample_fn,
                    loader=dataloader,
                    device=device,
                    total_view=num_views,
                    cond_num=cond_num,
                    val_num_batches=args.max_samples,
                    use_prope=use_prope,
                    rank=0,
                    world_size=1,
                    prope_image_size=(H, W),
                    predict_cls=False,
                    joint_ode=False,
                    ref_view_sampling=ref_view_sampling,
                    camera_mode=camera_mode,
                    is_concat_mode=is_concat_mode,
                    ray_pose_mode=ray_pose_mode,
                    pag_scale=pag_scale,
                    pag_layer_idx=pag_layer_idx,
                    cfg_scale=cfg_scale,
                    batch=batch,
                )
            feat_denormalized[1] = rae._denormalize(feat[1])  # [latent_norm] → [raw]

        if level >= 2: # level 2,3 need level 2 features
            print("\nGetting denoised features for level 2...")
            rae.level = 2
            rae._init_normalization(stat_path=stat_path[2])
            model2.to(device)
            feat[2] = get_denoised_features(
                rae=rae,
                model=model2,
                transport=transport,
                sampler=sample_fn,
                loader=dataloader,
                device=device,
                total_view=num_views,
                cond_num=cond_num,
                val_num_batches=args.max_samples,
                use_prope=use_prope,
                rank=0,
                world_size=1,
                prope_image_size=(H, W),
                predict_cls=False,
                joint_ode=False,
                ref_view_sampling=ref_view_sampling,
                camera_mode=camera_mode,
                is_concat_mode=is_concat_mode,
                    ray_pose_mode=ray_pose_mode,
                pag_scale=pag_scale,
                pag_layer_idx=pag_layer_idx,
                cfg_scale=cfg_scale,
                batch=batch,
            )
            feat_denormalized[2] = rae._denormalize(feat[2])  # [latent_norm] → [raw]

        if level >= 3: # level 3 need level 3 features
            print("\nGetting denoised features for level 3...")
            rae.level = 3
            rae._init_normalization(stat_path=stat_path[3])
            feat[3] = get_denoised_features(
                rae=rae,
                model=model3,
                transport=transport,
                sampler=sample_fn,
                loader=dataloader,
                device=device,
                total_view=num_views,
                cond_num=cond_num,
                val_num_batches=args.max_samples,
                use_prope=use_prope,
                rank=0,
                world_size=1,
                prope_image_size=(H, W),
                predict_cls=False,
                joint_ode=False,
                ref_view_sampling=ref_view_sampling,
                camera_mode=camera_mode,
                is_concat_mode=is_concat_mode,
                    ray_pose_mode=ray_pose_mode,
                pag_scale=pag_scale,
                pag_layer_idx=pag_layer_idx,
                cfg_scale=cfg_scale,
                batch=batch,
            )
            feat_denormalized[3] = rae._denormalize(feat[3])  # [latent_norm] → [raw]

        # ==================== ORIGINAL INDEPENDENT DIFFUSION END ====================

        # now models unloaded to cpu
        # model0.to('cpu')
        # model1.to('cpu')
        # model2.to('cpu')
        # model3.to('cpu')
        
        # # DEBUG: Compare generated features with GT
        # if i == 0:  # Only debug first sample
        #     from utils.debug_normalization import run_full_debug
        #     print("\n" + "="*60)
        #     print("RUNNING NORMALIZATION DEBUG...")
        #     print("="*60 + "\n")
        #     run_full_debug(
        #         rae=rae,
        #         generated_feats=feat,
        #         gt_images=batch['image'],
        #         stat_paths=stat_path,
        #         device=device,
        #     )

        # replace feat0's pred with gt feat
        # images = batch['image'].to(device)  # (B, V, C, H, W)
        # B, V, C, H, W = images.shape
        # rae.level = 0
        # rae._init_normalization(stat_path=stat_path[0])
        # images_norm = (images - rae.encoder_mean[None]) / rae.encoder_std[None]
        # images_flat = images_norm.reshape(B * V, C, H, W)
        # with torch.no_grad():
        #         gt_feat, _ = rae.encode(images_flat, return_cls=True, mode='single', level=0)

        # feat[0] = gt_feat
        # feat_denormalized[0] = rae._denormalize(gt_feat)

        # Pipeline modes: decode with level=1 so that decode_into_images handles
        # L1→L2,L3 propagation internally (latent_norm → layer_norm, no double norm).
        # Independent mode: use the original level.
        _decode_level = 1 if args.eval_mode in ['cascade'] else level

        outputs = decode_into_images(
            rae=rae,
            features=feat,                       # [latent_norm] (for propagation)
            feat_latent_denorm=feat_denormalized, # [raw] (→ apply_da3_norm → layer_norm)
            level=_decode_level,
            total_view=num_views,
            cond_num=cond_num,
            device=device,
            batch=batch,
            stat_path=stat_path,
            sample_idx=i,
        )

        img = outputs['rgb']
        pred_depth = outputs['depth']
        pred_depth_conf = outputs.get('depth_conf', None)
        pred_ray = outputs.get('ray', None)
        pred_ray_conf = outputs.get('ray_conf', None)

        # GT decode: only if all 4 levels available, otherwise skip
        gt_depth = None
        if all(lvl in feat_gt_norm for lvl in range(4)):
            gt_outputs = decode_into_images(
                rae=rae,
                features=feat_gt_norm,
                feat_latent_denorm=feat_gt_denorm,
                level=max(level, max(feat_gt_norm.keys())),
                total_view=num_views,
                cond_num=cond_num,
                device=device,
                batch=batch,
                stat_path=stat_path,
                sample_idx=i,
            )
            gt_depth = gt_outputs['depth']

        # Reshape from (BV, C, H, W) → (B, V, C, H, W) for metric computation
        B_eval = 1  # eval batch_size is always 1
        if img is not None and img.ndim == 4 and img.shape[0] == num_views:
            img = img.reshape(B_eval, num_views, *img.shape[1:])
        if pred_depth is not None:
            # DPT outputs (B, V, H, W) — already correct shape
            if pred_depth.ndim == 4 and pred_depth.shape[0] == B_eval and pred_depth.shape[1] == num_views:
                pred_depth = pred_depth.unsqueeze(2)  # (B, V, 1, H, W)
            elif pred_depth.ndim == 4:
                pred_depth = pred_depth.reshape(B_eval, num_views, *pred_depth.shape[1:])
            elif pred_depth.ndim == 5:
                pass  # already (B, V, C, H, W)
        if gt_depth is not None:
            if gt_depth.ndim == 4 and gt_depth.shape[0] == B_eval and gt_depth.shape[1] == num_views:
                gt_depth = gt_depth.unsqueeze(2)
            elif gt_depth.ndim == 4:
                gt_depth = gt_depth.reshape(B_eval, num_views, *gt_depth.shape[1:])
            elif gt_depth.ndim == 5:
                pass
        if pred_depth_conf is not None:
            if pred_depth_conf.ndim == 4 and pred_depth_conf.shape[0] == B_eval and pred_depth_conf.shape[1] == num_views:
                pred_depth_conf = pred_depth_conf.unsqueeze(2)
            elif pred_depth_conf.ndim == 4:
                pred_depth_conf = pred_depth_conf.reshape(B_eval, num_views, *pred_depth_conf.shape[1:])

        # img = decode_into_images_level3(
        #     rae=rae, 
        #     features=feat,
        #     feat_latent_denorm=feat_denormalized, 
        #     level=level, 
        #     total_view=num_views, 
        #     cond_num=cond_num,
        #     device=device, 
        #     batch=batch,
        #     stat_path=stat_path,
        #     sample_idx=i,
        # )

        if args.run_pca:
            # PCA visualization of features for debugging

            from utils.pca_visualization import visualize_feature_dict_pca
            
            pca_dir = os.path.join(vis_dir, f'pca_sample_{i:04d}')
            visualize_feature_dict_pca(
                feat_dict=feat_denormalized,
                save_dir=pca_dir,
                sample_idx=i,
                n_components=3,
                gt_images=batch['image'],  # GT images for row 1
                feat_gt_dict=feat_gt_denorm,  # GT features for row 3
            )
            
        # Save visualization (skip if RGB not available, e.g. VGGT depth-only)
        if img is not None:
            save_visualization(
                gt_images=batch['image'].squeeze(0),
                pred_images=img.squeeze(0),
                cond_num=cond_num,
                save_path=os.path.join(vis_dir, f'sample_{i:04d}.png')
            )
        
        # Save depth visualization
        if pred_depth is not None and gt_depth is not None:
            save_depth_visualization(
                gt_depth=gt_depth.squeeze(0),
                pred_depth=pred_depth.squeeze(0),
                cond_num=cond_num,
                save_path=os.path.join(vis_dir, f'sample_{i:04d}_depth.png')
            )
        
        # Save raw data for 3D reconstruction (matches da3rae NPZ convention)
        # Recover pred_c2w/pred_K from ray head here so reconstruct_npz_scene.py can use them directly.
        pred_c2w_np = None
        pred_K_np = None
        if pred_ray is not None:
            from utils.camera_from_ray import recover_poses
            ray_np = pred_ray.cpu().numpy() if torch.is_tensor(pred_ray) else pred_ray
            ray_conf_np = pred_ray_conf.cpu().numpy() if pred_ray_conf is not None and torch.is_tensor(pred_ray_conf) else pred_ray_conf
            H_img = batch['image'].shape[-2]
            W_img = batch['image'].shape[-1]
            pred_c2w_list, K_recovered = recover_poses(
                ray_np, ray_conf_np, ref_view=0, subsample=4,
                input_size=(H_img, W_img),
            )
            pred_c2w_np = np.stack(pred_c2w_list)  # (V, 4, 4)
            pred_K_np = K_recovered                 # (3, 3)

        raw_data = {
            'pred_rgb': img.squeeze(0).cpu().numpy() if img is not None else None,
            'gt_rgb': batch['image'].squeeze(0).cpu().numpy(),
            'pred_depth': pred_depth.squeeze(0).cpu().numpy() if pred_depth is not None else None,
            'gt_depth': gt_depth.squeeze(0).cpu().numpy() if gt_depth is not None else None,
            'intrinsics': batch.get('intrinsic', batch.get('fxfycxcy')).cpu().numpy() if 'intrinsic' in batch or 'fxfycxcy' in batch else None,
            'extrinsics': batch.get('c2w').cpu().numpy() if 'c2w' in batch else None,
            'cond_num': cond_num,
        }
        # Depth confidence (from DPT main head)
        if pred_depth_conf is not None:
            raw_data['pred_ray_conf'] = pred_depth_conf.squeeze(0).cpu().numpy()
        # Ray head outputs
        if pred_ray is not None:
            raw_data['pred_ray'] = pred_ray.cpu().numpy() if torch.is_tensor(pred_ray) else pred_ray
        if pred_ray_conf is not None:
            raw_data['pred_ray_conf_raw'] = pred_ray_conf.cpu().numpy() if torch.is_tensor(pred_ray_conf) else pred_ray_conf
        # Pre-computed camera from ray head (for reconstruct_npz_scene.py)
        if pred_c2w_np is not None:
            raw_data['pred_c2w'] = pred_c2w_np
        if pred_K_np is not None:
            raw_data['pred_K'] = pred_K_np
        np.savez_compressed(os.path.join(vis_dir, f'sample_{i:04d}_raw.npz'), **raw_data)
        
        # Compute metrics for target views only
        # batch['image']: (1, V, 3, H, W)
        # img: (1, V, 3, H, W) or None (depth-only mode, e.g. VGGT)
        psnr = 0.0
        ssim = 0.0
        lpips = 0.0
        if img is not None:
            gt_target = batch['image'][:, cond_num:, ...].to(device)  # (1, V-cond_num, 3, H, W)
            pred_target = img[:, cond_num:, ...].to(device)  # (1, V-cond_num, 3, H, W)
            # Reshape to (V-cond_num, 3, H, W) for metric computation
            # Metrics expect (batch, channel, height, width) and return (batch,)
            gt_target = gt_target.squeeze(0)  # (V-cond_num, 3, H, W)
            pred_target = pred_target.squeeze(0)  # (V-cond_num, 3, H, W)

            # Compute RGB metrics per view, then average
            psnr_per_view = compute_psnr(pred_target, gt_target)  # (V-cond_num,)
            ssim_per_view = compute_ssim(pred_target, gt_target)  # (V-cond_num,)
            lpips_per_view = compute_lpips(pred_target, gt_target)  # (V-cond_num,)

            # Average across views
            psnr = psnr_per_view.mean().item()
            ssim = ssim_per_view.mean().item()
            lpips = lpips_per_view.mean().item()
        
        # ========== Depth Metrics (Pseudo-GT) ==========
        abs_rel = 0.0
        depth_rmse = 0.0
        delta1 = 0.0
        if pred_depth is not None and gt_depth is not None:
            # Extract target views for depth
            # pred_depth, gt_depth: (B, V, 1, H, W) or (B, V, H, W)
            pred_depth_target = pred_depth[:, cond_num:, ...].squeeze(0).to(device)
            gt_depth_target = gt_depth[:, cond_num:, ...].squeeze(0).to(device)
            
            # Ensure both have same shape
            if pred_depth_target.dim() == 4 and pred_depth_target.shape[1] == 1:
                pred_depth_target = pred_depth_target.squeeze(1)
            if gt_depth_target.dim() == 4 and gt_depth_target.shape[1] == 1:
                gt_depth_target = gt_depth_target.squeeze(1)
            
            # Compute depth metrics
            abs_rel_per_view = compute_abs_rel(pred_depth_target, gt_depth_target)
            depth_rmse_per_view = compute_depth_rmse(pred_depth_target, gt_depth_target)
            delta1_per_view = compute_delta(pred_depth_target, gt_depth_target, threshold=1.25)
            
            abs_rel = abs_rel_per_view.mean().item()
            depth_rmse = depth_rmse_per_view.mean().item()
            delta1 = delta1_per_view.mean().item()
        
        # Extract scene name from batch (if available)
        # scene_name = None
        # if 'scene_name' in batch:
        #     scene_name = batch['scene_name'][0] if isinstance(batch['scene_name'], (list, tuple)) else str(batch['scene_name'])
        # elif 'scene' in batch:
        #     scene_name = batch['scene'][0] if isinstance(batch['scene'], (list, tuple)) else str(batch['scene'])
        # elif 'video_name' in batch:
        #     scene_name = batch['video_name'][0] if isinstance(batch['video_name'], (list, tuple)) else str(batch['video_name'])
        # else:
        scene_name = f'sample_{i:04d}'  # Fallback to sample index
        
        # Store per-sample metrics with scene information
        per_sample_metrics.append({
            'scene_name': scene_name,
            'sample_idx': i,
            'psnr': psnr,
            'ssim': ssim,
            'lpips': lpips,
            'abs_rel': abs_rel,
            'depth_rmse': depth_rmse,
            'delta1': delta1,
        })
        
        # Update progress bar
        pbar.set_postfix({
            'PSNR': f'{psnr:.2f}',
            'SSIM': f'{ssim:.4f}',
            'LPIPS': f'{lpips:.4f}',
            'AbsRel': f'{abs_rel:.4f}',
        })
    
    # Create stats dict for final results
    stats = {
        'per_sample_metrics': per_sample_metrics,
        'num_samples': len(per_sample_metrics),
    }
            
    # ========== Final results ==========
    if stats is None or len(stats.get('per_sample_metrics', [])) == 0:
        print("No samples evaluated!")
        return
    
    per_sample_metrics = stats['per_sample_metrics']
    
    # Aggregate metrics
    results = {
        'timestamp': datetime.now().isoformat(),
        'eval_config': args.eval_config,
        # 'checkpoint': args.checkpoint,
        'difficulty': difficulty,
        'level': level,
        'num_samples': len(per_sample_metrics),
        'per_sample_metrics': per_sample_metrics,  # Include detailed per-scene metrics
        'metrics': {}
    }
    
    # Collect metrics by type
    # if args.feature_only:
    #     # Feature-level metrics: MSE + CosSim
    #     mse_vals = [m['mse'] for m in per_sample_metrics if 'mse' in m]
    #     cos_vals = [m['cosine_similarity'] for m in per_sample_metrics if 'cosine_similarity' in m]
        
    #     if mse_vals:
    #         results['metrics'].update({
    #             'mse_mean': float(np.mean(mse_vals)),
    #             'mse_std': float(np.std(mse_vals)),
    #             'cosine_similarity_mean': float(np.mean(cos_vals)),
    #             'cosine_similarity_std': float(np.std(cos_vals)),
    #         })
        
    #     # Add FID if available
    #     if stats.get('fid') is not None:
    #         results['metrics']['fid'] = float(stats['fid'])
    # else:
    # RGB-level metrics: PSNR/SSIM/LPIPS
    psnr_vals = [m['psnr'] for m in per_sample_metrics if 'psnr' in m]
    ssim_vals = [m['ssim'] for m in per_sample_metrics if 'ssim' in m]
    lpips_vals = [m['lpips'] for m in per_sample_metrics if 'lpips' in m]
    
    # Depth metrics (Pseudo-GT)
    abs_rel_vals = [m['abs_rel'] for m in per_sample_metrics if 'abs_rel' in m]
    depth_rmse_vals = [m['depth_rmse'] for m in per_sample_metrics if 'depth_rmse' in m]
    delta1_vals = [m['delta1'] for m in per_sample_metrics if 'delta1' in m]
    
    if psnr_vals:
        results['metrics'].update({
            'psnr_mean': float(np.mean(psnr_vals)),
            'psnr_std': float(np.std(psnr_vals)),
            'ssim_mean': float(np.mean(ssim_vals)),
            'ssim_std': float(np.std(ssim_vals)),
            'lpips_mean': float(np.mean(lpips_vals)),
            'lpips_std': float(np.std(lpips_vals)),
        })
    
    if abs_rel_vals:
        results['metrics'].update({
            'abs_rel_mean': float(np.mean(abs_rel_vals)),
            'abs_rel_std': float(np.std(abs_rel_vals)),
            'depth_rmse_mean': float(np.mean(depth_rmse_vals)),
            'depth_rmse_std': float(np.std(depth_rmse_vals)),
            'delta1_mean': float(np.mean(delta1_vals)),
            'delta1_std': float(np.std(delta1_vals)),
        })
    
    # Print summary
    print("\n" + "=" * 50)
    print(f"Level: {level}")
    if args.feature_only and 'mse_mean' in results['metrics']:
        print(f"MSE:    {results['metrics']['mse_mean']:.6f} ± {results['metrics']['mse_std']:.6f}")
        print(f"CosSim: {results['metrics']['cosine_similarity_mean']:.4f} ± {results['metrics']['cosine_similarity_std']:.4f}")
        if 'fid' in results['metrics']:
            print(f"FID:    {results['metrics']['fid']:.4f}")
    elif 'psnr_mean' in results['metrics']:
        print("--- RGB Metrics ---")
        print(f"PSNR:   {results['metrics']['psnr_mean']:.4f} ± {results['metrics']['psnr_std']:.4f}")
        print(f"SSIM:   {results['metrics']['ssim_mean']:.4f} ± {results['metrics']['ssim_std']:.4f}")
        print(f"LPIPS:  {results['metrics']['lpips_mean']:.4f} ± {results['metrics']['lpips_std']:.4f}")
    if 'abs_rel_mean' in results['metrics']:
        print("--- Depth Metrics (Pseudo-GT) ---")
        print(f"AbsRel: {results['metrics']['abs_rel_mean']:.4f} ± {results['metrics']['abs_rel_std']:.4f}")
        print(f"RMSE:   {results['metrics']['depth_rmse_mean']:.4f} ± {results['metrics']['depth_rmse_std']:.4f}")
        print(f"δ₁<1.25:{results['metrics']['delta1_mean']:.4f} ± {results['metrics']['delta1_std']:.4f}")
    print("=" * 50)
    
    # Save
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save per-scene metrics to CSV
    # import csv
    # per_scene_csv_path = os.path.join(output_dir, 'per_scene_metrics.csv')
    # with open(per_scene_csv_path, 'w', newline='') as f:
    #     if per_sample_metrics:
    #         fieldnames = ['scene_name', 'sample_idx', 'psnr', 'ssim', 'lpips']
    #         writer = csv.DictWriter(f, fieldnames=fieldnames)
    #         writer.writeheader()
    #         for metrics in per_sample_metrics:
    #             writer.writerow(metrics)
    # print(f"Per-scene metrics saved to: {per_scene_csv_path}")
    
    with open(os.path.join(output_dir, 'results.txt'), 'w') as f:
        f.write(f"Level: {level}\n")
        if args.feature_only and 'mse_mean' in results['metrics']:
            f.write(f"MSE: {results['metrics']['mse_mean']:.6f} ± {results['metrics']['mse_std']:.6f}\n")
            f.write(f"CosSim: {results['metrics']['cosine_similarity_mean']:.4f} ± {results['metrics']['cosine_similarity_std']:.4f}\n")
            if 'fid' in results['metrics']:
                f.write(f"FID: {results['metrics']['fid']:.4f}\n")
        elif 'psnr_mean' in results['metrics']:
            f.write("--- RGB Metrics ---\n")
            f.write(f"PSNR: {results['metrics']['psnr_mean']:.4f} ± {results['metrics']['psnr_std']:.4f}\n")
            f.write(f"SSIM: {results['metrics']['ssim_mean']:.4f} ± {results['metrics']['ssim_std']:.4f}\n")
            f.write(f"LPIPS: {results['metrics']['lpips_mean']:.4f} ± {results['metrics']['lpips_std']:.4f}\n")
        if 'abs_rel_mean' in results['metrics']:
            f.write("--- Depth Metrics (Pseudo-GT) ---\n")
            f.write(f"AbsRel: {results['metrics']['abs_rel_mean']:.4f} ± {results['metrics']['abs_rel_std']:.4f}\n")
            f.write(f"RMSE: {results['metrics']['depth_rmse_mean']:.4f} ± {results['metrics']['depth_rmse_std']:.4f}\n")
            f.write(f"δ₁<1.25: {results['metrics']['delta1_mean']:.4f} ± {results['metrics']['delta1_std']:.4f}\n")
        f.write(f"ckpt path: {args.checkpoint_level1}")
    
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GLD NVS Evaluation")
    
    # Config
    parser.add_argument("--eval_config", type=str, required=True,
                        help="Path to eval config YAML (e.g., configs/eval/dl3dv.yaml)")
    parser.add_argument("--model_config", type=str, default=None,
                        help="Path to model config (overrides eval_config.model.config)")
    parser.add_argument("--model_config_level0", type=str, default=None, help="Specific model config for Level 0")
    parser.add_argument("--model_config_level1", type=str, default=None, help="Specific model config for Level 1")
    parser.add_argument("--model_config_level2", type=str, default=None, help="Specific model config for Level 2")
    parser.add_argument("--model_config_level3", type=str, default=None, help="Specific model config for Level 3")
    parser.add_argument("--checkpoint_level0", type=str, default=None,
                        help="Path to level 0 model checkpoint (not needed for --use_cascade)")
    parser.add_argument("--checkpoint_level1", type=str, default=None,
                        help="Path to level 1 model checkpoint")
    parser.add_argument("--checkpoint_level2", type=str, default=None,
                        help="Path to level 2 model checkpoint (not needed for --use_cascade)")
    parser.add_argument("--checkpoint_level3", type=str, default=None,
                        help="Path to level 3 model checkpoint (not needed for --use_cascade)")
    parser.add_argument("--level", type=int, default=0, choices=[0,1,2,3],
                        help="GLD level to evaluate (0-3)")
    parser.add_argument("--difficulty", type=str, default=None,
                        choices=['easy', 'medium', 'hard', 'extreme'],
                        help="Difficulty preset (overrides default_difficulty)")
    parser.add_argument("--run_pca", type=str, default=False,
                        help="Run PCA")
    parser.add_argument("--cfg_scale_ar", type=float, default=None,
                        help="CFG scale override for L1→L0 Feature Flow (auto-regressive, defaults to cfg_scale)")
    parser.add_argument("--cfg_scale", type=float, default=None,
                        help="CFG scale override (from config if not specified)")
    # NEW: CFG camera drop and unconditional modes
    parser.add_argument("--use_camera_drop", type=lambda x: (str(x).lower() == 'true'), default=None,
                        help="Camera drop for CFG (true/false, from config if not specified)")
    parser.add_argument("--use_camera_drop_l1", type=lambda x: (str(x).lower() == 'true'), default=None,
                        help="L1 Diffusion camera drop for CFG (overrides use_camera_drop if specified)")
    parser.add_argument("--use_camera_drop_cascade", type=lambda x: (str(x).lower() == 'true'), default=None,
                        help="Feature Flow camera drop for CFG (overrides use_camera_drop if specified)")
    parser.add_argument("--cfg_l1_uncond_mode", type=str, default=None, choices=['keep', 'ref_noise'],
                        help="L1 Diffusion CFG unconditional mode")
    parser.add_argument("--cfg_cascade_uncond_mode", type=str, default=None, 
                        choices=['keep', 'ref_noise', 'source_noise', 'dual_noise'],
                        help="L1→L0 Feature Flow CFG unconditional mode")
    parser.add_argument("--shift_dim", type=float, default=None,
                        help="Shift dimension for DiT (experimental)")
    parser.add_argument("--cond_num", type=int, default=None,
                    help="Shift dimension for DiT (experimental)")
    # Pipeline Modes
    parser.add_argument("--eval_mode", type=str, default="independent",
                        choices=['independent', 'cascade'],
                        help="Evaluation mode: independent, cascade (flow), or cascade (propagation)")
    parser.add_argument("--use_cascade", action="store_true", help="DEPRECATED: Use --eval_mode cascade")
    parser.add_argument("--checkpoint_cascade", type=str, default=None,
                        help="Path to mapping (L1→L0) model checkpoint")
    parser.add_argument("--model_config_cascade", type=str, default=None, help="Specific model config for Mapping model")
    parser.add_argument("--cascade_noise_tau", type=float, default=0.0,
                        help="Noise tau for xT mapping start (only for cascade)")
    
    # Output
    parser.add_argument("--output_dir", type=str, default="results/nvs_eval",
                        help="Output directory")
    
    # Runtime
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Max samples to evaluate (for testing)")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="DataLoader workers")
    parser.add_argument("--save_gif", action='store_true', help="Save GIF")
    parser.add_argument("--feature_only", action="store_true",
                        help="Use feature-level validation (MSE+CosSim) instead of RGB validation")

    # Attention Visualization
    parser.add_argument("--visualize_attention", action="store_true",
                        help="Enable 3D attention map visualization during Level 1 generation")
    parser.add_argument("--attention_timesteps", type=str, default=None,
                        help="Comma-separated timestep indices to capture (e.g., '0,25,49'). Default: first,middle,last")
    parser.add_argument("--attention_layers", type=str, default=None,
                        help="Comma-separated layer IDs to capture (e.g., '0,2,4,28,30'). Default: every 2nd layer")
    parser.add_argument("--attention_query_view", type=int, default=3,
                        help="View index to query from for attention visualization (default: 3, first generated view)")
    parser.add_argument("--attention_query_positions", type=str, default=None,
                        help="Comma-separated query positions as 'h,w' pairs (e.g., '16,16:16,20'). Default: center+corners")

    args = parser.parse_args()
    main(args)
