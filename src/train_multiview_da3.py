# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
DA3 MVDiffusion training script - trains diffusion model on DA3 feature space.
Supports level-specific training (-1, -2, -3, -4) for multi-level feature reconstruction.
"""
import os
import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import sys

import math
from torch.cuda.amp import autocast
from omegaconf import OmegaConf
from stage1 import RAE_DA3
from stage2.models import Stage2ModelProtocol
from stage2.transport.transport import Transport, ModelType, WeightType, PathType, Sampler
from utils.train_utils import (
    parse_configs,
    create_transport,
    update_ema,
    requires_grad,
    cleanup,
    create_logger,
    center_crop_arr,
)
from utils.model_utils import instantiate_from_config
from utils import wandb_utils
from utils.optim_utils import build_optimizer, build_scheduler
from utils.config_utils import init_config_defaults, get_image_size_from_config

# added import
from video.video_load import DA3VideoDataset_Pose, create_multiview_dataloader
from pathlib import Path
from typing import Dict, Optional, Tuple
from tqdm import tqdm
from einops import rearrange
from disc import (
    LPIPS
)
import torch.nn.functional as F
from utils.metrics import compute_psnr, compute_ssim, compute_lpips
import wandb

from utils.camera.camera import get_camera_embedding
from datetime import datetime

def prepare_data(
    rae, images, intrinsic, extrinsic, device, 
    random_cond_num: int = 1, 
    return_cls: bool = False, 
    camera_mode: str = "camray",
    ray_pose_mode: str = "c2w",
    return_scale: bool = False
):
    """
    Prepare data for training.
    For DA3: Image 518x518, camera embedding at 518x518, which gets patchified to 37x37 by camera_embedder.
    Args:
        camera_mode: "camray" for direction only (3ch), "plucker" for [d, o×d] (6ch)
        return_scale: Whether to return the translation normalization scale factor.
    """
    # B, V, C, H, W = images.shape # Removed resize logic as per user request
    B, V, C, H, W = images.shape
    # Move inputs to device immediately
    images = images.to(device, non_blocking=True)
    intrinsic = intrinsic.to(device, non_blocking=True)
    extrinsic = extrinsic.to(device, non_blocking=True)
    # -------------------------------------------------------------------------
    # Encode Images
    # -------------------------------------------------------------------------
    # ImageNet Normalization is required for DA3 encoder
    # Using buffers from rae (1, 3, 1, 1) -> (1, 1, 3, 1, 1) for broadcasting
    images_norm = (images - rae.encoder_mean[None]) / rae.encoder_std[None]
    # 1. encode image - RAE_DA3 now accepts 5D input (B, V, C, H, W) directly
    with torch.no_grad():
        if random_cond_num is None:
            raise ValueError("random_cond_num (cond_num) must be provided explicitly.")
        cond_views = int(random_cond_num)
        if cond_views < 0:
            raise ValueError(f"cond_num must be >= 0, got {cond_views}")
        if cond_views > V:
            raise ValueError(
                f"cond_num exceeds number of views in batch: cond_num={cond_views}, V={V}. "
                "Fix your multiview config or dataloader."
            )
        
        # A. Encode ALL views together (to get Target GTs with cross-view info)
        if return_cls:
            latents_all, cls_all = rae.encode(images_norm, return_cls=True)
            # Reshape and concatenate CLS token: (BV, C, H, W) + (BV, C) -> (BV, C, N+1, 1)
            BV, C_lat, h_lat, w_lat = latents_all.shape
            latents_all = torch.cat([cls_all.reshape(BV, C_lat, 1, 1), 
                                    latents_all.reshape(BV, C_lat, h_lat * w_lat, 1)], dim=2)
        else:
            latents_all = rae.encode(images_norm)
            cls_all = None
            
        # B. Encode ONLY Reference views (to get leakage-free conditioning)
        if cond_views > 0:
            if return_cls:
                latents_ref, cls_ref = rae.encode(images_norm[:, :cond_views], return_cls=True)

                BrV = latents_ref.shape[0]
                latents_ref = torch.cat([cls_ref.reshape(BrV, C_lat, 1, 1), 
                                        latents_ref.reshape(BrV, C_lat, h_lat * w_lat, 1)], dim=2)
            else:
                latents_ref = rae.encode(images_norm[:, :cond_views])
        else:
            latents_ref = None
            
        # C. Merge: Reference part from 'ref-only' pass, Target part from 'all' pass
    # latents_all is (B*V, C, ...)
    BV = latents_all.shape[0]
    # Handle both spatial (4D) and sequence (4D with last dim 1) formats
    if latents_all.ndim == 4 and latents_all.shape[3] == 1:
        # Sequence format (with CLS)
        _, C_lat, N_plus_1, _ = latents_all.shape
        latents_all_5d = latents_all.reshape(B, V, C_lat, N_plus_1, 1)
        latents_cond_5d = torch.zeros(B, V, C_lat, N_plus_1, 1, device=device, dtype=latents_all.dtype)
        if latents_ref is not None:
            latents_ref_5d = latents_ref.reshape(B, cond_views, C_lat, N_plus_1, 1)
            latents_cond_5d[:, :cond_views] = latents_ref_5d
            latents_cond = latents_cond_5d.reshape(BV, C_lat, N_plus_1, 1)
    else:
        # Spatial format
        _, C_lat, h_lat, w_lat = latents_all.shape
        latents_all_5d = latents_all.reshape(B, V, C_lat, h_lat, w_lat)
        latents_cond_5d = torch.zeros(B, V, C_lat, h_lat, w_lat, device=device, dtype=latents_all.dtype)
        if latents_ref is not None:
            latents_ref_5d = latents_ref.reshape(B, cond_views, C_lat, h_lat, w_lat)
            latents_cond_5d[:, :cond_views] = latents_ref_5d
        latents_cond = latents_cond_5d.reshape(BV, C_lat, h_lat, w_lat)

    # 2. get camera embedding at IMAGE resolution
    # The camera_embedder in DiT model will patchify this to match latent size
    if extrinsic.shape[-2:] == (4, 4):
        extrinsic = extrinsic[..., :3, :4]
    elif extrinsic.shape[-2:] != (3, 4):
        raise ValueError(f"Unexpected extrinsic shape {extrinsic.shape}.")

    if intrinsic.shape[-1] == 4:
        fx, fy, cx, cy = intrinsic.unbind(dim=-1)
        zeros = torch.zeros_like(fx)
        ones = torch.ones_like(fx)
        intrinsic_mat = torch.stack(
            (
                torch.stack((fx, zeros, cx), dim=-1),
                torch.stack((zeros, fy, cy), dim=-1),
                torch.stack((zeros, zeros, ones), dim=-1),
            ),
            dim=-2,
        )
    elif intrinsic.shape[-2:] == (3, 3):
        intrinsic_mat = intrinsic
    else:
        raise ValueError(f"Unexpected intrinsic shape {intrinsic.shape}.")

    extri_ = rearrange(extrinsic, "b f c1 c2 -> (b f) c1 c2", f=V)
    intri_ = rearrange(intrinsic_mat, "b f c1 c2 -> (b f) c1 c2", f=V)

    # Generate camera embedding at IMAGE resolution (H, W)
    # camera_embedder in DiT will patchify to latent resolution
    # camera_mode: "camray" = direction only (3ch), "plucker" = [d, o×d] (6ch)
    if return_scale:
        camera_embedding, scale = get_camera_embedding(intri_, extri_, B, V, H, W, mode=camera_mode, ray_pose_mode=ray_pose_mode, return_scale=True)
    else:
        camera_embedding = get_camera_embedding(intri_, extri_, B, V, H, W, mode=camera_mode, ray_pose_mode=ray_pose_mode) 
        scale = None
        
    camera_embedding = rearrange(camera_embedding, "b f c h w -> (b f) c h w")

    # 3. make mask for camera distinguishing cond or tgt (at IMAGE resolution)
    if random_cond_num is None:
        raise ValueError("random_cond_num (cond_num) must be provided explicitly.")
    cond_views = int(random_cond_num)
    if cond_views < 0:
        raise ValueError(f"cond_num must be >= 0, got {cond_views}")
    if cond_views > V:
        raise ValueError(
            f"cond_num exceeds number of views in batch: cond_num={cond_views}, V={V}. "
            "Fix your multiview config or dataloader."
        )
    random_masks = torch.ones((B, V, 1, H, W), device=device, dtype=latents_all.dtype)
    random_masks[:, :cond_views] = 0
    random_masks = random_masks.reshape(B * V, 1, H, W)

    camera_embedding = torch.cat([random_masks, camera_embedding], dim=1)  # (B*V, 7, H, W)

    # Return (latents_cond, latents_all, camera_embedding)
    # 1. latents_cond: [ref from ref-only pass | zeros for tgt]
    # 2. latents_all: all views from all-pass (GT for global features)
    # 3. camera_embedding: plucker/camray + mask

    # Construct latents_cond (handles both spatial and sequence/packed formats)
    if latents_all.ndim == 4 and latents_all.shape[3] == 1:
        # Sequence/packed format: (BV, C, K+N, 1)
        _, C_lat, seq_len, _ = latents_all.shape
        latents_cond_5d = torch.zeros(B, V, C_lat, seq_len, 1, device=device, dtype=latents_all.dtype)
        if latents_ref is not None:
            latents_ref_5d = latents_ref.reshape(B, cond_views, C_lat, seq_len, 1)
            latents_cond_5d[:, :cond_views] = latents_ref_5d
        latents_cond = latents_cond_5d.reshape(BV, C_lat, seq_len, 1)
    else:
        # Spatial format: (BV, C, h, w)
        _, C_lat, h_lat, w_lat = latents_all.shape
        latents_cond_5d = torch.zeros(B, V, C_lat, h_lat, w_lat, device=device, dtype=latents_all.dtype)
        if latents_ref is not None:
            latents_ref_5d = latents_ref.reshape(B, cond_views, C_lat, h_lat, w_lat)
            latents_cond_5d[:, :cond_views] = latents_ref_5d
        latents_cond = latents_cond_5d.reshape(BV, C_lat, h_lat, w_lat)

    return latents_cond, latents_all, camera_embedding

def make_vis_multiview(gt, recon):
    """
    gt, recon: shape (2, C, H, W)
    returns numpy image: (H*2, W*2, C)
    """

    rows = []
    for i in range(2):    # two views
        # (C, H, W) → (H, W, C)
        gt_i = gt[i].permute(1, 2, 0)
        rc_i = recon[i].permute(1, 2, 0)
        
        # horizontal concat: (H, W*2, C)
        row = torch.cat([gt_i, rc_i], dim=1)
        rows.append(row)

    # vertical concat final output: (H*2, W*2, C)
    grid = torch.cat(rows, dim=0)

    return grid.cpu().numpy()


################################################################################
#                                  Training Loop                                #
################################################################################


def main(args):
    """Trains a new SiT model using config-driven hyperparameters."""
    if not torch.cuda.is_available():
        raise RuntimeError("Training currently requires at least one GPU.")
    cfg = OmegaConf.load(args.config)
    (
        rae_config,
        model_config,
        transport_config,
        sampler_config,
        guidance_config_top,
        pag_config_top,
        misc_config,
        training_config,
        validation_config,
        multiview_config,
        dataset_config,
    ) = parse_configs(args.config)
    # Initialize derived config defaults (auto-compute input_size, cam_*, latent_size, etc.)
    # Supports 2D encoder sizes (H, W)
    # Note: init_config_defaults returns RAE encoder size. We use this as default for dataloader
    # unless dataset config explicitly overrides it.
    # Auto-detect backbone type from config target
    rae_target = rae_config.get('target', '')
    is_da3 = 'rae_da3' in rae_target.lower() or 'rae_vggt' in rae_target.lower()
    encoder_h, encoder_w = init_config_defaults(
        rae_config, model_config, misc_config, patch_size=14, is_da3=is_da3
    )

    # Dataset configuration with defaults for backward compatibility
    dataset_cfg = dataset_config or {}
    dataset_name = dataset_cfg.get("name", "re10k")
    train_video_path = dataset_cfg.get("train_video_path", "data/re10k/training_256")
    train_pose_path = dataset_cfg.get("train_pose_path", "data/re10k/training_poses")
    val_video_path = dataset_cfg.get("val_video_path", "data/re10k/test_256")
    val_pose_path = dataset_cfg.get("val_pose_path", "data/re10k/test_poses")

    # Dataset configuration with defaults for backward compatibility
    dataset_cfg = dataset_config or {}
    dataset_name = dataset_cfg.get("name", "re10k")
    train_video_path = dataset_cfg.get("train_video_path", "data/re10k/training_256")
    train_pose_path = dataset_cfg.get("train_pose_path", "data/re10k/training_poses")
    val_video_path = dataset_cfg.get("val_video_path", "data/re10k/test_256")
    val_pose_path = dataset_cfg.get("val_pose_path", "data/re10k/test_poses")

    # Determine Dataloader Image Size (STRICT MODE)
    # Priority:
    # 1. dataset.image_size (Supports int, [h,w], or list of candidates)
    # NO FALLBACKS allowed.
    if "image_size" not in dataset_cfg:
        raise ValueError(
            "Config validation failed: 'dataset.image_size' is missing. "
            "Please explicitly specify 'image_size' in the 'dataset' section of your config. "
            "It can be an integer, a [h, w] list, or a list of candidate resolutions."
        )
        
    dl_image_size = dataset_cfg["image_size"]
    if OmegaConf.is_config(dl_image_size):
        dl_image_size = OmegaConf.to_container(dl_image_size, resolve=True)

    # Prepare dataset kwargs for create_multiview_dataloader
    # Exclude 'image_size' as it's passed explicitly
    dataset_kwargs = OmegaConf.to_container(dataset_cfg, resolve=True)
    # Filter out keys that are explicitly passed to create_multiview_dataloader or not needed by dataset
    for key in ["name", "image_size", "num_views", "cond_num", "ref_view_sampling", "camera_mode", "normalize_pose"]:
        if key in dataset_kwargs:
            del dataset_kwargs[key]

    # Update args for logging (best effort)
    if isinstance(dl_image_size, (int, float)):
        args.image_size = int(dl_image_size)
        args.image_height = int(dl_image_size)
        args.image_width = int(dl_image_size)
    elif isinstance(dl_image_size, (list, tuple)) and len(dl_image_size) == 2 and isinstance(dl_image_size[0], (int, float)):
        # Single rectangular resolution
        args.image_height = int(dl_image_size[0])
        args.image_width = int(dl_image_size[1])
        args.image_size = args.image_height # Ambiguous but set to H
    else:
        # List of candidates or complex structure
        args.image_size = dl_image_size 
        args.image_height = -1 # Indicating dynamic/multiple
        args.image_width = -1

    if rae_config is None or model_config is None:
        raise ValueError("Config must provide both stage_1 and stage_2 sections.")

    def to_dict(cfg_section):
        if cfg_section is None:
            return {}
        if not OmegaConf.is_config(cfg_section):
            return cfg_section
        return OmegaConf.to_container(cfg_section, resolve=True)

    misc = to_dict(misc_config)
    transport_cfg = to_dict(transport_config)
    sampler_cfg = to_dict(sampler_config)
    training_cfg = to_dict(training_config)
    validation_cfg = to_dict(validation_config)
    multiview_cfg = to_dict(multiview_config)
    # Guidance and PAG (Prioritize validation section, then top-level)
    guidance_cfg = to_dict(validation_cfg.get("guidance", guidance_config_top))
    pag_cfg = to_dict(validation_cfg.get("pag", pag_config_top))
    #
    use_prope = False
    if 'use_prope' in model_config.get('params', {}):
        if model_config.params.use_prope:
            use_prope = True
            multiview_cfg['use_prope'] = True
            # logger.info("Enable ProPE from Config") # Logger not init yet
            print("Enable ProPE from Config")
    # Update args for consistency (optional but good for logging)
    args.use_prope = use_prope

    # training config parameters
    global_batch_size = int(training_cfg.get("global_batch_size", 256))
    grad_accum_steps = int(training_cfg.get("grad_accum_steps", 1))
    num_epochs = int(training_cfg.get("epochs", 100))
    ema_decay = float(training_cfg.get("ema_decay", 0.9999))
    num_workers = int(training_cfg.get("num_workers", 4))
    log_every = int(training_cfg.get("log_every", 100))
    clip_grad = float(training_cfg.get("clip_grad", 1.0))
    default_seed = int(training_cfg.get("global_seed", 0))

    # validation config parameters
    ckpt_every = int(validation_cfg.get("ckpt_every", training_cfg.get("ckpt_every", 5000)))
    sample_every = int(validation_cfg.get("sample_every", training_cfg.get("sample_every", 100)))
    val_num_batches = validation_cfg.get("val_num_batches", None)
    val_mode = validation_cfg.get("validation_mode", "propagation") 
    # Overrides
    cfg_scale_override = training_cfg.get("cfg_scale", None) 
    # minkyung: multi-view config (can also be in dataset section)
    total_view = int(multiview_cfg.get("total_view", dataset_cfg.get("num_views", 2)))
    latent_size = tuple(int(dim) for dim in misc.get("latent_size", (768, 16, 16)))


    #
    # SNR' = SNR / V 효과를 위해 shift_dim = (H*W*C) * V 로 설정
    shift_dim = misc.get("time_dist_shift_dim", math.prod(latent_size) * total_view)
    shift_base = misc.get("time_dist_shift_base", 4096)
    time_dist_shift = math.sqrt(shift_dim / shift_base)
    # Dynamic cond_num: supports int or string range like "2-4"
    cond_num_raw = multiview_cfg.get("cond_num", dataset_cfg.get("cond_num", 2))
    if isinstance(cond_num_raw, str) and "-" in cond_num_raw:
        cond_num_min, cond_num_max = map(int, cond_num_raw.split("-"))
        cond_num = None  # Will be sampled per batch
    else:
        cond_num = int(cond_num_raw)
        cond_num_min, cond_num_max = cond_num, cond_num
    if "ref_view_sampling" not in multiview_cfg and "ref_view_sampling" not in dataset_cfg:
        raise ValueError("Config must set multiview.ref_view_sampling or dataset.ref_view_sampling explicitly (prefix|random|interpolate).")
    ref_view_sampling = multiview_cfg.get("ref_view_sampling", dataset_cfg.get("ref_view_sampling", "prefix"))
    if not isinstance(ref_view_sampling, str):
        raise TypeError(f"multiview.ref_view_sampling must be str, got {type(ref_view_sampling)}")
    ref_view_sampling = ref_view_sampling.lower()
    if ref_view_sampling not in {"prefix", "random", "interpolate"}:
        raise ValueError(
            f"Unknown multiview.ref_view_sampling={ref_view_sampling}. Use one of: prefix, random, interpolate."
        )
    use_prope = bool(multiview_cfg.get("use_prope", False))
    camera_mode = multiview_cfg.get("camera_mode", dataset_cfg.get("camera_mode", "camray")).lower()
    if camera_mode not in {"camray", "plucker"}:
        raise ValueError(f"Unknown camera_mode={camera_mode}. Use 'camray' or 'plucker'.")
    # Training always uses c2w mode (datasets provide OpenCV c2w at load time)
    ray_pose_mode = "c2w"
    # Feature-to-Feature Flow Matching: Optional source level conditioning
    # source_level: If set, use features from this level (+ noise) as x0 instead of pure noise
    source_level = training_cfg.get("source_level", None)
    if source_level is not None:
        source_level = int(source_level)
    # source_level_stat_path: Normalization stat path for SOURCE level features
    # IMPORTANT: Source and target levels have DIFFERENT distributions, must use different norm stats!
    source_level_stat_path = training_cfg.get("source_level_stat_path", None)
    if source_level is not None and source_level_stat_path is None:
        # Use print since logger is not initialized yet
        print(
            f"[WARNING] source_level={source_level} is set but source_level_stat_path is not specified! "
            "This may cause incorrect normalization for Feature-to-Feature Flow."
        )
    # Cache source level stats at startup (avoid per-batch disk I/O)
    _cached_source_stats = None
    if source_level is not None and source_level_stat_path is not None:
        _cached_source_stats = torch.load(source_level_stat_path, map_location='cpu')
        print(f"[INFO] Cached source level stats from: {source_level_stat_path}")
    # NEW: Feature Propagation Mode (Image 1 Architecture)
    # source_condition_level: If set, use features from this level as ENCODER CONDITION (not x0 init)
    # This is different from source_level which uses features for x0 initialization
    source_condition_level = training_cfg.get("source_condition_level", None)
    if source_condition_level is not None:
        source_condition_level = int(source_condition_level)
    source_condition_stat_path = training_cfg.get("source_condition_stat_path", None)
    if source_condition_level is not None and source_condition_stat_path is None:
        # Use print since logger is not initialized yet
        print(
            f"[WARNING] source_condition_level={source_condition_level} is set but source_condition_stat_path is not specified! "
            "This may cause incorrect normalization for Feature Propagation Mode."
        )
    # Cache source condition stats at startup (avoid per-batch disk I/O)
    _cached_source_cond_stats = None
    if source_condition_level is not None and source_condition_stat_path is not None:
        _cached_source_cond_stats = torch.load(source_condition_stat_path, map_location='cpu')
        print(f"[INFO] Cached source condition stats from: {source_condition_stat_path}")
    # Noise augmentation: sigma ~ |N(0, tau^2)| (following Dinh et al., Ho et al., Zhai et al.)
    # This smooths the latent distribution for better decoder generalization
    noise_tau_gt_feat = float(training_cfg.get("noise_tau_gt_feat", 0.0))
    global_seed = args.global_seed if args.global_seed is not None else default_seed

    if grad_accum_steps < 1:
        raise ValueError("Gradient accumulation steps must be >= 1.")
    # DA3 uses patch_size=14, so image dimensions must be divisible by 14
    # Skip this check for multi-resolution mode (indicated by -1)
    if args.image_height != -1 and args.image_width != -1:
        if args.image_height % 14 != 0 or args.image_width % 14 != 0:
            raise ValueError(
                f"Image dimensions ({args.image_height}x{args.image_width}) must be divisible by 14 for DA3."
            )
    #
    # if global_batch_size % (world_size * grad_accum_steps) != 0:
    #     raise ValueError(f"Global batch size {global_batch_size} must be divisible by world_size * grad_accum_steps ({world_size} * {grad_accum_steps} = {world_size * grad_accum_steps})")
    dist.init_process_group("nccl")
    world_size = dist.get_world_size()

    rank = dist.get_rank()
    device_idx = rank % torch.cuda.device_count()
    torch.cuda.set_device(device_idx)
    device = torch.device("cuda", device_idx)

    seed = global_seed * world_size + rank
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if rank == 0:
        print(f"Starting rank={rank}, seed={seed}, world_size={world_size}.")

    micro_batch_size = global_batch_size // (world_size * grad_accum_steps)
    use_bf16 = args.precision == "bf16"
    if use_bf16 and not torch.cuda.is_bf16_supported():
        raise ValueError("Requested bf16 precision, but the current CUDA device does not support bfloat16.")
    autocast_kwargs = dict(dtype=torch.bfloat16, enabled=use_bf16)
    latent_dtype = autocast_kwargs["dtype"] if use_bf16 else torch.float32

    transport_params = dict(transport_cfg.get("params", {}))
    path_type = transport_params.get("path_type", "Linear")
    prediction = transport_params.get("prediction", "velocity")
    loss_weight = transport_params.get("loss_weight")
    transport_params.pop("time_dist_shift", None)
    is_concat_mode = model_config.get('params', {}).get('is_concat_mode', False)

    sampler_mode = sampler_cfg.get("mode", "ODE").upper()
    sampler_params = dict(sampler_cfg.get("params", {}))

    guidance_scale = float(guidance_cfg.get("scale", 1.0))
    if cfg_scale_override is not None:
        guidance_scale = float(cfg_scale_override)
    guidance_method = guidance_cfg.get("method", "cfg")

    def guidance_value(key: str, default: float) -> float:
        if key in guidance_cfg:
            return guidance_cfg[key]
        dashed_key = key.replace("_", "-")
        return guidance_cfg.get(dashed_key, default)

    t_min = float(guidance_value("t_min", 0.0))
    t_max = float(guidance_value("t_max", 1.0))

    # PAG config (from validation or top level)
    pag_scale = float(pag_cfg.get("scale", 0.0))
    pag_layer_idx = pag_cfg.get("layer_idx", 22)

    experiment_name = None
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        experiment_name = f"{args.run_name}-{timestamp}"

    # Broadcast experiment_name to all ranks to ensure consistent experiment_dir
    # This prevents the "experiment_dir is None" crash on non-zero ranks
    name_obj = [experiment_name]

    dist.broadcast_object_list(name_obj, src=0)
    experiment_name = name_obj[0]
 
    experiment_dir = os.path.join(args.results_dir, experiment_name)

    if rank == 0:
        checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)

    logger = create_logger(experiment_dir)
    if rank == 0:
        logger.info(f"Experiment directory created at {experiment_dir}")

    # ---------------------------------------------------------------------------------
    # 2. Stage-1 RAE (DA3) Setup
    # ---------------------------------------------------------------------------------
    #
    # Pass 'level' from stage2 config to stage1 config if present
    if 'params' in model_config and 'level' in model_config.params:
        if 'params' not in rae_config:
            rae_config.params = OmegaConf.create({})
        # Ensure we don't overwrite the entire params dict
        rae_config.params.level = model_config.params.level
        if rank == 0:
            logger.info(f"Passed level={rae_config.params.level} from Stage 2 to Stage 1 RAE.")

    # Inject cam_in_channels based on camera_mode (camray: 4ch, plucker: 7ch)
    cam_in_channels = 4 if camera_mode == "camray" else 7
    if 'params' not in model_config:
        model_config.params = OmegaConf.create({})
    model_config.params.cam_in_channels = cam_in_channels
    if rank == 0:
        logger.info(f"Camera mode: {camera_mode}, cam_in_channels: {cam_in_channels}")

    rae = instantiate_from_config(rae_config).to(device)

    rae.eval()
    lpips = LPIPS().to(device)
    lpips.eval()

    # Ensure model config is resolved to primitives (fixes ListConfig issues in ProPE)
    if OmegaConf.is_config(model_config):
        model_config = OmegaConf.to_container(model_config, resolve=True)

    model: Stage2ModelProtocol = instantiate_from_config(model_config).to(device)
    ema = deepcopy(model).to(device)
    requires_grad(ema, False)
    opt_state = None
    sched_state = None
    train_steps = 0
    ckpt_meta = None

    if args.pretrained is not None: 
        checkpoint = torch.load(args.pretrained, map_location="cpu")
        state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
        state_dict = {
            k: v for k, v in state_dict.items()
            if not k.startswith("y_embedder.")
        }
        for k in state_dict:
            state_dict[k] = state_dict[k].to(torch.bfloat16)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if args.ckpt is not None:
        checkpoint = torch.load(args.ckpt, map_location="cpu")
        ckpt_meta = checkpoint
        if "model" in checkpoint:
            model.load_state_dict(checkpoint["model"])
        if "ema" in checkpoint:
            ema.load_state_dict(checkpoint["ema"])
        opt_state = checkpoint.get("opt")
        sched_state = checkpoint.get("scheduler")
        train_steps = int(checkpoint.get("train_steps", 0))

    model_param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"Model Parameters: {model_param_count/1e6:.2f}M")

    model = DDP(model, device_ids=[device_idx], gradient_as_bucket_view=False)

    opt, opt_msg = build_optimizer(model.parameters(), training_cfg)
    if opt_state is not None:
        opt.load_state_dict(opt_state)

    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    # dataset = ImageFolder(args.data_path, transform=transform)
    # sampler = DistributedSampler(
    #     dataset,
    #     num_replicas=world_size,
    #     rank=rank,
    #     shuffle=True,
    #     seed=global_seed,
    #     )
    # loader = DataLoader(
    #     dataset,
    #     batch_size=micro_batch_size,
    #     shuffle=False,
    #     sampler=sampler,
    #     num_workers=num_workers,
    #     pin_memory=True,
    #     drop_last=True,
    # )
    if args.overfit: 
        #
        # Pass tuple (H, W) to ensure DummyMultiviewDataset picks the correct resolution
        # args.image_size might be int (legacy), but we want explicit (h, w)
        
        loader, sampler = create_multiview_dataloader(
            dataset_name=dataset_name,
            video_path=val_video_path,
            pose_path=val_pose_path,
            image_size=dl_image_size,
            num_views=total_view,
            cond_num=cond_num,
            ref_view_sampling=ref_view_sampling,
            batch_size=micro_batch_size,
            workers=num_workers,
            rank=rank,
            world_size=world_size,
            shuffle=False,
            overfit=True,
            desired_steps=1000,
        )
        val_loader, _ = create_multiview_dataloader(
            dataset_name=dataset_name,
            video_path=val_video_path,
            pose_path=val_pose_path,
            image_size=dl_image_size,
            num_views=total_view,
            cond_num=cond_num,
            ref_view_sampling=ref_view_sampling,
            batch_size=micro_batch_size,
            workers=num_workers,
            rank=rank,
            world_size=world_size,
            shuffle=False,
            overfit=True,
            desired_steps=1,
            **dataset_kwargs,
        )
    else:
        #
        loader, sampler = create_multiview_dataloader(
            dataset_name=dataset_name,
            video_path=train_video_path,
            pose_path=train_pose_path,
            image_size=dl_image_size,
            num_views=total_view,
            cond_num=cond_num,
            ref_view_sampling=ref_view_sampling,
            batch_size=micro_batch_size,
            workers=num_workers,
            rank=rank,
            world_size=world_size,
            **dataset_kwargs,
        )
        #
        val_num_views = int(validation_cfg.get("num_views", total_view))
        val_cond_num = validation_cfg.get("cond_num", cond_num)
        if val_cond_num is None: # 가변 범위인 경우 중간값 사용
            val_cond_num = (cond_num_min + cond_num_max) // 2
        else:
            val_cond_num = int(val_cond_num)
        val_test_baseline = int(validation_cfg.get("test_baseline", 40))
        val_min_interval = validation_cfg.get("min_interval", 1)
        val_max_interval = validation_cfg.get("max_interval", val_test_baseline) # Fallback to test_baseline if max_interval not set
        val_seed = validation_cfg.get("seed", None)
        

        val_batch_size = int(validation_cfg.get("batch_size", 1))
        
        # 전체 55개를 보장하기 위해 각 rank가 수행할 batch 개수 계산
        total_val_batches = int(validation_cfg.get("val_num_batches", 55))
        val_num_batches_per_rank = total_val_batches // world_size
        if rank < (total_val_batches % world_size):
            val_num_batches_per_rank += 1

        # 검증 시에는 멀티 해상도 리스트 중 첫 번째 해상도로 고정
        val_image_size = dl_image_size[0] if isinstance(dl_image_size, (list, tuple)) else dl_image_size
        
        # Prepare validation-specific dataset kwargs
        val_dataset_kwargs = deepcopy(dataset_kwargs)
        val_dataset_kwargs.update({
            "min_interval": val_min_interval,
            "max_interval": val_max_interval,
        })

        val_loader, _ = create_multiview_dataloader(
            dataset_name=dataset_name,
            video_path=val_video_path,
            pose_path=val_pose_path,
            image_size=val_image_size,
            num_views=val_num_views,
            cond_num=val_cond_num,
            ref_view_sampling=ref_view_sampling,
            batch_size=val_batch_size,
            workers=num_workers,
            rank=rank,
            world_size=world_size,
            shuffle=False,
            mode='test',
            test_baseline=val_test_baseline,
            seed=val_seed,
            **val_dataset_kwargs,
        )
        
        
    # logger.info(f"Dataset contains {len(dataset):,} images ({args.data_path})")
    logger.info(
        f"Gradient accumulation: steps={grad_accum_steps}, micro batch={micro_batch_size}, "
        f"per-GPU batch={micro_batch_size * grad_accum_steps}, global batch={global_batch_size}"
    )
    logger.info(f"Precision mode: {args.precision}")
    loader_batches = len(loader)
    if loader_batches % grad_accum_steps != 0:
        # raise ValueError("Number of loader batches must be divisible by grad_accum_steps when drop_last=True.")
        logger.warning("Number of loader batches must be divisible by grad_accum_steps when drop_last=True.")
    steps_per_epoch = loader_batches // grad_accum_steps
    if steps_per_epoch <= 0:
        raise ValueError("Gradient accumulation configuration results in zero optimizer steps per epoch.")
    schedl, sched_msg = build_scheduler(opt, steps_per_epoch, training_cfg, sched_state)
    if rank == 0:
        logger.info(f"Training configured for {num_epochs} epochs, {steps_per_epoch} steps per epoch.")
        logger.info(opt_msg + "\n" + sched_msg)

    # ---------------------------------------------------------------------------------
    # Resume logic (epoch-aware)
    #
    # - We checkpoint `train_steps` (optimizer steps), but the dataset/sampler are epoch-based.
    # - To resume deterministically, map:
    #     start_epoch = train_steps // steps_per_epoch
    #     step_in_epoch = train_steps % steps_per_epoch
    #     micro_batches_to_skip = step_in_epoch * grad_accum_steps
    # ---------------------------------------------------------------------------------
    start_epoch = 0
    micro_batches_to_skip = 0
    if args.ckpt is not None and train_steps > 0:
        start_epoch = train_steps // steps_per_epoch
        step_in_epoch = train_steps % steps_per_epoch
        micro_batches_to_skip = int(step_in_epoch * grad_accum_steps)
        if start_epoch >= num_epochs:
            raise ValueError(
                f"Resume requested but train_steps={train_steps} maps to start_epoch={start_epoch} which is >= num_epochs={num_epochs}. "
                f"steps_per_epoch={steps_per_epoch}, grad_accum_steps={grad_accum_steps}."
            )
        if micro_batches_to_skip >= loader_batches:
            raise ValueError(
                f"Resume skip exceeds epoch length: micro_batches_to_skip={micro_batches_to_skip} >= loader_batches={loader_batches}. "
                f"train_steps={train_steps}, step_in_epoch={step_in_epoch}, steps_per_epoch={steps_per_epoch}, grad_accum_steps={grad_accum_steps}."
            )
        if rank == 0:
            logger.info(
                f"Resuming from ckpt: train_steps={train_steps} -> start_epoch={start_epoch}, "
                f"skip_micro_batches={micro_batches_to_skip} (step_in_epoch={step_in_epoch}, grad_accum_steps={grad_accum_steps})."
            )

    transport = create_transport(
        **transport_params,
        time_dist_shift=time_dist_shift,
    )
    transport_sampler = Sampler(transport)

    if sampler_mode == "ODE":
        # Original RAE code already flips time in ode.__init__ with `1 - linspace()`
        # So no need for reverse=True here
        eval_sampler = transport_sampler.sample_ode(**sampler_params)
    elif sampler_mode == "SDE":
        eval_sampler = transport_sampler.sample_sde(**sampler_params)
    # elif sampler_mode == "ODE_MULTI":
    #     eval_sampler = transport_sampler.sample_ode_multiview(**sampler_params)
    else:
        raise NotImplementedError(f"Invalid sampling mode {sampler_mode}.")

    guid_model_forward = None
    if guidance_scale > 1.0 and guidance_method == "autoguidance":
        guidance_model_cfg = guidance_cfg.get("guidance_model")
        if guidance_model_cfg is None:
            raise ValueError("Please provide a guidance model config when using autoguidance.")
        guid_model: Stage2ModelProtocol = instantiate_from_config(guidance_model_cfg).to(device)
        guid_model.eval()
        guid_model_forward = guid_model.forward

    update_ema(ema, model.module, decay=0)
    model.train()
    ema.eval()

    # Overfit mode: compute mean/var from first batch for normalization - REMOVED per user request
    # if args.overfit:
    #     logger.info("Overfit mode: Computing normalization statistics from first batch...")
    #     ... (logic removed to use global stats instead)

    log_steps = 0
    running_loss = 0.0
    running_ref_loss = 0.0
    running_tgt_loss = 0.0
    wandb_initialized = False  # Track wandb initialization state for resume support
    start_time = time()

    # ys = torch.randint(num_classes, size=(micro_batch_size,), device=device)
    # using_cfg = guidance_scale > 1.0
    # n = ys.size(0)
    # zs = torch.randn(n, *latent_size, device=device, dtype=latent_dtype)

    # if using_cfg:
    #     zs = torch.cat([zs, zs], dim=0)
    #     y_null = torch.full((n,), null_label, device=device)
    #     ys = torch.cat([ys, y_null], dim=0)
    #     sample_model_kwargs = dict(
    #         y=ys,
    #         cfg_scale=guidance_scale,
    #         cfg_interval=(t_min, t_max),
    #     )
    #     if guidance_method == "autoguidance":
    #         if guid_model_forward is None:
    #             raise RuntimeError("Guidance model forward is not initialized.")
    #         sample_model_kwargs["additional_model_forward"] = guid_model_forward
    #         model_fn = ema.forward_with_autoguidance
    #     else:
    #         model_fn = ema.forward_with_cfg
    # else:
    #     sample_model_kwargs = dict(y=ys)
    #     model_fn = ema.forward

    logger.info(f"Training for {num_epochs} epochs...")
    for epoch in range(start_epoch, num_epochs):
        model.train()
        
        # Fast resume: compute batches to skip for this epoch
        batches_to_skip_this_epoch = micro_batches_to_skip if epoch == start_epoch else 0
        
        # Check if sampler supports fast resume (CUT3R BatchedRandomSampler)
        use_fast_resume = False
        if hasattr(sampler, 'set_epoch'):
            import inspect
            sig = inspect.signature(sampler.set_epoch)
            if 'start_batch_idx' in sig.parameters:
                use_fast_resume = True
                sampler.set_epoch(epoch, start_batch_idx=batches_to_skip_this_epoch)
                if batches_to_skip_this_epoch > 0 and rank == 0:
                    logger.info(f"[Fast Resume] Skipping {batches_to_skip_this_epoch} batches via sampler.set_epoch()")
            else:
                sampler.set_epoch(epoch)
        
        # Also set epoch for dataset if available (required for CUT3R multi-res)
        if hasattr(loader.dataset, "set_epoch"):
            loader.dataset.set_epoch(epoch)
        elif hasattr(loader.dataset, "dataset") and hasattr(loader.dataset.dataset, "set_epoch"):
             loader.dataset.dataset.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        opt.zero_grad()
        accum_counter = 0
        step_loss_accum = 0.0
        step_ref_loss_accum = 0.0
        step_tgt_loss_accum = 0.0
        if rank == 0:
        # Wrap the loader with tqdm on Rank 0 only
        # Set initial to show correct position after fast resume
            pbar = tqdm(loader, total=steps_per_epoch, initial=batches_to_skip_this_epoch, 
                       desc=f"Epoch {epoch}/{num_epochs}")
        else:
            # Use the raw loader on other ranks
            pbar = loader
        pbar_iter = iter(pbar)
        
        # Slow fallback: skip via next() if sampler doesn't support fast resume
        if not use_fast_resume and batches_to_skip_this_epoch > 0:
            if rank == 0:
                logger.info(f"[Slow Resume] Skipping {batches_to_skip_this_epoch} micro-batches via next()...")
            for _ in range(batches_to_skip_this_epoch):
                next(pbar_iter)
        
        # Clear skip count after first epoch
        micro_batches_to_skip = 0

        for batch in pbar_iter:
            # Dynamic cond_num: sample per batch if range was specified
            if cond_num is None:
                import random
                batch_cond_num = random.randint(cond_num_min, cond_num_max)
            else:
                batch_cond_num = cond_num
            
            # CUT3R batch conversion (List[Dict] -> Dict)
            if isinstance(batch, list) and isinstance(batch[0], dict):
                from video.cut3r_adapter import convert_cut3r_batch
                batch = convert_cut3r_batch(batch, batch_cond_num, ref_view_sampling)
            
            # image_dict, dict_keys(['enc_inp', 'gt_inp', 'fxfycxcy_252', 'c2w', 'video_id', 'frame_indices'])
            image = batch['gt_inp'] 
            
            intrinsic = batch['fxfycxcy']
            extrinsic = batch['c2w']

            # ------------------------------------------------------------------
            # Fail loudly on view-count mismatches (critical for multiview runs)
            # ------------------------------------------------------------------
            if image.ndim != 5:
                raise ValueError(f"Expected gt_inp shape (B, V, C, H, W), got {tuple(image.shape)}")
            B_batch, V_batch = int(image.shape[0]), int(image.shape[1])
            if V_batch != int(total_view):
                raise ValueError(
                    f"Config/Dataloader mismatch: total_view={int(total_view)} but dataloader returned V={V_batch}. "
                    "This is unsafe for DDP and indicates a dataset/sampler bug. Fix the dataloader to always return fixed V."
                )
            if intrinsic.ndim != 3:
                raise ValueError(f"Expected fxfycxcy shape (B, V, 4), got {tuple(intrinsic.shape)}")
            if extrinsic.ndim != 4:
                raise ValueError(f"Expected c2w shape (B, V, 4, 4), got {tuple(extrinsic.shape)}")
            if int(intrinsic.shape[0]) != B_batch or int(intrinsic.shape[1]) != V_batch:
                raise ValueError(
                    f"Intrinsic shape mismatch: expected (B={B_batch}, V={V_batch}, 4) but got {tuple(intrinsic.shape)}"
                )
            if int(extrinsic.shape[0]) != B_batch or int(extrinsic.shape[1]) != V_batch:
                raise ValueError(
                    f"Extrinsic shape mismatch: expected (B={B_batch}, V={V_batch}, 4, 4) but got {tuple(extrinsic.shape)}"
                )
            if not (1 <= int(batch_cond_num) < int(total_view)):
                raise ValueError(
                    f"Invalid cond_num/total_view: expected 1 <= cond_num < total_view but got cond_num={batch_cond_num}, total_view={total_view}"
                )
            
            predict_cls = training_cfg.get('predict_cls', False) or model_config.get('params', {}).get('predict_cls', False)
            
            if predict_cls:
                # For now, following user's strict return signature.
                x1_cond, x1_all, camera_embedding = prepare_data(
                    rae, image, intrinsic, extrinsic, device, 
                    random_cond_num=batch_cond_num, return_cls=True, 
                    camera_mode=camera_mode, ray_pose_mode=ray_pose_mode, return_scale=use_prope
                )
            else:
                x1_cond, x1_all, camera_embedding = prepare_data(
                    rae, image, intrinsic, extrinsic, device, 
                    random_cond_num=batch_cond_num, return_cls=False,
                    camera_mode=camera_mode, ray_pose_mode=ray_pose_mode, return_scale=use_prope
                )
            
            # Ensure cond_num is set to actual sampled value for model_kwargs
            cond_num_for_model = int(batch_cond_num)
            
            # Expected channels: 1 (mask) + 3 (camray) or 6 (plucker)
            expected_cam_ch = 4 if camera_mode == "camray" else 7
            assert camera_embedding.shape[1] == expected_cam_ch, f"Camera embedding should have {expected_cam_ch} channels for mode={camera_mode}, got {camera_embedding.shape[1]}"
            
            # Classifier-Free Guidance training: drop camera embedding (channels 1-6)
            camera_drop = training_cfg.get('camera_drop', 0.0)
            drop_mask = None
            if camera_drop > 0:
                V_num = V_batch
                B_num = B_batch
                if int(x1_cond.shape[0]) != B_num * V_num:
                    raise ValueError(
                        f"Latent batch size mismatch: x1_cond.shape[0]={int(x1_cond.shape[0])} but expected B*V={B_num}*{V_num}={B_num*V_num}. "
                    )
                # Generate mask for each sample in batch
                drop_mask_b = (torch.rand(B_num, 1, 1, 1, device=device) > camera_drop).float()
                # Expand to all views of the same sample
                drop_mask = drop_mask_b.repeat_interleave(V_num, dim=0)  # (B*V, 1, 1, 1)
                # Keep channel 0 (mask), drop 1-6
                camera_embedding[:, 1:] = camera_embedding[:, 1:] * drop_mask
                
            # Setup Model Args
            model_kwargs = dict(
                camera_embedding=camera_embedding, 
                total_view=total_view, 
                cond_num=cond_num_for_model,
                is_concat_mode=True,
                ref_cond=x1_cond,        # Conditioning part [clean_ref | zeros]
                x1_global=x1_all,        # Target part [all views clean]
                freeze_cond=False        
            )

            
            # Support ProPE (Fail-loud + consistent normalization)
            if use_prope:
                 # Use actual batch resolution for ProPE (supports multi-res)
                 model_kwargs['prope_image_size'] = (image.shape[-2], image.shape[-1])
                 # Compute viewmats (w2c) and Ks
                 with torch.no_grad():
                    # ------------------------------------------------------------------
                    # ProPE safety: translation normalization + outlier detection
                    #
                    # - camera_embedding(plucker) already normalizes translation scale (normalize_t=True),
                    #   but ProPE consumes raw viewmats (w2c). If we feed raw, huge translations can
                    #   explode activations and loss.
                    # - We therefore (1) require an explicit max_translation_norm config, (2) normalize
                    #   c2w translations by a per-sample scale, and (3) normalize extrinsics relative
                    #   to a reference view (match get_camera_embedding: normalize_extrinsic_tgt=-1).
                    # ------------------------------------------------------------------
                    max_t_allowed = multiview_cfg.get("max_translation_norm", None)
                    if max_t_allowed is None:
                        max_t_allowed = dataset_cfg.get("max_translation_norm", None)
                    if max_t_allowed is None:
                        raise ValueError(
                            "Config must explicitly set multiview.max_translation_norm (or dataset.max_translation_norm) "
                            "when use_prope=true, to fail loudly on pose scale outliers."
                        )
                    max_t_allowed = float(max_t_allowed)

                    # Intrinsic: (B, V, 4) -> (B, V, 3, 3) 
                    fx, fy, cx, cy = intrinsic.to(device).unbind(dim=-1)
                    zeros = torch.zeros_like(fx)
                    ones = torch.ones_like(fx)
                    Ks = torch.stack(
                        (
                            torch.stack((fx, zeros, cx), dim=-1),
                            torch.stack((zeros, fy, cy), dim=-1),
                            torch.stack((zeros, zeros, ones), dim=-1),
                        ),
                        dim=-2,
                    )

                    # Extrinsic: (B, V, 4, 4) or 3x4
                    c2w = extrinsic.to(device)
                    if c2w.shape[-2:] == (3, 4):
                        last_row = torch.zeros(c2w.shape[:-2] + (1, 4), device=device)
                        last_row[..., 3] = 1.0
                        c2w = torch.cat([c2w, last_row], dim=-2)
                    
                    # 1) Normalize extrinsics relative to reference view (last view)
                    ref_inv = torch.linalg.inv(c2w[:, -1])  # (B, 4, 4)
                    c2w = ref_inv.unsqueeze(1) @ c2w        # (B, V, 4, 4)
                    
                    # 2) Fail loudly on translation outliers (before any normalization)
                    t = c2w[:, :, :3, 3]
                    t_norm = torch.linalg.vector_norm(t, dim=-1)  # (B, V)
                    max_t = t_norm.max(dim=1).values              # (B,)
                    bad = max_t > max_t_allowed
                    if bad.any():
                        bad_b = torch.where(bad)[0][:10].tolist()
                        raise ValueError(
                            f"ProPE pose outlier detected: max|t| exceeds max_translation_norm={max_t_allowed}. "
                            f"bad batch indices (first up to 10): {bad_b}. "
                            f"max|t| values: {max_t[bad][:10].detach().cpu().tolist()}. "
                            f"cut3r_idx (if available): {batch.get('cut3r_idx', 'N/A')}"
                        )
                    
                    # 3) Translation normalization (match get_camera_embedding normalize_t)
                    # scale = 1 / (max_{v,xyz} |t| + eps)
                    farthest = t.abs().amax(dim=1).amax(dim=1, keepdim=True)  # (B,1)
                    scale = 1.0 / (farthest + 1e-8)
                    c2w = c2w.clone()
                    c2w[:, :, :3, 3] = c2w[:, :, :3, 3] * scale.unsqueeze(1)
                    
                    # Compute w2c = inv(c2w)
                    w2c = torch.linalg.inv(c2w)
                    
                    # CFG camera drop: Replace viewmats with identity for dropped samples
                    if drop_mask is not None:
                        # drop_mask: (B*V, 1, 1, 1), 0 = dropped, 1 = keep
                        # w2c: (B, V, 4, 4)
                        if w2c.shape[0] != B_num or w2c.shape[1] != V_num:
                            raise ValueError(
                                f"w2c shape mismatch: got {tuple(w2c.shape)} but expected (B={B_num}, V={V_num}, 4, 4). "
                                "This usually means total_view does not match the dataloader output."
                            )
                        drop_mask_bv = drop_mask.view(B_num, V_num, 1, 1)  # (B, V, 1, 1)
                        
                        # Create identity viewmats
                        identity_viewmat = torch.eye(4, device=device, dtype=w2c.dtype).unsqueeze(0).unsqueeze(0)  # (1, 1, 4, 4)
                        
                        # Replace viewmats with identity where dropped (drop_mask == 0)
                        w2c = drop_mask_bv * w2c + (1 - drop_mask_bv) * identity_viewmat
                        
                        # Replace Ks with identity where dropped (drop_mask == 0)
                        # Ks: (B, V, 3, 3)
                        # drop_mask_bv: (B, V, 1, 1) broadcastable
                        identity_K = torch.eye(3, device=device, dtype=Ks.dtype).unsqueeze(0).unsqueeze(0)
                        Ks = drop_mask_bv * Ks + (1 - drop_mask_bv) * identity_K
                    
                    model_kwargs['viewmats'] = w2c
                    model_kwargs['Ks'] = Ks
            
            with autocast(**autocast_kwargs):
                # Add debugging for timestep sampling
                try:
                    # Feature-to-Feature Flow: Construct x0 from source features + noise
                    x0_init = None
                    if source_level is not None:
                        # Extract source level features WITH SOURCE-LEVEL NORMALIZATION
                        # CRITICAL: Source and target levels have DIFFERENT distributions!
                        # We must normalize source features with source-level stats.
                        with torch.no_grad():
                            # Normalize images for encoder (ImageNet normalization)
                            images_norm = (image.to(device) - rae.encoder_mean[None]) / rae.encoder_std[None]
                            
                            # Save original state for safe restoration
                            original_do_norm = rae.do_normalization
                            original_mean = rae.latent_mean
                            original_var = rae.latent_var
                            
                            try:
                                if _cached_source_stats is not None:
                                    # Use cached stats (loaded once at startup)
                                    rae.latent_mean = _cached_source_stats.get('mean', None)
                                    rae.latent_var = _cached_source_stats.get('var', None)
                                    if rae.latent_mean is not None:
                                        rae.latent_mean = rae.latent_mean.to(device)
                                    if rae.latent_var is not None:
                                        rae.latent_var = rae.latent_var.to(device)
                                    rae.do_normalization = True
                                
                                # Encode with source-level normalization
                                latents_source = rae.encode(images_norm, mode='single', level=source_level)
                                # latents_source: (B*V, C, H, W) - normalized with source-level stats
                            finally:
                                # ALWAYS restore target-level normalization
                                rae.do_normalization = original_do_norm
                                rae.latent_mean = original_mean
                                rae.latent_var = original_var
                        
                        # Sample noise scale: sigma ~ |N(0, tau^2)| (following RAE paper)
                        # This stochastic sigma regularizes training and improves robustness
                        if noise_tau_gt_feat > 0:
                            noise_std = torch.abs(torch.randn(1, device=device) * noise_tau_gt_feat).item()
                        else:
                            noise_std = 0.0
                        noise = torch.randn_like(latents_source) * noise_std
                        x0_init = (latents_source + noise) / math.sqrt(1 + noise_std**2)

                    
                    # NEW: Feature Propagation Mode (Image 1 Architecture)
                    # Extract L1 features for encoder conditioning (NOT for x0 init)
                    if source_condition_level is not None:
                        with torch.no_grad():
                            # Normalize images for encoder (ImageNet normalization)
                            images_norm = (image.to(device) - rae.encoder_mean[None]) / rae.encoder_std[None]
                            
                            # Save original state for safe restoration
                            original_do_norm = rae.do_normalization
                            original_mean = rae.latent_mean
                            original_var = rae.latent_var
                            
                            try:
                                if _cached_source_cond_stats is not None:
                                    # Use cached stats (loaded once at startup)
                                    rae.latent_mean = _cached_source_cond_stats.get('mean', None)
                                    rae.latent_var = _cached_source_cond_stats.get('var', None)
                                    if rae.latent_mean is not None:
                                        rae.latent_mean = rae.latent_mean.to(device)
                                    if rae.latent_var is not None:
                                        rae.latent_var = rae.latent_var.to(device)
                                    rae.do_normalization = True
                                
                                # Encode L1 features with source-condition-level normalization
                                latents_source_cond = rae.encode(images_norm, mode='single', level=source_condition_level)
                                # latents_source_cond: (B*V, C, H, W)
                            finally:
                                # ALWAYS restore target-level normalization
                                rae.do_normalization = original_do_norm
                                rae.latent_mean = original_mean
                                rae.latent_var = original_var
                        
                        # Add noise to L1 condition features for training robustness
                        if noise_tau_gt_feat > 0:
                            cond_noise_std = torch.abs(torch.randn(1, device=device) * noise_tau_gt_feat).item()
                        else:
                            cond_noise_std = 0.0
                        cond_noise = torch.randn_like(latents_source_cond) * cond_noise_std
                        latents_source_cond_noisy = (latents_source_cond + cond_noise) / math.sqrt(1 + cond_noise_std**2)
                        
                        # Add noisy L1 features to model_kwargs for encoder conditioning
                        model_kwargs['source_condition'] = latents_source_cond_noisy
                    
                    # In concat mode: 
                    # - 2nd arg (x1) is the conditioning latent (x1_cond)
                    # - model_kwargs['x1_global'] is the target latent (x1_all)
                    loss_dict = transport.training_multiview_losses(model, x1_cond, total_view, batch_cond_num, model_kwargs, x0=x0_init)
                    loss_tensor = loss_dict["loss"].mean()
                    ref_loss_tensor = loss_dict.get("ref_loss", loss_tensor).mean()
                    tgt_loss_tensor = loss_dict.get("tgt_loss", loss_tensor).mean()
                except Exception as e:
                    logger.error(f"Exception during forward pass at step {train_steps}: {e}")
                    logger.error(f"Latent stats before forward - Mean: {x1_cond.mean().item():.6f}, Std: {x1_cond.std().item():.6f}")
                    logger.error(f"Latent range before forward - Min: {x1_cond.min().item():.6f}, Max: {x1_cond.max().item():.6f}")
                    raise

            # ============================================================
            # Loss sanity check: skip abnormal loss without backward/logging (DDP-safe)
            #
            # NOTE:
            # - If only one rank skips backward while others backprop, DDP can deadlock on all-reduce.
            # - We therefore all-reduce a bad-flag and, if ANY rank reports abnormal loss, ALL ranks
            #   skip this batch (no loss accumulation, no logging, no backward, no optimizer step).
            # ============================================================
            loss_isfinite = torch.isfinite(loss_tensor).all()
            loss_value = loss_tensor.detach().float().item() if bool(loss_isfinite) else float("nan")
            local_bad = (not bool(loss_isfinite)) or (loss_value > 1000.0)

            if dist.is_available() and dist.is_initialized():
                bad_flag = torch.tensor(int(local_bad), device=device, dtype=torch.int32)
                dist.all_reduce(bad_flag, op=dist.ReduceOp.MAX)
                bad_any = bool(bad_flag.item())
            else:
                bad_any = bool(local_bad)

            if bad_any:
                if rank == 0:
                    video_id = batch.get("video_id", "N/A")
                    frame_indices = batch.get("frame_indices", "N/A")
                    if torch.is_tensor(frame_indices):
                        frame_info = f"tensor(shape={tuple(frame_indices.shape)}, dtype={frame_indices.dtype})"
                    else:
                        frame_info = str(frame_indices)
                    logger.warning(
                        f"[step={train_steps}] Abnormal loss detected on at least one rank; "
                        "skipping this batch (no backward/optimizer/logging). "
                        f"batch video_id={video_id} frame_indices={frame_info}"
                    )
                opt.zero_grad()
                accum_counter = 0
                step_loss_accum = 0.0
                step_ref_loss_accum = 0.0
                step_tgt_loss_accum = 0.0
                continue

            # ============================================================
            # BACKWARD PASS with Gradient Accumulation (Efficient DDP)
            # ============================================================
            step_loss_accum += loss_tensor.item()
            step_ref_loss_accum += ref_loss_tensor.item()
            step_tgt_loss_accum += tgt_loss_tensor.item()
            
            # Use no_sync() for all but the last accumulation step to avoid unnecessary communication
            if (accum_counter + 1) % grad_accum_steps != 0:
                with model.no_sync():
                    (loss_tensor / grad_accum_steps).backward()
            else:
                (loss_tensor / grad_accum_steps).backward()
            
            # ============================================================
            # PROTECTION: Check for NaN gradients after backward pass
            # ============================================================
            has_nan_grad = False
            for name, param in model.named_parameters():
                if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                    has_nan_grad = True
                    logger.error(f"NaN/Inf gradient detected in {name} at step {train_steps}")
                    break
            
            if has_nan_grad:
                logger.error(f"NaN gradient detected at step {train_steps}, zeroing gradients and skipping step")
                opt.zero_grad()
                accum_counter = 0
                step_loss_accum = 0.0
                step_ref_loss_accum = 0.0
                step_tgt_loss_accum = 0.0
                continue
            
            accum_counter += 1

            if accum_counter < grad_accum_steps:
                continue

            if clip_grad > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                # Log if gradient was clipped significantly
                if grad_norm > clip_grad * 10:
                    logger.warning(f"[Step {train_steps}] Large gradient norm before clipping: {grad_norm:.2f}")
                
            opt.step()
            schedl.step()
            update_ema(ema, model.module, decay=ema_decay)
            opt.zero_grad()

            running_loss += step_loss_accum / grad_accum_steps
            running_ref_loss += step_ref_loss_accum / grad_accum_steps
            running_tgt_loss += step_tgt_loss_accum / grad_accum_steps
            log_steps += 1
            train_steps += 1
            accum_counter = 0
            step_loss_accum = 0.0
            step_ref_loss_accum = 0.0
            step_tgt_loss_accum = 0.0

            # Delay WandB initialization to avoid clutter from failed starts
            # Use >= 5 instead of == 5 to handle resume from checkpoint
            if not wandb_initialized and train_steps >= 0:
                if rank == 0 and args.wandb:
                    entity = os.environ.get("ENTITY", "gld")
                    project = os.environ.get("PROJECT", "gld-mvdiffusion")
                    wandb_utils.initialize(args, entity, experiment_name, project)
                wandb_initialized = True

            if train_steps % log_every == 0:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                avg_ref_loss = torch.tensor(running_ref_loss / log_steps, device=device)
                avg_tgt_loss = torch.tensor(running_tgt_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                dist.all_reduce(avg_ref_loss, op=dist.ReduceOp.SUM)
                dist.all_reduce(avg_tgt_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / world_size
                avg_ref_loss = avg_ref_loss.item() / world_size
                avg_tgt_loss = avg_tgt_loss.item() / world_size
                logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Ref: {avg_ref_loss:.4f}, Tgt: {avg_tgt_loss:.4f}, Steps/Sec: {steps_per_sec:.2f}")
                if args.wandb:
                    wandb_utils.log(
                        {
                            "train loss": avg_loss, 
                            "ref loss": avg_ref_loss,
                            "tgt loss": avg_tgt_loss,
                            "train steps/sec": steps_per_sec
                        },
                        step=train_steps,
                    )
                running_loss = 0.0
                running_ref_loss = 0.0
                running_tgt_loss = 0.0
                log_steps = 0
                start_time = time()

            if train_steps % ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "scheduler": schedl.state_dict(),
                        "train_steps": train_steps,
                        "epoch": epoch,
                        "steps_per_epoch": steps_per_epoch,
                        "config_path": args.config,
                        "training_cfg": training_cfg,
                        "cli_overrides": {
                            "data_path": args.data_path,
                            "results_dir": args.results_dir,
                            "image_size": args.image_size,
                            "precision": args.precision,
                            "global_seed": global_seed,
                        },
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                dist.barrier()

        # if train_steps % sample_every == 0 or train_steps == 1:
        #     logger.info("Generating EMA samples...")
        #     with torch.no_grad():
        #         with autocast(**autocast_kwargs):
        #             samples = eval_sampler(zs, model_fn, **sample_model_kwargs)[-1]
        #         dist.barrier()

        #         if using_cfg:
        #             samples, _ = samples.chunk(2, dim=0)
        #         samples = rae.decode(samples.to(torch.float32))
        #         out_samples = torch.zeros(
        #             (global_batch_size // grad_accum_steps, 3, args.image_size, args.image_size),
        #             device=device,
        #         )
        #         dist.all_gather_into_tensor(out_samples, samples)
        #         if args.wandb:
        #             wandb_utils.log_image(out_samples, train_steps)
        #     logger.info("Generating EMA samples done.")
            if train_steps % sample_every == 0 or train_steps == 1:
                # All ranks participate in validation for distributed aggregation
                logger.info(f"Running validation at Step {train_steps}...") 
                # Use EMA model for validation instead of training model
                
                # Import shared validation
                from utils.da3_validation import validate_da3_multiview
                
                val_stats = validate_da3_multiview(
                    rae=rae, 
                    model=ema, 
                    transport=transport,
                    sampler=eval_sampler, 
                    loader=val_loader, 
                    device=device, 
                    total_view=val_num_views, 
                    cond_num=val_cond_num, 
                    compute_loss=True,
                    val_num_batches=val_num_batches_per_rank, 
                    use_prope=use_prope,
                    validation_mode=val_mode,
                    output_dir=os.path.join(experiment_dir, f"validation_step_{train_steps}"),
                    pag_scale=pag_scale if pag_scale > 0 else None,
                    pag_layer_idx=pag_layer_idx,
                    run_config=to_dict(cfg),
                    # prope_image_size=args.image_size, # Removed: uses batch resolution internally
                    predict_cls=predict_cls,
                    ref_view_sampling=ref_view_sampling,
                    camera_mode=camera_mode,
                    ray_pose_mode=ray_pose_mode,
                    rank=rank,
                    world_size=world_size,
                    joint_ode=False,  # Disabled when using concat mode
                    is_concat_mode=is_concat_mode,
                    source_level=source_level,  # Feature-to-Feature Flow
                    source_level_stat_path=source_level_stat_path,  # Source-level normalization stats
                    noise_tau_gt_feat=noise_tau_gt_feat / 4.0,  # Fixed noise at tau/4 for validation
                    # NEW: Feature Propagation Mode (Image 1 Architecture)
                    source_condition_level=source_condition_level,
                    source_condition_stat_path=source_condition_stat_path,
                ) 
                # remove images from printed log
                log_copy = {k: v for k, v in val_stats.items() if k != "val/images"}
                if rank == 0:
                    logger.info(f"[Validation @ Step {train_steps}] " + ", ".join(f"{k}: {v:.4f}" for k, v in log_copy.items()))
                    if args.wandb:
                        wandb_utils.log(log_copy, step=train_steps)
                        if val_stats["val/images"] is not None:
                            # Log list of images
                            images_to_log = [wandb.Image(img) for img in val_stats["val/images"]]
                            wandb_utils.log({"val/reconstructions": images_to_log}, step=train_steps)
            
            # Ensure all ranks are synchronized after validation before moving to next training step
            if dist.is_initialized():
                dist.barrier()

        if accum_counter != 0:
            raise RuntimeError("Gradient accumulation counter not zero at epoch end.")

    model.eval()
    logger.info("Done!")
    cleanup()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DA3 MVDiffusion Training")
    parser.add_argument("--config", type=str, required=True, help="Path to the config file.")
    parser.add_argument("--data-path", type=str, required=True, help="Path to the training dataset root.")
    parser.add_argument("--results-dir", type=str, default="results", help="Directory to store training outputs.")
    # NOTE: image-size is now read from config (stage_1.params.encoder_input_size)
    parser.add_argument("--precision", type=str, choices=["fp32", "bf16"], default="fp32", help="Compute precision for training.")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")
    #
    parser.add_argument("--level", type=int, default=-1, help="DA3 feature level to train (-1, -2, -3, -4)")


    parser.add_argument("--pretrained", type=str, default=None, help="Resume from PRETRAINED DDT, only loading for model")
    parser.add_argument("--overfit",  action="store_true", help="overfit to a single sample")
    # Check for duplicates or cleanup if needed.
    # We already have run_name logic in config possibly? But keep it here.
    parser.add_argument("--run_name", type=str, default="da3_mvdiff", help="name of the run")
    parser.add_argument("--global-seed", type=int, default=0, help="Global seed")
    parser.add_argument("--vae-type", type=str, default="RAE", help="VAE type: RAE or VAE")
    parser.add_argument("--ckpt", type=str, default=None, help="Resume from checkpoint (model+ema+opt)")
    parser.add_argument("--git-hash", type=str, default=None, help="Git hash for current run")

    args = parser.parse_args()
    # Log level info
    print(f"Training DA3 MVDiffusion at level {args.level}")
    main(args)
