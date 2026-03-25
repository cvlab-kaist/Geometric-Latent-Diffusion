"""
Feature-level validation function for DA3 MVDiffusion.

This module provides a simplified validation function that directly compares
generated features with GT features using MSE and Cosine Similarity metrics,
Feature-level validation.
"""

import torch
import numpy as np
import scipy.linalg
from einops import rearrange
from tqdm import tqdm
import os
import cv2
import matplotlib.pyplot as plt

def save_visualization(gt_images, pred_images, cond_num, save_path):
    """
    Save visualization: GT row on top, prediction row on bottom.
    
    Args:
        gt_images: (V, 3, H, W) ground truth in [0, 1]
        pred_images: (V, 3, H, W) predictions in [0, 1]
        cond_num: number of conditioning views
        save_path: path to save visualization
    """
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


def get_denoised_features(
    rae,
    model,
    transport,
    sampler,
    loader,
    device,
    total_view,
    cond_num,
    val_num_batches=None,
    use_prope=False,
    rank=0,
    world_size=1,
    prope_image_size=None,
    predict_cls=False,
    joint_ode=False,
    ref_view_sampling='prefix',
    camera_mode='plucker',
    is_concat_mode=False,
    pag_scale=None,
    pag_layer_idx=None,
    cfg_scale=None,
    use_camera_drop=True,  # NEW
    cfg_uncond_mode='keep',  # NEW
    batch=None,
):
    """
    Diffusion sampling to generate denoised features at a single level.

    Returns: [latent_norm] (BV, C, h, w) — diffusion model output in latent_norm space.
    """
    from utils.metrics import compute_mse, compute_cosine_similarity
    from utils.camera.camera import get_camera_embedding
    
    print(f"[Feature denoising] Level: {rae.level}, Mode: {camera_mode}, Concat: {is_concat_mode}")
    model.eval()
    rae.eval()
    
    per_sample_metrics = []
    
    # Accumulate features for FID computation
    all_gt_features = []
    all_pred_features = []

    # Get model reference for checking parameters
    m = model.module if hasattr(model, 'module') else model
    
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
    
    # ImageNet normalization
    images_norm = (images - rae.encoder_mean[None].to('cuda')) / rae.encoder_std[None].to('cuda')
    
    # ========== 1. Encode GT Features ==========
    with torch.no_grad():
        # All views together (for GT comparison)
        latents_all = rae.encode(images_norm, level=rae.level)  # (B*V, C, h, w)
        # Reference views only (for conditioning)
        latents_ref = rae.encode(images_norm[:, :cond_num], level=rae.level)  # (B*cond, C, h, w)
    
    # ========== 2. Prepare Camera Embedding ==========
    # Handle extrinsic format
    if extrinsic.shape[-2:] == (4, 4):
        extrinsic = extrinsic[..., :3, :4]
    
    # Handle intrinsic format
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
    
    # Generate camera embedding
    if use_prope:
        camera_embedding, scale = get_camera_embedding(
            intri_, extri_, B, V, H, W, mode=camera_mode, return_scale=True
        )
    else:
        camera_embedding = get_camera_embedding(
            intri_, extri_, B, V, H, W, mode=camera_mode, return_scale=False
        )
    
    camera_embedding = rearrange(camera_embedding, "b v c h w -> (b v) c h w")
    
    # Add condition mask
    random_masks = torch.ones((B, V, 1, H, W), device=device, dtype=latents_all.dtype)
    random_masks[:, :cond_num] = 0
    random_masks = random_masks.reshape(B * V, 1, H, W)
    camera_embedding = torch.cat([random_masks, camera_embedding], dim=1)
    
    # ========== 3. Prepare Sampling Input ==========
    _, C_lat, h_lat, w_lat = latents_all.shape
    
    # Create merged latents: ref from latents_ref, targets as zeros
    latents_5d = latents_all.reshape(B, V, C_lat, h_lat, w_lat)
    latents_ref_5d = latents_ref.reshape(B, cond_num, C_lat, h_lat, w_lat)
    
    latents_cond_5d = torch.zeros(B, V, C_lat, h_lat, w_lat, device=device, dtype=latents_all.dtype)
    latents_cond_5d[:, :cond_num] = latents_ref_5d
    latents_cond = latents_cond_5d.reshape(B * V, C_lat, h_lat, w_lat)
    
    # Prepare sample input: ALL views start as noise at t=1 (matching training)
    # In training, x0 (noise) is sampled for ALL views, not just target views
    # The model learns to denoise all views from noise → clean features
    sample_input_flat = torch.randn(B * V, C_lat, h_lat, w_lat, device=device, dtype=latents_all.dtype)
    
    # Model kwargs
    model_kwargs = dict(
        camera_embedding=camera_embedding,
        total_view=total_view,
        cond_num=cond_num,
    )
    
    if use_prope:
        model_kwargs['prope_image_size'] = prope_image_size if prope_image_size is not None else (H, W)
        with torch.no_grad():
            Ks = intrinsic_mat # (B, V, 3, 3)
            c2w = extrinsic # (B, V, 3, 4)
            last_row = torch.zeros(c2w.shape[:-2] + (1, 4), device=device)
            last_row[..., 3] = 1.0
            c2w_4x4 = torch.cat([c2w, last_row], dim=-2)
            w2c = torch.linalg.inv(c2w_4x4)
            
            # Normalize translation scale if scale is provided (from Plucker normalization)
            if scale is not None:
                w2c[..., :3, 3] = w2c[..., :3, 3] * scale
            
            model_kwargs['viewmats'] = w2c
            model_kwargs['Ks'] = Ks
    
    # Add PAG scale if enabled
    if pag_scale is not None:
        model_kwargs['pag_scale'] = pag_scale
        model_kwargs['pag_layer_idx'] = pag_layer_idx
    
    # Add CFG scale if enabled
    if cfg_scale is not None and cfg_scale > 1.0:
        model_kwargs['cfg_scale'] = cfg_scale
        model_kwargs['use_camera_drop'] = use_camera_drop  # NEW
        model_kwargs['uncond_mode'] = cfg_uncond_mode  # NEW
    
    # Concat mode handling
    if is_concat_mode:
        sample_input_flat = torch.cat([latents_cond, sample_input_flat], dim=1)
        model_kwargs['is_concat_mode'] = True
        model_kwargs['ref_cond'] = latents_cond
    
    # ========== 4. Diffusion Sampling ==========
    with torch.no_grad():
        samples = sampler(sample_input_flat, model, **model_kwargs)[-1]
    
    # Extract denoised latents from concat mode
    if is_concat_mode:
        samples = samples[:, C_lat:]
    
    return samples # denoised feature

def get_denoised_features_vae(
    rae,
    model,
    transport,
    sampler,
    batch,
    device,
    total_view,
    cond_num,
    val_num_batches=None,
    use_prope=False,
    validation_mode="propagation",  # 'propagation', 'replacement', 'naive'
    use_lvsm=False,  # Enable LVSM for upward feature propagation
    rank=0,
    world_size=1,
    output_dir=None, # If provided, save images
    pag_scale=None,  # PAG (Perturbed Attention Guidance) scale. None = disabled.
    pag_layer_idx=None, # Layer to apply PAG to
    cfg_scale=None,  # CFG (Classifier-Free Guidance) scale. None or 1.0 = disabled.
    run_config=None,  # Optional dict with run configuration to save
    prope_image_size=None,  # Image size for ProPE (dynamic, avoid hardcoding, if None uses batch H,W)
    predict_cls=False,  # If True, also predict/generate CLS token
    ref_view_sampling='prefix', # sampling method for reference views
    camera_mode='camray', # 'camray' or 'plucker'
    is_concat_mode=False,  # If True, use concat mode: [cond|noisy] input
    is_dino=False
):
    """
    Shared validation/inference logic for DA3 MVDiffusion.
    
    Args:
        validation_mode:
            - 'naive': Just decode the generated latent using RAE.decode() (MAE decoder).
            - 'replacement': Replace target level in GT features with generated latent, decode with DPT.
            - 'propagation': Use generated latent to compute rest, decode with DPT.
                             For Level -4: Uses downward propagation (backbone forward)
        run_config: Optional dict with run configuration (checkpoint, pag_scale, etc.) to save with results.
    """

    print(f"Validation mode: {validation_mode}")
    model.eval()
    rae.eval()
    

    all_images = []
    saved_sample_count = 0  # Counter for saved samples
    
    # Track metrics if needed
    metrics_acc = {}
    per_sample_metrics = []  # List of per-sample metrics

    val_loss_sum = torch.tensor(0.0, device=device)
    val_loss_count = torch.tensor(0.0, device=device)
    
    # Track metrics for global aggregation
    psnr_sum = torch.tensor(0.0, device=device)
    ssim_sum = torch.tensor(0.0, device=device)
    lpips_sum = torch.tensor(0.0, device=device)
    metrics_count = torch.tensor(0.0, device=device)

    # Determine if model expects CLS token from its own attribute if predict_cls is not explicitly set
    m = model.module if hasattr(model, 'module') else model
    if predict_cls is None:
        predict_cls = getattr(m, 'predict_cls', False)

        
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
    print(images.min())
    B, V, C, H, W = images.shape
    
    
    # ImageNet Normalization
    if is_dino:
        images_norm = (images - rae.encoder_mean[None].to('cuda')) / rae.encoder_std[None].to('cuda')
    else:
        images_norm = 2*images-1
    
    # Import camera embedding function
    from utils.camera.camera import get_camera_embedding

    # Prepare camera embedding (Logic from train_multiview_da3.py)
    intrinsic = intrinsic.to(device, non_blocking=True) # (b, v, 4)]
    extrinsic = extrinsic.to(device, non_blocking=True) # (b, v, 4, 4) or (b, v, 3, 4)

    if extrinsic.shape[-2:] == (4, 4):
        extrinsic = extrinsic[..., :3, :4]
    elif extrinsic.shape[-2:] != (3, 4):
            # Try to handle if already flattened? No, assume (B, V, ...)
            if extrinsic.dim() == 3 and extrinsic.shape[-1] == 12:
                extrinsic = extrinsic.reshape(B, V, 3, 4)
            elif extrinsic.dim() == 3 and extrinsic.shape[-1] == 16:
                extrinsic = extrinsic.reshape(B, V, 4, 4)[..., :3, :4]
            else:
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

    # Generate camera embedding at IMAGE resolution
    if use_prope:
        camera_embedding, scale = get_camera_embedding(intri_, extri_, B, V, H, W, mode=camera_mode, return_scale=True)
    else:
        camera_embedding = get_camera_embedding(intri_, extri_, B, V, H, W, mode=camera_mode, return_scale=False)
        scale = None
        
    camera_embedding = rearrange(camera_embedding, "b f c h w -> (b f) c h w")
    
    # Validation: first 'cond_num' views are condition.
    random_masks = torch.ones((B, V, 1, H, W), device=device, dtype=extrinsic.dtype)
    random_masks[:, :cond_num] = 0
    random_masks = random_masks.reshape(B * V, 1, H, W)

    camera_embedding = torch.cat([random_masks, camera_embedding], dim=1)  # (B*V, 4 or 7, H, W)

    # Fail loudly if channel count doesn't match model expectations
    expected_ch = m.camera_embedder.proj.in_channels
    if camera_embedding.shape[1] != expected_ch:
            raise ValueError(
            f"Camera embedding channel mismatch in validation: "
            f"expected {expected_ch} (from model), but got {camera_embedding.shape[1]} (mode={camera_mode})."
        )

    # ProPE kwargs
    model_kwargs = dict(
        camera_embedding=camera_embedding,
        total_view=total_view,
        cond_num=cond_num  # For integrators to zero out cond view velocity
    )
    
    target_view=total_view-cond_num

    # Add prope_image_size for ProPE dynamic image size
    if use_prope:
        model_kwargs['prope_image_size'] = prope_image_size if prope_image_size is not None else (H, W)
    
    # Add PAG scale if enabled
    if pag_scale is not None:
        model_kwargs['pag_scale'] = pag_scale
        model_kwargs['pag_layer_idx'] = pag_layer_idx
        
    # Add CFG scale if enabled
    if cfg_scale is not None and cfg_scale > 1.0:
        model_kwargs['cfg_scale'] = cfg_scale
        
    if use_prope:
            # Compute viewmats (w2c) and Ks
        with torch.no_grad():
            # Intrinsic: (B, V, 4) -> (B, V, 3, 3) 
            # (Reusing intrinsic_mat calculated above)
            Ks = intrinsic_mat # (B, V, 3, 3)
            
            # Extrinsic for ProPE needs to be w2c
            # Current extrinsic is c2w (3x4)
            c2w = extrinsic # (B, V, 3, 4)
            last_row = torch.zeros(c2w.shape[:-2] + (1, 4), device=device)
            last_row[..., 3] = 1.0
            c2w_4x4 = torch.cat([c2w, last_row], dim=-2)
            
            w2c = torch.linalg.inv(c2w_4x4)
            
            # Normalize translation scale if scale is provided (from Plucker normalization)
            if scale is not None:
                # scale: (B, 1, 1). w2c: (B, V, 4, 4)
                w2c[..., :3, 3] = w2c[..., :3, 3] * scale
            
            model_kwargs['viewmats'] = w2c
            model_kwargs['Ks'] = Ks

    # 2. Diffusion Sampling
    # CRITICAL: Condition views start from GT, only target views get noise
    
    # Encode GT images to get condition view latents
    with torch.no_grad():
        
        latents_all = rae.encode(images_norm)
        
        latents_ref = latents_all[:,:cond_num]
        C_lat, h_lat, w_lat = latents_all.shape[2:]
        gt_latents=latents_all.clone()
        gt_cls_toks = None

    

    # Derive dimensions from actual encoded latents
    latent_dim = gt_latents.shape[2]
    h_lat = gt_latents.shape[3]
    w_lat = gt_latents.shape[4]
    


    latents_ref_5d = latents_ref.reshape(B, cond_num, latent_dim, h_lat, w_lat)
    
    # Full noise for all views
    sample_input = torch.randn(B, target_view, latent_dim, h_lat, w_lat, device=device, dtype=gt_latents.dtype)
    sample_input = torch.concat([latents_ref_5d,sample_input],dim=1)
    cond_channel=latents_ref_5d
    # Build condition channel: clean for ref, zeros for tgt

    # Flatten for sampling
    sample_input_flat = sample_input.reshape(B * V, latent_dim , h_lat, w_lat)
    model_kwargs['is_concat_mode'] = False
    model_kwargs['ref_cond'] = cond_channel.reshape(B * cond_num, latent_dim, h_lat, w_lat)
    model_kwargs['x1_global'] = latents_all.reshape(B * V, latent_dim, h_lat, w_lat) 


    with torch.no_grad():
        # Pass model directly (transport now handles positional total_view)
        samples = sampler(sample_input_flat, model, **model_kwargs)[-1]  # (B*V, C, H, W) or (B*V, C, N+1, 1)
    
    with torch.no_grad():
        if is_dino:
            decoded_images = rae.decode(samples,H=H,W=W)
        else:
            decoded_images = rae.decode(samples)

    # 4. Visualization / Logging
    # decoded_images: Expect (B*V, 3, H, W) usually
    if decoded_images.ndim == 4:
        # (B*V, C, H, W) -> (B, V, C, H, W)
        # Ensure B and V are correct
        if decoded_images.shape[0] == B * V:
            decoded_images = rearrange(decoded_images, "(b v) c h w -> b v c h w", b=B, v=V)
        elif decoded_images.shape[0] == B:
                # Case where it might have lost V dim? Or just 1 view?
                if V == 1:
                    decoded_images = decoded_images.unsqueeze(1)
                else:
                    raise ValueError(f"Shape mismatch: decoded_images {decoded_images.shape}, expected B*V={B*V} or B={B} with V={V}")
    elif decoded_images.ndim == 5:
        pass # Already (B, V, C, H, W)
    else:
        raise ValueError(f"Unexpected decoded_images shape: {decoded_images.shape}")
    
    # Save results if output_dir is active

    # GT images are already [0,1] from video_load.py (no ImageNet normalization)
    # decoded_images are denormalized in rae.decode()
    images_denorm = images  # Already [0, 1]
    if is_dino:
        decoded_images_denorm = decoded_images
    else: 
        decoded_images_denorm = (decoded_images.clamp(-1, 1) + 1) / 2  # Already denormalized in rae.decode()
    # Clamp
    images_denorm = torch.clamp(images_denorm, 0, 1)
    decoded_images_denorm = torch.clamp(decoded_images_denorm, 0, 1)
    
    return decoded_images_denorm





def apply_da3_norm(rae, feat):
    """
    raw → layer_norm: Apply DINOv2 final LayerNorm to second 768 dims.

    Same operation as DA3 backbone's _forward_from_layer (trans.norm).

    Args:
        feat: [raw] (..., 1536) = [local(768), current(768)]
    Returns:
        [layer_norm] (..., 1536) = [local(768), LayerNorm(current)(768)]
    """
    embed_dim = 768 
    
    local = feat[..., :embed_dim]
    current = feat[..., embed_dim:]
    
    # Access norm layer from backbone (DepthAnything3 -> model -> backbone -> pretrained -> norm)
    # Note: 'pretrained' is the Dinov2 model inside DA3 wrapper
    norm_layer = rae.encoder.backbone.pretrained.norm
    current_norm = norm_layer(current)
    
    return torch.cat([local, current_norm], dim=-1)


def decode_into_images(
    rae,
    features,
    feat_latent_denorm,
    level,
    total_view,
    cond_num,
    device='cuda',
    batch=None, 
    stat_path=None,
    sample_idx=0,
    feat_gt_denorm=None,
):
    """
    Decode features into RGB + Depth using RAE decoder.

    Norm flow:
        feat_latent_denorm (raw) → apply_da3_norm → layer_norm (for levels in dict)
        features (latent_norm)   → propagate_features → layer_norm (for deeper levels)
        layer_norm features      → rae.decode() → RGB + Depth

    Args:
        features:         [latent_norm] dict {level: (BV, C, h, w)} — used for propagation
        feat_latent_denorm: [raw] dict {level: (BV, C, h, w)} — converted to layer_norm via apply_da3_norm
        level: Decode starting level (0-3), or -1 to skip propagation (all levels in feat_latent_denorm must be raw)

    Returns:
        dict with 'rgb', 'depth'
    """
    rae.eval()

    if batch is None:
        raise ValueError("batch must be provided")
    
    # Extract image dimensions from batch (needed for decode call)
    images = batch['image'].to(device)  # (B, V, C, H, W)
    B, V, C, H, W = images.shape

    if stat_path is not None:
        if level == 0:
            rae.level=0
            rae._init_normalization(stat_path=stat_path[0])
        elif level == 1:
            rae.level=1
            rae._init_normalization(stat_path=stat_path[1])
        elif level == 2:
            rae.level=2
            rae._init_normalization(stat_path=stat_path[2])
        elif level == 3:
            rae.level=3
            rae._init_normalization(stat_path=stat_path[3])
    else:
        raise ValueError("stat_path must be provided")

    if level != 3 and level !=-1: # run propagation
        images_norm = (images - rae.encoder_mean[None].to('cuda')) / rae.encoder_std[None].to('cuda')

        # Encode ref images to get CLS token [latent_norm]
        ref_images_norm = images_norm[:, :cond_num].reshape(B * cond_num, C, H, W)
        _, ref_gt_cls = rae.encode(ref_images_norm, return_cls=True, mode='single', level=level)
        merged_cls = ref_gt_cls.reshape(B, cond_num, -1)[:, :1].expand(-1, V, -1).reshape(B * V, -1)

        # latent_norm → propagate → [layer_norm patches, raw cls]
        propagated_feats = rae.propagate_features(
            features[level],    # [latent_norm]
            from_level=level,
            total_view=total_view,
            cls_token=merged_cls,  # [latent_norm]
        )  # returns: List[(layer_norm patches, raw cls)]
    else:
        print('already propagated')
        level=3

    # Convert feat_latent_denorm [raw] → [layer_norm] via apply_da3_norm
    # For VGGT: skip apply_da3_norm — DPT head has its own LayerNorm internally
    is_vggt = not hasattr(rae.encoder, 'backbone')
    dpt_features = {}
    for layer, f in feat_latent_denorm.items():
        BV, C, h, w = f.shape
        B = 1
        V = total_view
        feat_5d = f.reshape(B, V, C, h, w)
        feat_5d = feat_5d.permute(0, 1, 3, 4, 2)
        N = h * w
        feat_4d = feat_5d.reshape(B, V, N, C)       # [raw]
        if is_vggt:
            dpt_features[layer] = feat_4d            # [raw] — DPT head norms internally
        else:
            patches_4d_da_norm = apply_da3_norm(rae, feat_4d)  # raw → layer_norm
            dpt_features[layer] = patches_4d_da_norm     # [layer_norm] (B, V, N, C)
    
    # Propagate features from specified level
    # propagate_features returns List[(patches_4d, cls_3d)] for levels >= from_level
    if level == 0: 
        # Propagate from level 0 -> returns [level0, level1, level2, level3]
        # propagated_feats [level0, level1, level2, level3], each has [feat, cls]
        print("in level 0!")
    elif level == 1:
        # Assemble 4-level features for DPT decoder:
        #   L0: dpt_features[0] [layer_norm] (from raw → apply_da3_norm)
        #   L1: dpt_features[1] [layer_norm] (from raw → apply_da3_norm)
        #   L2: propagated_feats[1] [layer_norm] (from backbone forward)
        #   L3: propagated_feats[2] [layer_norm] (from backbone forward)
        B = 1
        BV, C_lat, h_lat, w_lat = features[0].shape
        V = total_view

        gt_patches_0 = dpt_features[0]  # [layer_norm] L0
        gt_patches_1 = dpt_features[1]  # [layer_norm] L1

        dummy_cls_0 = torch.zeros(B, V, C_lat, device=features[0].device, dtype=features[0].dtype)

        # Replace propagated L1 patches with diffusion L1 (from dpt_features), keep propagated CLS
        prop_cls_1 = propagated_feats[0][1]  # [raw] cls from propagation
        propagated_feats[0] = (gt_patches_1, prop_cls_1)

        # Prepend L0
        level0_tuple = (gt_patches_0, dummy_cls_0)
        propagated_feats = [level0_tuple] + propagated_feats
        
    elif level == 2:
        # Assemble: L0,L1 from dpt_features [layer_norm], L2,L3 from propagation [layer_norm]
        B = 1
        V = total_view

        BV, C_lat, h_lat, w_lat = features[0].shape
        gt_patches_0 = dpt_features[0]  # [layer_norm]
        dummy_cls_0 = torch.zeros(B, V, C_lat, device=features[0].device, dtype=features[0].dtype)
        level0_tuple = (gt_patches_0, dummy_cls_0)

        BV, C_lat, h_lat, w_lat = features[1].shape
        gt_patches_1 = dpt_features[1]  # [layer_norm]
        dummy_cls_1 = torch.zeros(B, V, C_lat, device=features[1].device, dtype=features[1].dtype)
        level1_tuple = (gt_patches_1, dummy_cls_1)

        propagated_feats = [level0_tuple, level1_tuple] + propagated_feats

    elif level == 3:
        # All 4 levels from dpt_features [layer_norm], no propagation
        # ⚠️ feat_latent_denorm의 모든 값이 raw 상태여야 apply_da3_norm이 정확함
        B = 1
        V = total_view

        propagated_feats = []
        for lvl in range(4):
            BV, C_lat, h_lat, w_lat = features[lvl].shape
            gt_patches = dpt_features[lvl]  # [layer_norm] (raw → apply_da3_norm)
            dummy_cls = torch.zeros(B, V, C_lat, device=features[lvl].device, dtype=features[lvl].dtype)
            propagated_feats.append((gt_patches, dummy_cls))
    # elif level == -1:
    #     print('already have all the features')
    else:
        raise ValueError(f"Invalid level: {level}")
        
    # Ensure dtype matches decoder
    if rae.rae_cl_decoder is not None:
        decoder_dtype = next(rae.rae_cl_decoder.parameters()).dtype
    else:
        decoder_dtype = torch.float32
        
    propagated_feats_casted = []
    for p, c in propagated_feats:
        p = p.to(dtype=decoder_dtype)
        if c is not None:
            c = c.to(dtype=decoder_dtype)
        propagated_feats_casted.append((p, c))

    # CHECK IF DECODER HANDLES level0 vs level1, WHICH BETTER?
    # if level == 0:
    #     # zero out all featues except level 0 
    #     feature_zeros = torch.zeros_like(propagated_feats_casted[0][0])
    #     cls_zeros = torch.zeros_like(propagated_feats_casted[0][1])

    #     propagated_feats_casted[0] = (feature_zeros, cls_zeros)
    #     propagated_feats_casted[2] = (feature_zeros, cls_zeros)
    #     propagated_feats_casted[3] = (feature_zeros, cls_zeros)

    # elif level == 1:
    #     # zero out all features except level 1
    #     feature_zeros = torch.zeros_like(propagated_feats_casted[1][0])
    #     cls_zeros = torch.zeros_like(propagated_feats_casted[1][1])

    #     propagated_feats_casted[0] = (feature_zeros, cls_zeros)
    #     propagated_feats_casted[2] = (feature_zeros, cls_zeros)
    #     propagated_feats_casted[3] = (feature_zeros, cls_zeros)


    # CHECK IF REPLACING 'BEFORE FEATURE' W/ GT SHOWS IMPROVEMENT
    # if feat_gt_denorm is not None:
    #     print("Replacing 'before features' to GT features")
    #     cls_zeros = torch.zeros_like(propagated_feats_casted[0][1])
    #     # da3_norm for gt features
    #     # feat_gt_denorm shape (BV, C, H, W)

    #     for lvl in range(4):
    #         BV,C,h,w = feat_gt_denorm[lvl].shape
    #         B=1
    #         V=total_view
    #         feat_gt_denorm[lvl] = feat_gt_denorm[lvl].reshape(B, V, C, h * w)
    #         feat_gt_denorm[lvl] = feat_gt_denorm[lvl].permute(0, 1, 3, 2) # B, V, N, C
    #         feat_gt_denorm[lvl] = apply_da3_norm(rae, feat_gt_denorm[lvl])
        
        
    #     if level==0:
    #         pass
    #     elif level==1:
    #         print("Replacing level 0 to GT")
    #         propagated_feats_casted[0] = (feat_gt_denorm[0], cls_zeros)
    #     elif level==2:
    #         print("Replacing level 0 and 1 to GT")
    #         propagated_feats_casted[0] = (feat_gt_denorm[0], cls_zeros)
    #         propagated_feats_casted[1] = (feat_gt_denorm[1], cls_zeros)
    #     elif level==3:
    #         print("Replacing level 0, 1, and 2 to GT")
    #         propagated_feats_casted[0] = (feat_gt_denorm[0], cls_zeros)
    #         propagated_feats_casted[1] = (feat_gt_denorm[1], cls_zeros)
    #         propagated_feats_casted[2] = (feat_gt_denorm[2], cls_zeros)

    with torch.no_grad():
        res = rae.decode(propagated_feats_casted, H, W)
        decoded_images = res['rgb']
        decoded_depth = res.get('depth', None)
        decoded_depth_conf = res.get('depth_conf', None)
        decoded_ray = res.get('ray', None)
        decoded_ray_conf = res.get('ray_conf', None)

    return {
        'rgb': decoded_images, 'depth': decoded_depth, 'depth_conf': decoded_depth_conf,
        'ray': decoded_ray, 'ray_conf': decoded_ray_conf,
    }

# def get_propagated_feats(rae, images, cond_num, features, level, total_view):
#     '''
#     get propagated feats from RAE
#     Args:
#     rae: RAE_DA3 model, stat initialized for feature level 
#     images: (B, V, C, H, W)
#     features: Latent features at given level, shape (B*V, C, h, w)
#     level: level to propagate from
#     total_view: total number of views
#     '''
#     B,V,C,H,W = images.shape
#     # need cls token for propagation (when predict_cls=False)
#     images_norm = (images - rae.encoder_mean[None]) / rae.encoder_std[None]
#     ref_images_norm = images_norm[:, :cond_num].reshape(B * cond_num, C, H, W)
#     _, ref_gt_cls = rae.encode(ref_images_norm, return_cls=True, mode='single', level=level)
#     merged_cls = ref_gt_cls.reshape(B, cond_num, -1)[:, :1].expand(-1, V, -1).reshape(B * V, -1)
#     import pdb; pdb.set_trace()
#     propagated_feats = rae.propagate_features(
#         features,  # Latent Normalized input
#         from_level=level,
#         total_view=total_view,
#         cls_token=merged_cls,  # Latent Normalized CLS
#     ) # DA3 normalized propagated features
    
#     return propagated_feats


# def decode_into_images_level3(
#     rae,
#     features,
#     feat_latent_denorm,
#     level,
#     total_view,
#     cond_num,
#     device='cuda',
#     batch=None, 
#     stat_path=None,
#     sample_idx=0,
# ):
#     """
#     Decode features into images using RAE decoder.
#     Uses consistent level0 normalization throughout to match decode_into_images behavior.
    
#     Args:
#         rae: RAE_DA3 model
#         features: Latent features, dict with levels as keys, each of shape (B*V, C, h, w)
#         feat_latent_denorm: Denormalized features, dict with levels as keys, each of shape (B*V, C, h, w)
#         level: Level to decode (currently only level=3 is supported)
#         device: Device
        
#     Returns:
#         Decoded images (B, 3, H, W)
#     """
#     rae.eval()
    
#     if batch is None:
#         raise ValueError("batch must be provided")
#     import pdb; pdb.set_trace()
#     # Extract image dimensions from batch (needed for decode call)
#     images = batch['image'].to(device)  # (B, V, C, H, W)
#     B, V, C, H, W = images.shape

#     if stat_path is None:
#         raise ValueError("stat_path must be provided for feature propagation")
    
#     # Use ONLY level0 normalization for consistent behavior
#     # This matches decode_into_images which sets level0 normalization once
#     rae.level = 0
#     rae._init_normalization(stat_path=stat_path[0])
    
#     # Propagate ONLY from level0 with level0 normalization
#     # This gives us [level0, level1, level2, level3] all from consistent level0 context
#     propagated_feats_level0 = get_propagated_feats(rae, images, cond_num, features[0], 0, total_view)
#     # propagated_feats_level0 = [level0_tuple, level1_tuple, level2_tuple, level3_tuple]

#     if level == 3:
#         # Use all 4 levels from level0 propagation
#         # propagated_feats_level0[0] = (level0_patches, level0_cls)
#         # propagated_feats_level0[1] = (level1_patches, level1_cls)
#         # propagated_feats_level0[2] = (level2_patches, level2_cls)
#         # propagated_feats_level0[3] = (level3_patches, level3_cls)
        
#         # Get level1 from propagation (same as decode_into_images with level=0 keeping level1)
#         level0_feats = propagated_feats_level0[0][0]  # (B, V, N, C)
#         level1_feats = propagated_feats_level0[1][0]  # (B, V, N, C)
#         level2_feats = propagated_feats_level0[2][0]  # (B, V, N, C)
#         level3_feats = propagated_feats_level0[3][0]  # (B, V, N, C)
        
#         B_f, V_f, N, C_feat = level1_feats.shape
        
#         # For now, keep only level1 and zero out others (matching decode_into_images level=0 behavior)
#         level0_avg = torch.zeros(B_f, V_f, N, C_feat, device=level1_feats.device, dtype=level1_feats.dtype)
#         level1_avg = level1_feats  # Keep level1
#         level2_avg = torch.zeros(B_f, V_f, N, C_feat, device=level1_feats.device, dtype=level1_feats.dtype)
#         level3_avg = torch.zeros(B_f, V_f, N, C_feat, device=level1_feats.device, dtype=level1_feats.dtype)

#         final_feats = {0: level0_avg, 1: level1_avg, 2: level2_avg, 3: level3_avg}

#         # Create list with dummy CLS tokens
#         propagated_feats = []
#         for lvl in range(4):
#             B_l, V_l, N_l, C_l = final_feats[lvl].shape
#             dummy_cls = torch.zeros(B_l, V_l, C_l, device=final_feats[lvl].device, dtype=final_feats[lvl].dtype)
#             propagated_feats.append((final_feats[lvl], dummy_cls))
#     else:
#         raise ValueError(f"Invalid level: {level}")
        
#     # Ensure dtype matches decoder
#     if rae.rae_cl_decoder is not None:
#         decoder_dtype = next(rae.rae_cl_decoder.parameters()).dtype
#     else:
#         decoder_dtype = torch.float32
        
#     propagated_feats_casted = []
#     for p, c in propagated_feats:
#         p = p.to(dtype=decoder_dtype)
#         if c is not None:
#             c = c.to(dtype=decoder_dtype)
#         propagated_feats_casted.append((p, c))

#     with torch.no_grad():
#         res = rae.decode(propagated_feats_casted, H, W)
#         decoded_images = res['rgb']

#     return decoded_images