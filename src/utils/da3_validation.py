
import torch
import torch.distributed as dist
import torchvision
from einops import rearrange
import logging, math
import os

def make_vis_multiview(src_imgs, tgt_imgs, cond_num: int):
    """
    Visualize source and target images side-by-side.
    """
    if cond_num is None:
        raise ValueError("cond_num must be provided explicitly to make_vis_multiview.")
    if not isinstance(cond_num, int):
        raise TypeError(f"cond_num must be int, got {type(cond_num)}")
    V = src_imgs.shape[0]
    if not (0 <= cond_num <= int(V)):
        raise ValueError(f"cond_num must satisfy 0 <= cond_num <= V, got cond_num={cond_num}, V={int(V)}")
    vis_list = []
    
    for v in range(V):
        src = src_imgs[v]
        tgt = tgt_imgs[v]
        pair = torch.cat([src, tgt], dim=2) # (C, H, 2W)
        vis_list.append(pair)
        if v == cond_num - 1 and cond_num < V:
            sep = torch.zeros((pair.shape[0], 4, pair.shape[2]), dtype=pair.dtype, device=pair.device)
            vis_list.append(sep)
    
    if len(vis_list) == 0:
        return torch.zeros(3, 256, 512)
    # Stack vertically (C, sumH, 2W)
    grid = torch.cat(vis_list, dim=1)
        
    return grid

def validate_da3_multiview(
    rae,
    model,
    transport,
    sampler,
    loader,
    device,
    total_view,
    cond_num,
    compute_loss,
    val_num_batches=None,
    use_prope=False,
    validation_mode="propagation",  # 'propagation', 'replacement', 'naive'
    rank=0,
    world_size=1,
    output_dir=None, # If provided, save images
    pag_scale=None,  # PAG (Perturbed Attention Guidance) scale. None = disabled.
    pag_layer_idx=None, # Layer to apply PAG to
    cfg_scale=None,  # CFG (Classifier-Free Guidance) scale. None or 1.0 = disabled.
    run_config=None,  # Optional dict with run configuration to save
    prope_image_size=None,  # Image size for ProPE (dynamic, avoid hardcoding, if None uses batch H,W)
    predict_cls=False,  # If True, also predict/generate CLS token
    joint_ode=False,  # If True, use Joint ODE inference (ref views start from F_r^ind, not clamped)
    ref_view_sampling='prefix', # sampling method for reference views
    camera_mode='camray', # 'camray' or 'plucker'
    ray_pose_mode='c2w',
    is_concat_mode=False,  # If True, use concat mode: [cond|noisy] input
    source_level=None,  # Feature-to-Feature Flow: source level for x0 initialization
    source_level_stat_path=None,  # Normalization stat path for source level (must differ from target)
    noise_tau_gt_feat=0.0,  # Tau for noise augmentation: sigma ~ |N(0, tau^2)|
    # NEW: Feature Propagation Mode (Image 1 Architecture)
    source_condition_level=None,  # L1 features for encoder conditioning (not x0 init)
    source_condition_stat_path=None,  # Normalization stat path for source condition level
):
    """
    Shared validation/inference logic for DA3 MVDiffusion.
    
    Args:
        validation_mode:
            - 'naive': Just decode the generated latent using RAE.decode() (MAE decoder).
            - 'replacement': Replace target level in GT features with generated latent, decode with DPT.
            - 'propagation': Use generated latent to compute rest, decode with DPT.
                             For Level -4: Uses downward propagation (backbone forward)
            - 'propagation_0': Similar to propagation, but uses Zero features for upward propagation (features < current level).
        run_config: Optional dict with run configuration (checkpoint, pag_scale, etc.) to save with results.
    """

    print(f"Validation mode: {validation_mode}")
    
    all_images = []
    saved_sample_count = 0  # Counter for saved samples
    
    # Track metrics if needed
    metrics_acc = {}
    per_sample_metrics = []  # List of per-sample metrics

    val_loss_sum = torch.tensor(0.0, device=device)
    val_ref_loss_sum = torch.tensor(0.0, device=device) # Split Loss
    val_tgt_loss_sum = torch.tensor(0.0, device=device) # Split Loss
    val_loss_count = torch.tensor(0.0, device=device)
    
    # Track metrics for global aggregation (Separated)
    # Target (Default)
    psnr_tgt_sum = torch.tensor(0.0, device=device)
    ssim_tgt_sum = torch.tensor(0.0, device=device)
    lpips_tgt_sum = torch.tensor(0.0, device=device)
    mse_tgt_sum = torch.tensor(0.0, device=device)
    metrics_tgt_count = torch.tensor(0.0, device=device)

    # Reference
    psnr_ref_sum = torch.tensor(0.0, device=device)
    ssim_ref_sum = torch.tensor(0.0, device=device)
    lpips_ref_sum = torch.tensor(0.0, device=device)
    mse_ref_sum = torch.tensor(0.0, device=device)
    metrics_ref_count = torch.tensor(0.0, device=device)
    
    from tqdm import tqdm
    
    # Use tqdm for progress bar
    # If val_num_batches is set, use it as total, otherwise loader length
    total_steps = len(loader)
    if val_num_batches is not None:
        if not isinstance(val_num_batches, int) or val_num_batches <= 0:
            raise ValueError(f"val_num_batches must be a positive int or None, got {val_num_batches} ({type(val_num_batches)})")
        total_steps = min(total_steps, val_num_batches)
        print(f"[validate] Limiting batches: val_num_batches={val_num_batches} / len(loader)={len(loader)}")
        
    pbar = tqdm(enumerate(loader), total=total_steps, desc="Validation")
    
    # Determine if model expects CLS token from its own attribute if predict_cls is not explicitly set
    m = model.module if hasattr(model, 'module') else model
    if predict_cls is None:
        predict_cls = getattr(m, 'predict_cls', False)
    
    for i, batch in pbar:
        if val_num_batches is not None and i >= val_num_batches:
            break
            
        # CUT3R batch conversion (List[Dict] -> Dict)
        if isinstance(batch, list) and isinstance(batch[0], dict):
            from video.cut3r_adapter import convert_cut3r_batch
            batch = convert_cut3r_batch(batch, cond_num, ref_view_sampling)
        
        if all([key in batch for key in ['gt_inp', 'fxfycxcy', 'c2w']]):
            images = batch['gt_inp'].to(device) # (B, V, C, H, W)
            intrinsic = batch['fxfycxcy'].to(device)
            extrinsic = batch['c2w'].to(device)
        elif all([key in batch for key in ['image', 'c2w', 'intrinsic']]):
            images = batch['image'].to(device) # (B, V, C, H, W)
            intrinsic = batch['intrinsic'].to(device)
            extrinsic = batch['c2w'].to(device)
        else:
            raise ValueError("Batch must contain 'gt_inp', 'fxfycxcy', 'c2w' or 'image', 'extrinsic', 'intrinsic' keys")
        B, V, C, H, W = images.shape
        
        # Debug: Log batch resolution
        if i < 5 or i % 20 == 0:
            print(f"[VAL batch {i}] Resolution: {H}x{W}, B={B}, V={V}")
        
        # 1. Condition & Camera Setup (Common)
        
        # ImageNet Normalization
        images_norm = (images - rae.encoder_mean[None]) / rae.encoder_std[None]
        
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
            camera_embedding, scale = get_camera_embedding(intri_, extri_, B, V, H, W, mode=camera_mode, ray_pose_mode=ray_pose_mode, return_scale=True)
        else:
            camera_embedding = get_camera_embedding(intri_, extri_, B, V, H, W, mode=camera_mode, ray_pose_mode=ray_pose_mode, return_scale=False)
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
            if predict_cls:
                # [LEAKAGE FIX]
                # A. All views together (for Target GTs)
                latents_all, cls_all = rae.encode(images_norm, return_cls=True)
                # B. Reference views only (for Conditioning)
                latents_ref, cls_ref = rae.encode(images_norm[:, :cond_num], return_cls=True)
                
                # C. Merge
                C_lat, h_lat, w_lat = latents_all.shape[1:]
                C_cls = cls_all.shape[1]
                
                latents_merged_5d = latents_all.reshape(B, V, C_lat, h_lat, w_lat).clone()
                latents_merged_5d[:, :cond_num] = latents_ref.reshape(B, cond_num, C_lat, h_lat, w_lat)
                gt_latents = latents_merged_5d.reshape(B * V, C_lat, h_lat, w_lat)
                
                cls_merged_3d = cls_all.reshape(B, V, C_cls).clone()
                cls_merged_3d[:, :cond_num] = cls_ref.reshape(B, cond_num, C_cls)
                gt_cls_toks = cls_merged_3d.reshape(B * V, C_cls)
            else:
                # [LEAKAGE FIX]
                latents_all = rae.encode(images_norm)
                latents_ref = rae.encode(images_norm[:, :cond_num])
                
                C_lat, h_lat, w_lat = latents_all.shape[1:]
                latents_merged_5d = latents_all.reshape(B, V, C_lat, h_lat, w_lat).clone()
                latents_merged_5d[:, :cond_num] = latents_ref.reshape(B, cond_num, C_lat, h_lat, w_lat)
                gt_latents = latents_merged_5d.reshape(B * V, C_lat, h_lat, w_lat)
                gt_cls_toks = None
        
        # Derive dimensions from actual encoded latents
        latent_dim = gt_latents.shape[1]
        h_lat = gt_latents.shape[2]
        w_lat = gt_latents.shape[3]
        
        if predict_cls:
            # Reshape patches to (B, V, C, H*W, 1) and CLS to (B, V, C, 1, 1)
            gt_patches_flat = gt_latents.reshape(B, V, latent_dim, h_lat * w_lat, 1)
            gt_cls_flat = gt_cls_toks.reshape(B, V, latent_dim, 1, 1)
            gt_latents_combined = torch.cat([gt_cls_flat, gt_patches_flat], dim=3) # (B, V, C, N+1, 1)
            
            # Create sampling input
            sample_input = gt_latents_combined.clone()
            # Noise for target views (including CLS)
            target_noise = torch.randn(B, V - cond_num, latent_dim, h_lat * w_lat + 1, 1, device=device, dtype=gt_latents.dtype)
            sample_input[:, cond_num:] = target_noise
            
            sample_input_flat = sample_input.reshape(B * V, latent_dim, h_lat * w_lat + 1, 1)
        else:
            # Reshape to (B, V, C, h, w)
            # Reshape to (B, V, C, h, w)
            gt_latents_5d = gt_latents.reshape(B, V, latent_dim, h_lat, w_lat)
            
            if is_concat_mode:
                # Concat Mode: [cond_channel | noise_or_source] for all views
                # Ref views: [clean_ref | noise_or_source]
                # Tgt views: [zeros | noise_or_source]
                # Get clean ref features (leakage-free)
                
                latents_ref_5d = latents_ref.reshape(B, cond_num, latent_dim, h_lat, w_lat)
                
                # Feature-to-Feature Flow: use source features + noise as x0
                # Standard diffusion: use pure noise as x0
                if source_level is not None:
                    # Encode source features with source-level normalization
                    with torch.no_grad():
                        # Temporarily switch RAE to source-level normalization
                        original_do_norm = rae.do_normalization
                        original_mean = rae.latent_mean
                        original_var = rae.latent_var
                        
                        if source_level_stat_path is not None:
                            source_stats = torch.load(source_level_stat_path, map_location=device)
                            rae.latent_mean = source_stats.get('mean', None)
                            rae.latent_var = source_stats.get('var', None)
                            if rae.latent_mean is not None:
                                rae.latent_mean = rae.latent_mean.to(device)
                            if rae.latent_var is not None:
                                rae.latent_var = rae.latent_var.to(device)
                            rae.do_normalization = True
                        
                        latents_source = rae.encode(images_norm, mode='single', level=source_level)
                        latents_source_5d = latents_source.reshape(B, V, latent_dim, h_lat, w_lat)
                        
                        # Restore target-level normalization
                        rae.do_normalization = original_do_norm
                        rae.latent_mean = original_mean
                        rae.latent_var = original_var
                    
                    # Add noise: sigma ~ |N(0, tau^2)|
                    if noise_tau_gt_feat > 0:
                        noise_std = torch.abs(torch.randn(1, device=device) * noise_tau_gt_feat).item()
                    else:
                        noise_std = 0.0
                    noise = torch.randn_like(latents_source_5d) * noise_std
                    noisy_part = (latents_source_5d + noise) / math.sqrt(1 + noise_std**2)
                else:
                    # Standard diffusion: pure noise
                    noisy_part = torch.randn(B, V, latent_dim, h_lat, w_lat, device=device, dtype=gt_latents.dtype)
                
                # NEW: Feature Propagation Mode (Image 1 Architecture)
                # Extract L1 features for encoder conditioning (NOT for x0 init)
                if source_condition_level is not None:
                    with torch.no_grad():
                        # Temporarily switch RAE to source-condition-level normalization
                        original_do_norm = rae.do_normalization
                        original_mean = rae.latent_mean
                        original_var = rae.latent_var
                        
                        if source_condition_stat_path is not None:
                            source_cond_stats = torch.load(source_condition_stat_path, map_location=device)
                            rae.latent_mean = source_cond_stats.get('mean', None)
                            rae.latent_var = source_cond_stats.get('var', None)
                            if rae.latent_mean is not None:
                                rae.latent_mean = rae.latent_mean.to(device)
                            if rae.latent_var is not None:
                                rae.latent_var = rae.latent_var.to(device)
                            rae.do_normalization = True
                        
                        # Encode L1 features with source-condition-level normalization
                        latents_source_cond = rae.encode(images_norm, mode='single', level=source_condition_level)
                        # latents_source_cond: (B*V, C, H, W)
                        
                        # Restore target-level normalization
                        rae.do_normalization = original_do_norm
                        rae.latent_mean = original_mean
                        rae.latent_var = original_var
                    
                    # Add noise to L1 condition features (reduced for validation determinism)
                    # Note: noise_tau_gt_feat is already divided by 4 before being passed here
                    if noise_tau_gt_feat > 0:
                        cond_noise_std = torch.abs(torch.randn(1, device=device) * noise_tau_gt_feat).item()
                    else:
                        cond_noise_std = 0.0
                    cond_noise = torch.randn_like(latents_source_cond) * cond_noise_std
                    latents_source_cond_noisy = (latents_source_cond + cond_noise) / math.sqrt(1 + cond_noise_std**2)
                    
                    # Add noisy L1 features to model_kwargs for encoder conditioning
                    model_kwargs['source_condition'] = latents_source_cond_noisy
                
                # Build condition channel: clean for ref, zeros for tgt
                cond_channel = torch.zeros(B, V, latent_dim, h_lat, w_lat, device=device, dtype=gt_latents.dtype)
                cond_channel[:, :cond_num] = latents_ref_5d  # Ref views get clean features
                # Tgt views remain zeros
                
                # Concatenate: [cond_channel | noisy_part] along channel dim
                sample_input = torch.cat([cond_channel, noisy_part], dim=2)  # (B, V, 2*C, h, w)

                # Flatten for sampling
                sample_input_flat = sample_input.reshape(B * V, latent_dim * 2, h_lat, w_lat)
                model_kwargs['is_concat_mode'] = True
                model_kwargs['ref_cond'] = cond_channel.reshape(B * V, latent_dim, h_lat, w_lat)
                model_kwargs['x1_global'] = latents_all.reshape(B * V, latent_dim, h_lat, w_lat)
                
            elif joint_ode:
                # Joint ODE: Ref views start from F_r^ind, Target views start from noise
                # latents_ref is already F_r^ind (ref-only encoding) (but might have different shape depending on usage above)
                # Ensure latents_ref has correct 5d shape
                latents_ref_5d = latents_ref.reshape(B, cond_num, latent_dim, h_lat, w_lat)
                target_noise = torch.randn(B, V - cond_num, latent_dim, h_lat, w_lat, device=device, dtype=gt_latents.dtype)
                sample_input = torch.cat([latents_ref_5d, target_noise], dim=1)  # [F_r^ind, noise]
                
                # Flatten for sampling
                sample_input_flat = sample_input.reshape(B * V, latent_dim, h_lat, w_lat)
            else:
                # Target-only: Ref views = GT (F_r^ind clamped), Target views = noise OR source features
                sample_input = gt_latents_5d.clone()
                
                # Feature-to-Feature Flow: use source features + noise as x0
                # CRITICAL: Source and target levels have DIFFERENT distributions!
                # Must use source-level normalization stats for proper encoding.
                if source_level is not None:
                    with torch.no_grad():
                        # Temporarily switch RAE to source-level normalization
                        original_do_norm = rae.do_normalization
                        original_mean = rae.latent_mean
                        original_var = rae.latent_var
                        
                        if source_level_stat_path is not None:
                            # Load source-level stats
                            source_stats = torch.load(source_level_stat_path, map_location=device)
                            rae.latent_mean = source_stats.get('mean', None)
                            rae.latent_var = source_stats.get('var', None)
                            if rae.latent_mean is not None:
                                rae.latent_mean = rae.latent_mean.to(device)
                            if rae.latent_var is not None:
                                rae.latent_var = rae.latent_var.to(device)
                            rae.do_normalization = True
                        
                        # Encode with source-level normalization
                        latents_source = rae.encode(images_norm, mode='single', level=source_level)
                        # latents_source: (B*V, C, H, W) - normalized with source-level stats
                        latents_source_5d = latents_source.reshape(B, V, latent_dim, h_lat, w_lat)
                        
                        # Restore target-level normalization
                        rae.do_normalization = original_do_norm
                        rae.latent_mean = original_mean
                        rae.latent_var = original_var
                    
                    # Sample noise: sigma ~ |N(0, tau^2)| (but for validation, typically use tau=0 for determinism)
                    if noise_tau_gt_feat > 0:
                        noise_std = torch.abs(torch.randn(1, device=device) * noise_tau_gt_feat).item()
                    else:
                        noise_std = 0.0
                    noise = torch.randn_like(latents_source_5d) * noise_std
                    sample_input[:, cond_num:] = (latents_source_5d[:, cond_num:] + noise[:, cond_num:]) / math.sqrt(1 + noise_std**2)
                else:
                    # Standard: target views get pure noise
                    target_noise = torch.randn(B, V - cond_num, latent_dim, h_lat, w_lat, device=device, dtype=gt_latents.dtype)
                    sample_input[:, cond_num:] = target_noise
                
                # Flatten for sampling (B*V, C, H, W)
                sample_input_flat = sample_input.reshape(B * V, latent_dim, h_lat, w_lat)

        if compute_loss:
            if transport is None:
                raise ValueError("compute_loss=True requires a non-None `transport` argument.")
            loss_latents = gt_latents
            if predict_cls:
                if gt_cls_toks is None:
                    raise ValueError("predict_cls=True but gt_cls_toks is None.")
                BV_loss, C_loss, H_loss, W_loss = loss_latents.shape
                loss_latents_flat = loss_latents.reshape(BV_loss, C_loss, H_loss * W_loss, 1)
                loss_cls_flat = gt_cls_toks.reshape(BV_loss, C_loss, 1, 1)
                loss_latents = torch.cat([loss_cls_flat, loss_latents_flat], dim=2)  # (BV, C, N+1, 1)

            with torch.no_grad():
                # Fix: Use a copy of model_kwargs to prevent training_multiview_losses from popping keys 
                # that are needed for sampling (like is_concat_mode)
                m_kwargs_loss = model_kwargs.copy()
                # [LEAKAGE FIX] Use cond_channel (Ref=GT, Tgt=Zero) as Condition (x1 arg)
                # The REAL Target (GT) is passed via model_kwargs['x1_global'] (set to latents_all above)
                if is_concat_mode:
                    # cond_channel is (B, V, C, H, W) but training_multiview_losses expects (BV, C, H, W)
                    loss_input = cond_channel.reshape(B * V, latent_dim, h_lat, w_lat)
                else:
                    loss_input = loss_latents

                loss_dict = transport.training_multiview_losses(
                    model=model,
                    x1=loss_input,
                    total_view=total_view,
                    cond_num=cond_num,
                    model_kwargs=m_kwargs_loss,
                    t_override=0.5,  # Fixed timestep for validation
                )
                loss_batch = loss_dict["loss"].mean()
                if not torch.isfinite(loss_batch).all():
                    raise RuntimeError(f"Non-finite validation loss detected: {loss_batch.item()}")
                
                # Split Loss Accumulation
                ref_loss_batch = loss_dict.get("ref_loss", loss_dict["loss"]).mean()
                tgt_loss_batch = loss_dict.get("tgt_loss", loss_dict["loss"]).mean()
                
                val_loss_sum += loss_batch * float(loss_latents.shape[0])
                val_ref_loss_sum += ref_loss_batch * float(loss_latents.shape[0])
                val_tgt_loss_sum += tgt_loss_batch * float(loss_latents.shape[0])
                val_loss_count += float(loss_latents.shape[0])

        with torch.no_grad():
            # Pass model directly (transport now handles positional total_view)
            samples = sampler(sample_input_flat, model, **model_kwargs)[-1]  # (B*V, C, H, W) or (B*V, C, N+1, 1)

        if is_concat_mode:
            # Extract only the denoised part (second half of channels)
            # samples shape: (BV, 2*C, ...) -> (BV, C, ...)
            samples = samples[:, latent_dim:]

        # Detect packed format with special tokens (BV, C, K+N, 1)
        num_special = getattr(rae, 'num_special_tokens', 0)
        has_special_tokens = (num_special > 0 and samples.ndim == 4 and samples.shape[3] == 1
                              and samples.shape[2] == num_special + h_lat * w_lat)

        if predict_cls:
            # Extract CLS and patches from samples (B*V, C, N+1, 1)
            samples_cls = samples[:, :, :1, 0].squeeze(-1) # (B*V, C)
            samples_patches = samples[:, :, 1:, 0].reshape(B * V, latent_dim, h_lat, w_lat)
            samples = samples_patches # Use patches for standard decoding logic below
            # samples_cls will be used in propagation/replacement if needed
        else:
            samples_cls = None

        def apply_da3_norm(rae, feat):
            """
            Apply DA3 backbone LayerNorm to raw features.
            feat: (..., 1536) -> [local(768), current(768)]
            Returns: [local(768), norm(current)(768)]
            """
            embed_dim = 768 
            
            local = feat[..., :embed_dim]
            current = feat[..., embed_dim:]
            
            # Access norm layer from backbone (DepthAnything3 -> model -> backbone -> pretrained -> norm)
            # Note: 'pretrained' is the Dinov2 model inside DA3 wrapper
            norm_layer = rae.encoder.backbone.pretrained.norm
            current_norm = norm_layer(current)
            
            return torch.cat([local, current_norm], dim=-1)

        # 3. Decoding Strategy

        # Prepare GT features (for Replacement or Propagation logic)
        # encode(mode='all') returns Dict[level_idx -> raw features] (DA3 Denorm / Raw state)
        gt_feats_all = rae.encode(images_norm, mode='all') 
        decoded_images_list = []

        def _normalize_gt_feats_to_neg_levels(gt_feats_dict):
            """
            Normalize gt feature dict keys to negative DA3 level notation: {-4, -3, -2, -1}.
            Supported inputs:
              - keys {0,1,2,3}  (level index)
              - keys {-4,-3,-2,-1} (negative level)
            Fail loudly on any other key set.
            """
            keys = set(gt_feats_dict.keys())
            if keys.issubset({0, 1, 2, 3}):
                out = {int(k) - 4: v for k, v in gt_feats_dict.items()}
            elif keys.issubset({-4, -3, -2, -1}):
                out = {int(k): v for k, v in gt_feats_dict.items()}
            else:
                raise ValueError(
                    f"Unexpected gt_feats_all keys={sorted(list(keys))}. "
                    "Expected subset of {0,1,2,3} or {-4,-3,-2,-1}. "
                    "This indicates a mismatch between RAE.encode(mode='all') contract and validation logic."
                )
            # Ensure all required levels exist
            for lvl in (-4, -3, -2, -1):
                if lvl not in out:
                    raise ValueError(
                        f"gt_feats_all is missing required level {lvl}. "
                        f"Available keys after normalization: {sorted(list(out.keys()))}"
                    )
            return out
        
        if validation_mode == "replacement":
            # REPLACEMENT MODE:
            # - Use GT features for ALL 4 levels (Raw state)
            # - Replace ONLY target level (and only target views) with generated latent
            
            # NOTE: gt_feats_all are Raw Features (no LayerNorm, no DA3 Norm)
            # samples are DA3 Norm -> Denormalize to get Raw Features
            samples_raw = rae._denormalize(samples)
            
            # Reshape to (B*V, N, C)
            bv, c, h_lat, w_lat = samples_raw.shape
            generated_seq = samples_raw.reshape(bv, c, -1).transpose(1, 2)  # (B*V, N, C)
            
            if samples_cls is not None:
                samples_cls_raw = rae._denormalize(samples_cls) # Denorm CLS
                generated_seq = torch.cat([samples_cls_raw.unsqueeze(1), generated_seq], dim=1) # (B*V, N+1, C)
            
            # Create view mask - only replace target views
            view_mask = torch.zeros(B, V, dtype=torch.bool, device=device)
            view_mask[:, cond_num:] = True
            
            # Replace target level with generated features (Raw state)
            modified_feats = rae.replace_level_features(
                gt_feats=gt_feats_all,
                generated_latent=generated_seq,
                level=rae.level,
                view_mask=view_mask,
            )
            
            # Convert Dict to DPT decoder format: List[Tuple[patches_4d, cls_3d]]
            # AND Apply LayerNorm because DPT Decoder expects Normed features
            dpt_feats = []
            for level_idx in range(4):
                feat_raw = modified_feats[level_idx]  # (B*V, N+1, C) Raw
                
                # Reshape to (B, V, N+1, C)
                feat_5d = feat_raw.reshape(B, V, *feat_raw.shape[1:])
                
                # Split CLS and patches (Raw)
                cls_3d_raw = feat_5d[:, :, 0, :]      # (B, V, C)
                patches_4d_raw = feat_5d[:, :, 1:, :] # (B, V, N, C)
                
                # Apply LayerNorm MANUALLY to patches ONLY
                # DPT Decoder expects: (Raw CLS, Normed Patches)
                patches_4d_norm = apply_da3_norm(rae, patches_4d_raw)
                
                # dpt_feats.append((patches_4d_norm, cls_3d_raw))
                # Re-enabled:
                dpt_feats.append((patches_4d_norm, cls_3d_raw))
            
            res = rae.decode(dpt_feats, H, W)
            decoded_images = res['rgb']
            
        if validation_mode in ["propagation", "propagation_gt", "propagation_0"]:
            # Step 1: Get GT shallow features WITH CLS at the target level ONLY for Reference views
            # NOTE: gt_shallow is DA3 Norm (규칙 3: encode output is DA3 Norm)
            with torch.no_grad():
                # Only encode reference views: (B, cond_num, C, H, W) -> (B * cond_num, C, H, W)
                ref_images_norm = images_norm[:, :cond_num].reshape(B * cond_num, C, H, W)
                ref_gt_shallow, ref_gt_cls = rae.encode(ref_images_norm, return_cls=True, mode='single', level=rae.level)
            
            # Step 2: Prepare Merged patches & CLS
            # Both gt_shallow and samples are in DA3 Norm state!
            # - ref_gt_shallow: DA3 Norm (from encode)
            # - samples: DA3 Norm (from DDT)

            if has_special_tokens:
                # Packed format: (BV, C, K+N, 1)
                feat_shape = ref_gt_shallow.shape  # (B*cond_num, C, K+N, 1)
                C_feat = feat_shape[1]
                seq_len = feat_shape[2]  # K+N
                samples_5d = samples.reshape(B, V, C_feat, seq_len, 1)
                merged_shallow = samples_5d.clone()
                if not (joint_ode or is_concat_mode):
                    merged_shallow[:, :cond_num] = ref_gt_shallow.reshape(B, cond_num, C_feat, seq_len, 1)
                merged_shallow_flat = merged_shallow.reshape(B * V, C_feat, seq_len, 1)
            else:
                # Spatial format: (BV, C, h, w)
                _, C_feat, h_feat, w_feat = ref_gt_shallow.shape
                samples_5d = samples.reshape(B, V, C_feat, h_feat, w_feat)
                merged_shallow = samples_5d.clone()
                if not (joint_ode or is_concat_mode):
                    merged_shallow[:, :cond_num] = ref_gt_shallow.reshape(B, cond_num, C_feat, h_feat, w_feat)
                merged_shallow_flat = merged_shallow.reshape(B * V, C_feat, h_feat, w_feat)
            
            # Step 3: Merge CLS
            if samples_cls is not None:
                # Initialize with samples_cls: (B*V, C)
                merged_cls = samples_cls.reshape(B, V, -1).clone()
                # Joint ODE or Concat Mode: Use predicted CLS directly (no overwrite)
                # Target-only: Overwrite ref CLS with GT
                if not (joint_ode or is_concat_mode):
                    merged_cls[:, :cond_num] = ref_gt_cls.reshape(B, cond_num, -1)
                merged_cls = merged_cls.reshape(B * V, -1)
            else:
                # If no samples_cls, use reference GT CLS for all views (expand/tile)
                # (B, cond_num, C) -> (B, 1, C) -> (B, V, C) -> (B*V, C)
                merged_cls = ref_gt_cls.reshape(B, cond_num, -1)[:, :1].expand(-1, V, -1).reshape(B * V, -1)
            
            current_level = rae.level
            
            # Normalize current_level to negative notation
            current_level_neg = current_level
            if current_level >= 0:
                current_level_neg = current_level - 4  # 0->-4, 1->-3, 2->-2, 3->-1

            use_gt_for_upward = (validation_mode == "propagation_gt")
            use_zero_for_upward = (validation_mode == "propagation_0")
            
            if current_level == -4 or current_level == 0 or (not use_gt_for_upward and not use_zero_for_upward):
                print("[propagation] Standard downward propagation")
                # NOTE: merged_shallow_flat is DA3 Norm - propagate_features() handles denorm internally
                # Returns (Raw CLS, Normed Patches)
                propagated_feats = rae.propagate_features(
                    merged_shallow_flat,  # DA3 Norm input
                    from_level=rae.level,
                    total_view=total_view,
                    cls_token=merged_cls,  # Merged DA3 Norm
                )
            else:
                # Combined Upward (GT or Zeros) + DA3 (downward) propagation
                # E.g., Level -3: GT generates -4, DA3 generates -2, -1
                
                # Step 4a: Get upward features (current -> ... -> -4)
                upward_feats = {}
                
                if use_gt_for_upward:
                    print(f"[propagation_gt] Using GT features for upward propagation (levels < {current_level_neg})")
                    # Use GT features for levels shallower than current level (e.g. if -3, use -4 from GT)
                    gt_feats_neg = _normalize_gt_feats_to_neg_levels(gt_feats_all)
                    
                    for lvl in [-4, -3, -2, -1]:
                        if lvl < current_level_neg:
                            # (B*V, N+1, C) Raw
                            feat_raw = gt_feats_neg[lvl]
                            if feat_raw.ndim != 3:
                                raise ValueError(
                                    f"[propagation_gt] Expected GT feature tensor to be 3D (B*V, N+1, C) at level {lvl}, "
                                    f"got shape={tuple(feat_raw.shape)}"
                                )
                            _BV, _N_plus_1, _C = feat_raw.shape
                            patches_raw = feat_raw[:, 1:, :]  # (B*V, N, C)
                            expected_tokens = int(h_lat) * int(w_lat)
                            if int(patches_raw.shape[1]) != expected_tokens:
                                raise ValueError(
                                    f"[propagation_gt] Token grid mismatch at level {lvl}: "
                                    f"expected N={expected_tokens} (h_lat={h_lat}, w_lat={w_lat}) but got N={int(patches_raw.shape[1])}. "
                                    "Refusing to reshape with an implicit fallback."
                                )
                            upward_feats[lvl] = patches_raw.transpose(1, 2).reshape(_BV, _C, int(h_lat), int(w_lat))

                elif use_zero_for_upward:
                    print(f"[propagation_0] Using Zero features for upward propagation (levels < {current_level_neg})")
                    # upward_feats remains empty, loop below will handle missing features with zeros
                
                # Step 4b: Get downward features using DA3 propagate (current -> ... -> -1)
                # This returns list starting from current level
                # propogate_features -> _forward_from_layer returns (Raw CLS, Normed Patches)
                downward_feats = rae.propagate_features(
                    merged_shallow_flat,  # DA3 Norm input
                    from_level=rae.level,
                    total_view=total_view,
                    cls_token=merged_cls  # DA3 Norm CLS token (merged: ref=GT, tgt=diffusion)
                )
                
                # Step 4c: Combine into full 4-level list for DPT decoder
                # Order must be: [-4, -3, -2, -1] (or Layer 5, 7, 9, 11)
                propagated_feats = []
                
                for level in [-4, -3, -2, -1]:
                    if level < current_level_neg:
                        if level in upward_feats:
                            upward_feat = upward_feats[level]  # (B*V, C, H, W) Raw Patches
                            
                            # Convert to DPT format (B, V, N, C)
                            _B_V = upward_feat.shape[0]
                            _B = _B_V // total_view
                            _V = total_view
                            _C, _H, _W = upward_feat.shape[1:]
                            
                            patches_4d_raw = upward_feat.reshape(_B, _V, _C, _H * _W).transpose(-1, -2) # (B, V, N, C) Raw
                            
                            # Apply LayerNorm to Patches
                            patches_4d_norm = apply_da3_norm(rae, patches_4d_raw)
                            
                            # Use GT CLS (Raw)
                            gt_feat = gt_feats_neg[level]  # (B*V, N+1, C) Raw
                            gt_feat_5d = gt_feat.reshape(B, V, *gt_feat.shape[1:])
                            cls_3d_raw = gt_feat_5d[:, :, 0, :]   # (B, V, C) Raw
                            
                            propagated_feats.append((patches_4d_norm, cls_3d_raw))
                        else:
                            # Feature missing
                            if len(downward_feats) > 0:
                                ref_p, ref_c = downward_feats[0]
                                propagated_feats.append((torch.zeros_like(ref_p), torch.zeros_like(ref_c)))
                            else:
                                # Fallback
                                _feat_dim = rae.latent_dim if hasattr(rae, 'latent_dim') else 1536
                                _n_patches = rae.num_patches if hasattr(rae, 'num_patches') else 1369
                                propagated_feats.append((torch.zeros(B, total_view, _n_patches, _feat_dim, device=images.device),
                                                            torch.zeros(B, total_view, _feat_dim, device=images.device)))
                    
                    else:
                        # Level is CURRENT or ABOVE -> come from DA3 downward
                        # downward_feats[0] corresponds to current_level_neg
                        # downward_feats are already (Raw CLS, Normed Patches)
                        idx = level - current_level_neg
                        if idx < len(downward_feats):
                            propagated_feats.append(downward_feats[idx])
                        else:
                            ref_p, ref_c = downward_feats[0]
                            propagated_feats.append((torch.zeros_like(ref_p), torch.zeros_like(ref_c)))
            
            # --- CRITICAL FIX: Force Replace Reference Views with GT ---
            # To ensure consistency and perfect Reference visualization, we overwrite Reference parts with GT for ALL levels.
            # EXCEPTION: In concat mode, we use the model's full prediction to maintain consistency.
            
            if not is_concat_mode:
                final_propagated_feats = []
                for i, (p, c) in enumerate(propagated_feats):
                    # p: (B, V, N, C) Normed Patches
                    # c: (B, V, C) or None Raw CLS
                    level_idx = i # 0->Level-4, 1->Level-3 ...
                    
                    # Get GT feature for this level
                    if level_idx in gt_feats_all:
                        gt_feat = gt_feats_all[level_idx] # (B*V, N+1, C) Raw
                        
                        # Reshape GT to match p, c
                        gt_feat_5d = gt_feat.reshape(B, V, *gt_feat.shape[1:])
                        gt_cls_raw = gt_feat_5d[:, :, 0, :]      # (B, V, C) Raw
                        gt_patches_raw = gt_feat_5d[:, :, 1:, :] # (B, V, N, C) Raw
                        
                        # Apply LayerNorm to GT Patches
                        gt_patches_norm = apply_da3_norm(rae, gt_patches_raw)
                        
                        # Overwrite Reference Views (0...cond_num)
                        p_new = p.clone()
                        
                        if c is not None:
                            c_new = c.clone()
                        else:
                            c_new = gt_cls_raw.clone() 
                            
                        p_new[:, :cond_num] = gt_patches_norm[:, :cond_num] # Patches: Normed <- Normed
                        if c_new is not None:
                            c_new[:, :cond_num] = gt_cls_raw[:, :cond_num] # CLS: Raw <- Raw
                            
                        final_propagated_feats.append((p_new, c_new))
                    else:
                        final_propagated_feats.append((p, c))
                        
                propagated_feats = final_propagated_feats
            
            # Ensure dtype matches decoder
            if rae.rae_cl_decoder is not None:
                decoder_dtype = next(rae.rae_cl_decoder.parameters()).dtype
            else:
                # Fallback to model dtype if decoder is missing (though validation will likely fail later)
                decoder_dtype = next(model.parameters()).dtype
            propagated_feats_casted = []
            for p, c in propagated_feats:
                p = p.to(dtype=decoder_dtype)
                if c is not None:
                    c = c.to(dtype=decoder_dtype)
                propagated_feats_casted.append((p, c))
            
            res = rae.decode(propagated_feats_casted, H, W)
            decoded_images = res['rgb']
                
        # 4. Visualization / Logging
        # decoded_images: Expect (B*V, 3, H, W) usually, or None (e.g. VGGT has no RGB decoder)
        has_rgb = decoded_images is not None
        if has_rgb and decoded_images.ndim == 4:
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
        elif has_rgb and decoded_images.ndim == 5:
            pass # Already (B, V, C, H, W)
        elif has_rgb:
            raise ValueError(f"Unexpected decoded_images shape: {decoded_images.shape}")

        # Save results if output_dir is active
        if output_dir is not None:
             from torchvision.utils import save_image
             os.makedirs(output_dir, exist_ok=True)

        # GT images are already [0,1] from video_load.py (no ImageNet normalization)
        # decoded_images are denormalized in rae.decode()
        images_denorm = images  # Already [0, 1]
        decoded_images_denorm = decoded_images  # Already denormalized in rae.decode()

        # Clamp
        images_denorm = torch.clamp(images_denorm, 0, 1)
        if has_rgb:
            decoded_images_denorm = torch.clamp(decoded_images_denorm, 0, 1)

        # Save each view vs GT ?
        # Save grid for each sample in batch
        if output_dir is not None and has_rgb:
             # Create visualizations subdirectory
             vis_dir = f"{output_dir}/visualizations"
             os.makedirs(vis_dir, exist_ok=True)

             for bi in range(B):
                 # Construct grid: Top row GT, Bottom row Gen
                 gt_views = images_denorm[bi] # (V, 3, H, W)
                 gen_views = decoded_images_denorm[bi] # (V, 3, H, W)

                 # Concat V views horizontally
                 gt_row = torch.cat([v for v in gt_views], dim=2) # (3, H, W*V)
                 gen_row = torch.cat([v for v in gen_views], dim=2)

                 # Concat GT and Gen vertically
                 grid = torch.cat([gt_row, gen_row], dim=1) # (3, H*2, W*V)

                 # Filename: include rank to avoid collisions in multi-node/GPU training
                 save_path = f"{output_dir}/sample_rank{rank}_{saved_sample_count:05d}.png"
                 save_image(grid.float().cpu(), save_path)
                 saved_sample_count += 1

        # Compute RGB Metrics (skip when decoded_images is None, e.g. VGGT)
        if has_rgb:
            # Flatten -> (B*V, 3, H, W)
            from utils.metrics import compute_psnr, compute_ssim, compute_lpips

            gt_targets = images_denorm[:, cond_num:].reshape(-1, 3, H, W)
            gen_targets = decoded_images_denorm[:, cond_num:].reshape(-1, 3, H, W)

            if gt_targets.numel() > 0:
                 psnr = compute_psnr(gt_targets, gen_targets)
                 ssim = compute_ssim(gt_targets, gen_targets)
                 lpips_val = compute_lpips(gt_targets, gen_targets)
                 mse_val = ((gt_targets - gen_targets) ** 2).mean(dim=[1, 2, 3]) # (B,)

                 # Accumulate for global average (Target)
                 num_samples = float(gt_targets.shape[0])
                 psnr_tgt_sum += psnr.sum()
                 ssim_tgt_sum += ssim.sum()
                 lpips_tgt_sum += lpips_val.sum()
                 mse_tgt_sum += mse_val.sum()
                 metrics_tgt_count += num_samples

                 if "psnr" not in metrics_acc: metrics_acc["psnr"] = []
                 if "ssim" not in metrics_acc: metrics_acc["ssim"] = []
                 if "lpips" not in metrics_acc: metrics_acc["lpips"] = []

                 metrics_acc["psnr"].append(psnr.mean().item())
                 metrics_acc["ssim"].append(ssim.mean().item())
                 metrics_acc["lpips"].append(lpips_val.mean().item())

            # Compute Reference Metrics (For completeness and debugging)
            gt_refs = images_denorm[:, :cond_num].reshape(-1, 3, H, W)
            gen_refs = decoded_images_denorm[:, :cond_num].reshape(-1, 3, H, W)

            if gt_refs.numel() > 0:
                 psnr_ref = compute_psnr(gt_refs, gen_refs)
                 ssim_ref = compute_ssim(gt_refs, gen_refs)
                 lpips_ref = compute_lpips(gt_refs, gen_refs)
                 mse_ref = ((gt_refs - gen_refs) ** 2).mean(dim=[1, 2, 3])

                 psnr_ref_sum += psnr_ref.sum()
                 ssim_ref_sum += ssim_ref.sum()
                 lpips_ref_sum += lpips_ref.sum()
                 mse_ref_sum += mse_ref.sum()
                 metrics_ref_count += float(gt_refs.shape[0])

        # Depth Evaluation (Pseudo-GT fallback when batch['depth'] unavailable)
        depth_metrics = {}
        if 'depth' in res and res['depth'] is not None:
            depth_pred = res['depth']

            if depth_pred.ndim == 3:
                H_d, W_d = depth_pred.shape[-2:]
                depth_pred_bv = depth_pred.reshape(B, V, H_d, W_d)
            elif depth_pred.ndim == 4 and depth_pred.shape[1] == 1:
                depth_pred = depth_pred.squeeze(1)
                H_d, W_d = depth_pred.shape[-2:]
                depth_pred_bv = depth_pred.reshape(B, V, H_d, W_d)
            elif depth_pred.ndim == 4:
                H_d, W_d = depth_pred.shape[-2:]
                depth_pred_bv = depth_pred
            else:
                depth_pred_bv = None

            gt_depth_bv = None

            if 'depth' in batch:
                gt_depth = batch['depth'].to(device)
                if gt_depth.ndim == 5:
                    gt_depth = gt_depth.squeeze(2)
                gt_depth_bv = gt_depth.reshape(B, V, H_d, W_d)
            else:
                if rae.rae_cl_decoder is not None and validation_mode in ["propagation", "propagation_gt", "propagation_0"]:
                    try:
                        with torch.no_grad():
                            gt_dpt_feats = []
                            gt_feats_neg = _normalize_gt_feats_to_neg_levels(gt_feats_all)

                            for level_idx in range(4):
                                neg_level = level_idx - 4
                                feat_raw = gt_feats_neg[neg_level]

                                _BV, _N_plus_1, _C = feat_raw.shape
                                feat_5d = feat_raw.reshape(B, V, _N_plus_1, _C)

                                cls_3d_raw = feat_5d[:, :, 0, :]
                                patches_4d_raw = feat_5d[:, :, 1:, :]

                                patches_4d_norm = apply_da3_norm(rae, patches_4d_raw)

                                gt_dpt_feats.append((patches_4d_norm, cls_3d_raw))

                            decoder_dtype = next(rae.rae_cl_decoder.parameters()).dtype
                            gt_dpt_feats_casted = []
                            for p, c in gt_dpt_feats:
                                p = p.to(dtype=decoder_dtype)
                                if c is not None:
                                    c = c.to(dtype=decoder_dtype)
                                gt_dpt_feats_casted.append((p, c))

                            gt_res = rae.decode(gt_dpt_feats_casted, H, W)
                            if 'depth' in gt_res and gt_res['depth'] is not None:
                                gt_depth_pseudo = gt_res['depth']

                                if gt_depth_pseudo.ndim == 3:
                                    gt_depth_bv = gt_depth_pseudo.reshape(B, V, H_d, W_d)
                                elif gt_depth_pseudo.ndim == 4 and gt_depth_pseudo.shape[1] == 1:
                                    gt_depth_bv = gt_depth_pseudo.squeeze(1).reshape(B, V, H_d, W_d)
                                elif gt_depth_pseudo.ndim == 4:
                                    gt_depth_bv = gt_depth_pseudo

                                if i == 0 and rank == 0:
                                    print(f"[Depth Eval] Using Pseudo-GT depth from DPT decoder (no batch['depth'])")
                    except Exception as e:
                        if i == 0 and rank == 0:
                            print(f"[Depth Eval] Failed to generate Pseudo-GT depth: {e}")
                        gt_depth_bv = None

            if depth_pred_bv is not None and gt_depth_bv is not None:
                depth_pred_tgt = depth_pred_bv[:, cond_num:]
                gt_depth_tgt = gt_depth_bv[:, cond_num:]

                from utils.metrics import compute_abs_rel, compute_depth_rmse, compute_delta

                pred_depth_flat = depth_pred_tgt.reshape(-1, H_d, W_d)
                gt_depth_flat = gt_depth_tgt.reshape(-1, H_d, W_d)

                if pred_depth_flat.numel() > 0:
                    abs_rel = compute_abs_rel(pred_depth_flat, gt_depth_flat)
                    rmse = compute_depth_rmse(pred_depth_flat, gt_depth_flat)
                    delta1 = compute_delta(pred_depth_flat, gt_depth_flat, threshold=1.25)

                    if "abs_rel" not in metrics_acc: metrics_acc["abs_rel"] = []
                    if "rmse" not in metrics_acc: metrics_acc["rmse"] = []
                    if "delta1" not in metrics_acc: metrics_acc["delta1"] = []

                    metrics_acc["abs_rel"].append(abs_rel.mean().item())
                    metrics_acc["rmse"].append(rmse.mean().item())
                    metrics_acc["delta1"].append(delta1.mean().item())

                    depth_metrics['abs_rel'] = abs_rel
                    depth_metrics['rmse'] = rmse
                    depth_metrics['delta1'] = delta1

        # Per-sample metrics (for each sample in batch)
        num_target_views = total_view - cond_num
        for bi in range(B):
            sample_start = bi * num_target_views
            sample_end = sample_start + num_target_views

            sample_dict = {
                'rank': rank,
                'sample_idx': saved_sample_count - B + bi,
            }

            # Add RGB metrics if available
            if has_rgb and 'psnr' in dir() and psnr is not None:
                sample_dict['psnr'] = psnr[sample_start:sample_end].mean().item() if sample_end <= len(psnr) else psnr.mean().item()
                sample_dict['ssim'] = ssim[sample_start:sample_end].mean().item() if sample_end <= len(ssim) else ssim.mean().item()
                sample_dict['lpips'] = lpips_val[sample_start:sample_end].mean().item() if sample_end <= len(lpips_val) else lpips_val.mean().item()

            # Add depth metrics if available
            if depth_metrics:
                for k, v in depth_metrics.items():
                    if sample_end <= len(v):
                        sample_val = v[sample_start:sample_end].mean().item()
                    else:
                        sample_val = v.mean().item()
                    sample_dict[k] = sample_val

            per_sample_metrics.append(sample_dict)
                     
                
        # Pick first batch for visualization return (wandb) - use denormalized
        # Or accumulate a few samples?
        num_logging_views = 8
        if has_rgb and len(all_images) < num_logging_views:
            for bi in range(min(B, num_logging_views)):
                 src_views = images_denorm[bi] # (V, C, H, W)
                 tgt_views = decoded_images_denorm[bi] # (V, C, H, W)
                 vis = make_vis_multiview(src_views.cpu(), tgt_views.cpu().float(), cond_num=int(cond_num))
                 all_images.append(vis)
                 if len(all_images) >= num_logging_views: break
        
            

    # Aggregate all metrics across ranks
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(val_loss_sum, op=dist.ReduceOp.SUM)
        # Split Loss aggregation
        dist.all_reduce(val_ref_loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_tgt_loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_loss_count, op=dist.ReduceOp.SUM)
        
        # Split Metrics aggregation
        dist.all_reduce(psnr_tgt_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(ssim_tgt_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(lpips_tgt_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(mse_tgt_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(metrics_tgt_count, op=dist.ReduceOp.SUM)
        
        dist.all_reduce(psnr_ref_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(ssim_ref_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(lpips_ref_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(mse_ref_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(metrics_ref_count, op=dist.ReduceOp.SUM)

        # Gather all per_sample_metrics to rank 0 for results.json
        # Must be called on all ranks to avoid hang
        all_gathered = [None] * dist.get_world_size()
        dist.all_gather_object(all_gathered, per_sample_metrics)
        if rank == 0:
            per_sample_metrics = [item for sublist in all_gathered for item in sublist]
            total_saved_sample_count = len(per_sample_metrics)
        else:
            total_saved_sample_count = saved_sample_count 
    else:
        total_saved_sample_count = saved_sample_count

    # Return stats
    stats = {
        "val/images": all_images if all_images else None
    }
    
    # Calculate global averages (Target)
    if metrics_tgt_count.item() > 0:
        stats["val/psnr"] = (psnr_tgt_sum / metrics_tgt_count).item()      # Default (Target)
        stats["val/ssim"] = (ssim_tgt_sum / metrics_tgt_count).item()      # Default (Target)
        stats["val/lpips"] = (lpips_tgt_sum / metrics_tgt_count).item()    # Default (Target)
        stats["val/mse_tgt"] = (mse_tgt_sum / metrics_tgt_count).item()
        
        # Explicit Target keys
        stats["val/psnr_tgt"] = stats["val/psnr"]
        stats["val/ssim_tgt"] = stats["val/ssim"]
        stats["val/lpips_tgt"] = stats["val/lpips"]
        
        # Add depth averages if available
        if "abs_rel" in metrics_acc and len(metrics_acc["abs_rel"]) > 0:
             stats["val/abs_rel"] = sum(metrics_acc["abs_rel"]) / len(metrics_acc["abs_rel"])
        if "rmse" in metrics_acc and len(metrics_acc["rmse"]) > 0:
             stats["val/rmse"] = sum(metrics_acc["rmse"]) / len(metrics_acc["rmse"])
        if "delta1" in metrics_acc and len(metrics_acc["delta1"]) > 0:
             stats["val/delta1"] = sum(metrics_acc["delta1"]) / len(metrics_acc["delta1"])

    # Calculate global averages (Reference)
    if metrics_ref_count.item() > 0:
        stats["val/psnr_ref"] = (psnr_ref_sum / metrics_ref_count).item()
        stats["val/ssim_ref"] = (ssim_ref_sum / metrics_ref_count).item()
        stats["val/lpips_ref"] = (lpips_ref_sum / metrics_ref_count).item()
        stats["val/mse_ref"] = (mse_ref_sum / metrics_ref_count).item()

    if compute_loss:
        if val_loss_count.item() > 0:
            stats["val/loss"] = (val_loss_sum / val_loss_count).item()
            stats["val/loss_ref"] = (val_ref_loss_sum / val_loss_count).item()
            stats["val/loss_tgt"] = (val_tgt_loss_sum / val_loss_count).item()
        else:
            print("[Warning] Validation loss requested but no samples processed.")
    
    # Save results to JSON if output_dir provided (Rank 0 only)
    if output_dir is not None and rank == 0:
        os.makedirs(output_dir, exist_ok=True)
        import json
        from datetime import datetime
        
        # Build results dict
        results = {
            'timestamp': datetime.now().isoformat(),
            'run_config': run_config or {},
            'settings': {
                'validation_mode': validation_mode,
                'use_prope': use_prope,
                'pag_scale': pag_scale,
                'pag_layer_idx': pag_layer_idx,
                'cfg_scale': cfg_scale,
                'total_view': total_view,
                'cond_num': cond_num,
                'num_batches': val_num_batches,
                'rae_level': rae.level if hasattr(rae, 'level') else None,
                'world_size': world_size,
            },
            'summary': {
                'total_samples': total_saved_sample_count,
                'psnr_mean': stats.get('val/psnr'),
                'ssim_mean': stats.get('val/ssim'),
                'lpips_mean': stats.get('val/lpips'),
                'loss_mean': stats.get('val/loss'),
            },
            'per_sample_metrics': per_sample_metrics
        }
        
        # Save JSON
        if output_dir is not None and rank == 0:
            os.makedirs(output_dir, exist_ok=True)
            results_path = f"{output_dir}/results.json"
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)
        print(f"Results saved to {results_path}")
        psnr_val = results['summary']['psnr_mean']
        loss_val = results['summary']['loss_mean']
        psnr_str = f"{psnr_val:.4f}" if psnr_val is not None else "N/A"
        loss_str = f"{loss_val:.4f}" if loss_val is not None else "N/A"
        print(f"Summary (Global) - PSNR: {psnr_str}, Loss: {loss_str}")

    if dist.is_available() and dist.is_initialized():
        dist.barrier()
            
    return stats
