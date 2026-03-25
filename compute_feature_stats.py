
import torch
import argparse
from tqdm import tqdm
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from utils.train_utils import parse_configs
from utils.model_utils import instantiate_from_config
from train_multiview_da3 import prepare_dataloader

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/training/DA3_level1.yaml")
    parser.add_argument("--level", type=int, default=-4, help="DA3 Level to extract stats for")
    parser.add_argument("--save-path", type=str, default="model_stats/da3/normalization_stats.pt", help="Output path for .pt file")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for generic iteration")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--max-batches", type=int, default=None, help="Limit number of batches for quick test")
    
    # Data paths - Defaults to same as training script
    parser.add_argument("--video-path", type=str, default='data/re10k/training_256')
    parser.add_argument("--pose-path", type=str, default='data/re10k/training_poses')
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Load Config & Model
    print(f"Parsing config: {args.config}")
    # We only need stage_1 config actually, but parse_configs returns tuple
    rae_config, _, _, _, _, _, _, _ = parse_configs(args.config)
    
    # Override level
    print(f"Overriding RAE level to: {args.level}")
    rae_config.params.level = args.level
    
    print("Instantiating RAE Model...")
    # Instantiate
    # Note: RAE_DA3 should have 'normalization_stat_path' as None or similar to avoid loading bad stats
    if 'normalization_stat_path' in rae_config.params:
        rae_config.params.normalization_stat_path = None
        
    rae = instantiate_from_config(rae_config).to(device).to(torch.bfloat16)
    rae.eval()
    
    # Ensure RAE does NOT normalize internally while we compute raw stats
    rae.do_normalization = False
    
    print("Model loaded in bfloat16 for faster inference")
    
    # 2. Prepare DataLoader
    image_size = 518 # DA3 Fixed Size
    print(f"Preparing DataLoader with Image Size {image_size}...")
    loader, _ = prepare_dataloader(
        args.video_path, args.pose_path, image_size, 
        args.batch_size, args.num_workers, 
        rank=0, world_size=1, shuffle=False # No shuffle needed for stats
    )
    
    # 3. Compute Stats for ALL/Multiple Levels
    # DA3 levels: 0, 1, 2, 3 (Mapped to layers 5, 7, 9, 11)
    target_levels = [0, 1, 2, 3]
    # Corresponding negative levels for logging if needed: -4, -3, -2, -1
    
    print(f"Computing statistics for levels: {target_levels} (Sequential Passthrough)")
    
    count = 0
    # Dictionaries to store accumulators for each level
    # Key: level index (0, 1, 2, 3)
    accumulators = {
        lvl: {'sum_x': None, 'sum_sq_x': None} 
        for lvl in target_levels
    }
    
    total_steps = len(loader)
    if args.max_batches:
        total_steps = min(total_steps, args.max_batches)
        
    pbar = tqdm(enumerate(loader), total=total_steps)
    
    with torch.no_grad():
        for i, batch in pbar:
            if args.max_batches is not None and i >= args.max_batches:
                break
                
            # images: (B, V, 3, H, W)
            images = batch['gt_inp'].to(device, dtype=torch.bfloat16)
            current_batch_n = 0
            
            # Extract ALL levels in a single forward pass (efficient!)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                all_feats = rae.encoder.forward(images, mode='all')
            
            # all_feats: Dict[level_idx -> (B*V, N, 1536)]
            for level, feats in all_feats.items():
                if level not in target_levels:
                    continue
                
                # Flatten Logic
                # feats: (B*V, N, C) -> (Total_Pixels, C)
                if isinstance(feats, (tuple, list)):
                     feats = feats[0]

                # Check shape
                if feats.dim() == 4:
                     # (B, C, H, W)
                     c = feats.shape[1]
                     feats_flat = feats.permute(0, 2, 3, 1).reshape(-1, c)
                elif feats.dim() == 3:
                     # (B, N, C)
                     c = feats.shape[-1]
                     feats_flat = feats.reshape(-1, c)
                else:
                    print(f"Unexpected shape {feats.shape} for level {level}. Skipping.")
                    continue
                
                # Double precision accum
                feats_flat = feats_flat.double()
                
                if current_batch_n == 0:
                    current_batch_n = feats_flat.shape[0]
                
                # Initialize accumulators if first time
                if accumulators[level]['sum_x'] is None:
                    feature_dim = feats_flat.shape[1]
                    accumulators[level]['sum_x'] = torch.zeros(feature_dim, dtype=torch.double, device=device)
                    accumulators[level]['sum_sq_x'] = torch.zeros(feature_dim, dtype=torch.double, device=device)
                    if i == 0: 
                        print(f"[Level {level}] Detected Feature Dim (C): {feature_dim}")
                
                accumulators[level]['sum_x'] += feats_flat.sum(dim=0)
                accumulators[level]['sum_sq_x'] += (feats_flat ** 2).sum(dim=0)

            count += current_batch_n
            
            if i % 20 == 0 and count > 0:
                # Log mean of level 3 (deepest) just for progress
                deep_sum = accumulators[3]['sum_x'] # Level 3
                if deep_sum is not None:
                    curr_mean = deep_sum / count
                    pbar.set_postfix({
                        "Mean(L3)": f"{curr_mean.mean().item():.5f}"
                    })

    for level in target_levels:
        sum_x = accumulators[level]['sum_x']
        sum_sq_x = accumulators[level]['sum_sq_x']
        
        if sum_x is None: 
            print(f"Warning: No data for Level {level}")
            continue

        # Final Calculation
        mean = sum_x / count
        var = (sum_sq_x / count) - (mean ** 2)
        
        # Handle numerical instability
        var = torch.clamp(var, min=0.0)
        std = torch.sqrt(var)
        
        # Convert back to float
        mean = mean.float()
        var = var.float()
        std = std.float()
        
        # Reshape to (1, C, 1, 1) ALWAYS
        mean_save = mean.view(1, -1, 1, 1)
        var_save = var.view(1, -1, 1, 1)
        std_save = std.view(1, -1, 1, 1)
            
        stats = {
            'mean': mean_save,
            'var': var_save,
            'std': std_save, 
            'count': count,
            'level': level
        }
        
        # Determine filename
        # e.g., normalization_stats_level0.pt
        base_name = os.path.splitext(args.save_path)[0]
        # Remove any existing suffix if present in base_name/args.save_path logic, 
        # but here the user usually passes 'normalization_stats.pt'
        # We want 'normalization_stats_levelX.pt'
        
        # Just use directory of save_path + fixed filename structure
        save_dir = os.path.dirname(args.save_path)
        out_path = os.path.join(save_dir, f"normalization_stats_level{level}.pt")
        
        print(f"\n[Level {level}] Correction completed.")
        print(f"Count: {count}")
        print(f"Mean Range: {mean.min().item():.4f} ~ {mean.max().item():.4f}")
        print(f"Std Range: {std.min().item():.4f} ~ {std.max().item():.4f}")
        print(f"Saving statistics to: {out_path}")
        
        torch.save(stats, out_path)
    
    print("\nAll saved successfully.")

if __name__ == "__main__":
    main()
