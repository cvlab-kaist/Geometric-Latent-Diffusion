"""
CUT3R DataLoader Adapter for gld pipeline.

Converts CUT3R batch format (List[Dict]) to gld format (Dict).
Handles:
1. View reordering based on ref_view_sampling
2. ImageNet denormalization to [0, 1]
3. Intrinsics format conversion (3x3) -> [fx, fy, cx, cy]
"""
import torch
from typing import List, Dict, Any, Optional
import hashlib


def convert_cut3r_batch(
    cut3r_batch: List[Dict],
    cond_num: int,
    ref_view_sampling: str = "prefix"
) -> Dict[str, Any]:
    """
    Convert CUT3R batch format to gld format.
    
    CUT3R returns: batch = [view0_dict, view1_dict, ...]
    gld expects: batch = {'gt_inp': (B,V,C,H,W), 'fxfycxcy': (B,V,4), ...}
    
    Args:
        cut3r_batch: List of view dicts from CUT3R DataLoader.
            Each dict contains:
                - 'img': (B, C, H, W) - ImageNet normalized
                - 'camera_pose': (B, 4, 4) - c2w matrix
                - 'camera_intrinsics': (B, 3, 3) - intrinsic matrix
                - 'idx': tuple - (sample_idx, ar_idx, view_idx)
        cond_num: Number of reference (conditioning) views.
        ref_view_sampling: How to select reference views.
            - "prefix": First cond_num views are references.
            - "interpolate": First and last views are references (cond_num=2).
            - "random": Randomly select cond_num views as references.
    
    Returns:
        Dict with keys:
            - 'gt_inp': (B, V, C, H, W) in [0, 1] range
            - 'fxfycxcy': (B, V, 4) intrinsics [fx, fy, cx, cy]
            - 'c2w': (B, V, 4, 4) camera-to-world extrinsics
            - 'video_id': str identifier
            - 'frame_indices': (B, V) frame indices
    """
    V = len(cut3r_batch)
    B = cut3r_batch[0]['img'].shape[0]
    
    # 1. Determine view order based on ref_view_sampling
    order = _get_view_order(V, cond_num, ref_view_sampling, cut3r_batch)
    
    # Reorder views
    reordered = [cut3r_batch[i] for i in order]
    
    # 2. Stack images: List[(B,C,H,W)] → (B,V,C,H,W)
    imgs = torch.stack([v['img'] for v in reordered], dim=1)  # (B,V,C,H,W)
    
    # 3. Denormalize ImageNet → [0,1]
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1).to(imgs)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1).to(imgs)
    gt_inp = imgs * std + mean
    gt_inp = gt_inp.clamp(0, 1)
    
    # 4. Convert intrinsics (B,3,3) → (B,4) for each view, then stack
    fxfycxcy_list = []
    for v in reordered:
        K = v['camera_intrinsics']  # (B, 3, 3)
        fx = K[:, 0, 0]
        fy = K[:, 1, 1]
        cx = K[:, 0, 2]
        cy = K[:, 1, 2]
        fxfycxcy_list.append(torch.stack([fx, fy, cx, cy], dim=-1))  # (B, 4)
    fxfycxcy = torch.stack(fxfycxcy_list, dim=1)  # (B, V, 4)
    
    # 5. Stack poses (camera_pose is c2w)
    c2w = torch.stack([v['camera_pose'] for v in reordered], dim=1)  # (B, V, 4, 4)
    
    # 6. Frame indices 
    frame_indices = torch.tensor([order], dtype=torch.long).expand(B, -1)
    
    # 7. Preserve CUT3R provenance (critical for debugging / reproducibility)
    # Each view dict contains 'idx' = (sample_idx, ar_idx, view_idx) as tensors.
    cut3r_idx = [v.get('idx', None) for v in reordered]
    
    return {
        'gt_inp': gt_inp,           # (B, V, C, H, W) in [0,1]
        'fxfycxcy': fxfycxcy,       # (B, V, 4)
        'c2w': c2w,                 # (B, V, 4, 4)
        'video_id': 'cut3r_batch',
        'frame_indices': frame_indices,
        'cut3r_idx': cut3r_idx,     # List[V] of CUT3R (sample_idx, ar_idx, view_idx)
    }


def _get_view_order(
    num_views: int,
    cond_num: int,
    ref_view_sampling: str,
    batch: Optional[List[Dict]] = None
) -> List[int]:
    """
    Determine view reordering so that first cond_num views are references.
    
    Args:
        num_views: Total number of views.
        cond_num: Number of reference views.
        ref_view_sampling: Sampling strategy.
        batch: Original batch (used for deterministic hashing in 'random' mode).
    
    Returns:
        List of indices representing the new view order.
    """
    V = num_views
    
    if ref_view_sampling == "prefix":
        # First cond_num views are references (no reordering needed)
        return list(range(V))
    
    elif ref_view_sampling == "interpolate":
        # First and last views are references
        if cond_num != 2:
            raise ValueError(
                f"ref_view_sampling='interpolate' requires cond_num=2, got {cond_num}"
            )
        # Order: [first, last, middle views...]
        return [0, V - 1] + list(range(1, V - 1))
    
    elif ref_view_sampling == "random":
        # Randomly select reference views (deterministic based on batch)
        if cond_num < 1:
            raise ValueError(
                f"ref_view_sampling='random' requires cond_num>=1, got {cond_num}"
            )
        
        # Create deterministic seed from batch info
        if batch is not None and 'idx' in batch[0]:
            batch_hash = hashlib.md5(str(batch[0]['idx']).encode()).hexdigest()
            seed = int(batch_hash[:8], 16)
        else:
            seed = 0
        
        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(V, generator=g).tolist()
        ref_pos = sorted(perm[:cond_num])
        tgt_pos = [i for i in range(V) if i not in set(ref_pos)]
        return ref_pos + tgt_pos
    
    else:
        raise ValueError(f"Unknown ref_view_sampling: {ref_view_sampling}")


def is_cut3r_batch(batch: Any) -> bool:
    """
    Check if the batch is from CUT3R DataLoader.
    
    CUT3R returns List[Dict], gld returns Dict.
    """
    return isinstance(batch, list) and len(batch) > 0 and isinstance(batch[0], dict)
