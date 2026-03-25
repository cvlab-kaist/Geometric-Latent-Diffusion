"""
CUT3R-style NVS Dataset Adapter for GLD Evaluation.

This module provides dataset wrappers that load data from CUT3R-style datasets
(DL3DV, HyperSim, RE10K) and convert to Matrix3D evaluation format.

NO video-based datasets - only CUT3R format.
"""

import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Any, Optional, Union, Tuple


class CUT3RNVSDataset(Dataset):
    """
    Wrapper for CUT3R-style datasets (DL3DV, HyperSim, RE10K).
    
    Converts CUT3R batch format to Matrix3D format for NVS evaluation.
    
    CUT3R format (per sample):
        - List[Dict] with V views, each containing:
            - 'img': (C, H, W) ImageNet normalized
            - 'camera_pose': (4, 4) c2w
            - 'camera_intrinsics': (3, 3) K matrix
            
    Output format (Matrix3D):
        - 'image': (V, 3, H, W) in [0,1] float32
        - 'cond_image': (V, 3, H, W) in [-1,1] float32
        - 'extrinsic': (V, 3, 4) w2c
        - 'intrinsic': (V, 3, 3) K matrix
        - 'scene_id': str
        - 'view_id': (V,)
    """
    
    SUPPORTED_DATASETS = ['cut3r_dl3dv', 'cut3r_hypersim', 'cut3r_re10k', 'cut3r_eth3d', 'cut3r_mipnerf360']
    
    def __init__(
        self,
        dataset_name: str,
        root: str,
        num_views: int = 4,
        cond_num: int = 2,
        image_size: Union[int, List[int], List[Tuple[int, int]]] = 512,
        ref_view_sampling: str = "prefix",
        mode: str = "test",
        seed: int = 42,
    ):
        """
        Args:
            dataset_name: One of 'cut3r_dl3dv', 'cut3r_hypersim', 'cut3r_re10k'
            root: Path to dataset root directory
            num_views: Total number of views per sample
            cond_num: Number of conditioning (reference) views
            image_size: Target resolution(s). Can be int, [H, W], or list of resolutions
            ref_view_sampling: How to select reference views ("prefix", "random", "interpolate")
            mode: "train" or "test"
            seed: Random seed for deterministic view selection in test mode
        """
        from cut3r_data import DL3DV_Multi, HyperSim_Multi, RE10K_Multi

        dataset_name = dataset_name.lower()
        if dataset_name not in self.SUPPORTED_DATASETS:
            raise ValueError(f"Unknown dataset: {dataset_name}. Supported: {self.SUPPORTED_DATASETS}")
        
        # Parse image_size to resolution list
        # Handle OmegaConf ListConfig from YAML configs
        from omegaconf import OmegaConf
        if OmegaConf.is_config(image_size):
            image_size = OmegaConf.to_container(image_size, resolve=True)
        
        if isinstance(image_size, int):
            resolution = [(image_size, image_size)]
        elif isinstance(image_size, list) and len(image_size) == 2 and isinstance(image_size[0], int):
            resolution = [tuple(image_size)]
        elif isinstance(image_size, list) and all(isinstance(r, (list, tuple)) for r in image_size):
            resolution = [tuple(r) for r in image_size]
        else:
            resolution = [(512, 512)]
        
        # Use fixed seed for test mode determinism
        test_seed = seed if mode == 'test' else None
        
        # Create the appropriate CUT3R dataset
        if dataset_name == 'cut3r_dl3dv' or dataset_name == 'cut3r_mipnerf360':
            self.cut3r_dataset = DL3DV_Multi(
                ROOT=root,
                resolution=resolution,
                num_views=num_views,
                split='train' if mode == 'train' else None,
                allow_repeat=False,
                seed=test_seed,
            )
        elif dataset_name == 'cut3r_hypersim':
            self.cut3r_dataset = HyperSim_Multi(
                ROOT=root,
                resolution=resolution,
                num_views=num_views,
                split='train' if mode == 'train' else None,
                allow_repeat=False,
                seed=test_seed,
            )
        elif dataset_name == 'cut3r_re10k':
            self.cut3r_dataset = RE10K_Multi(
                ROOT=root,
                resolution=resolution,
                num_views=num_views,
                split=None,  # RE10K doesn't use split
                allow_repeat=False,
                seed=test_seed,
            )
        elif dataset_name == 'cut3r_eth3d':
            from cut3r_data.datasets.eth3d import ETH3D_Multi
            self.cut3r_dataset = ETH3D_Multi(
                ROOT=root,
                resolution=resolution,
                num_views=num_views,
                split=None,
                allow_repeat=False,
                seed=test_seed,
            )

        self.dataset_name = dataset_name
        self.num_views = num_views
        self.cond_num = cond_num
        self.ref_view_sampling = ref_view_sampling
        self.mode = mode
        
        # ImageNet normalization constants
        self.imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.imagenet_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        
    def __len__(self):
        return len(self.cut3r_dataset)
    
    def _get_view_order(self, num_views: int, cond_num: int, idx: int) -> List[int]:
        """Determine view order based on ref_view_sampling strategy."""
        V = num_views
        
        if self.ref_view_sampling == "prefix":
            return list(range(V))
        
        elif self.ref_view_sampling == "interpolate":
            # Interpolation: uniformly distributed condition views
            # Select cond_num views uniformly across all V views
            if cond_num >= V:
                raise ValueError(f"cond_num ({cond_num}) must be < num_views ({V})")
            
            # Uniform sampling: select indices evenly spaced
            step = (V - 1) / (cond_num - 1) if cond_num > 1 else 0
            cond_indices = [int(round(i * step)) for i in range(cond_num)]
            
            # Remaining indices are targets
            tgt_indices = [i for i in range(V) if i not in cond_indices]
            
            # Return: condition views first, then target views
            # This ensures model receives condition views in first cond_num positions
            return cond_indices + tgt_indices
        
        elif self.ref_view_sampling == "random":
            # Deterministic random based on sample index
            import hashlib
            h = hashlib.md5(f"{idx}_{self.dataset_name}".encode()).hexdigest()
            seed = int(h[:8], 16)
            g = torch.Generator().manual_seed(seed)
            perm = torch.randperm(V, generator=g).tolist()
            ref_pos = sorted(perm[:cond_num])
            tgt_pos = [i for i in range(V) if i not in set(ref_pos)]
            return ref_pos + tgt_pos
        
        else:
            raise ValueError(f"Unknown ref_view_sampling: {self.ref_view_sampling}")
    
    def _denormalize_imagenet(self, img: torch.Tensor) -> torch.Tensor:
        """Convert from ImageNet normalization to [0, 1] range."""
        return (img * self.imagenet_std.to(img.device) + self.imagenet_mean.to(img.device)).clamp(0, 1)
    
    def __getitem__(self, idx):
        # Get CUT3R sample (List[Dict], one dict per view)
        views = self.cut3r_dataset[idx]
        
        V = len(views)
        
        # Reorder views based on ref_view_sampling
        order = self._get_view_order(V, self.cond_num, idx)
        views = [views[i] for i in order]
        
        # Stack images
        imgs = torch.stack([torch.from_numpy(v['img']) if isinstance(v['img'], np.ndarray) 
                           else v['img'] for v in views])  # (V, C, H, W)
        
        # Denormalize from ImageNet to [0, 1]
        images = self._denormalize_imagenet(imgs.float())
        
        # Stack camera poses (c2w)
        c2w = torch.stack([torch.from_numpy(v['camera_pose']) if isinstance(v['camera_pose'], np.ndarray)
                          else v['camera_pose'] for v in views])  # (V, 4, 4)

        # OpenGL→OpenCV flip is now handled inside CUT3R dataset (dl3dv.py)

        # Stack intrinsics
        K = torch.stack([torch.from_numpy(v['camera_intrinsics']) if isinstance(v['camera_intrinsics'], np.ndarray)
                        else v['camera_intrinsics'] for v in views])  # (V, 3, 3)

        # Convert c2w to w2c for Matrix3D
        w2c = torch.linalg.inv(c2w.float())
        extrinsics = w2c[:, :3, :]  # (V, 3, 4)

        # cond_image in [-1, 1]
        cond_image = images * 2.0 - 1.0

        result = {
            'image': images.float(),                # (V, 3, H, W) in [0,1]
            'cond_image': cond_image.float(),       # (V, 3, H, W) in [-1,1]
            'extrinsic': extrinsics.float(),        # (V, 3, 4) w2c
            'intrinsic': K.float(),                 # (V, 3, 3)
            'c2w': c2w.float(),                     # (V, 4, 4)
            'scene_id': f"{self.dataset_name}_{idx}",
            'view_id': torch.arange(V),
            'cond_num': self.cond_num,
        }

        # Include ground-truth depth if the dataset provides it (e.g. ETH3D)
        if 'depthmap' in views[0]:
            depths = []
            for v in views:
                d = v['depthmap']
                if isinstance(d, np.ndarray):
                    d = torch.from_numpy(d)
                depths.append(d.float())
            result['depth'] = torch.stack(depths)  # (V, H, W), metric metres; 0 = invalid

        return result


def create_nvs_dataloader(
    dataset_name: str,
    root: str,
    num_views: int = 4,
    cond_num: int = 2,
    image_size: Union[int, List] = 512,
    batch_size: int = 1,
    num_workers: int = 4,
    mode: str = "test",
    ref_view_sampling: str = "prefix",
    seed: int = 42,
    max_samples: Optional[int] = None,
    samples_per_scene: Optional[int] = None,
    min_interval: int = 1,
    max_interval: int = 20,
) -> DataLoader:
    """
    Factory function to create NVS evaluation dataloader.
    
    Args:
        dataset_name: 'cut3r_dl3dv', 'cut3r_hypersim', or 'cut3r_re10k'
        root: Path to dataset root
        num_views: Total views per sample
        cond_num: Number of conditioning views
        image_size: Target resolution
        batch_size: Batch size
        num_workers: Dataloader workers
        mode: 'train' or 'test'
        ref_view_sampling: View selection strategy
        seed: Random seed
        max_samples: Optional limit on number of samples (takes first N)
        samples_per_scene: If set, takes N samples per scene (ensures all scenes covered)
                          This is better for evaluation. E.g., samples_per_scene=1 gives
                          exactly 1 sample per scene, covering all scenes.
        min_interval: Minimum frame spacing between views (default: 1)
        max_interval: Maximum frame spacing between views (default: 20)
        
    Returns:
        DataLoader for NVS evaluation
    """
    dataset = CUT3RNVSDataset(
        dataset_name=dataset_name,
        root=root,
        num_views=num_views,
        cond_num=cond_num,
        image_size=image_size,
        ref_view_sampling=ref_view_sampling,
        mode=mode,
        seed=seed,
    )
    
    # Apply interval settings to underlying CUT3R dataset
    if hasattr(dataset.cut3r_dataset, 'max_interval'):
        dataset.cut3r_dataset.max_interval = max_interval
    if hasattr(dataset.cut3r_dataset, 'min_interval'):
        dataset.cut3r_dataset.min_interval = min_interval
    # Per-scene sampling for evaluation (ensures scenes are covered evenly)
    if samples_per_scene is not None:
        from torch.utils.data import Subset
        
        # Get scene information from the underlying CUT3R dataset
        cut3r_ds = dataset.cut3r_dataset
        
        if hasattr(cut3r_ds, 'scenes') and hasattr(cut3r_ds, 'sceneids') and hasattr(cut3r_ds, 'start_img_ids'):
            # Group sample indices by scene
            num_scenes = len(cut3r_ds.scenes)
            scene_to_samples = {i: [] for i in range(num_scenes)}
            for sample_idx, start_id in enumerate(cut3r_ds.start_img_ids):
                if "re10k" in dataset_name:
                    start_id = start_id[-1]
                scene_id = cut3r_ds.sceneids[start_id]
                scene_to_samples[scene_id].append(sample_idx)
            
            # Determine how many scenes to sample
            # if max_samples is not None:
            #     # max_samples limits the number of scenes (not total samples)
            #     num_scenes_to_sample = min(max_samples, num_scenes)
            # else:
            num_scenes_to_sample = num_scenes
            
            # Take N samples per scene (evenly spaced) from first M scenes
            selected_indices = []
            for scene_id in range(num_scenes_to_sample):
                scene_samples = scene_to_samples[scene_id]
                if len(scene_samples) == 0:
                    continue
                
                if samples_per_scene >= len(scene_samples):
                    # Take all samples from this scene
                    selected_indices.extend(scene_samples)
                else:
                    # Take evenly spaced samples
                    step = len(scene_samples) / samples_per_scene
                    for i in range(samples_per_scene):
                        idx = int(i * step)
                        selected_indices.append(scene_samples[idx])
            
            if max_samples is not None:
                selected_indices = selected_indices[:max_samples]

            print(f"[Evaluation Mode] samples_per_scene={samples_per_scene}, max_samples={max_samples}: "
                  f"Selected {len(selected_indices)} samples from {num_scenes_to_sample} scenes "
                  f"(total {num_scenes} scenes available)")
            dataset = Subset(dataset, selected_indices)
        else:
            print(f"[Warning] Dataset doesn't have scene info, falling back to max_samples")
            if max_samples is not None and max_samples < len(dataset):
                indices = list(range(min(max_samples, len(dataset))))
                dataset = Subset(dataset, indices)
    
    # Simple first-N selection (only if samples_per_scene not used)
    elif max_samples is not None and max_samples < len(dataset):
        from torch.utils.data import Subset
        indices = list(range(min(max_samples, len(dataset))))
        dataset = Subset(dataset, indices)
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=True,
    )
    
    return loader


# Backward compatibility aliases
DA3NVSDataset = CUT3RNVSDataset


def create_nvs_dataloader_from_config(
    eval_cfg,
    difficulty: Optional[str] = None,
    max_samples: Optional[int] = None,
    batch_size: int = 1,
    num_workers: int = 4,
):
    """
    Create NVS dataloader directly from evaluation config.
    
    This is a convenience wrapper that extracts all parameters from the config,
    making it easier to use and more compatible with other code.
    
    Args:
        eval_cfg: OmegaConf config loaded from YAML
        difficulty: Difficulty preset to use (overrides default_difficulty in config)
        max_samples: Optional limit on samples (for debugging)
        batch_size: Batch size (default: 1)
        num_workers: Number of workers (default: 4)
    
    Returns:
        DataLoader configured according to eval_cfg
    
    Example:
        >>> from omegaconf import OmegaConf
        >>> eval_cfg = OmegaConf.load('configs/eval/dl3dv.yaml')
        >>> dataloader = create_nvs_dataloader_from_config(eval_cfg, difficulty='medium')
    """
    # Extract base parameters
    dataset_name = eval_cfg.dataset.name
    root = eval_cfg.dataset.root
    
    # Extract sampling parameters
    num_views = eval_cfg.sampling.num_views
    cond_num = eval_cfg.sampling.cond_num
    ref_view_sampling = eval_cfg.sampling.ref_view_sampling
    samples_per_scene = eval_cfg.sampling.get('samples_per_scene', None)
    seed = eval_cfg.sampling.get('seed', 42)
    
    # Extract image parameters
    image_size = eval_cfg.image.size
    
    # Determine difficulty and extract interval settings
    if difficulty is None:
        difficulty = eval_cfg.get('default_difficulty', 'medium')
    
    if 'difficulties' in eval_cfg and difficulty in eval_cfg.difficulties:
        difficulty_cfg = eval_cfg.difficulties[difficulty]
        min_interval = difficulty_cfg.get('min_interval', 1)
        max_interval = difficulty_cfg.get('max_interval', 20)
    else:
        # Fallback to base config or defaults
        min_interval = eval_cfg.sampling.get('min_interval', 1)
        max_interval = eval_cfg.sampling.get('max_interval', 20)
    
    # Create dataloader
    return create_nvs_dataloader(
        dataset_name=dataset_name,
        root=root,
        num_views=num_views,
        cond_num=cond_num,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
        mode="test",
        ref_view_sampling=ref_view_sampling,
        seed=seed,
        max_samples=max_samples,
        samples_per_scene=samples_per_scene,
        min_interval=min_interval,
        max_interval=max_interval,
    )
