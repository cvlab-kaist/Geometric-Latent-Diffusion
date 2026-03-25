# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only)
# Modified for GLD project - BaseMultiViewDataset

import PIL
import numpy as np
import torch
import random
import itertools
from cut3r_data.base.easy_dataset import EasyDataset
from cut3r_data.utils.transforms import ImgNorm, SeqColorJitter
from cut3r_data.utils.geometry import depthmap_to_absolute_camera_coordinates
import cut3r_data.utils.cropping as cropping


class BaseMultiViewDataset(EasyDataset):
    """Base class for multi-view datasets.

    Usage:
        class MyDataset(BaseMultiViewDataset):
            def _get_views(self, idx, resolution, rng, num_views):
                views = []
                views.append(dict(img=..., camera_pose=..., camera_intrinsics=...))
                return views
    """

    def __init__(
        self,
        *,
        num_views=None,
        split=None,
        resolution=None,
        transform=ImgNorm,
        aug_crop=False,
        n_corres=0,
        nneg=0,
        seed=None,
        allow_repeat=False,
        seq_aug_crop=False,
        # ---------------------------------------------------------------------
        # Pose validity filtering (explicit; never silent)
        # ---------------------------------------------------------------------
        skip_bad_poses: bool = False,
        max_translation_norm: float = None,
        max_pose_retries: int = None,
        log_bad_pose_every: int = None,
    ):
        assert num_views is not None, "undefined num_views"

        self.num_views = num_views
        self.split = split
        self._set_resolutions(resolution)

        self.n_corres = n_corres
        self.nneg = nneg

        self.is_seq_color_jitter = False
        if isinstance(transform, str):
            transform = eval(transform)
        if transform == SeqColorJitter:
            transform = SeqColorJitter()
            self.is_seq_color_jitter = True
        self.transform = transform

        self.aug_crop = aug_crop
        self.seed = seed
        self.allow_repeat = allow_repeat
        self.seq_aug_crop = seq_aug_crop
        
        # Pose filtering configuration (must be explicit if enabled)
        if skip_bad_poses:
            if max_translation_norm is None:
                raise ValueError(
                    "skip_bad_poses=true requires explicit max_translation_norm (no implicit default)."
                )
            if max_pose_retries is None:
                raise ValueError(
                    "skip_bad_poses=true requires explicit max_pose_retries (no implicit default)."
                )
            if log_bad_pose_every is None:
                raise ValueError(
                    "skip_bad_poses=true requires explicit log_bad_pose_every (no implicit default)."
                )
        self.skip_bad_poses = bool(skip_bad_poses)
        self.max_translation_norm = max_translation_norm if max_translation_norm is None else float(max_translation_norm)
        self.max_pose_retries = max_pose_retries if max_pose_retries is None else int(max_pose_retries)
        self.log_bad_pose_every = log_bad_pose_every if log_bad_pose_every is None else int(log_bad_pose_every)
        self._bad_pose_skip_count = 0

    def __len__(self):
        return len(self.scenes)

    @staticmethod
    def blockwise_shuffle(x, rng, block_shuffle):
        if block_shuffle is None:
            return rng.permutation(x).tolist()
        else:
            assert block_shuffle > 0
            blocks = [x[i : i + block_shuffle] for i in range(0, len(x), block_shuffle)]
            shuffled_blocks = [rng.permutation(block).tolist() for block in blocks]
            shuffled_list = [item for block in shuffled_blocks for item in block]
            return shuffled_list

    def get_seq_from_start_id_novid(
        self,
        num_views,
        id_ref,
        ids_all,
        rng,
        min_interval=1,
        max_interval=25,
        video_prob=0.5,
        fix_interval_prob=0.5,
        block_shuffle=None,
    ):
        """Get sequence of view positions from start id (no video mode).
        
        Adaptive behavior:
        - If enough frames exist with default interval: use random intervals
        - If not enough frames: adaptively compute max interval for even sampling
        """
        assert min_interval > 0
        assert min_interval <= max_interval
        assert id_ref in ids_all
        pos_ref = ids_all.index(id_ref)
        
        # Total available frames from reference position
        total_available = len(ids_all) - pos_ref
        
        # Case 1: Enough frames for requested views
        if total_available >= num_views:
            remaining_sum = total_available - 1
            
            if remaining_sum == num_views - 1:
                # Exact match - use consecutive frames
                return [pos_ref + i for i in range(num_views)]
            
            # Compute adaptive max_interval based on available frames
            # This ensures we can always get num_views frames
            adaptive_max = remaining_sum // (num_views - 1)
            max_interval = min(max_interval, adaptive_max)
            max_interval = max(max_interval, min_interval)  # Ensure valid range
            
            intervals = [
                rng.choice(range(min_interval, max_interval + 1))
                for _ in range(num_views - 1)
            ]

            pos = list(itertools.accumulate([pos_ref] + intervals))
            pos = [p for p in pos if p < len(ids_all)]
            
            # Fill missing positions from available candidates
            if len(pos) < num_views:
                all_possible_pos = list(range(pos_ref, len(ids_all)))
                pos_candidates = [p for p in all_possible_pos if p not in pos]
                need_more = num_views - len(pos)
                if len(pos_candidates) >= need_more:
                    pos = pos + rng.choice(pos_candidates, need_more, replace=False).tolist()
                else:
                    # Fallback: even spacing when not enough candidates
                    pos = self._evenly_spaced_positions(pos_ref, len(ids_all) - 1, num_views)
            
            pos = self.blockwise_shuffle(pos, rng, block_shuffle)
        
        # Case 2: Not enough frames - return None to signal skip
        else:
            return None
        
        assert len(pos) == num_views, f"Expected {num_views} positions, got {len(pos)}"
        return pos
    
    def _evenly_spaced_positions(self, start: int, end: int, num_views: int) -> list:
        """Generate evenly spaced positions.
        
        Args:
            start: Start index (inclusive)
            end: End index (inclusive)
            num_views: Number of positions to generate
            
        Returns:
            List of positions, evenly spaced. Returns None if not enough frames.
        """
        available = end - start + 1
        
        if available >= num_views:
            # Enough frames - evenly space without repeat
            step = (available - 1) / (num_views - 1) if num_views > 1 else 0
            positions = [start + int(round(i * step)) for i in range(num_views)]
            return positions
        else:
            # Not enough frames - return None to signal skip
            return None

    def get_img_and_ray_masks(self, is_metric, v, rng, p=[0.8, 0.15, 0.05]):
        if v == 0 or (not is_metric):
            return True, False
        else:
            rand_val = rng.random()
            if rand_val < p[0]:
                return True, False
            elif rand_val < p[0] + p[1]:
                return False, True
            else:
                return True, True

    def get_stats(self):
        return f"{len(self)} groups of views"

    def __repr__(self):
        resolutions_str = "[" + ";".join(f"{w}x{h}" for w, h in self._resolutions) + "]"
        return f"{type(self).__name__}({self.get_stats()}, num_views={self.num_views}, split={self.split}, resolutions={resolutions_str})"

    def _get_views(self, idx, resolution, rng, num_views):
        raise NotImplementedError()

    def __getitem__(self, idx):
        if isinstance(idx, (tuple, list, np.ndarray)):
            idx, ar_idx, nview = idx
        else:
            ar_idx = 0
            nview = self.num_views

        assert nview >= 1 and nview <= self.num_views
        
        if self.seed:
            self._rng = np.random.default_rng(seed=self.seed + idx)
        elif not hasattr(self, "_rng"):
            seed = torch.randint(0, 2**32, (1,)).item()
            self._rng = np.random.default_rng(seed=seed)

        if self.aug_crop > 1 and self.seq_aug_crop:
            self.delta_target_resolution = self._rng.integers(0, self.aug_crop)

        resolution = self._resolutions[ar_idx]
        
        # If pose filtering is enabled, retry sampling when an invalid pose is detected.
        # This is NOT silent: we keep a counter and log periodically, and we cap retries.
        retries = 0
        while True:
            views = self._get_views(idx, resolution, self._rng, nview)
            assert len(views) == nview
            
            # Validate camera poses early (before returning to training).
            # We check translation magnitude only; rotation validity is covered in offline audit.
            if self.max_translation_norm is None:
                break
            
            ok = True
            for v_i, view in enumerate(views):
                pose = view.get("camera_pose", None)
                if pose is None:
                    ok = False
                    break
                pose = np.asarray(pose, dtype=np.float64)
                if pose.shape != (4, 4) or (not np.isfinite(pose).all()):
                    ok = False
                    break
                t = pose[:3, 3]
                t_norm = float(np.linalg.norm(t))
                if not np.isfinite(t_norm) or t_norm > float(self.max_translation_norm):
                    ok = False
                    break
            
            if ok:
                break
            
            # invalid pose
            if not self.skip_bad_poses:
                raise ValueError(
                    f"Invalid pose detected (and skip_bad_poses=false). "
                    f"idx={idx}, ar_idx={ar_idx}, nview={nview}, resolution={resolution}, "
                    f"max_translation_norm={self.max_translation_norm}"
                )
            
            self._bad_pose_skip_count += 1
            retries += 1
            if self.log_bad_pose_every > 0 and (self._bad_pose_skip_count % self.log_bad_pose_every == 0):
                print(
                    f"[CUT3R PoseFilter] skipped_bad_pose={self._bad_pose_skip_count} "
                    f"(latest idx={idx}, ar_idx={ar_idx}, nview={nview}, res={resolution}, "
                    f"max_translation_norm={self.max_translation_norm})"
                )
            
            if retries >= int(self.max_pose_retries):
                raise RuntimeError(
                    f"Exceeded max_pose_retries={self.max_pose_retries} while trying to avoid bad poses. "
                    f"Last idx={idx}, ar_idx={ar_idx}, nview={nview}, res={resolution}. "
                    "Fix dataset or tighten offline filtering."
                )
            
            # Resample a different index using RNG (explicitly controlled by dataset rng/seed).
            idx = int(self._rng.integers(low=0, high=len(self)))

        if "camera_pose" not in views[0]:
            views[0]["camera_pose"] = np.ones((4, 4), dtype=np.float32)
        transform = SeqColorJitter() if self.is_seq_color_jitter else self.transform

        for v, view in enumerate(views):
            view["idx"] = (idx, ar_idx, v)
            width, height = view["img"].size
            view["true_shape"] = np.int32((height, width))
            view["img"] = transform(view["img"])

            assert "camera_intrinsics" in view
            if "camera_pose" not in view:
                view["camera_pose"] = np.full((4, 4), np.nan, dtype=np.float32)

            for key, val in view.items():
                res, _ = is_good_type(key, val)
                assert res

        for view in views:
            view["rng"] = int.from_bytes(self._rng.bytes(4), "big")
        return views

    def _set_resolutions(self, resolutions):
        assert resolutions is not None, "undefined resolution"

        if not isinstance(resolutions, list):
            resolutions = [resolutions]

        self._resolutions = []
        for resolution in resolutions:
            if isinstance(resolution, int):
                width = height = resolution
            else:
                width, height = resolution
            self._resolutions.append((width, height))

    def _crop_resize_if_necessary(self, image, depthmap, intrinsics, resolution, rng=None, info=None):
        """Crop and resize image with proper intrinsic adjustment."""
        if not isinstance(image, PIL.Image.Image):
            image = PIL.Image.fromarray(image)
        
        W, H = image.size
        cx, cy = intrinsics[:2, 2].round().astype(int)
        min_margin_x = min(cx, W - cx)
        min_margin_y = min(cy, H - cy)
        assert min_margin_x > W / 5, f"Bad principal point in view={info}"
        assert min_margin_y > H / 5

        l, t = cx - min_margin_x, cy - min_margin_y
        r, b = cx + min_margin_x, cy + min_margin_y
        crop_bbox = (l, t, r, b)
        image, depthmap, intrinsics = cropping.crop_image_depthmap(image, depthmap, intrinsics, crop_bbox)

        W, H = image.size

        target_resolution = np.array(resolution)
        if self.aug_crop > 1:
            target_resolution += (
                rng.integers(0, self.aug_crop)
                if not self.seq_aug_crop
                else self.delta_target_resolution
            )
        
        image, depthmap, intrinsics = cropping.rescale_image_depthmap(
            image, depthmap, intrinsics, target_resolution
        )

        intrinsics2 = cropping.camera_matrix_of_crop(
            intrinsics, image.size, resolution, offset_factor=0.5
        )
        crop_bbox = cropping.bbox_from_intrinsics_in_out(intrinsics, intrinsics2, resolution)
        image, depthmap, intrinsics2 = cropping.crop_image_depthmap(
            image, depthmap, intrinsics, crop_bbox
        )

        return image, depthmap, intrinsics2


def is_good_type(key, v):
    """Check if value type is acceptable."""
    if isinstance(v, (str, int, tuple)):
        return True, None
    if hasattr(v, 'dtype') and v.dtype not in (np.float32, torch.float32, bool, np.int32, np.int64, np.uint8):
        return False, f"bad {v.dtype=}"
    return True, None
