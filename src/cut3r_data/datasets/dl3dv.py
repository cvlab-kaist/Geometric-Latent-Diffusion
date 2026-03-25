# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only)
# Modified for GLD project - DL3DV Multi-View Dataset

import os
import os.path as osp
import cv2
import numpy as np
from tqdm import tqdm

from cut3r_data.base.base_multiview_dataset import BaseMultiViewDataset
from cut3r_data.utils.image import imread_cv2


class DL3DV_Multi(BaseMultiViewDataset):
    def __init__(self, *args, split, ROOT, **kwargs):
        self.ROOT = ROOT
        self.video = True
        self.min_interval = kwargs.get('min_interval', 1)
        self.max_interval = kwargs.get('max_interval', 20)
        self.is_metric = False
        super().__init__(*args, **kwargs)
        self.loaded_data = self._load_data()

    def _load_data(self):
        level_dirs = sorted(
            d for d in os.listdir(self.ROOT)
            if os.path.isdir(osp.join(self.ROOT, d))
        )
        
        subscenes = []
        for level in level_dirs:
            level_dir = osp.join(self.ROOT, level)
            subscenes.extend([
                osp.join(level, scene)
                for scene in os.listdir(level_dir)
                if os.path.isdir(osp.join(level_dir, scene))
            ])

        offset = 0
        scenes = []
        sceneids = []
        images = []
        scene_img_list = []
        start_img_ids = []
        j = 0

        for scene in tqdm(subscenes, desc="Loading DL3DV"):
            scene_dir = osp.join(self.ROOT, scene)
            
            # Support both directory structures:
            # 1. DL3DV-10K (training): scene_hash/images_4/
            # 2. DL3DV-Evaluation: scene_hash/nerfstudio/images_4/
            img_dir = osp.join(scene_dir, "images_4")
            nerfstudio_dir = osp.join(scene_dir, "nerfstudio")
            
            if osp.isdir(nerfstudio_dir):
                # Use nerfstudio subdirectory (DL3DV-Evaluation structure)
                scene_dir = nerfstudio_dir
                img_dir = osp.join(scene_dir, "images_4")

            if not osp.isdir(img_dir):
                continue
            transforms_path = osp.join(scene_dir, "transforms.json")
            if not osp.isfile(transforms_path):
                print(f"Skipping {scene}: missing transforms.json")
                continue
            rgb_paths = sorted(f for f in os.listdir(img_dir) if f.endswith(".png"))

            if len(rgb_paths) == 0:
                continue

            num_imgs = len(rgb_paths)
            # Use minimum cut_off of 2 - adaptive sampling handles the rest
            cut_off = max(2, self.num_views // 3) if not self.allow_repeat else max(2, self.num_views // 3)

            if num_imgs < cut_off:
                print(f"Skipping {scene} dl3dv (only {num_imgs} frames)")
                continue

            img_ids = list(np.arange(num_imgs) + offset)
            start_img_ids_ = img_ids[: num_imgs - cut_off + 1]

            # Store the actual scene directory (relative to ROOT), including nerfstudio/ if used
            # This ensures _get_views can find files correctly for both structures
            actual_scene_dir = osp.relpath(scene_dir, self.ROOT)
            scenes.append(actual_scene_dir)
            scene_img_list.append(img_ids)
            sceneids.extend([j] * num_imgs)
            images.extend([osp.join("images_4", f) for f in rgb_paths])
            start_img_ids.extend(start_img_ids_)
            offset += num_imgs
            j += 1

        self.scenes = scenes
        self.sceneids = sceneids
        self.images = images
        self.start_img_ids = start_img_ids
        self.scene_img_list = scene_img_list

    def __len__(self):
        return len(self.start_img_ids)

    def get_image_num(self):
        return len(self.images)

    def _get_views(self, idx, resolution, rng, num_views):
        max_retries = 100
        retry_count = 0
        
        while retry_count < max_retries:
            retry_count += 1
            start_id = self.start_img_ids[idx]
            scene_id = self.sceneids[start_id]
            all_image_ids = self.scene_img_list[scene_id]
            pos = self.get_seq_from_start_id_novid(
                num_views,
                start_id,
                all_image_ids,
                rng,
                min_interval=self.min_interval,
                max_interval=self.max_interval,
                block_shuffle=25,
            )
            
            # Skip if not enough frames
            if pos is None:
                idx = rng.integers(low=0, high=len(self.start_img_ids))
                continue
            
            image_idxs = np.array(all_image_ids)[pos]

            views = []
            try:
                for view_idx in image_idxs:
                    scene_id = self.sceneids[view_idx]
                    scene_dir = osp.join(self.ROOT, self.scenes[scene_id])
                    rgb_path = self.images[view_idx]
                    basename = rgb_path[:-4]
                    
                    # Check if npz file exists
                    npz_path = osp.join(scene_dir, basename + ".npz")
                    if not osp.exists(npz_path):
                        raise FileNotFoundError(f"Missing camera file: {npz_path}")
                    
                    rgb_image = imread_cv2(osp.join(scene_dir, rgb_path), cv2.IMREAD_COLOR)
                    cam_file = np.load(npz_path)
                    intrinsics = cam_file["intrinsic"].astype(np.float32)
                    camera_pose = cam_file["pose"].astype(np.float32)
                    # Expects OpenCV c2w (X-right, Y-down, Z-forward).
                    # If your data uses OpenGL convention, convert beforehand.

                    depthmap = np.zeros_like(rgb_image)
                    rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                        rgb_image, depthmap, intrinsics, resolution, rng=rng, info=view_idx
                    )

                    views.append(dict(
                        img=rgb_image,
                        camera_pose=camera_pose.astype(np.float32),
                        camera_intrinsics=intrinsics.astype(np.float32),
                    ))
                
                if len(views) == num_views:
                    return views
            except FileNotFoundError as e:
                # Skip this sample and try another
                idx = rng.integers(low=0, high=len(self.start_img_ids))
                continue
        
        raise RuntimeError(f"Could not find scene with enough frames for {num_views} views")
