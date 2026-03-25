# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only)
# Modified for GLD project - RealEstate10K Multi-View Dataset

import os
import os.path as osp
import cv2
import numpy as np
from tqdm import tqdm

from cut3r_data.base.base_multiview_dataset import BaseMultiViewDataset
from cut3r_data.utils.image import imread_cv2


class RE10K_Multi(BaseMultiViewDataset):
    def __init__(self, *args, ROOT, **kwargs):
        self.ROOT = ROOT
        self.video = True
        self.is_metric = False
        self.min_interval = kwargs.get('min_interval', 1)
        self.max_interval = kwargs.get('max_interval', 128)
        super().__init__(*args, **kwargs)
        self.loaded_data = self._load_data()

    def _load_data(self):
        self.scenes = [
            d for d in os.listdir(self.ROOT)
            if osp.isdir(osp.join(self.ROOT, d))
        ]

        offset = 0
        scenes = []
        sceneids = []
        scene_img_list = []
        images = []
        start_img_ids = []

        j = 0
        for scene in tqdm(self.scenes, desc="Loading RE10K"):
            scene_dir = osp.join(self.ROOT, scene)
            basenames = sorted(
                [f[:-4] for f in os.listdir(scene_dir) if f.endswith(".png")],
                key=lambda x: int(x),
            )

            num_imgs = len(basenames)
            if num_imgs == 0:
                continue

            img_ids = list(np.arange(num_imgs) + offset)
            # Use minimum cut_off of 2 - adaptive sampling handles the rest
            # This allows scenes with fewer frames than num_views to still be used
            cut_off = max(2, self.num_views // 3) if not self.allow_repeat else max(2, self.num_views // 3)

            if num_imgs < cut_off:
                print(f"Skipping {scene} re10k (only {num_imgs} frames, need at least {cut_off})")
                continue

            start_img_ids_ = img_ids[: num_imgs - cut_off + 1]
            start_img_ids.extend([(scene, id_) for id_ in start_img_ids_])
            sceneids.extend([j] * num_imgs)
            images.extend(basenames)
            scenes.append(scene)
            scene_img_list.append(img_ids)

            offset += num_imgs
            j += 1

        self.scenes = scenes
        self.sceneids = sceneids
        self.images = images
        self.start_img_ids = start_img_ids
        self.scene_img_list = scene_img_list
        self.invalid_scenes = {scene: False for scene in self.scenes}

    def __len__(self):
        return len(self.start_img_ids)

    def get_image_num(self):
        return len(self.images)

    def _get_views(self, idx, resolution, rng, num_views):
        invalid_seq = True
        scene, start_id = self.start_img_ids[idx]
        max_retries = 100  # Prevent infinite loop

        retry_count = 0
        while invalid_seq and retry_count < max_retries:
            retry_count += 1
            
            while self.invalid_scenes[scene]:
                idx = rng.integers(low=0, high=len(self.start_img_ids))
                scene, start_id = self.start_img_ids[idx]

            all_image_ids = self.scene_img_list[self.sceneids[start_id]]
            pos = self.get_seq_from_start_id_novid(
                num_views, start_id, all_image_ids, rng, 
                min_interval=self.min_interval,
                max_interval=self.max_interval
            )
            
            # Skip scene if not enough frames
            if pos is None:
                idx = rng.integers(low=0, high=len(self.start_img_ids))
                scene, start_id = self.start_img_ids[idx]
                continue
            
            image_idxs = np.array(all_image_ids)[pos]

            views = []
            for view_idx in image_idxs:
                scene_id = self.sceneids[view_idx]
                scene_dir = osp.join(self.ROOT, self.scenes[scene_id])
                basename = self.images[view_idx]
                
                try:
                    rgb_image = imread_cv2(osp.join(scene_dir, basename + ".png"))
                    cam = np.load(osp.join(scene_dir, basename + "_cam.npz"))
                    intrinsics = cam["intrinsics"]
                    camera_pose = cam["pose"]
                except Exception as e:
                    raise ValueError(f"Error loading {scene} {basename}: {e}")
                
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
                invalid_seq = False
        
        if retry_count >= max_retries:
            raise RuntimeError(f"Could not find scene with enough frames for {num_views} views after {max_retries} retries")
        return views
