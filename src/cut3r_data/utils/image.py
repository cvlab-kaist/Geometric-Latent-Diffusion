# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only)
# Modified for GLD project - minimal subset of CUT3R/dust3r utilities

import os
import numpy as np
import PIL.Image
import cv2
import torchvision.transforms as tvf

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

# Standard image normalization
ImgNorm = tvf.Compose([
    tvf.ToTensor(), 
    tvf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def imread_cv2(path, options=cv2.IMREAD_COLOR):
    """Open an image or a depthmap with opencv-python."""
    if path.endswith((".exr", "EXR")):
        options = cv2.IMREAD_ANYDEPTH
    img = cv2.imread(path, options)
    if img is None:
        raise IOError(f"Could not load image={path} with {options=}")
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img
