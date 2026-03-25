# CUT3R Data Package for GLD
# Self-contained dataset utilities, adapted from CUT3R/dust3r
# No external dependencies on CUT3R repository

import torch
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from accelerate import Accelerator

# Base classes
from cut3r_data.base import (
    EasyDataset,
    MulDataset,
    ResizedDataset,
    CatDataset,
    BatchedRandomSampler,
    CustomRandomSampler,
    BaseMultiViewDataset,
)

# Dataset implementations
from cut3r_data.datasets import DL3DV_Multi, HyperSim_Multi, RE10K_Multi, TartanAir_Multi


def get_data_loader(
    dataset,
    batch_size,
    num_workers=8,
    shuffle=True,
    drop_last=True,
    pin_mem=True,
    world_size: int = 1,
    rank: int = 0,
    fixed_length=False,
):
    """Create a data loader for the given dataset.
    
    Args:
        dataset: Dataset or string representation
        batch_size: Batch size
        num_workers: Number of workers for data loading
        shuffle: Whether to shuffle data
        drop_last: Whether to drop last incomplete batch
        pin_mem: Whether to pin memory
        world_size: Number of distributed processes (DDP world size). Must be explicit.
        rank: Rank of the current process in [0, world_size). Must be explicit.
        fixed_length: If True, use fixed view count (for test mode)
        
    Returns:
        DataLoader instance
    """
    if isinstance(dataset, str):
        dataset = eval(dataset)

    try:
        sampler = dataset.make_sampler(
            batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            world_size=world_size,
            rank=rank,
            fixed_length=fixed_length
        )
        shuffle = False

        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_mem,
        )

    except (AttributeError, NotImplementedError):
        sampler = None

        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_mem,
            drop_last=drop_last,
        )
    return data_loader


__all__ = [
    # Base classes
    "EasyDataset",
    "MulDataset", 
    "ResizedDataset",
    "CatDataset",
    "BatchedRandomSampler",
    "CustomRandomSampler",
    "BaseMultiViewDataset",
    # Datasets
    "DL3DV_Multi",
    "HyperSim_Multi",
    "RE10K_Multi",
    "TartanAir_Multi",
    # Functions
    "get_data_loader",
]
