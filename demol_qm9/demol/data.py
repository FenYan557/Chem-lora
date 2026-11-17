"""DataLoader plumbing on top of :mod:`demol.qm9`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import torch

from .losses import TargetStandardizer
from .qm9 import QM9Dataset, qm9_collate

__all__ = [
    "DataBundle",
    "build_datasets",
    "build_loaders",
    "cycling_loader",
    "fit_standardizer",
]


@dataclass
class DataBundle:
    """Datasets plus their loaders for the three official QM9 splits."""

    train: torch.utils.data.DataLoader
    val: torch.utils.data.DataLoader
    test: torch.utils.data.DataLoader
    datasets: dict[str, QM9Dataset]


def build_datasets(cache_dir: str) -> dict[str, QM9Dataset]:
    """Train / validation / test datasets using the paper's 10k / 10,831 split."""
    return {
        "train": QM9Dataset(cache_dir, split_ids=(0,)),
        "val": QM9Dataset(cache_dir, split_ids=(1,)),
        "test": QM9Dataset(cache_dir, split_ids=(2,)),
    }


def fit_standardizer(dataset: QM9Dataset) -> TargetStandardizer:
    """Label standardisation statistics from the *training* split only."""
    targets = np.asarray(dataset.arrays["targets"])[dataset.rows]
    return TargetStandardizer.fit(targets)


def build_loaders(
    cache_dir: str,
    batch_size: int,
    num_workers: int = 0,
    shuffle_train: bool = True,
    seed: int = 42,
    rank: int = 0,
    world_size: int = 1,
) -> DataBundle:
    datasets = build_datasets(cache_dir)
    common = dict(collate_fn=qm9_collate, num_workers=num_workers)
    if num_workers > 0:
        common.update(persistent_workers=True, pin_memory=True)
    sampler = None
    generator = None
    if world_size > 1:
        # DDP：每个 rank 一个不相交的分片；DistributedSampler 只做索引运算，
        # 不依赖进程组，单进程下也能用来测试分片是否正确。
        from torch.utils.data.distributed import DistributedSampler

        sampler = DistributedSampler(
            datasets["train"],
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle_train,
            seed=seed,
            drop_last=True,
        )
    elif shuffle_train:
        generator = torch.Generator()
        generator.manual_seed(seed)
    train = torch.utils.data.DataLoader(
        datasets["train"],
        batch_size=batch_size,
        shuffle=shuffle_train and sampler is None,
        sampler=sampler,
        drop_last=True,
        generator=generator,
        **common,
    )
    val = torch.utils.data.DataLoader(
        datasets["val"], batch_size=batch_size, shuffle=False, **common
    )
    test = torch.utils.data.DataLoader(
        datasets["test"], batch_size=batch_size, shuffle=False, **common
    )
    return DataBundle(train=train, val=val, test=test, datasets=datasets)


def cycling_loader(loader: torch.utils.data.DataLoader) -> Iterator[dict[str, torch.Tensor]]:
    """Infinite iterator over ``loader`` so training is driven by step count.

    With a ``DistributedSampler`` each pass calls ``set_epoch`` so every rank
    re-shuffles its shard differently on every sweep through the dataset.
    """
    sampler = getattr(loader, "sampler", None)
    epoch = 0
    while True:
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        for batch in loader:
            yield batch
        epoch += 1
