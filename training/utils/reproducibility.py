"""Reproducible RNG seeding for training scripts."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int, *, rank: int = 0, deterministic: bool = True) -> int:
    """
    Seed Python, NumPy, and Torch RNGs.

    Uses ``seed + rank`` so distributed workers differ in dropout noise while the
    run remains a pure function of ``(seed, rank)``. Returns the effective seed.
    """
    effective = int(seed) + int(rank)
    random.seed(effective)
    np.random.seed(effective % (2**32))
    torch.manual_seed(effective)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(effective)
    os.environ["PYTHONHASHSEED"] = str(effective)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return effective


def seed_worker(worker_id: int) -> None:
    """DataLoader ``worker_init_fn`` so augmentation / shuffle in workers is seeded."""
    worker_seed = (torch.initial_seed() + int(worker_id)) % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def make_torch_generator(seed: int, *, device: str = "cpu") -> torch.Generator:
    """Generator for DataLoader ``generator=`` (shuffle reproducibility)."""
    g = torch.Generator(device=device)
    g.manual_seed(int(seed))
    return g
