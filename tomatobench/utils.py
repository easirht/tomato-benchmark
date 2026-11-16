"""Seeding, wall-clock budget and small helpers."""
from __future__ import annotations

import os
import random
import time

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed python, numpy and torch.

    deterministic=True gives (nearly) bit-exact repeatability but is slower; for the main
    benchmark we use False (faster) and rely on multiple seeds instead.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.benchmark = True


def seed_worker(worker_id: int) -> None:
    """DataLoader worker_init_fn so augmentation randomness is seeded per worker."""
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator(seed: int) -> torch.Generator:
    """A torch Generator for DataLoader shuffling. Use seed = base_seed * 1000 + epoch so a
    resumed run shuffles exactly like an uninterrupted one."""
    g = torch.Generator()
    g.manual_seed(int(seed))
    return g


class TimeBudget:
    """Stop gracefully before the session limit so the last checkpoint is always saved.

    Usage inside the training loop (after saving the epoch checkpoint):
        if budget.should_stop(est_epoch_seconds): break
    Kaggle/Colab kill the session without warning; this avoids losing a half-finished epoch.
    """

    def __init__(self, max_hours: float, safety_minutes: float = 10.0):
        self.start = time.time()
        self.max_seconds = max_hours * 3600.0
        self.safety = safety_minutes * 60.0

    def elapsed(self) -> float:
        return time.time() - self.start

    def remaining(self) -> float:
        return self.max_seconds - self.elapsed()

    def should_stop(self, est_next_epoch_seconds: float = 0.0) -> bool:
        return self.remaining() < (self.safety + est_next_epoch_seconds)


def count_params(model: torch.nn.Module, trainable_only: bool = False) -> int:
    return sum(p.numel() for p in model.parameters() if (p.requires_grad or not trainable_only))


def human_time(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600}h{(s % 3600) // 60:02d}m{s % 60:02d}s"
