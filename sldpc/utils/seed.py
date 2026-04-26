"""Deterministic seeding utilities.

All randomness in SLDPC training flows through :func:`set_seed` to ensure
the 5-seed results reported in the paper are reproducible bit-for-bit
given the same hardware and PyTorch build.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


__all__ = ["set_seed", "seed_worker"]


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and PyTorch RNGs.

    Parameters
    ----------
    seed : int
        Master seed. Applied to :mod:`random`, :mod:`numpy.random`,
        :mod:`torch`, and ``torch.cuda``.
    deterministic : bool, optional
        If ``True`` (default), also set :mod:`torch.backends.cudnn` to
        deterministic mode and disable benchmark autotuning. This makes
        runs reproducible at a small throughput cost.

    Notes
    -----
    - ``PYTHONHASHSEED`` is set so that Python string hashing is also
      deterministic across process restarts.
    - Full bitwise determinism across different GPU models is not
      guaranteed even with ``deterministic=True``; only runs on the
      same hardware with the same PyTorch build should match exactly.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    """PyTorch ``DataLoader`` ``worker_init_fn`` for deterministic shuffling.

    Pair this with a seeded ``torch.Generator`` passed to the
    ``DataLoader`` to fully pin down data-iteration order across
    multi-worker loading.
    """
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)
