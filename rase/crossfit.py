from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass
class CrossFitConfig:
    num_folds: int = 3
    fold_iql_steps: int = 100000
    fold_fqe_steps: int = 100000
    train_fqe: bool = False
    heldout_eval_states: int = 2048
    candidate_ms: Iterable[int] = (1, 4, 16, 64, 256)


def make_kfold_indices(n: int, k: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    if k < 2:
        raise ValueError("num_folds must be at least 2")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    folds = np.array_split(perm, k)
    out = []
    for i in range(k):
        valid = np.asarray(folds[i], dtype=np.int64)
        train = np.concatenate([folds[j] for j in range(k) if j != i]).astype(np.int64)
        out.append((train, valid))
    return out
