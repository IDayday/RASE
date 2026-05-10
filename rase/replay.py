from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np

# D4RL imports several optional domains (Flow, Kitchen, CARLA, Bullet) at import
# time. They are not needed for D4RL MuJoCo / AntMaze Phase-0 experiments.
os.environ.setdefault("D4RL_SUPPRESS_IMPORT_ERROR", "1")


@dataclass
class Normalizer:
    mean: np.ndarray
    std: np.ndarray

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    def denormalize(self, x: np.ndarray) -> np.ndarray:
        return x * self.std + self.mean


class D4RLReplayBuffer:
    def __init__(self, dataset: Dict[str, np.ndarray], normalize_obs: bool = True):
        required = ["observations", "actions", "next_observations", "rewards", "terminals"]
        missing = [k for k in required if k not in dataset]
        if missing:
            raise KeyError(f"D4RL dataset is missing keys: {missing}")

        self.raw_observations = dataset["observations"].astype(np.float32)
        self.raw_next_observations = dataset["next_observations"].astype(np.float32)
        self.actions = dataset["actions"].astype(np.float32)
        self.rewards = dataset["rewards"].astype(np.float32).reshape(-1, 1)
        self.terminals = dataset["terminals"].astype(np.float32).reshape(-1, 1)
        if "timeouts" in dataset:
            self.timeouts = dataset["timeouts"].astype(np.float32).reshape(-1, 1)
        else:
            self.timeouts = np.zeros_like(self.terminals)

        if normalize_obs:
            mean = self.raw_observations.mean(axis=0, keepdims=True)
            std = self.raw_observations.std(axis=0, keepdims=True) + 1e-6
        else:
            mean = np.zeros((1, self.raw_observations.shape[-1]), dtype=np.float32)
            std = np.ones((1, self.raw_observations.shape[-1]), dtype=np.float32)
        self.obs_normalizer = Normalizer(mean.astype(np.float32), std.astype(np.float32))
        self.observations = self.obs_normalizer.normalize(self.raw_observations).astype(np.float32)
        self.next_observations = self.obs_normalizer.normalize(self.raw_next_observations).astype(np.float32)

        self.size = self.observations.shape[0]
        self.obs_dim = self.observations.shape[-1]
        self.act_dim = self.actions.shape[-1]

    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        idx = np.random.randint(0, self.size, size=batch_size)
        return self.batch_by_indices(idx)

    def sample_indices(self, n: int) -> np.ndarray:
        # Evaluation states should be unique when possible. Sampling with
        # replacement adds avoidable Monte Carlo noise to the M sweep.
        if n <= self.size:
            return np.random.choice(self.size, size=n, replace=False)
        return np.random.randint(0, self.size, size=n)

    def batch_by_indices(self, idx: np.ndarray) -> Dict[str, np.ndarray]:
        idx = np.asarray(idx, dtype=np.int64)
        return {
            "observations": self.observations[idx],
            "actions": self.actions[idx],
            "next_observations": self.next_observations[idx],
            "rewards": self.rewards[idx],
            "terminals": self.terminals[idx],
            "timeouts": self.timeouts[idx],
            "indices": idx.astype(np.int64),
        }

    def raw_batch_by_indices(self, idx: np.ndarray) -> Dict[str, np.ndarray]:
        """Return unnormalized observations for simulator diagnostics."""
        idx = np.asarray(idx, dtype=np.int64)
        return {
            "raw_observations": self.raw_observations[idx],
            "raw_next_observations": self.raw_next_observations[idx],
            "actions": self.actions[idx],
            "rewards": self.rewards[idx],
            "terminals": self.terminals[idx],
            "timeouts": self.timeouts[idx],
            "indices": idx.astype(np.int64),
        }

    def view(self, indices: np.ndarray) -> "D4RLReplayView":
        """Lightweight view used for cross-fitted critics without copying data."""
        return D4RLReplayView(self, indices)


class D4RLReplayView:
    """Index view over a D4RLReplayBuffer.

    The view preserves the parent buffer's observation normalizer. This is
    important for cross-fitting: all folds should be evaluated on the same
    feature scale even if each fold trains on a disjoint subset.
    """

    def __init__(self, parent: D4RLReplayBuffer, indices: np.ndarray):
        self.parent = parent
        self.indices = np.asarray(indices, dtype=np.int64)
        self.size = len(self.indices)
        self.obs_dim = parent.obs_dim
        self.act_dim = parent.act_dim
        self.obs_normalizer = parent.obs_normalizer

    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        local = np.random.randint(0, self.size, size=batch_size)
        return self.batch_by_indices(local)

    def sample_indices(self, n: int) -> np.ndarray:
        if n <= self.size:
            return np.random.choice(self.size, size=n, replace=False)
        return np.random.randint(0, self.size, size=n)

    def batch_by_indices(self, idx: np.ndarray) -> Dict[str, np.ndarray]:
        idx = np.asarray(idx, dtype=np.int64)
        parent_idx = self.indices[idx]
        out = self.parent.batch_by_indices(parent_idx)
        out["view_indices"] = idx.astype(np.int64)
        return out

    def raw_batch_by_indices(self, idx: np.ndarray) -> Dict[str, np.ndarray]:
        idx = np.asarray(idx, dtype=np.int64)
        parent_idx = self.indices[idx]
        out = self.parent.raw_batch_by_indices(parent_idx)
        out["view_indices"] = idx.astype(np.int64)
        return out


def _d4rl_cache_roots() -> Iterable[Path]:
    roots = []
    for env_var in ["D4RL_DATASET_DIR", "D4RL_DATASETDIR"]:
        val = os.environ.get(env_var)
        if val:
            roots.append(Path(val).expanduser())
    roots.extend([
        Path.home() / ".d4rl" / "datasets",
        Path.home() / ".d4rl",
    ])
    seen = set()
    for root in roots:
        root = root.resolve()
        if root not in seen and root.exists():
            seen.add(root)
            yield root


def _matching_d4rl_hdf5_files(env_name: str) -> list[Path]:
    # Typical D4RL cache names contain the env id, e.g. antmaze-umaze-v2.hdf5.
    env_token = env_name.lower().replace("_", "-")
    base_token = env_token.rsplit("-v", 1)[0]
    matches: list[Path] = []
    for root in _d4rl_cache_roots():
        for path in root.rglob("*.hdf5"):
            name = path.name.lower().replace("_", "-")
            if env_token in name or base_token in name:
                matches.append(path)
    return sorted(set(matches))


def _remove_probably_corrupt_d4rl_files(env_name: str) -> None:
    matches = _matching_d4rl_hdf5_files(env_name)
    if not matches:
        print(
            "[D4RL] Could not locate a matching cached HDF5 file. "
            "Check ~/.d4rl/datasets or $D4RL_DATASET_DIR manually."
        )
        return
    for path in matches:
        try:
            print(f"[D4RL] Removing possibly corrupt cached dataset: {path}")
            path.unlink()
        except FileNotFoundError:
            pass


def load_d4rl(env_name: str, normalize_obs: bool = True) -> Tuple[object, D4RLReplayBuffer]:
    # Import lazily so the package can be linted without D4RL installed.
    import gym
    import d4rl  # noqa: F401

    env = gym.make(env_name)
    try:
        dataset = d4rl.qlearning_dataset(env)
    except OSError as exc:
        msg = str(exc).lower()
        if "truncated file" in msg or ("unable to" in msg and "open file" in msg):
            print(
                "[D4RL] HDF5 open failed. This almost always means the cached D4RL "
                "dataset is partially downloaded/corrupted, often because multiple "
                "parallel jobs tried to download the same file."
            )
            try:
                env.close()
            except Exception:
                pass
            _remove_probably_corrupt_d4rl_files(env_name)
            print("[D4RL] Retrying dataset download/load once...")
            env = gym.make(env_name)
            dataset = d4rl.qlearning_dataset(env)
        else:
            raise
    replay = D4RLReplayBuffer(dataset, normalize_obs=normalize_obs)
    return env, replay
