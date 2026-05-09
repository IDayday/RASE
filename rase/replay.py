from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


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
        return np.random.randint(0, self.size, size=n)

    def batch_by_indices(self, idx: np.ndarray) -> Dict[str, np.ndarray]:
        return {
            "observations": self.observations[idx],
            "actions": self.actions[idx],
            "next_observations": self.next_observations[idx],
            "rewards": self.rewards[idx],
            "terminals": self.terminals[idx],
            "timeouts": self.timeouts[idx],
            "indices": idx.astype(np.int64),
        }


def load_d4rl(env_name: str, normalize_obs: bool = True) -> Tuple[object, D4RLReplayBuffer]:
    # Import lazily so the package can be linted without D4RL installed.
    import gym
    import d4rl  # noqa: F401

    env = gym.make(env_name)
    dataset = d4rl.qlearning_dataset(env)
    replay = D4RLReplayBuffer(dataset, normalize_obs=normalize_obs)
    return env, replay
