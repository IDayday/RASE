from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import torch
import torch.nn.functional as F
from torch.optim import Adam

from .networks import GaussianPolicy, TwinQ
from .utils import hard_update, soft_update, torchify


@dataclass
class FQEConfig:
    gamma: float = 0.99
    tau: float = 0.005
    lr: float = 3e-4
    hidden_dim: int = 256


class FQEAgent:
    """Twin-Q FQE for a fixed continuation policy.

    The diagnostic treats FQE as an empirical proxy for the first-action value
    under the IQL continuation policy. A single Q estimator can itself overestimate
    OOD candidate actions; the twin/min value is a cheap conservative proxy that
    makes false-positive diagnostics less brittle.
    """

    CHECKPOINT_VERSION = 2

    def __init__(self, obs_dim: int, act_dim: int, cfg: FQEConfig, ref_policy: GaussianPolicy, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.ref_policy = ref_policy.to(device)
        self.ref_policy.eval()
        for p in self.ref_policy.parameters():
            p.requires_grad_(False)
        hidden = (cfg.hidden_dim, cfg.hidden_dim)
        self.q = TwinQ(obs_dim, act_dim, hidden).to(device)
        self.q_target = TwinQ(obs_dim, act_dim, hidden).to(device)
        hard_update(self.q_target, self.q)
        self.opt = Adam(self.q.parameters(), lr=cfg.lr)

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        obs = batch["observations"]
        act = batch["actions"]
        next_obs = batch["next_observations"]
        rew = batch["rewards"]
        done = batch["terminals"]
        with torch.no_grad():
            next_act = self.ref_policy.sample(next_obs, deterministic=True)
            target = rew + self.cfg.gamma * (1.0 - done) * self.q_target.q_min(next_obs, next_act)
        q1, q2 = self.q(obs, act)
        loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()
        soft_update(self.q_target, self.q, self.cfg.tau)
        with torch.no_grad():
            q_min = torch.minimum(q1, q2)
            q_disagreement = torch.abs(q1 - q2).mean()
        return {
            "fqe/loss": float(loss.detach().cpu()),
            "fqe/q_mean": float(q_min.mean().detach().cpu()),
            "fqe/q_disagreement": float(q_disagreement.detach().cpu()),
        }

    def batch_to_torch(self, batch_np: Dict[str, object]) -> Dict[str, torch.Tensor]:
        keys = ["observations", "actions", "next_observations", "rewards", "terminals"]
        return {k: torchify(batch_np[k], self.device) for k in keys}

    @torch.no_grad()
    def value(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        return self.q.q_min(obs, act)

    @torch.no_grad()
    def disagreement(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        q1, q2 = self.q(obs, act)
        return torch.abs(q1 - q2)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "version": self.CHECKPOINT_VERSION,
            "q": self.q.state_dict(),
            "q_target": self.q_target.state_dict(),
            "cfg": self.cfg.__dict__,
        }, path)

    def load(self, path: str | Path) -> None:
        ckpt = torch.load(path, map_location=self.device)
        if ckpt.get("version", 1) != self.CHECKPOINT_VERSION:
            raise RuntimeError(
                "FQE checkpoint is from an older incompatible version. "
                "Use --force_retrain or delete fqe_iql_ref.pt."
            )
        self.q.load_state_dict(ckpt["q"])
        self.q_target.load_state_dict(ckpt.get("q_target", ckpt["q"]))
