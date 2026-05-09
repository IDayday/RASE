from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch.optim import Adam

from .networks import GaussianPolicy, TwinQ, ValueNet, expectile_loss
from .utils import hard_update, soft_update, torchify


@dataclass
class IQLConfig:
    gamma: float = 0.99
    tau: float = 0.005
    expectile: float = 0.7
    beta: float = 3.0
    exp_adv_max: float = 100.0
    lr: float = 3e-4
    hidden_dim: int = 256


class IQLAgent:
    def __init__(self, obs_dim: int, act_dim: int, cfg: IQLConfig, device: torch.device):
        self.cfg = cfg
        self.device = device
        hidden = (cfg.hidden_dim, cfg.hidden_dim)
        self.actor = GaussianPolicy(obs_dim, act_dim, hidden).to(device)
        self.q = TwinQ(obs_dim, act_dim, hidden).to(device)
        self.q_target = TwinQ(obs_dim, act_dim, hidden).to(device)
        self.v = ValueNet(obs_dim, hidden).to(device)
        hard_update(self.q_target, self.q)
        self.actor_opt = Adam(self.actor.parameters(), lr=cfg.lr)
        self.q_opt = Adam(self.q.parameters(), lr=cfg.lr)
        self.v_opt = Adam(self.v.parameters(), lr=cfg.lr)

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        obs = batch["observations"]
        act = batch["actions"]
        next_obs = batch["next_observations"]
        rew = batch["rewards"]
        done = batch["terminals"]

        with torch.no_grad():
            target_q = self.q_target.q_min(obs, act)
        v = self.v(obs)
        value_loss = expectile_loss(target_q - v, self.cfg.expectile).mean()
        self.v_opt.zero_grad(set_to_none=True)
        value_loss.backward()
        self.v_opt.step()

        with torch.no_grad():
            next_v = self.v(next_obs)
            q_target = rew + self.cfg.gamma * (1.0 - done) * next_v
        q1, q2 = self.q(obs, act)
        q_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)
        self.q_opt.zero_grad(set_to_none=True)
        q_loss.backward()
        self.q_opt.step()

        with torch.no_grad():
            q_min = self.q.q_min(obs, act)
            adv = q_min - self.v(obs)
            weights = torch.exp(self.cfg.beta * adv).clamp(max=self.cfg.exp_adv_max)
        log_prob = self.actor.log_prob(obs, act)
        actor_loss = -(weights * log_prob).mean()
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        soft_update(self.q_target, self.q, self.cfg.tau)

        return {
            "iql/value_loss": float(value_loss.detach().cpu()),
            "iql/q_loss": float(q_loss.detach().cpu()),
            "iql/actor_loss": float(actor_loss.detach().cpu()),
            "iql/adv_mean": float(adv.mean().detach().cpu()),
            "iql/weight_mean": float(weights.mean().detach().cpu()),
        }

    def batch_to_torch(self, batch_np: Dict[str, object]) -> Dict[str, torch.Tensor]:
        keys = ["observations", "actions", "next_observations", "rewards", "terminals"]
        return {k: torchify(batch_np[k], self.device) for k in keys}

    @torch.no_grad()
    def score(self, obs: torch.Tensor, act: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q = self.q.q_min(obs, act)
        v = self.v(obs)
        adv = q - v
        return q, v, adv

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "actor": self.actor.state_dict(),
            "q": self.q.state_dict(),
            "q_target": self.q_target.state_dict(),
            "v": self.v.state_dict(),
            "cfg": self.cfg.__dict__,
        }, path)

    def load(self, path: str | Path) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.q.load_state_dict(ckpt["q"])
        self.q_target.load_state_dict(ckpt.get("q_target", ckpt["q"]))
        self.v.load_state_dict(ckpt["v"])
