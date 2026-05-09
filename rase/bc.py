from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import torch
from torch.optim import Adam

from .networks import GaussianPolicy
from .utils import torchify


@dataclass
class BCConfig:
    lr: float = 3e-4
    hidden_dim: int = 256


class BCAgent:
    def __init__(self, obs_dim: int, act_dim: int, cfg: BCConfig, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.policy = GaussianPolicy(obs_dim, act_dim, (cfg.hidden_dim, cfg.hidden_dim)).to(device)
        self.opt = Adam(self.policy.parameters(), lr=cfg.lr)

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        obs = batch["observations"]
        act = batch["actions"]
        loss = -self.policy.log_prob(obs, act).mean()
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()
        return {"bc/loss": float(loss.detach().cpu())}

    def batch_to_torch(self, batch_np: Dict[str, object]) -> Dict[str, torch.Tensor]:
        return {
            "observations": torchify(batch_np["observations"], self.device),
            "actions": torchify(batch_np["actions"], self.device),
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"policy": self.policy.state_dict(), "cfg": self.cfg.__dict__}, path)

    def load(self, path: str | Path) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(ckpt["policy"])
