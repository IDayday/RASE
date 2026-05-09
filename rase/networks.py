from __future__ import annotations

from typing import Iterable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: Iterable[int] = (256, 256), activation=nn.ReLU):
        super().__init__()
        dims = [input_dim, *list(hidden_dims)]
        layers = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(a, b), activation()]
        layers += [nn.Linear(dims[-1], output_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TwinQ(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_dims=(256, 256)):
        super().__init__()
        self.q1 = MLP(obs_dim + act_dim, 1, hidden_dims)
        self.q2 = MLP(obs_dim + act_dim, 1, hidden_dims)

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([obs, act], dim=-1)
        return self.q1(x), self.q2(x)

    def q_min(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        q1, q2 = self(obs, act)
        return torch.minimum(q1, q2)


class QNet(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_dims=(256, 256)):
        super().__init__()
        self.q = MLP(obs_dim + act_dim, 1, hidden_dims)

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        return self.q(torch.cat([obs, act], dim=-1))


class ValueNet(nn.Module):
    def __init__(self, obs_dim: int, hidden_dims=(256, 256)):
        super().__init__()
        self.v = MLP(obs_dim, 1, hidden_dims)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.v(obs)


class GaussianPolicy(nn.Module):
    """Diagonal Gaussian policy with clipped actions.

    This is intentionally simple for Phase 0 diagnostics. It is suitable for
    behavior cloning, IQL policy extraction, FQE continuation, and proposal
    sampling. For paper-level experiments, replace this with a stronger policy
    or flow proposal after the Phase 0 go/no-go decision.
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden_dims=(256, 256), log_std_bounds=(-5.0, 2.0)):
        super().__init__()
        self.backbone = MLP(obs_dim, act_dim * 2, hidden_dims)
        self.log_std_bounds = log_std_bounds
        self.act_dim = act_dim

    def mean_logstd(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.backbone(obs)
        mean, log_std = torch.chunk(out, 2, dim=-1)
        lo, hi = self.log_std_bounds
        log_std = torch.tanh(log_std)
        log_std = lo + 0.5 * (hi - lo) * (log_std + 1.0)
        return mean, log_std

    def distribution(self, obs: torch.Tensor) -> Normal:
        mean, log_std = self.mean_logstd(obs)
        return Normal(mean, log_std.exp())

    def sample(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        dist = self.distribution(obs)
        raw = dist.mean if deterministic else dist.rsample()
        return torch.clamp(raw, -1.0, 1.0)

    def log_prob(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        dist = self.distribution(obs)
        # Phase-0 approximation: log-probability in action space without tanh correction.
        return dist.log_prob(actions).sum(dim=-1, keepdim=True)

    @torch.no_grad()
    def act_np(self, obs_np, device: torch.device, deterministic: bool = True):
        obs = torch.as_tensor(obs_np, dtype=torch.float32, device=device)
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)
        act = self.sample(obs, deterministic=deterministic)
        return act.cpu().numpy()[0]


def expectile_loss(diff: torch.Tensor, expectile: float) -> torch.Tensor:
    weight = torch.where(diff > 0, expectile, 1.0 - expectile)
    return weight * diff.pow(2)
