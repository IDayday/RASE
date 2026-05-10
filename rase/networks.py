from __future__ import annotations

from typing import Iterable, Tuple

import torch
import torch.nn as nn
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

    def q_stack(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        q1, q2 = self(obs, act)
        return torch.cat([q1, q2], dim=-1)


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


def atanh_clamped(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Numerically stable inverse tanh for actions in [-1, 1]."""
    x = torch.clamp(x, -1.0 + eps, 1.0 - eps)
    return 0.5 * torch.log((1.0 + x) / (1.0 - x))


class GaussianPolicy(nn.Module):
    """Tanh-squashed diagonal Gaussian policy.

    Earlier drafts used raw Gaussian samples followed by hard clipping. That is fast,
    but it makes the behavior log-probability and support-NLL diagnostic inconsistent:
    a clipped sample is not distributed as the unclipped Normal whose log_prob is used.
    This implementation uses the standard tanh transform and exact change-of-variables
    correction, which is still cheap and gives better support diagnostics.
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
        return torch.tanh(raw)

    def sample_n(self, obs: torch.Tensor, n: int, deterministic: bool = False) -> torch.Tensor:
        """Return [B, n, act_dim] actions without building Python loops."""
        b = obs.shape[0]
        obs_rep = obs[:, None, :].expand(b, n, obs.shape[-1]).reshape(b * n, obs.shape[-1])
        actions = self.sample(obs_rep, deterministic=deterministic)
        return actions.reshape(b, n, self.act_dim)

    def log_prob(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        dist = self.distribution(obs)
        raw = atanh_clamped(actions)
        logp_raw = dist.log_prob(raw).sum(dim=-1, keepdim=True)
        # Change-of-variables correction for tanh(raw).
        correction = torch.log(torch.clamp(1.0 - actions.pow(2), min=1e-6)).sum(dim=-1, keepdim=True)
        return logp_raw - correction

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
