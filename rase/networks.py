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
    """Diagonal Gaussian policy with configurable action squashing.

    `squash_mode="tanh"` is the audited v3/v4+ behavior: samples are transformed
    with tanh and log_prob uses the exact change-of-variables correction.

    `squash_mode="clip"` reproduces the legacy Phase-0 behavior used by the first
    code releases: raw Gaussian samples are hard-clipped and log_prob is computed
    in the raw action space. This mode is intentionally retained only so existing
    checkpoints under `outputs/rase_phase0/` can be diagnosed without changing the
    behavior of the saved IQL/BC policies.
    """

    VALID_SQUASH_MODES = {"tanh", "clip"}

    def __init__(self, obs_dim: int, act_dim: int, hidden_dims=(256, 256), log_std_bounds=(-5.0, 2.0), squash_mode: str = "tanh"):
        super().__init__()
        if squash_mode not in self.VALID_SQUASH_MODES:
            raise ValueError(f"Unknown squash_mode={squash_mode!r}; expected one of {sorted(self.VALID_SQUASH_MODES)}")
        self.backbone = MLP(obs_dim, act_dim * 2, hidden_dims)
        self.log_std_bounds = log_std_bounds
        self.act_dim = act_dim
        self.squash_mode = str(squash_mode)

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
        if self.squash_mode == "clip":
            return torch.clamp(raw, -1.0, 1.0)
        return torch.tanh(raw)

    def sample_n(self, obs: torch.Tensor, n: int, deterministic: bool = False) -> torch.Tensor:
        """Return [B, n, act_dim] actions without building Python loops."""
        b = obs.shape[0]
        obs_rep = obs[:, None, :].expand(b, int(n), obs.shape[-1]).reshape(b * int(n), obs.shape[-1])
        actions = self.sample(obs_rep, deterministic=deterministic)
        return actions.reshape(b, int(n), self.act_dim)

    def log_prob(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        dist = self.distribution(obs)
        if self.squash_mode == "clip":
            # Legacy approximation used by the original Phase-0 checkpoints.
            return dist.log_prob(actions).sum(dim=-1, keepdim=True)
        raw = atanh_clamped(actions)
        logp_raw = dist.log_prob(raw).sum(dim=-1, keepdim=True)
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
