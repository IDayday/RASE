from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import numpy as np
import torch

from .bc import BCAgent
from .fqe import FQEAgent
from .iql import IQLAgent
from .replay import D4RLReplayBuffer
from .utils import torchify


@dataclass
class SweepConfig:
    candidate_ms: Iterable[int] = (1, 4, 16, 64, 256)
    n_eval_states: int = 4096
    batch_size: int = 512
    source: str = "bc"  # bc | iql | random | perturb
    perturb_std: float = 0.1
    rase_lambda_support: float = 0.05
    thresholds: Iterable[float] = (-2, -1, -0.5, 0, 0.5, 1, 2)


def _sample_candidates(
    source: str,
    obs: torch.Tensor,
    base_act: torch.Tensor,
    M: int,
    iql: IQLAgent,
    bc: BCAgent,
    perturb_std: float,
) -> torch.Tensor:
    B, act_dim = base_act.shape
    obs_rep = obs[:, None, :].expand(B, M, obs.shape[-1]).reshape(B * M, obs.shape[-1])
    if source == "bc":
        cand = bc.policy.sample(obs_rep, deterministic=False)
    elif source == "iql":
        cand = iql.actor.sample(obs_rep, deterministic=False)
    elif source == "random":
        cand = torch.empty(B * M, act_dim, device=obs.device).uniform_(-1.0, 1.0)
    elif source == "perturb":
        base = base_act[:, None, :].expand(B, M, act_dim).reshape(B * M, act_dim)
        cand = torch.clamp(base + perturb_std * torch.randn_like(base), -1.0, 1.0)
    else:
        raise ValueError(f"Unknown candidate source: {source}")
    return cand.reshape(B, M, act_dim)


@torch.no_grad()
def run_candidate_sweep(
    replay: D4RLReplayBuffer,
    iql: IQLAgent,
    bc: BCAgent,
    fqe: FQEAgent,
    cfg: SweepConfig,
    device: torch.device,
) -> Dict[str, List[Dict[str, float]]]:
    idx = replay.sample_indices(cfg.n_eval_states)
    all_rows: List[Dict[str, float]] = []
    risk_rows: List[Dict[str, float]] = []

    thresholds = list(cfg.thresholds)

    for M in cfg.candidate_ms:
        selected_pred_adv: List[np.ndarray] = []
        selected_pred_q: List[np.ndarray] = []
        selected_fqe_adv: List[np.ndarray] = []
        selected_support_nll: List[np.ndarray] = []
        selected_z: List[np.ndarray] = []

        for start in range(0, len(idx), cfg.batch_size):
            batch_idx = idx[start : start + cfg.batch_size]
            b_np = replay.batch_by_indices(batch_idx)
            obs = torchify(b_np["observations"], device)
            base_act = torchify(b_np["actions"], device)
            B = obs.shape[0]

            cand = _sample_candidates(cfg.source, obs, base_act, M, iql, bc, cfg.perturb_std)
            obs_rep = obs[:, None, :].expand(B, M, obs.shape[-1]).reshape(B * M, obs.shape[-1])
            cand_flat = cand.reshape(B * M, -1)

            q, v, adv = iql.score(obs_rep, cand_flat)
            q = q.reshape(B, M)
            adv = adv.reshape(B, M)
            best = torch.argmax(q, dim=1)
            gather_idx = best[:, None, None].expand(B, 1, cand.shape[-1])
            best_act = cand.gather(1, gather_idx).squeeze(1)
            best_q = q.gather(1, best[:, None]).squeeze(1)
            best_adv = adv.gather(1, best[:, None]).squeeze(1)

            base_fqe = fqe.value(obs, base_act).squeeze(1)
            cand_fqe = fqe.value(obs, best_act).squeeze(1)
            fqe_adv = cand_fqe - base_fqe
            support_nll = -bc.policy.log_prob(obs, best_act).squeeze(1)
            z = best_adv - cfg.rase_lambda_support * support_nll

            selected_pred_q.append(best_q.cpu().numpy())
            selected_pred_adv.append(best_adv.cpu().numpy())
            selected_fqe_adv.append(fqe_adv.cpu().numpy())
            selected_support_nll.append(support_nll.cpu().numpy())
            selected_z.append(z.cpu().numpy())

        pred_q_np = np.concatenate(selected_pred_q)
        pred_adv_np = np.concatenate(selected_pred_adv)
        fqe_adv_np = np.concatenate(selected_fqe_adv)
        support_np = np.concatenate(selected_support_nll)
        z_np = np.concatenate(selected_z)
        pred_positive = pred_adv_np > 0.0
        true_positive = fqe_adv_np > 0.0
        false_positive = np.logical_and(pred_positive, ~true_positive)
        row = {
            "M": int(M),
            "source": cfg.source,
            "n": int(len(pred_q_np)),
            "pred_q_mean": float(pred_q_np.mean()),
            "pred_adv_mean": float(pred_adv_np.mean()),
            "fqe_adv_mean": float(fqe_adv_np.mean()),
            "pred_empirical_gap": float((pred_adv_np - fqe_adv_np).mean()),
            "pred_positive_rate": float(pred_positive.mean()),
            "fpi_rate_unconditional": float(false_positive.mean()),
            "fpi_rate_cond_pred_positive": float(false_positive.sum() / max(pred_positive.sum(), 1)),
            "support_nll_mean": float(support_np.mean()),
        }
        all_rows.append(row)

        for tau in thresholds:
            accepted = z_np >= tau
            if accepted.sum() == 0:
                precision = float("nan")
                fpr = float("nan")
                accepted_fqe_adv = float("nan")
            else:
                precision = float(true_positive[accepted].mean())
                fpr = float((~true_positive[accepted]).mean())
                accepted_fqe_adv = float(fqe_adv_np[accepted].mean())
            risk_rows.append({
                "M": int(M),
                "source": cfg.source,
                "tau": float(tau),
                "coverage": float(accepted.mean()),
                "precision": precision,
                "fpr": fpr,
                "accepted_fqe_adv_mean": accepted_fqe_adv,
            })

    return {"sweep": all_rows, "risk_coverage": risk_rows}
