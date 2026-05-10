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
    """Vectorized candidate sampler returning [B, M, act_dim]."""
    B, act_dim = base_act.shape
    if source == "bc":
        return bc.policy.sample_n(obs, M, deterministic=False)
    if source == "iql":
        return iql.actor.sample_n(obs, M, deterministic=False)
    if source == "random":
        return torch.empty(B, M, act_dim, device=obs.device).uniform_(-1.0, 1.0)
    if source == "perturb":
        base = base_act[:, None, :].expand(B, M, act_dim)
        return torch.clamp(base + perturb_std * torch.randn_like(base), -1.0, 1.0)
    raise ValueError(f"Unknown candidate source: {source}")


def _mean_std_se(x: np.ndarray) -> tuple[float, float, float]:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(x.mean())
    std = float(x.std(ddof=1)) if x.size > 1 else 0.0
    se = float(std / np.sqrt(x.size)) if x.size > 1 else 0.0
    return mean, std, se


@torch.no_grad()
def run_candidate_sweep(
    replay: D4RLReplayBuffer,
    iql: IQLAgent,
    bc: BCAgent,
    fqe: FQEAgent,
    cfg: SweepConfig,
    device: torch.device,
) -> Dict[str, List[Dict[str, float]]]:
    """Run a nested candidate-pool-size sweep.

    Important implementation choices:
    1. All candidate pool sizes for a state share the same max-M candidate list and
       use prefixes. This directly measures the effect of enlarging the same pool
       instead of adding independent Monte Carlo noise at every M.
    2. False-positive policy improvement is defined with a consistent pairwise
       baseline: predicted gap Q_IQL(s,a*) - Q_IQL(s,a_data) versus empirical proxy
       gap FQE(s,a*) - FQE(s,a_data). The older draft mixed Q(s,a*)-V(s) with a
       pairwise FQE gap, which shifted the FPI label.
    """
    candidate_ms = sorted({int(m) for m in cfg.candidate_ms})
    if not candidate_ms or candidate_ms[0] <= 0:
        raise ValueError(f"candidate_ms must contain positive integers, got {cfg.candidate_ms}")
    max_m = candidate_ms[-1]

    idx = replay.sample_indices(cfg.n_eval_states)
    thresholds = list(cfg.thresholds)

    per_m: dict[int, dict[str, list[np.ndarray]]] = {
        M: {
            "pred_q": [],
            "pred_pair_gap": [],
            "iql_adv_vs_v": [],
            "fqe_pair_gap": [],
            "fqe_disagreement": [],
            "support_nll": [],
            "rase_score": [],
        }
        for M in candidate_ms
    }

    for start in range(0, len(idx), cfg.batch_size):
        batch_idx = idx[start : start + cfg.batch_size]
        b_np = replay.batch_by_indices(batch_idx)
        obs = torchify(b_np["observations"], device)
        base_act = torchify(b_np["actions"], device)
        B = obs.shape[0]

        base_iql_q, base_iql_v, _ = iql.score(obs, base_act)
        base_iql_q = base_iql_q.squeeze(1)
        base_iql_v = base_iql_v.squeeze(1)
        base_fqe = fqe.value(obs, base_act).squeeze(1)

        cand = _sample_candidates(cfg.source, obs, base_act, max_m, iql, bc, cfg.perturb_std)
        obs_rep = obs[:, None, :].expand(B, max_m, obs.shape[-1]).reshape(B * max_m, obs.shape[-1])
        cand_flat = cand.reshape(B * max_m, -1)

        q_flat, _, adv_flat = iql.score(obs_rep, cand_flat)
        q_all = q_flat.reshape(B, max_m)
        adv_vs_v_all = adv_flat.reshape(B, max_m)
        pred_pair_gap_all = q_all - base_iql_q[:, None]

        for M in candidate_ms:
            q_prefix = q_all[:, :M]
            best = torch.argmax(q_prefix, dim=1)
            gather_idx = best[:, None, None].expand(B, 1, cand.shape[-1])
            best_act = cand[:, :M, :].gather(1, gather_idx).squeeze(1)

            row_idx = torch.arange(B, device=device)
            best_q = q_prefix[row_idx, best]
            best_pair_gap = pred_pair_gap_all[:, :M][row_idx, best]
            best_adv_vs_v = adv_vs_v_all[:, :M][row_idx, best]

            cand_fqe = fqe.value(obs, best_act).squeeze(1)
            fqe_pair_gap = cand_fqe - base_fqe
            fqe_disagreement = fqe.disagreement(obs, best_act).squeeze(1)
            support_nll = -bc.policy.log_prob(obs, best_act).squeeze(1)
            rase_score = best_pair_gap - cfg.rase_lambda_support * support_nll

            per_m[M]["pred_q"].append(best_q.cpu().numpy())
            per_m[M]["pred_pair_gap"].append(best_pair_gap.cpu().numpy())
            per_m[M]["iql_adv_vs_v"].append(best_adv_vs_v.cpu().numpy())
            per_m[M]["fqe_pair_gap"].append(fqe_pair_gap.cpu().numpy())
            per_m[M]["fqe_disagreement"].append(fqe_disagreement.cpu().numpy())
            per_m[M]["support_nll"].append(support_nll.cpu().numpy())
            per_m[M]["rase_score"].append(rase_score.cpu().numpy())

    all_rows: List[Dict[str, float]] = []
    risk_rows: List[Dict[str, float]] = []

    for M in candidate_ms:
        data = {k: np.concatenate(v) for k, v in per_m[M].items()}
        pred_q_np = data["pred_q"]
        pred_gap_np = data["pred_pair_gap"]
        adv_v_np = data["iql_adv_vs_v"]
        fqe_gap_np = data["fqe_pair_gap"]
        fqe_dis_np = data["fqe_disagreement"]
        support_np = data["support_nll"]
        z_np = data["rase_score"]

        pred_positive = pred_gap_np > 0.0
        true_positive = fqe_gap_np > 0.0
        false_positive = np.logical_and(pred_positive, ~true_positive)

        pred_gap_mean, pred_gap_std, pred_gap_se = _mean_std_se(pred_gap_np)
        fqe_gap_mean, fqe_gap_std, fqe_gap_se = _mean_std_se(fqe_gap_np)
        row = {
            "M": int(M),
            "source": cfg.source,
            "n": int(len(pred_q_np)),
            "pred_q_mean": float(pred_q_np.mean()),
            # Backward-compatible column name, now using the correct pairwise baseline.
            "pred_adv_mean": pred_gap_mean,
            "pred_adv_std": pred_gap_std,
            "pred_adv_se": pred_gap_se,
            "iql_adv_vs_v_mean": float(adv_v_np.mean()),
            "fqe_adv_mean": fqe_gap_mean,
            "fqe_adv_std": fqe_gap_std,
            "fqe_adv_se": fqe_gap_se,
            "pred_empirical_gap": float((pred_gap_np - fqe_gap_np).mean()),
            "pred_positive_rate": float(pred_positive.mean()),
            "empirical_positive_rate": float(true_positive.mean()),
            "fpi_rate_unconditional": float(false_positive.mean()),
            "fpi_rate_cond_pred_positive": float(false_positive.sum() / max(int(pred_positive.sum()), 1)),
            "support_nll_mean": float(support_np.mean()),
            "fqe_disagreement_mean": float(fqe_dis_np.mean()),
            "rase_score_mean": float(z_np.mean()),
        }
        all_rows.append(row)

        for tau in thresholds:
            accepted = z_np >= tau
            if accepted.sum() == 0:
                precision = float("nan")
                fpr = float("nan")
                accepted_fqe_adv = float("nan")
                accepted_pred_adv = float("nan")
                accepted_support_nll = float("nan")
            else:
                precision = float(true_positive[accepted].mean())
                fpr = float((~true_positive[accepted]).mean())
                accepted_fqe_adv = float(fqe_gap_np[accepted].mean())
                accepted_pred_adv = float(pred_gap_np[accepted].mean())
                accepted_support_nll = float(support_np[accepted].mean())
            risk_rows.append({
                "M": int(M),
                "source": cfg.source,
                "tau": float(tau),
                "coverage": float(accepted.mean()),
                "precision": precision,
                "fpr": fpr,
                "accepted_pred_adv_mean": accepted_pred_adv,
                "accepted_fqe_adv_mean": accepted_fqe_adv,
                "accepted_support_nll_mean": accepted_support_nll,
            })

    return {"sweep": all_rows, "risk_coverage": risk_rows}
