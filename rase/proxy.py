from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable

import numpy as np
import torch

from .metrics import auprc_from_scores, auroc_from_scores, precision_at_coverage, risk_coverage_rows
from .replay import D4RLReplayBuffer


@dataclass
class ProxyConfig:
    knn_ref_size: int = 20000
    knn_batch_size: int = 512
    obs_scale: float = 1.0
    action_scale: float = 1.0
    coverages: Iterable[float] = (0.2, 0.3, 0.5, 0.7)
    composite_lambda_support: float = 0.05
    composite_lambda_iql_dis: float = 0.10
    composite_lambda_fqe_dis: float = 0.10
    composite_lambda_knn: float = 0.10


def _make_joint_features(obs: np.ndarray, act: np.ndarray, obs_scale: float, action_scale: float) -> np.ndarray:
    return np.concatenate([obs_scale * obs.astype(np.float32), action_scale * act.astype(np.float32)], axis=-1)


def knn_support_distance(
    replay: D4RLReplayBuffer,
    obs: np.ndarray,
    act: np.ndarray,
    cfg: ProxyConfig,
    device: torch.device,
    seed: int = 0,
) -> np.ndarray:
    """Approximate support distance in normalized (s,a) space.

    We sample a fixed reference set from the dataset and compute the distance to
    the nearest dataset state-action pair. This is intentionally a diagnostic
    proxy, not a learned density model.
    """
    rng = np.random.default_rng(seed)
    ref_n = min(int(cfg.knn_ref_size), replay.size)
    ref_idx = rng.choice(replay.size, size=ref_n, replace=False)
    ref = _make_joint_features(
        replay.observations[ref_idx], replay.actions[ref_idx], float(cfg.obs_scale), float(cfg.action_scale)
    )
    query = _make_joint_features(obs, act, float(cfg.obs_scale), float(cfg.action_scale))

    ref_t = torch.as_tensor(ref, dtype=torch.float32, device=device)
    dists = []
    for start in range(0, query.shape[0], int(cfg.knn_batch_size)):
        q = torch.as_tensor(query[start : start + int(cfg.knn_batch_size)], dtype=torch.float32, device=device)
        # [B, ref_n]. cdist is fast enough for the intended 4k diagnostic states.
        d = torch.cdist(q, ref_t, p=2).min(dim=1).values
        dists.append(d.cpu().numpy())
    return np.concatenate(dists, axis=0).astype(np.float32)


def add_knn_proxy_to_selected(
    selected: Dict[int, Dict[str, np.ndarray]],
    replay: D4RLReplayBuffer,
    cfg: ProxyConfig,
    device: torch.device,
    seed: int = 0,
) -> Dict[int, Dict[str, np.ndarray]]:
    for M, data in selected.items():
        data["knn_sa_distance"] = knn_support_distance(
            replay, data["obs"], data["selected_action"], cfg, device, seed=seed + int(M)
        )
        data["rase_score_v2"] = (
            data["pred_pair_gap"]
            - float(cfg.composite_lambda_support) * data["support_nll"]
            - float(cfg.composite_lambda_iql_dis) * data["iql_q_disagreement"]
            - float(cfg.composite_lambda_fqe_dis) * data["fqe_disagreement"]
            - float(cfg.composite_lambda_knn) * data["knn_sa_distance"]
        ).astype(np.float32)
    return selected


def proxy_alignment_summary(selected: Dict[int, Dict[str, np.ndarray]], source: str, cfg: ProxyConfig) -> list[dict]:
    """Evaluate proxy scores against FQE-positive labels.

    Positive label is `fqe_pair_gap > 0`. This is still a proxy label; Phase 0.5
    rollout diagnostics can be used to replace or validate it.
    """
    rows: list[dict] = []
    score_specs = [
        ("pred_pair_gap", +1.0),
        ("rase_score", +1.0),
        ("rase_score_v2", +1.0),
        ("neg_support_nll", -1.0, "support_nll"),
        ("neg_knn_sa_distance", -1.0, "knn_sa_distance"),
        ("neg_iql_q_disagreement", -1.0, "iql_q_disagreement"),
        ("neg_fqe_disagreement", -1.0, "fqe_disagreement"),
        ("neg_action_l2_to_data", -1.0, "action_l2_to_data"),
    ]
    for M, data in selected.items():
        labels = data["fqe_pair_gap"] > 0.0
        for spec in score_specs:
            if len(spec) == 2:
                score_name, sign = spec
                key = score_name
            else:
                score_name, sign, key = spec
            if key not in data:
                continue
            scores = sign * data[key]
            base = {
                "M": int(M),
                "source": source,
                "score": score_name,
                "n": int(len(labels)),
                "positive_rate": float(labels.mean()),
                "auroc_fqe_positive": auroc_from_scores(labels, scores),
                "auprc_fqe_positive": auprc_from_scores(labels, scores),
            }
            for cov in cfg.coverages:
                base[f"precision_at_cov_{cov:g}"] = precision_at_coverage(labels, scores, float(cov))
            rows.append(base)
    return rows


def calibrated_rows(selected: Dict[int, Dict[str, np.ndarray]], source: str, cfg: ProxyConfig) -> list[dict]:
    rows: list[dict] = []
    for M, data in selected.items():
        labels = data["fqe_pair_gap"] > 0.0
        for score_name in ["pred_pair_gap", "rase_score", "rase_score_v2"]:
            if score_name not in data:
                continue
            rows.extend(risk_coverage_rows(
                labels,
                data[score_name],
                cfg.coverages,
                prefix={"M": int(M), "source": source, "score": score_name},
            ))
    return rows
