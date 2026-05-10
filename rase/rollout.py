from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Literal

import numpy as np
import torch

from .bc import BCAgent
from .fqe import FQEAgent
from .iql import IQLAgent
from .replay import D4RLReplayBuffer
from .selection import CandidateSelectionConfig, collect_selected_candidates
from .metrics import safe_corr

ContinuationPolicy = Literal["iql", "bc"]


@dataclass
class RolloutDiagnosticConfig:
    candidate_ms: Iterable[int] = (1, 16, 64, 256)
    n_eval_states: int = 256
    batch_size: int = 256
    source: str = "bc"
    perturb_std: float = 0.1
    rase_lambda_support: float = 0.05
    rollout_horizon: int = 50
    rollout_repeats: int = 3
    gamma: float = 0.99
    max_pairs_per_m: int = 128
    only_pred_positive: bool = True
    continuation_policy: ContinuationPolicy = "iql"
    deterministic_continuation: bool = True


def _env_step(env, action: np.ndarray):
    out = env.step(action.astype(np.float32))
    if len(out) == 5:  # gymnasium API
        obs, reward, terminated, truncated, info = out
        done = bool(terminated or truncated)
    else:
        obs, reward, done, info = out
    return np.asarray(obs, dtype=np.float32), float(reward), bool(done), info


def reset_env(env, seed: int | None = None):
    try:
        if seed is None:
            out = env.reset()
        else:
            out = env.reset(seed=seed)
    except TypeError:
        if seed is not None and hasattr(env, "seed"):
            env.seed(seed)
        out = env.reset()
    if isinstance(out, tuple):
        return out[0]
    return out


def infer_mujoco_qpos_qvel(env, raw_obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Best-effort inverse of common D4RL MuJoCo observation functions.

    Locomotion observations usually omit the root x position but include qpos[1:]
    and all qvel. Setting qpos[0]=0 is acceptable for the short-horizon action
    replacement diagnostic because absolute x is not part of the observation and
    the reward depends on velocity / displacement over the rollout.
    """
    unwrapped = getattr(env, "unwrapped", env)
    if not hasattr(unwrapped, "model"):
        raise RuntimeError("Environment does not expose a MuJoCo model; state reset diagnostic is unavailable.")
    nq = int(unwrapped.model.nq)
    nv = int(unwrapped.model.nv)
    raw_obs = np.asarray(raw_obs, dtype=np.float64).reshape(-1)

    if raw_obs.size == nq + nv:
        qpos = raw_obs[:nq].copy()
        qvel = raw_obs[nq : nq + nv].copy()
    elif raw_obs.size == (nq - 1) + nv:
        qpos = np.zeros(nq, dtype=np.float64)
        qpos[1:] = raw_obs[: nq - 1]
        qvel = raw_obs[nq - 1 : nq - 1 + nv].copy()
    elif raw_obs.size > nq + nv:
        # Some envs append goal or contact features. Use the leading qpos/qvel block.
        qpos = raw_obs[:nq].copy()
        qvel = raw_obs[nq : nq + nv].copy()
    else:
        raise RuntimeError(
            f"Cannot infer MuJoCo state from observation length {raw_obs.size}; model has nq={nq}, nv={nv}."
        )
    return qpos, qvel


def set_mujoco_state_from_obs(env, raw_obs: np.ndarray) -> None:
    unwrapped = getattr(env, "unwrapped", env)
    qpos, qvel = infer_mujoco_qpos_qvel(env, raw_obs)
    if hasattr(unwrapped, "set_state"):
        unwrapped.set_state(qpos, qvel)
    elif hasattr(unwrapped, "sim"):
        unwrapped.sim.data.qpos[:] = qpos
        unwrapped.sim.data.qvel[:] = qvel
        unwrapped.sim.forward()
    else:
        raise RuntimeError("Environment does not expose set_state or sim for state reset.")
    if hasattr(unwrapped, "sim"):
        unwrapped.sim.forward()


def _policy_action_np(policy, replay: D4RLReplayBuffer, obs_raw: np.ndarray, device: torch.device, deterministic: bool) -> np.ndarray:
    obs_norm = replay.obs_normalizer.normalize(obs_raw.reshape(1, -1).astype(np.float32))
    obs_t = torch.as_tensor(obs_norm, dtype=torch.float32, device=device)
    with torch.no_grad():
        act = policy.sample(obs_t, deterministic=deterministic)
    return act.cpu().numpy()[0]


def rollout_first_action_then_policy(
    env,
    replay: D4RLReplayBuffer,
    raw_obs: np.ndarray,
    first_action: np.ndarray,
    continuation_policy,
    device: torch.device,
    horizon: int,
    gamma: float,
    deterministic_continuation: bool,
    seed: int | None = None,
) -> float:
    reset_env(env, seed=seed)
    set_mujoco_state_from_obs(env, raw_obs)
    obs, reward, done, _ = _env_step(env, first_action)
    ret = float(reward)
    discount = float(gamma)
    for _ in range(1, int(horizon)):
        if done:
            break
        act = _policy_action_np(continuation_policy, replay, obs, device, deterministic_continuation)
        obs, reward, done, _ = _env_step(env, act)
        ret += discount * float(reward)
        discount *= float(gamma)
    return ret


def _pick_pairs(data: Dict[str, np.ndarray], max_pairs: int, only_pred_positive: bool, seed: int) -> np.ndarray:
    mask = np.ones(len(data["indices"]), dtype=bool)
    if only_pred_positive:
        mask &= data["pred_pair_gap"] > 0.0
    candidates = np.where(mask)[0]
    if candidates.size == 0:
        return candidates
    rng = np.random.default_rng(seed)
    if candidates.size > max_pairs:
        candidates = rng.choice(candidates, size=max_pairs, replace=False)
    return np.sort(candidates)


def run_action_replacement_rollout_diagnostic(
    env,
    replay: D4RLReplayBuffer,
    iql: IQLAgent,
    bc: BCAgent,
    fqe: FQEAgent,
    cfg: RolloutDiagnosticConfig,
    device: torch.device,
    seed: int = 0,
) -> dict[str, list[dict]]:
    """Validate FQE pairwise labels with short simulator rollouts.

    For each selected candidate action a*, compare a short-horizon rollout from
    the same approximate MuJoCo state after taking either the dataset action a0 or
    a*. Both branches then follow the same continuation policy.
    """
    sel_cfg = CandidateSelectionConfig(
        candidate_ms=cfg.candidate_ms,
        n_eval_states=cfg.n_eval_states,
        batch_size=cfg.batch_size,
        source=cfg.source,
        perturb_std=cfg.perturb_std,
        rase_lambda_support=cfg.rase_lambda_support,
    )
    selected = collect_selected_candidates(replay, iql, bc, fqe, sel_cfg, device, include_raw_obs=True)
    cont = iql.actor if cfg.continuation_policy == "iql" else bc.policy

    pair_rows: list[dict] = []
    summary_rows: list[dict] = []
    for M, data in selected.items():
        chosen = _pick_pairs(data, int(cfg.max_pairs_per_m), bool(cfg.only_pred_positive), seed + int(M))
        if chosen.size == 0:
            summary_rows.append({
                "M": int(M), "source": cfg.source, "n_pairs": 0,
                "rollout_adv_mean": float("nan"), "fqe_adv_mean": float("nan"),
                "pred_adv_mean": float("nan"), "rollout_fpi_rate": float("nan"),
                "fqe_rollout_corr": float("nan"), "pred_rollout_corr": float("nan"),
            })
            continue

        rollout_adv = []
        fqe_adv = []
        pred_adv = []
        for local_i, j in enumerate(chosen):
            raw_obs = data["raw_obs"][j]
            base_action = data["base_action"][j]
            cand_action = data["selected_action"][j]
            base_returns = []
            cand_returns = []
            for rep in range(int(cfg.rollout_repeats)):
                base_returns.append(rollout_first_action_then_policy(
                    env, replay, raw_obs, base_action, cont, device,
                    int(cfg.rollout_horizon), float(cfg.gamma), bool(cfg.deterministic_continuation),
                    seed=seed + 100000 * int(M) + 1000 * local_i + rep,
                ))
                cand_returns.append(rollout_first_action_then_policy(
                    env, replay, raw_obs, cand_action, cont, device,
                    int(cfg.rollout_horizon), float(cfg.gamma), bool(cfg.deterministic_continuation),
                    seed=seed + 200000 * int(M) + 1000 * local_i + rep,
                ))
            base_mean = float(np.mean(base_returns))
            cand_mean = float(np.mean(cand_returns))
            ra = cand_mean - base_mean
            fa = float(data["fqe_pair_gap"][j])
            pa = float(data["pred_pair_gap"][j])
            rollout_adv.append(ra)
            fqe_adv.append(fa)
            pred_adv.append(pa)
            pair_rows.append({
                "M": int(M),
                "source": cfg.source,
                "index": int(data["indices"][j]),
                "pred_pair_gap": pa,
                "fqe_pair_gap": fa,
                "rollout_adv": ra,
                "rollout_base_return": base_mean,
                "rollout_candidate_return": cand_mean,
                "support_nll": float(data["support_nll"][j]),
                "rase_score": float(data["rase_score"][j]),
                "iql_q_disagreement": float(data["iql_q_disagreement"][j]),
                "fqe_disagreement": float(data["fqe_disagreement"][j]),
                "pred_positive": int(pa > 0.0),
                "fqe_positive": int(fa > 0.0),
                "rollout_positive": int(ra > 0.0),
                "fpi_rollout": int((pa > 0.0) and (ra <= 0.0)),
                "fpi_fqe": int((pa > 0.0) and (fa <= 0.0)),
            })

        rollout_adv_np = np.asarray(rollout_adv)
        fqe_adv_np = np.asarray(fqe_adv)
        pred_adv_np = np.asarray(pred_adv)
        pred_pos = pred_adv_np > 0.0
        rollout_fpi = np.logical_and(pred_pos, rollout_adv_np <= 0.0)
        summary_rows.append({
            "M": int(M),
            "source": cfg.source,
            "n_pairs": int(len(rollout_adv_np)),
            "rollout_adv_mean": float(rollout_adv_np.mean()),
            "rollout_adv_std": float(rollout_adv_np.std(ddof=1)) if len(rollout_adv_np) > 1 else 0.0,
            "fqe_adv_mean": float(fqe_adv_np.mean()),
            "pred_adv_mean": float(pred_adv_np.mean()),
            "pred_rollout_gap_mean": float((pred_adv_np - rollout_adv_np).mean()),
            "fqe_rollout_gap_mean": float((fqe_adv_np - rollout_adv_np).mean()),
            "rollout_fpi_rate": float(rollout_fpi.sum() / max(int(pred_pos.sum()), 1)),
            "rollout_positive_rate": float((rollout_adv_np > 0.0).mean()),
            "fqe_rollout_corr": safe_corr(fqe_adv_np, rollout_adv_np),
            "pred_rollout_corr": safe_corr(pred_adv_np, rollout_adv_np),
        })

    return {"pairs": pair_rows, "summary": summary_rows}
