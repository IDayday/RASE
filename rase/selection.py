from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Union

import numpy as np
import torch

from .bc import BCAgent
from .fqe import FQEAgent
from .iql import IQLAgent
from .replay import D4RLReplayBuffer, D4RLReplayView
from .utils import torchify

ReplayLike = Union[D4RLReplayBuffer, D4RLReplayView]


@dataclass
class CandidateSelectionConfig:
    candidate_ms: Iterable[int] = (1, 4, 16, 64, 256)
    n_eval_states: int = 4096
    batch_size: int = 512
    source: str = "bc"  # bc | iql | random | perturb
    perturb_std: float = 0.1
    rase_lambda_support: float = 0.05
    use_deterministic_candidates: bool = False


def sample_candidates(
    source: str,
    obs: torch.Tensor,
    base_act: torch.Tensor,
    M: int,
    iql: IQLAgent,
    bc: BCAgent,
    perturb_std: float,
    deterministic: bool = False,
) -> torch.Tensor:
    """Vectorized candidate sampler returning [B, M, act_dim]."""
    B, act_dim = base_act.shape
    if source == "bc":
        return bc.policy.sample_n(obs, M, deterministic=deterministic)
    if source == "iql":
        return iql.actor.sample_n(obs, M, deterministic=deterministic)
    if source == "random":
        return torch.empty(B, M, act_dim, device=obs.device).uniform_(-1.0, 1.0)
    if source == "perturb":
        base = base_act[:, None, :].expand(B, M, act_dim)
        return torch.clamp(base + perturb_std * torch.randn_like(base), -1.0, 1.0)
    raise ValueError(f"Unknown candidate source: {source}")


@torch.no_grad()
def collect_selected_candidates(
    replay: ReplayLike,
    iql: IQLAgent,
    bc: BCAgent,
    fqe: FQEAgent,
    cfg: CandidateSelectionConfig,
    device: torch.device,
    indices: np.ndarray | None = None,
    include_raw_obs: bool = False,
) -> Dict[int, Dict[str, np.ndarray]]:
    """Collect selected actions and diagnostic arrays for each nested M.

    Smaller candidate pools use prefixes of the same max-M sample, so M-sweep
    comparisons isolate the selection effect rather than independent sampling
    noise. Returned arrays are keyed by candidate pool size.
    """
    candidate_ms = sorted({int(m) for m in cfg.candidate_ms})
    if not candidate_ms or candidate_ms[0] <= 0:
        raise ValueError(f"candidate_ms must contain positive integers, got {cfg.candidate_ms}")
    max_m = candidate_ms[-1]
    if indices is None:
        indices = replay.sample_indices(int(cfg.n_eval_states))
    else:
        indices = np.asarray(indices, dtype=np.int64)

    fields = [
        "indices",
        "obs",
        "base_action",
        "selected_action",
        "pred_q",
        "pred_pair_gap",
        "iql_adv_vs_v",
        "iql_q_disagreement",
        "fqe_pair_gap",
        "fqe_disagreement",
        "support_nll",
        "action_l2_to_data",
        "rase_score",
        "pred_positive",
        "fqe_positive",
        "fpi_fqe",
    ]
    if include_raw_obs:
        fields.append("raw_obs")
    out: dict[int, dict[str, list[np.ndarray]]] = {M: {k: [] for k in fields} for M in candidate_ms}

    for start in range(0, len(indices), int(cfg.batch_size)):
        batch_idx = indices[start : start + int(cfg.batch_size)]
        b_np = replay.batch_by_indices(batch_idx)
        obs = torchify(b_np["observations"], device)
        base_act = torchify(b_np["actions"], device)
        B = obs.shape[0]

        raw_obs_np = None
        if include_raw_obs and hasattr(replay, "raw_batch_by_indices"):
            raw_obs_np = replay.raw_batch_by_indices(batch_idx)["raw_observations"]

        base_iql_q, base_iql_v, _ = iql.score(obs, base_act)
        base_iql_q = base_iql_q.squeeze(1)
        base_iql_v = base_iql_v.squeeze(1)
        base_fqe = fqe.value(obs, base_act).squeeze(1)

        cand = sample_candidates(
            cfg.source,
            obs,
            base_act,
            max_m,
            iql,
            bc,
            cfg.perturb_std,
            deterministic=bool(cfg.use_deterministic_candidates),
        )
        obs_rep = obs[:, None, :].expand(B, max_m, obs.shape[-1]).reshape(B * max_m, obs.shape[-1])
        cand_flat = cand.reshape(B * max_m, -1)

        q_flat, _, adv_flat = iql.score(obs_rep, cand_flat)
        q_all = q_flat.reshape(B, max_m)
        adv_vs_v_all = adv_flat.reshape(B, max_m)
        q_stack_flat = iql.q.q_stack(obs_rep, cand_flat)
        iql_dis_all = torch.abs(q_stack_flat[:, 0] - q_stack_flat[:, 1]).reshape(B, max_m)
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
            best_iql_dis = iql_dis_all[:, :M][row_idx, best]

            cand_fqe = fqe.value(obs, best_act).squeeze(1)
            fqe_pair_gap = cand_fqe - base_fqe
            fqe_disagreement = fqe.disagreement(obs, best_act).squeeze(1)
            support_nll = -bc.policy.log_prob(obs, best_act).squeeze(1)
            action_l2 = torch.linalg.norm(best_act - base_act, dim=1)
            rase_score = best_pair_gap - float(cfg.rase_lambda_support) * support_nll

            pred_pos = best_pair_gap > 0.0
            fqe_pos = fqe_pair_gap > 0.0
            fpi = torch.logical_and(pred_pos, ~fqe_pos)

            out[M]["indices"].append(np.asarray(b_np["indices"], dtype=np.int64))
            out[M]["obs"].append(obs.cpu().numpy())
            out[M]["base_action"].append(base_act.cpu().numpy())
            out[M]["selected_action"].append(best_act.cpu().numpy())
            out[M]["pred_q"].append(best_q.cpu().numpy())
            out[M]["pred_pair_gap"].append(best_pair_gap.cpu().numpy())
            out[M]["iql_adv_vs_v"].append(best_adv_vs_v.cpu().numpy())
            out[M]["iql_q_disagreement"].append(best_iql_dis.cpu().numpy())
            out[M]["fqe_pair_gap"].append(fqe_pair_gap.cpu().numpy())
            out[M]["fqe_disagreement"].append(fqe_disagreement.cpu().numpy())
            out[M]["support_nll"].append(support_nll.cpu().numpy())
            out[M]["action_l2_to_data"].append(action_l2.cpu().numpy())
            out[M]["rase_score"].append(rase_score.cpu().numpy())
            out[M]["pred_positive"].append(pred_pos.cpu().numpy().astype(np.float32))
            out[M]["fqe_positive"].append(fqe_pos.cpu().numpy().astype(np.float32))
            out[M]["fpi_fqe"].append(fpi.cpu().numpy().astype(np.float32))
            if include_raw_obs:
                if raw_obs_np is None:
                    raw_obs_np = replay.obs_normalizer.denormalize(obs.cpu().numpy())
                out[M]["raw_obs"].append(raw_obs_np.astype(np.float32))

    return {M: {k: np.concatenate(v, axis=0) for k, v in vals.items()} for M, vals in out.items()}


def selected_dict_to_rows(selected: Dict[int, Dict[str, np.ndarray]], source: str, max_action_dims: int = 0) -> list[dict]:
    """Convert selected candidate arrays to compact CSV rows.

    Action vectors are omitted by default to keep CSVs small. Set max_action_dims
    if detailed action columns are needed for debugging. Any 1-D numeric array of
    length n is automatically emitted, so proxy modules can add fields without
    rewriting this converter.
    """
    rows: list[dict] = []
    skip = {"obs", "raw_obs", "base_action", "selected_action"}
    int_fields = {"indices", "pred_positive", "fqe_positive", "fpi_fqe"}
    for M, data in selected.items():
        n = len(data["indices"])
        one_dim_fields = []
        for key, arr in data.items():
            if key in skip:
                continue
            arr = np.asarray(arr)
            if arr.ndim == 1 and len(arr) == n:
                one_dim_fields.append(key)
        for i in range(n):
            row = {"M": int(M), "source": source}
            for key in one_dim_fields:
                out_key = "index" if key == "indices" else key
                val = data[key][i]
                row[out_key] = int(val) if key in int_fields else float(val)
            if max_action_dims > 0:
                for j, x in enumerate(data["selected_action"][i][:max_action_dims]):
                    row[f"selected_action_{j}"] = float(x)
                for j, x in enumerate(data["base_action"][i][:max_action_dims]):
                    row[f"base_action_{j}"] = float(x)
            rows.append(row)
    return rows
