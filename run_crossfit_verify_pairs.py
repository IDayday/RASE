from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch

from rase.crossfit import make_kfold_indices
from rase.iql import IQLAgent, IQLConfig
from rase.io_utils import apply_policy_squash, require_checkpoint, resolve_policy_squash, train_loop, write_csv
from rase.metrics import auprc_from_scores, auroc_from_scores, precision_at_coverage
from rase.replay import load_d4rl
from rase.utils import get_device, load_yaml, save_json, set_seed, torchify


def _parse_sources(text: str) -> list[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def _action_columns(df: pd.DataFrame) -> list[str]:
    if "action_dim" in df.columns:
        dim = int(np.nanmax(df["action_dim"].to_numpy()))
        cols = [f"selected_action_{i}" for i in range(dim)]
    else:
        cols = sorted([c for c in df.columns if c.startswith("selected_action_")], key=lambda x: int(x.rsplit("_", 1)[1]))
    missing = [c for c in cols if c not in df.columns]
    if missing or not cols:
        raise RuntimeError(
            "Rollout pair CSV does not contain saved selected actions. Re-run run_rollout_diagnostic.py "
            "with --save_action_dims -1 before crossfit verification."
        )
    return cols


@torch.no_grad()
def _score_fold_pairs(
    fold_iql: IQLAgent,
    replay,
    df_fold: pd.DataFrame,
    action_cols: list[str],
    device: torch.device,
    batch_size: int,
) -> dict[str, np.ndarray]:
    n = len(df_fold)
    out = {
        "crossfit_pair_gap": np.empty(n, dtype=np.float32),
        "crossfit_selected_q": np.empty(n, dtype=np.float32),
        "crossfit_base_q": np.empty(n, dtype=np.float32),
        "crossfit_selected_adv_vs_v": np.empty(n, dtype=np.float32),
        "crossfit_q_disagreement": np.empty(n, dtype=np.float32),
    }
    indices = df_fold["index"].to_numpy(dtype=np.int64)
    actions = df_fold[action_cols].to_numpy(dtype=np.float32)
    for start in range(0, n, int(batch_size)):
        sl = slice(start, min(start + int(batch_size), n))
        b = replay.batch_by_indices(indices[sl])
        obs = torchify(b["observations"], device)
        base_act = torchify(b["actions"], device)
        cand_act = torch.as_tensor(actions[sl], dtype=torch.float32, device=device)

        q_c, v, adv_c = fold_iql.score(obs, cand_act)
        q_b, _, _ = fold_iql.score(obs, base_act)
        q_stack = fold_iql.q.q_stack(obs, cand_act)
        dis = torch.abs(q_stack[:, 0] - q_stack[:, 1])
        out["crossfit_pair_gap"][sl] = (q_c - q_b).squeeze(1).cpu().numpy()
        out["crossfit_selected_q"][sl] = q_c.squeeze(1).cpu().numpy()
        out["crossfit_base_q"][sl] = q_b.squeeze(1).cpu().numpy()
        out["crossfit_selected_adv_vs_v"][sl] = adv_c.squeeze(1).cpu().numpy()
        out["crossfit_q_disagreement"][sl] = dis.cpu().numpy()
    return out


def _summary_rows(df: pd.DataFrame, source: str, coverages: Iterable[float]) -> list[dict]:
    rows = []
    score_names = [
        "pred_pair_gap",
        "fqe_pair_gap",
        "rase_score",
        "rase_score_v2",
        "crossfit_pair_gap",
        "min_pred_crossfit_gap",
        "neg_crossfit_q_disagreement",
    ]
    if "knn_sa_distance" in df.columns:
        score_names.append("neg_knn_sa_distance")
    if "support_nll" in df.columns:
        score_names.append("neg_support_nll")
    labels = df["rollout_positive"].astype(bool).to_numpy()
    for M, g in df.groupby("M"):
        labels_g = g["rollout_positive"].astype(bool).to_numpy()
        for score in score_names:
            if score.startswith("neg_"):
                raw = score[4:]
                if raw not in g.columns:
                    continue
                scores = -g[raw].to_numpy(dtype=float)
            else:
                if score not in g.columns:
                    continue
                scores = g[score].to_numpy(dtype=float)
            row = {
                "source": source,
                "M": int(M),
                "score": score,
                "n": int(len(g)),
                "rollout_positive_rate": float(labels_g.mean()) if len(labels_g) else float("nan"),
                "auroc_rollout_positive": auroc_from_scores(labels_g, scores),
                "auprc_rollout_positive": auprc_from_scores(labels_g, scores),
            }
            for cov in coverages:
                row[f"precision_at_cov_{cov:g}"] = precision_at_coverage(labels_g, scores, float(cov))
            rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Score rollout-labeled candidate pairs with cross-fitted IQL verifier critics.")
    parser.add_argument("--config", type=str, default="configs/phase0_d4rl.yaml")
    parser.add_argument("--env_name", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--sources", type=str, default="bc,iql", help="Comma-separated sources to verify.")
    parser.add_argument("--continuation_policy", type=str, default="iql")
    parser.add_argument("--num_folds", type=int, default=None)
    parser.add_argument("--fold_iql_steps", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--policy_squash", type=str, default="auto", choices=["auto", "tanh", "clip"])
    parser.add_argument("--force_retrain_folds", action="store_true")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    cfg["env_name"] = args.env_name
    cfg["seed"] = int(args.seed)
    cfg["device"] = args.device
    if args.out_dir is not None:
        cfg["out_dir"] = args.out_dir
    set_seed(int(cfg["seed"]))
    device = get_device(cfg["device"])
    run_dir = Path(cfg["out_dir"]) / cfg["env_name"] / f"seed{cfg['seed']}"
    policy_squash = resolve_policy_squash(cfg, run_dir, requested=args.policy_squash, ignore_existing=False)
    cfg = apply_policy_squash(cfg, policy_squash)
    print(f"[policy] policy_squash={policy_squash}")
    print(f"[crossfit verify] env={cfg['env_name']} seed={cfg['seed']} run_dir={run_dir}")

    cf_raw = dict(cfg.get("crossfit", {}) or {})
    if args.num_folds is not None:
        cf_raw["num_folds"] = int(args.num_folds)
    if args.fold_iql_steps is not None:
        cf_raw["fold_iql_steps"] = int(args.fold_iql_steps)
    num_folds = int(cf_raw.get("num_folds", 3))
    fold_iql_steps = int(cf_raw.get("fold_iql_steps", 100000))
    batch_size = int(args.batch_size or cfg.get("eval_batch_size", 512))

    print(f"Loading D4RL environment: {cfg['env_name']}")
    env, replay = load_d4rl(cfg["env_name"], normalize_obs=bool(cfg.get("normalize_obs", True)))
    diag_dir = run_dir / "diagnostics"
    cross_dir = run_dir / "crossfit"
    cross_dir.mkdir(parents=True, exist_ok=True)
    save_json({"cfg": cfg, "num_folds": num_folds, "fold_iql_steps": fold_iql_steps}, cross_dir / "crossfit_verify_config.json")

    folds = make_kfold_indices(replay.size, num_folds, int(cfg["seed"]))
    fold_id_for_index = np.full(replay.size, -1, dtype=np.int16)
    fold_agents: list[IQLAgent] = []
    for fold_id, (train_idx, valid_idx) in enumerate(folds):
        fold_id_for_index[valid_idx] = int(fold_id)
        fold_dir = cross_dir / f"fold{fold_id}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        ckpt = fold_dir / "iql_fold.pt"
        agent = IQLAgent(replay.obs_dim, replay.act_dim, IQLConfig(**cfg["iql"]), device)
        if ckpt.exists() and not args.force_retrain_folds:
            print(f"Loading fold {fold_id} IQL checkpoint: {ckpt}")
            agent.load(ckpt)
        else:
            print(f"Training fold {fold_id}: train={len(train_idx)} valid={len(valid_idx)} steps={fold_iql_steps}")
            train_view = replay.view(train_idx)
            train_loop(agent, train_view, fold_iql_steps, int(cfg["batch_size"]), int(cfg["log_every"]), f"IQL_fold{fold_id}")
            agent.save(ckpt)
        fold_agents.append(agent)

    all_summary = []
    for source in _parse_sources(args.sources):
        in_path = diag_dir / f"rollout_pairs_{source}_{args.continuation_policy}.csv"
        if not in_path.exists():
            raise FileNotFoundError(f"Missing rollout pair file: {in_path}. Run certificate rollout first.")
        df = pd.read_csv(in_path)
        if "rollout_positive" not in df.columns:
            raise RuntimeError(f"{in_path} is missing rollout_positive labels.")
        action_cols = _action_columns(df)
        indices = df["index"].to_numpy(dtype=np.int64)
        df["crossfit_fold"] = fold_id_for_index[indices].astype(int)
        if (df["crossfit_fold"] < 0).any():
            raise RuntimeError("Some rollout pair indices were not assigned to a crossfit fold.")

        for col in ["crossfit_pair_gap", "crossfit_selected_q", "crossfit_base_q", "crossfit_selected_adv_vs_v", "crossfit_q_disagreement"]:
            df[col] = np.nan
        for fold_id, agent in enumerate(fold_agents):
            mask = df["crossfit_fold"].to_numpy(dtype=int) == fold_id
            if not mask.any():
                continue
            sub = df.loc[mask].copy()
            scored = _score_fold_pairs(agent, replay, sub, action_cols, device, batch_size=batch_size)
            for col, vals in scored.items():
                df.loc[mask, col] = vals

        df["min_pred_crossfit_gap"] = np.minimum(df["pred_pair_gap"].astype(float), df["crossfit_pair_gap"].astype(float))
        df["crossfit_positive"] = (df["crossfit_pair_gap"] > 0.0).astype(int)
        df["pred_and_crossfit_positive"] = ((df["pred_pair_gap"] > 0.0) & (df["crossfit_pair_gap"] > 0.0)).astype(int)
        out_path = diag_dir / f"crossfit_verified_pairs_{source}_{args.continuation_policy}.csv"
        df.to_csv(out_path, index=False)
        print(f"Saved crossfit-verified pairs: {out_path}")
        rows = _summary_rows(df, source, coverages=[0.2, 0.3, 0.5, 0.7])
        for r in rows:
            r["env"] = cfg["env_name"]
            r["seed"] = int(cfg["seed"])
        write_csv(rows, diag_dir / f"crossfit_verification_summary_{source}_{args.continuation_policy}.csv")
        all_summary.extend(rows)

    write_csv(all_summary, cross_dir / f"crossfit_verification_summary_{args.continuation_policy}.csv")
    try:
        env.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
