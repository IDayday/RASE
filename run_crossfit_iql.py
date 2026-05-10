from __future__ import annotations

import argparse
from pathlib import Path

from rase.bc import BCAgent, BCConfig
from rase.candidate import SweepConfig, run_candidate_sweep
from rase.crossfit import CrossFitConfig, make_kfold_indices
from rase.fqe import FQEAgent, FQEConfig
from rase.iql import IQLAgent, IQLConfig
from rase.io_utils import apply_policy_squash, ensure_fqe_checkpoint, train_loop, write_csv, require_checkpoint, resolve_policy_squash
from rase.replay import load_d4rl
from rase.utils import get_device, load_yaml, save_json, set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="RASE Phase-1 cross-fitted IQL selection diagnostic.")
    parser.add_argument("--config", type=str, default="configs/phase0_d4rl.yaml")
    parser.add_argument("--env_name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--candidate_source", type=str, default=None, choices=["bc", "iql", "random", "perturb"])
    parser.add_argument("--num_folds", type=int, default=None)
    parser.add_argument("--fold_iql_steps", type=int, default=None)
    parser.add_argument("--heldout_eval_states", type=int, default=None)
    parser.add_argument("--fqe_steps", type=int, default=None)
    parser.add_argument("--policy_squash", type=str, default="auto", choices=["auto", "tanh", "clip"])
    parser.add_argument("--force_retrain_fqe", action="store_true")
    parser.add_argument("--no_auto_retrain_fqe", action="store_true")
    parser.add_argument("--force_retrain_folds", action="store_true")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    for k in ["env_name", "seed", "device", "out_dir", "fqe_steps"]:
        v = getattr(args, k)
        if v is not None:
            cfg[k] = v
    if args.candidate_source is not None:
        cfg["candidate_source"] = args.candidate_source
    set_seed(int(cfg["seed"]))
    device = get_device(cfg["device"])

    run_dir = Path(cfg["out_dir"]) / cfg["env_name"] / f"seed{cfg['seed']}"
    policy_squash = resolve_policy_squash(cfg, run_dir, requested=args.policy_squash, ignore_existing=False)
    cfg = apply_policy_squash(cfg, policy_squash)
    print(f"[policy] policy_squash={policy_squash}")

    cf_raw = dict(cfg.get("crossfit", {}) or {})
    if args.num_folds is not None:
        cf_raw["num_folds"] = args.num_folds
    if args.fold_iql_steps is not None:
        cf_raw["fold_iql_steps"] = args.fold_iql_steps
    if args.heldout_eval_states is not None:
        cf_raw["heldout_eval_states"] = args.heldout_eval_states
    cf_raw.setdefault("candidate_ms", cfg["candidate_ms"])
    cross_cfg = CrossFitConfig(**cf_raw)

    diag_dir = run_dir / "crossfit"
    diag_dir.mkdir(parents=True, exist_ok=True)
    cfg_to_save = dict(cfg)
    cfg_to_save["crossfit_resolved"] = cross_cfg.__dict__
    save_json(cfg_to_save, diag_dir / f"crossfit_{cfg['candidate_source']}_config.json")

    print(f"Loading D4RL environment: {cfg['env_name']}")
    env, replay = load_d4rl(cfg["env_name"], normalize_obs=bool(cfg.get("normalize_obs", True)))

    full_iql = IQLAgent(replay.obs_dim, replay.act_dim, IQLConfig(**cfg["iql"]), device)
    full_iql.load(require_checkpoint(run_dir / "iql.pt"))
    bc = BCAgent(replay.obs_dim, replay.act_dim, BCConfig(**cfg["bc"]), device)
    bc.load(require_checkpoint(run_dir / "bc.pt"))
    fqe = FQEAgent(replay.obs_dim, replay.act_dim, FQEConfig(**cfg["fqe"]), ref_policy=full_iql.actor, device=device)
    ensure_fqe_checkpoint(
        fqe,
        replay,
        run_dir / "fqe_iql_ref.pt",
        steps=int(cfg["fqe_steps"]),
        batch_size=int(cfg["batch_size"]),
        log_every=int(cfg["log_every"]),
        force_retrain=bool(args.force_retrain_fqe),
        auto_retrain_incompatible=not bool(args.no_auto_retrain_fqe),
    )

    folds = make_kfold_indices(replay.size, int(cross_cfg.num_folds), int(cfg["seed"]))
    all_rows = []
    for fold_id, (train_idx, valid_idx) in enumerate(folds):
        fold_dir = diag_dir / f"fold{fold_id}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        fold_iql = IQLAgent(replay.obs_dim, replay.act_dim, IQLConfig(**cfg["iql"]), device)
        ckpt = fold_dir / "iql_fold.pt"
        if ckpt.exists() and not args.force_retrain_folds:
            print(f"Loading fold {fold_id} IQL checkpoint: {ckpt}")
            fold_iql.load(ckpt)
        else:
            print(f"Training fold {fold_id}: train={len(train_idx)} valid={len(valid_idx)}")
            train_view = replay.view(train_idx)
            train_loop(fold_iql, train_view, int(cross_cfg.fold_iql_steps), int(cfg["batch_size"]), int(cfg["log_every"]), f"IQL_fold{fold_id}")
            fold_iql.save(ckpt)

        if len(valid_idx) > int(cross_cfg.heldout_eval_states):
            valid_eval = valid_idx[: int(cross_cfg.heldout_eval_states)]
        else:
            valid_eval = valid_idx
        proposal_iql = fold_iql if cfg["candidate_source"] == "iql" else full_iql
        sweep_cfg = SweepConfig(
            candidate_ms=cross_cfg.candidate_ms,
            n_eval_states=len(valid_eval),
            batch_size=int(cfg["eval_batch_size"]),
            source=cfg["candidate_source"],
            perturb_std=float(cfg["perturb_std"]),
            rase_lambda_support=float(cfg["rase_lambda_support"]),
            thresholds=cfg["thresholds"],
        )
        valid_view = replay.view(valid_eval)
        selector_iql = proposal_iql if cfg["candidate_source"] == "iql" else fold_iql
        out = run_candidate_sweep(valid_view, selector_iql, bc, fqe, sweep_cfg, device)
        for row in out["sweep"]:
            row["fold"] = fold_id
            row["n_train"] = int(len(train_idx))
            row["n_valid"] = int(len(valid_eval))
            all_rows.append(row)

    write_csv(all_rows, diag_dir / f"crossfit_sweep_{cfg['candidate_source']}.csv")
    print(f"Saved cross-fit diagnostics to {diag_dir}")
    try:
        env.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
