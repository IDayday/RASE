from __future__ import annotations

import argparse
from pathlib import Path

from rase.bc import BCAgent, BCConfig
from rase.fqe import FQEAgent, FQEConfig
from rase.iql import IQLAgent, IQLConfig
from rase.io_utils import apply_policy_squash, ensure_fqe_checkpoint, write_csv, require_checkpoint, resolve_policy_squash
from rase.replay import load_d4rl
from rase.rollout import RolloutDiagnosticConfig, run_action_replacement_rollout_diagnostic
from rase.utils import get_device, load_yaml, save_json, set_seed


def _parse_int_list(text: str):
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def load_agents(cfg, replay, device, run_dir: Path, force_retrain_fqe: bool, auto_retrain_fqe: bool):
    iql = IQLAgent(replay.obs_dim, replay.act_dim, IQLConfig(**cfg["iql"]), device)
    bc = BCAgent(replay.obs_dim, replay.act_dim, BCConfig(**cfg["bc"]), device)
    iql.load(require_checkpoint(run_dir / "iql.pt"))
    bc.load(require_checkpoint(run_dir / "bc.pt"))
    fqe = FQEAgent(replay.obs_dim, replay.act_dim, FQEConfig(**cfg["fqe"]), ref_policy=iql.actor, device=device)
    ensure_fqe_checkpoint(
        fqe,
        replay,
        run_dir / "fqe_iql_ref.pt",
        steps=int(cfg["fqe_steps"]),
        batch_size=int(cfg["batch_size"]),
        log_every=int(cfg["log_every"]),
        force_retrain=force_retrain_fqe,
        auto_retrain_incompatible=auto_retrain_fqe,
    )
    return iql, bc, fqe


def main() -> None:
    parser = argparse.ArgumentParser(description="RASE Phase-0.5 short-rollout action replacement diagnostic.")
    parser.add_argument("--config", type=str, default="configs/phase0_d4rl.yaml")
    parser.add_argument("--env_name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--candidate_source", type=str, default=None, choices=["bc", "iql", "random", "perturb"])
    parser.add_argument("--candidate_ms", type=str, default=None, help="Comma-separated candidate pool sizes, e.g. 16,64,256")
    parser.add_argument("--n_eval_states", type=int, default=None)
    parser.add_argument("--fqe_steps", type=int, default=None)
    parser.add_argument("--rollout_horizon", type=int, default=None)
    parser.add_argument("--rollout_repeats", type=int, default=None)
    parser.add_argument("--max_pairs_per_m", type=int, default=None)
    parser.add_argument("--continuation_policy", type=str, default=None, choices=["iql", "bc"])
    parser.add_argument("--add_proxy_features", action="store_true", help="Add kNN/support/disagreement composite features to rollout pair CSVs.")
    parser.add_argument("--knn_ref_size", type=int, default=None)
    parser.add_argument("--knn_batch_size", type=int, default=None)
    parser.add_argument("--save_action_dims", type=int, default=None, help="-1 saves all action dims; 0 saves none; positive saves prefix.")
    parser.add_argument("--policy_squash", type=str, default="auto", choices=["auto", "tanh", "clip"])
    parser.add_argument("--force_retrain_fqe", action="store_true", help="Retrain the audited TwinQ FQE evaluator before rollout diagnostics.")
    parser.add_argument("--no_auto_retrain_fqe", action="store_true", help="Do not auto-retrain if the cached FQE checkpoint is incompatible.")
    parser.add_argument("--all_pairs", action="store_true", help="Do not restrict diagnostic pairs to pred-positive selections.")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    for k in ["env_name", "seed", "device", "out_dir", "n_eval_states", "fqe_steps"]:
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

    diag_dir = run_dir / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)

    rollout_cfg_raw = dict(cfg.get("rollout_diagnostic", {}) or {})
    rollout_cfg_raw.setdefault("candidate_ms", [1, 16, 64, 256])
    rollout_cfg_raw.setdefault("n_eval_states", min(int(cfg.get("n_eval_states", 4096)), 256))
    rollout_cfg_raw.setdefault("batch_size", int(cfg.get("eval_batch_size", 512)))
    rollout_cfg_raw.setdefault("source", cfg["candidate_source"])
    rollout_cfg_raw["source"] = cfg["candidate_source"]
    rollout_cfg_raw.setdefault("perturb_std", float(cfg.get("perturb_std", 0.1)))
    rollout_cfg_raw.setdefault("rase_lambda_support", float(cfg.get("rase_lambda_support", 0.05)))
    proxy_raw = dict(cfg.get("proxy", {}) or {})
    for proxy_key in [
        "knn_ref_size", "knn_batch_size", "obs_scale", "action_scale",
        "composite_lambda_support", "composite_lambda_iql_dis",
        "composite_lambda_fqe_dis", "composite_lambda_knn",
    ]:
        if proxy_key in proxy_raw:
            rollout_cfg_raw.setdefault(proxy_key, proxy_raw[proxy_key])
    if args.candidate_ms is not None:
        rollout_cfg_raw["candidate_ms"] = _parse_int_list(args.candidate_ms)
    if args.add_proxy_features:
        rollout_cfg_raw["add_proxy_features"] = True
    if args.knn_ref_size is not None:
        rollout_cfg_raw["knn_ref_size"] = int(args.knn_ref_size)
    if args.knn_batch_size is not None:
        rollout_cfg_raw["knn_batch_size"] = int(args.knn_batch_size)
    if args.save_action_dims is not None:
        rollout_cfg_raw["save_action_dims"] = int(args.save_action_dims)
    if args.rollout_horizon is not None:
        rollout_cfg_raw["rollout_horizon"] = args.rollout_horizon
    if args.rollout_repeats is not None:
        rollout_cfg_raw["rollout_repeats"] = args.rollout_repeats
    if args.max_pairs_per_m is not None:
        rollout_cfg_raw["max_pairs_per_m"] = args.max_pairs_per_m
    if args.continuation_policy is not None:
        rollout_cfg_raw["continuation_policy"] = args.continuation_policy
    if args.all_pairs:
        rollout_cfg_raw["only_pred_positive"] = False
    if args.n_eval_states is not None:
        rollout_cfg_raw["n_eval_states"] = args.n_eval_states
    rollout_cfg = RolloutDiagnosticConfig(**rollout_cfg_raw)

    cfg_to_save = dict(cfg)
    cfg_to_save["rollout_diagnostic_resolved"] = rollout_cfg.__dict__
    save_json(cfg_to_save, diag_dir / f"rollout_{cfg['candidate_source']}_config.json")

    print(f"Loading D4RL environment: {cfg['env_name']}")
    env, replay = load_d4rl(cfg["env_name"], normalize_obs=bool(cfg.get("normalize_obs", True)))
    iql, bc, fqe = load_agents(
        cfg,
        replay,
        device,
        run_dir,
        force_retrain_fqe=bool(args.force_retrain_fqe),
        auto_retrain_fqe=not bool(args.no_auto_retrain_fqe),
    )

    out = run_action_replacement_rollout_diagnostic(env, replay, iql, bc, fqe, rollout_cfg, device, seed=int(cfg["seed"]))
    prefix = f"{cfg['candidate_source']}_{rollout_cfg.continuation_policy}"
    write_csv(out["pairs"], diag_dir / f"rollout_pairs_{prefix}.csv")
    write_csv(out["summary"], diag_dir / f"rollout_summary_{prefix}.csv")
    print(f"Saved rollout diagnostics to {diag_dir}")
    try:
        env.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
