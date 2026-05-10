from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

from rase.bc import BCAgent, BCConfig
from rase.candidate import SweepConfig, run_candidate_sweep
from rase.fqe import FQEAgent, FQEConfig
from rase.iql import IQLAgent, IQLConfig
from rase.io_utils import apply_policy_squash, ensure_fqe_checkpoint, write_csv, train_loop, resolve_policy_squash
from rase.plotting import plot_risk_coverage, plot_sweep
from rase.replay import load_d4rl
from rase.utils import ensure_dir, get_device, load_yaml, save_json, set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="RASE Phase-0 candidate-pool sweep.")
    parser.add_argument("--config", type=str, default="configs/phase0_d4rl.yaml")
    parser.add_argument("--env_name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--iql_steps", type=int, default=None)
    parser.add_argument("--bc_steps", type=int, default=None)
    parser.add_argument("--fqe_steps", type=int, default=None)
    parser.add_argument("--n_eval_states", type=int, default=None)
    parser.add_argument("--candidate_source", type=str, default=None, choices=["bc", "iql", "random", "perturb"])
    parser.add_argument("--policy_squash", type=str, default="auto", choices=["auto", "tanh", "clip"],
                        help="auto preserves legacy outputs/rase_phase0 checkpoints; tanh is the audited default for new runs.")
    parser.add_argument("--force_retrain", action="store_true", help="Ignore cached checkpoints and retrain all modules.")
    parser.add_argument("--force_retrain_fqe", action="store_true", help="Retrain only FQE even if IQL/BC checkpoints are loaded.")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    for k in ["env_name", "seed", "device", "out_dir", "iql_steps", "bc_steps", "fqe_steps", "n_eval_states"]:
        v = getattr(args, k)
        if v is not None:
            cfg[k] = v
    if args.candidate_source is not None:
        cfg["candidate_source"] = args.candidate_source
    if args.force_retrain:
        cfg["force_retrain"] = True

    set_seed(int(cfg["seed"]))
    device = get_device(cfg["device"])
    out_dir = ensure_dir(Path(cfg["out_dir"]) / cfg["env_name"] / f"seed{cfg['seed']}")
    policy_squash = resolve_policy_squash(cfg, out_dir, requested=args.policy_squash, ignore_existing=bool(cfg.get("force_retrain", False)))
    cfg = apply_policy_squash(cfg, policy_squash)
    save_json(cfg, out_dir / "resolved_config.json")
    print(f"[policy] policy_squash={policy_squash}")

    print(f"Loading D4RL environment: {cfg['env_name']}")
    env, replay = load_d4rl(cfg["env_name"], normalize_obs=bool(cfg.get("normalize_obs", True)))
    obs_dim, act_dim = replay.obs_dim, replay.act_dim
    print(f"Dataset size={replay.size}, obs_dim={obs_dim}, act_dim={act_dim}")

    iql = IQLAgent(obs_dim, act_dim, IQLConfig(**cfg["iql"]), device)
    bc = BCAgent(obs_dim, act_dim, BCConfig(**cfg["bc"]), device)

    iql_ckpt = out_dir / "iql.pt"
    bc_ckpt = out_dir / "bc.pt"
    if iql_ckpt.exists() and not cfg.get("force_retrain", False):
        print(f"Loading IQL checkpoint: {iql_ckpt}")
        iql.load(iql_ckpt)
    else:
        train_loop(iql, replay, int(cfg["iql_steps"]), int(cfg["batch_size"]), int(cfg["log_every"]), "IQL")
        iql.save(iql_ckpt)

    if bc_ckpt.exists() and not cfg.get("force_retrain", False):
        print(f"Loading BC checkpoint: {bc_ckpt}")
        bc.load(bc_ckpt)
    else:
        train_loop(bc, replay, int(cfg["bc_steps"]), int(cfg["batch_size"]), int(cfg["log_every"]), "BC")
        bc.save(bc_ckpt)

    fqe = FQEAgent(obs_dim, act_dim, FQEConfig(**cfg["fqe"]), ref_policy=iql.actor, device=device)
    fqe_ckpt = out_dir / "fqe_iql_ref.pt"
    ensure_fqe_checkpoint(
        fqe,
        replay,
        fqe_ckpt,
        steps=int(cfg["fqe_steps"]),
        batch_size=int(cfg["batch_size"]),
        log_every=int(cfg["log_every"]),
        force_retrain=bool(cfg.get("force_retrain", False) or args.force_retrain_fqe),
        auto_retrain_incompatible=True,
    )

    sweep_cfg = SweepConfig(
        candidate_ms=cfg["candidate_ms"],
        n_eval_states=int(cfg["n_eval_states"]),
        batch_size=int(cfg["eval_batch_size"]),
        source=cfg["candidate_source"],
        perturb_std=float(cfg["perturb_std"]),
        rase_lambda_support=float(cfg["rase_lambda_support"]),
        thresholds=cfg["thresholds"],
    )
    result = run_candidate_sweep(replay, iql, bc, fqe, sweep_cfg, device)
    sweep_csv = out_dir / f"sweep_{sweep_cfg.source}.csv"
    risk_csv = out_dir / f"risk_coverage_{sweep_cfg.source}.csv"
    write_csv(result["sweep"], sweep_csv)
    write_csv(result["risk_coverage"], risk_csv)
    plot_sweep(sweep_csv, out_dir / "plots")
    plot_risk_coverage(risk_csv, out_dir / "plots")
    print(f"Saved outputs to {out_dir}")
    try:
        env.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
