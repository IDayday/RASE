from __future__ import annotations

import argparse
from pathlib import Path

from rase.fqe import FQEAgent, FQEConfig
from rase.iql import IQLAgent, IQLConfig
from rase.io_utils import apply_policy_squash, ensure_fqe_checkpoint, require_checkpoint, resolve_policy_squash
from rase.replay import load_d4rl
from rase.utils import get_device, load_yaml, set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh or create audited TwinQ/min-Q FQE checkpoints.")
    parser.add_argument("--config", type=str, default="configs/phase0_d4rl.yaml")
    parser.add_argument("--env_name", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--fqe_steps", type=int, default=None)
    parser.add_argument("--policy_squash", type=str, default="auto", choices=["auto", "tanh", "clip"])
    parser.add_argument("--force_retrain_fqe", action="store_true")
    parser.add_argument("--no_auto_retrain_fqe", action="store_true")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    cfg["env_name"] = args.env_name
    cfg["seed"] = int(args.seed)
    cfg["device"] = args.device
    if args.out_dir is not None:
        cfg["out_dir"] = args.out_dir
    if args.fqe_steps is not None:
        cfg["fqe_steps"] = int(args.fqe_steps)

    set_seed(int(cfg["seed"]))
    device = get_device(cfg["device"])
    run_dir = Path(cfg["out_dir"]) / cfg["env_name"] / f"seed{cfg['seed']}"
    policy_squash = resolve_policy_squash(cfg, run_dir, requested=args.policy_squash, ignore_existing=False)
    cfg = apply_policy_squash(cfg, policy_squash)
    print(f"[policy] policy_squash={policy_squash}")
    print(f"[FQE refresh] env={cfg['env_name']} seed={cfg['seed']} run_dir={run_dir}")

    print(f"Loading D4RL environment: {cfg['env_name']}")
    env, replay = load_d4rl(cfg["env_name"], normalize_obs=bool(cfg.get("normalize_obs", True)))
    iql = IQLAgent(replay.obs_dim, replay.act_dim, IQLConfig(**cfg["iql"]), device)
    iql.load(require_checkpoint(run_dir / "iql.pt"))
    fqe = FQEAgent(replay.obs_dim, replay.act_dim, FQEConfig(**cfg["fqe"]), ref_policy=iql.actor, device=device)
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
    print("[FQE refresh] done")
    try:
        env.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
