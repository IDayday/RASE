from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from tqdm import trange

from rase.bc import BCAgent, BCConfig
from rase.candidate import SweepConfig, run_candidate_sweep
from rase.fqe import FQEAgent, FQEConfig
from rase.iql import IQLAgent, IQLConfig
from rase.plotting import plot_risk_coverage, plot_sweep
from rase.replay import load_d4rl
from rase.utils import ensure_dir, get_device, load_yaml, save_json, set_seed


def write_csv(rows: List[Dict[str, float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def train_loop(agent, replay, steps: int, batch_size: int, log_every: int, name: str) -> None:
    pbar = trange(steps, desc=name, dynamic_ncols=True)
    last = {}
    for step in pbar:
        batch_np = replay.sample(batch_size)
        batch = agent.batch_to_torch(batch_np)
        metrics = agent.update(batch)
        if step % log_every == 0:
            last = metrics
            pbar.set_postfix({k.split("/")[-1]: f"{v:.3g}" for k, v in metrics.items()})
    if last:
        print(f"[{name}] final metrics:", last)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/phase0_d4rl.yaml")
    parser.add_argument("--env_name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--iql_steps", type=int, default=None)
    parser.add_argument("--bc_steps", type=int, default=None)
    parser.add_argument("--fqe_steps", type=int, default=None)
    parser.add_argument("--n_eval_states", type=int, default=None)
    parser.add_argument("--candidate_source", type=str, default=None, choices=[None, "bc", "iql", "random", "perturb"])
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    for k in ["env_name", "seed", "device", "out_dir", "iql_steps", "bc_steps", "fqe_steps", "n_eval_states"]:
        v = getattr(args, k)
        if v is not None:
            cfg[k] = v
    if args.candidate_source is not None:
        cfg["candidate_source"] = args.candidate_source

    set_seed(int(cfg["seed"]))
    device = get_device(cfg["device"])
    out_dir = ensure_dir(Path(cfg["out_dir"]) / cfg["env_name"] / f"seed{cfg['seed']}")
    save_json(cfg, out_dir / "resolved_config.json")

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

    # FQE evaluates a candidate first action followed by the IQL policy.
    fqe = FQEAgent(obs_dim, act_dim, FQEConfig(**cfg["fqe"]), ref_policy=iql.actor, device=device)
    fqe_ckpt = out_dir / "fqe_iql_ref.pt"
    if fqe_ckpt.exists() and not cfg.get("force_retrain", False):
        print(f"Loading FQE checkpoint: {fqe_ckpt}")
        fqe.load(fqe_ckpt)
    else:
        train_loop(fqe, replay, int(cfg["fqe_steps"]), int(cfg["batch_size"]), int(cfg["log_every"]), "FQE")
        fqe.save(fqe_ckpt)

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
