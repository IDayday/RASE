from __future__ import annotations

import argparse
from pathlib import Path

from rase.bc import BCAgent, BCConfig
from rase.fqe import FQEAgent, FQEConfig
from rase.iql import IQLAgent, IQLConfig
from rase.io_utils import apply_policy_squash, ensure_fqe_checkpoint, write_csv, require_checkpoint, resolve_policy_squash
from rase.proxy import ProxyConfig, add_knn_proxy_to_selected, calibrated_rows, proxy_alignment_summary
from rase.replay import load_d4rl
from rase.selection import CandidateSelectionConfig, collect_selected_candidates, selected_dict_to_rows
from rase.utils import get_device, load_yaml, save_json, set_seed


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
    parser = argparse.ArgumentParser(description="RASE Phase-1 proxy alignment diagnostic.")
    parser.add_argument("--config", type=str, default="configs/phase0_d4rl.yaml")
    parser.add_argument("--env_name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--candidate_source", type=str, default=None, choices=["bc", "iql", "random", "perturb"])
    parser.add_argument("--n_eval_states", type=int, default=None)
    parser.add_argument("--fqe_steps", type=int, default=None)
    parser.add_argument("--knn_ref_size", type=int, default=None)
    parser.add_argument("--max_action_dims_csv", type=int, default=0)
    parser.add_argument("--policy_squash", type=str, default="auto", choices=["auto", "tanh", "clip"])
    parser.add_argument("--force_retrain_fqe", action="store_true")
    parser.add_argument("--no_auto_retrain_fqe", action="store_true")
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
    save_json(cfg, diag_dir / f"proxy_alignment_{cfg['candidate_source']}_config.json")

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

    sel_cfg = CandidateSelectionConfig(
        candidate_ms=cfg["candidate_ms"],
        n_eval_states=int(cfg["n_eval_states"]),
        batch_size=int(cfg["eval_batch_size"]),
        source=cfg["candidate_source"],
        perturb_std=float(cfg["perturb_std"]),
        rase_lambda_support=float(cfg["rase_lambda_support"]),
    )
    selected = collect_selected_candidates(replay, iql, bc, fqe, sel_cfg, device)

    proxy_cfg_raw = dict(cfg.get("proxy", {}) or {})
    if args.knn_ref_size is not None:
        proxy_cfg_raw["knn_ref_size"] = args.knn_ref_size
    proxy_cfg = ProxyConfig(**proxy_cfg_raw)
    selected = add_knn_proxy_to_selected(selected, replay, proxy_cfg, device, seed=int(cfg["seed"]))

    detail_rows = selected_dict_to_rows(selected, cfg["candidate_source"], max_action_dims=int(args.max_action_dims_csv))
    summary = proxy_alignment_summary(selected, cfg["candidate_source"], proxy_cfg)
    calib = calibrated_rows(selected, cfg["candidate_source"], proxy_cfg)

    prefix = f"{cfg['candidate_source']}"
    write_csv(detail_rows, diag_dir / f"proxy_details_{prefix}.csv")
    write_csv(summary, diag_dir / f"proxy_alignment_{prefix}.csv")
    write_csv(calib, diag_dir / f"calibrated_risk_coverage_{prefix}.csv")
    print(f"Saved proxy diagnostics to {diag_dir}")
    try:
        env.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
