from __future__ import annotations

import csv
import json
import shutil
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional

import torch
from tqdm import trange

from .fqe import IncompatibleFQECheckpointError


def write_csv(rows: List[Dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = []
    for row in rows:
        for k in row.keys():
            if k not in fieldnames:
                fieldnames.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def train_loop(agent, replay, steps: int, batch_size: int, log_every: int, name: str) -> None:
    pbar = trange(int(steps), desc=name, dynamic_ncols=True)
    last = {}
    for step in pbar:
        batch_np = replay.sample(int(batch_size))
        batch = agent.batch_to_torch(batch_np)
        metrics = agent.update(batch)
        if step % int(log_every) == 0:
            last = metrics
            pbar.set_postfix({k.split("/")[-1]: f"{v:.3g}" for k, v in metrics.items()})
    if last:
        print(f"[{name}] final metrics:", last)


def require_checkpoint(path: str | Path) -> Path:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {path}. Run run_pipeline.py first or pass the correct --out_dir.")
    return path


def torch_load(path: str | Path, device: torch.device):
    return torch.load(path, map_location=device)


def _safe_torch_load_cpu(path: Path):
    try:
        return torch.load(path, map_location="cpu")
    except Exception:
        return None


def _read_json(path: Path) -> Dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def detect_policy_squash_from_run_dir(run_dir: str | Path) -> Optional[str]:
    """Best-effort detection for legacy checkpoints.

    The original Phase-0 checkpoints used a clipped Gaussian policy and did not
    store a policy version. v3/v4+ checkpoints should either be in a new output
    directory or store `policy_squash`. If an old single-Q FQE checkpoint is found,
    we infer `clip` so diagnostics reproduce the saved policy behavior.
    """
    run_dir = Path(run_dir)
    resolved = _read_json(run_dir / "resolved_config.json")
    for key in ["policy_squash", "policy_squash_mode"]:
        val = resolved.get(key)
        if val in {"tanh", "clip"}:
            return str(val)
    for subkey in ["iql", "bc"]:
        val = (resolved.get(subkey) or {}).get("policy_squash") if isinstance(resolved.get(subkey), dict) else None
        if val in {"tanh", "clip"}:
            return str(val)

    for ckpt_name in ["iql.pt", "bc.pt"]:
        ckpt = _safe_torch_load_cpu(run_dir / ckpt_name)
        if isinstance(ckpt, dict):
            val = ckpt.get("policy_squash") or (ckpt.get("cfg") or {}).get("policy_squash")
            if val in {"tanh", "clip"}:
                return str(val)

    fqe = _safe_torch_load_cpu(run_dir / "fqe_iql_ref.pt")
    if isinstance(fqe, dict):
        val = fqe.get("ref_policy_squash")
        if val in {"tanh", "clip"}:
            return str(val)
        # Legacy FQE has no version and a QNet key pattern q.net.* rather than q1/q2.
        if fqe.get("version") is None:
            return "clip"
    return None


def resolve_policy_squash(cfg: Dict, run_dir: str | Path, requested: str = "auto", ignore_existing: bool = False) -> str:
    """Resolve the policy squashing mode for IQL/BC agents.

    `requested="auto"` preserves old output directories by detecting legacy
    checkpoints, otherwise falls back to cfg['policy_squash'] or tanh.
    """
    if requested not in {"auto", "tanh", "clip"}:
        raise ValueError("--policy_squash must be one of: auto, tanh, clip")
    if requested in {"tanh", "clip"}:
        return requested
    if not ignore_existing:
        detected = detect_policy_squash_from_run_dir(run_dir)
        if detected is not None:
            return detected
    return str(cfg.get("policy_squash", "tanh"))


def apply_policy_squash(cfg: Dict, policy_squash: str) -> Dict:
    out = deepcopy(cfg)
    out["policy_squash"] = policy_squash
    out.setdefault("iql", {})["policy_squash"] = policy_squash
    out.setdefault("bc", {})["policy_squash"] = policy_squash
    return out


def backup_checkpoint(path: str | Path, reason: str) -> Optional[Path]:
    """Copy an incompatible checkpoint aside before overwriting it."""
    path = Path(path)
    if not path.exists():
        return None
    reason = reason.replace("/", "_").replace(" ", "_")
    backup = path.with_suffix(path.suffix + f".{reason}.bak")
    i = 1
    while backup.exists():
        backup = path.with_suffix(path.suffix + f".{reason}.bak{i}")
        i += 1
    shutil.copy2(path, backup)
    return backup


def ensure_fqe_checkpoint(
    fqe,
    replay,
    ckpt_path: str | Path,
    steps: int,
    batch_size: int,
    log_every: int,
    force_retrain: bool = False,
    auto_retrain_incompatible: bool = True,
) -> None:
    """Load a compatible FQE checkpoint or train one in-place.

    This fixes the common v2->v4/v5 failure mode where `fqe_iql_ref.pt` exists
    but is the old single-Q FQE checkpoint. In that case we back it up and train
    the current TwinQ/min-Q evaluator.
    """
    ckpt_path = Path(ckpt_path)
    if ckpt_path.exists() and not force_retrain:
        print(f"Loading FQE checkpoint: {ckpt_path}")
        try:
            fqe.load(ckpt_path)
            return
        except IncompatibleFQECheckpointError as exc:
            if not auto_retrain_incompatible:
                raise
            print(f"[FQE] {exc}")
            backup = backup_checkpoint(ckpt_path, "legacy_or_incompatible_fqe")
            if backup is not None:
                print(f"[FQE] Backed up incompatible checkpoint to: {backup}")
            print("[FQE] Retraining audited TwinQ/min-Q FQE evaluator.")
    elif force_retrain and ckpt_path.exists():
        backup = backup_checkpoint(ckpt_path, "force_retrain_fqe")
        if backup is not None:
            print(f"[FQE] Backed up old FQE checkpoint to: {backup}")

    train_loop(fqe, replay, int(steps), int(batch_size), int(log_every), "FQE")
    fqe.save(ckpt_path)
    print(f"[FQE] Saved compatible FQE checkpoint: {ckpt_path}")
