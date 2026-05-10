from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List

import torch
from tqdm import trange


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
