from __future__ import annotations

import csv
from pathlib import Path
from typing import List, Dict

import matplotlib.pyplot as plt


def read_csv_rows(path: str | Path) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def plot_sweep(csv_path: str | Path, out_dir: str | Path) -> None:
    rows = read_csv_rows(csv_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    xs = [int(r["M"]) for r in rows]

    for key, ylabel in [
        ("pred_adv_mean", "Predicted advantage"),
        ("fqe_adv_mean", "FQE advantage"),
        ("pred_empirical_gap", "Predicted - FQE gap"),
        ("fpi_rate_cond_pred_positive", "Conditional FPI rate"),
    ]:
        ys = [float(r[key]) for r in rows]
        plt.figure()
        plt.plot(xs, ys, marker="o")
        plt.xscale("log", base=2)
        plt.xlabel("Candidate pool size M")
        plt.ylabel(ylabel)
        plt.title(key)
        plt.tight_layout()
        plt.savefig(out / f"{key}.png", dpi=180)
        plt.close()


def plot_risk_coverage(csv_path: str | Path, out_dir: str | Path) -> None:
    rows = read_csv_rows(csv_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    Ms = sorted({int(r["M"]) for r in rows})
    for M in Ms:
        sub = [r for r in rows if int(r["M"]) == M]
        coverage = [float(r["coverage"]) for r in sub]
        fpr = [float(r["fpr"]) for r in sub]
        plt.figure()
        plt.plot(coverage, fpr, marker="o")
        plt.xlabel("Coverage")
        plt.ylabel("False-positive rate")
        plt.title(f"Risk-coverage, M={M}")
        plt.tight_layout()
        plt.savefig(out / f"risk_coverage_M{M}.png", dpi=180)
        plt.close()
