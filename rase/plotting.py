from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt


def read_csv_rows(path: str | Path) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _float_or_nan(x: str) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _plot_xy(xs, ys, xlabel, ylabel, title, path: Path) -> None:
    if not xs or all(math.isnan(y) for y in ys):
        return
    plt.figure()
    plt.plot(xs, ys, marker="o")
    plt.xscale("log", base=2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_sweep(csv_path: str | Path, out_dir: str | Path) -> None:
    rows = read_csv_rows(csv_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    xs = [int(r["M"]) for r in rows]

    for key, ylabel in [
        ("pred_adv_mean", "Predicted pairwise gap"),
        ("iql_adv_vs_v_mean", "IQL advantage vs V"),
        ("fqe_adv_mean", "FQE pairwise gap"),
        ("pred_empirical_gap", "Predicted - FQE pairwise gap"),
        ("fpi_rate_cond_pred_positive", "Conditional FPI rate"),
        ("fpi_rate_unconditional", "Unconditional FPI rate"),
        ("support_nll_mean", "Behavior NLL of selected action"),
        ("fqe_disagreement_mean", "FQE twin-Q disagreement"),
        ("rase_score_mean", "RASE score"),
    ]:
        if key not in rows[0]:
            continue
        ys = [_float_or_nan(r[key]) for r in rows]
        _plot_xy(xs, ys, "Candidate pool size M", ylabel, key, out / f"{key}.png")


def plot_risk_coverage(csv_path: str | Path, out_dir: str | Path) -> None:
    rows = read_csv_rows(csv_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    Ms = sorted({int(r["M"]) for r in rows})
    for M in Ms:
        sub = [r for r in rows if int(r["M"]) == M]
        # Sort by coverage to make the curve readable even if thresholds are not monotone
        # after numerical ties.
        points = sorted((_float_or_nan(r["coverage"]), _float_or_nan(r["fpr"])) for r in sub)
        coverage = [p[0] for p in points]
        fpr = [p[1] for p in points]
        if all(math.isnan(y) for y in fpr):
            continue
        plt.figure()
        plt.plot(coverage, fpr, marker="o")
        plt.xlabel("Coverage")
        plt.ylabel("False-positive rate")
        plt.title(f"Risk-coverage, M={M}")
        plt.tight_layout()
        plt.savefig(out / f"risk_coverage_M{M}.png", dpi=180)
        plt.close()
