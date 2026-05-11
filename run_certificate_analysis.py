from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from rase.metrics import auprc_from_scores, auroc_from_scores, precision_at_coverage, risk_coverage_rows, threshold_at_coverage


def _parse_csv_list(text: str) -> list[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def _parse_ints(text: str) -> list[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def _score_specs(df: pd.DataFrame) -> list[tuple[str, str, float]]:
    specs = [
        ("pred_pair_gap", "pred_pair_gap", +1.0),
        ("fqe_pair_gap", "fqe_pair_gap", +1.0),
        ("rase_score", "rase_score", +1.0),
        ("rase_score_v2", "rase_score_v2", +1.0),
        ("min_pred_fqe_gap", "min_pred_fqe_gap", +1.0),
        ("crossfit_pair_gap", "crossfit_pair_gap", +1.0),
        ("min_pred_crossfit_gap", "min_pred_crossfit_gap", +1.0),
        ("min_pred_fqe_crossfit_gap", "min_pred_fqe_crossfit_gap", +1.0),
        ("neg_support_nll", "support_nll", -1.0),
        ("neg_knn_sa_distance", "knn_sa_distance", -1.0),
        ("neg_iql_q_disagreement", "iql_q_disagreement", -1.0),
        ("neg_fqe_disagreement", "fqe_disagreement", -1.0),
        ("neg_crossfit_q_disagreement", "crossfit_q_disagreement", -1.0),
        ("neg_action_l2_to_data", "action_l2_to_data", -1.0),
    ]
    return [(name, col, sign) for name, col, sign in specs if col in df.columns]


def _add_derived_scores(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if {"pred_pair_gap", "fqe_pair_gap"}.issubset(df.columns):
        df["min_pred_fqe_gap"] = np.minimum(df["pred_pair_gap"].astype(float), df["fqe_pair_gap"].astype(float))
    if {"pred_pair_gap", "crossfit_pair_gap"}.issubset(df.columns):
        df["min_pred_crossfit_gap"] = np.minimum(df["pred_pair_gap"].astype(float), df["crossfit_pair_gap"].astype(float))
    if {"pred_pair_gap", "fqe_pair_gap", "crossfit_pair_gap"}.issubset(df.columns):
        df["min_pred_fqe_crossfit_gap"] = np.minimum(
            np.minimum(df["pred_pair_gap"].astype(float), df["fqe_pair_gap"].astype(float)),
            df["crossfit_pair_gap"].astype(float),
        )
    return df


def _load_pairs(out_dir: Path, envs: list[str], seeds: list[int], sources: list[str], continuation: str) -> pd.DataFrame:
    rows = []
    for env in envs:
        for seed in seeds:
            diag = out_dir / env / f"seed{seed}" / "diagnostics"
            for source in sources:
                verified = diag / f"crossfit_verified_pairs_{source}_{continuation}.csv"
                base = diag / f"rollout_pairs_{source}_{continuation}.csv"
                if verified.exists():
                    path = verified
                    has_crossfit = 1
                elif base.exists():
                    path = base
                    has_crossfit = 0
                else:
                    print(f"[warn] missing pair file for env={env} seed={seed} source={source}: {base}")
                    continue
                df = pd.read_csv(path)
                df["env"] = env
                df["seed"] = int(seed)
                df["source"] = source
                df["has_crossfit"] = int(has_crossfit)
                rows.append(df)
    if not rows:
        raise SystemExit("No rollout pair CSVs found.")
    out = pd.concat(rows, ignore_index=True)
    out = _add_derived_scores(out)
    return out


def _alignment_rows(df: pd.DataFrame, coverages: Iterable[float], group_cols: list[str]) -> list[dict]:
    rows = []
    for keys, g in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        prefix = dict(zip(group_cols, keys))
        labels = g["rollout_positive"].astype(bool).to_numpy()
        for score_name, col, sign in _score_specs(g):
            scores = sign * g[col].to_numpy(dtype=float)
            row = dict(prefix)
            row.update({
                "score": score_name,
                "n": int(len(g)),
                "rollout_positive_rate": float(labels.mean()) if len(labels) else float("nan"),
                "rollout_fpi_rate": float((~labels).mean()) if len(labels) else float("nan"),
                "auroc_rollout_positive": auroc_from_scores(labels, scores),
                "auprc_rollout_positive": auprc_from_scores(labels, scores),
                "score_mean": float(np.nanmean(scores)) if len(scores) else float("nan"),
            })
            for cov in coverages:
                row[f"precision_at_cov_{cov:g}"] = precision_at_coverage(labels, scores, float(cov))
            rows.append(row)
    return rows


def _risk_rows(df: pd.DataFrame, coverages: Iterable[float], group_cols: list[str]) -> list[dict]:
    rows = []
    for keys, g in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        prefix_base = dict(zip(group_cols, keys))
        labels = g["rollout_positive"].astype(bool).to_numpy()
        for score_name, col, sign in _score_specs(g):
            scores = sign * g[col].to_numpy(dtype=float)
            prefix = dict(prefix_base)
            prefix["score"] = score_name
            prefix["n"] = int(len(g))
            rows.extend(risk_coverage_rows(labels, scores, coverages, prefix=prefix))
    return rows


def _calibrated_split_rows(
    df: pd.DataFrame,
    target_fprs: Iterable[float],
    group_cols: list[str],
    seed: int = 2026,
    min_cal: int = 20,
    min_test: int = 20,
) -> list[dict]:
    rows = []
    rng = np.random.default_rng(seed)
    for keys, g in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        prefix_base = dict(zip(group_cols, keys))
        g = g.copy().reset_index(drop=True)
        if len(g) < min_cal + min_test:
            continue
        perm = rng.permutation(len(g))
        cal_idx = perm[: len(g) // 2]
        test_idx = perm[len(g) // 2 :]
        cal = g.iloc[cal_idx]
        test = g.iloc[test_idx]
        y_cal = cal["rollout_positive"].astype(bool).to_numpy()
        y_test = test["rollout_positive"].astype(bool).to_numpy()
        if len(y_cal) < min_cal or len(y_test) < min_test:
            continue
        for score_name, col, sign in _score_specs(g):
            s_cal = sign * cal[col].to_numpy(dtype=float)
            s_test = sign * test[col].to_numpy(dtype=float)
            finite_cal = np.isfinite(s_cal)
            finite_test = np.isfinite(s_test)
            if finite_cal.sum() < min_cal or finite_test.sum() < min_test:
                continue
            # Sort candidate thresholds by score descending. Pick the largest coverage
            # on calibration whose observed false-positive rate is <= target_fpr.
            order = np.argsort(-s_cal[finite_cal], kind="mergesort")
            s_sorted = s_cal[finite_cal][order]
            y_sorted = y_cal[finite_cal][order]
            tp = np.cumsum(y_sorted)
            k = np.arange(1, len(y_sorted) + 1)
            precision = tp / k
            fpr = 1.0 - precision
            coverage = k / len(y_sorted)
            for target_fpr in target_fprs:
                ok = np.where(fpr <= float(target_fpr))[0]
                row = dict(prefix_base)
                row.update({
                    "score": score_name,
                    "target_fpr": float(target_fpr),
                    "n_cal": int(finite_cal.sum()),
                    "n_test": int(finite_test.sum()),
                })
                if ok.size == 0:
                    row.update({
                        "tau": float("inf"),
                        "cal_coverage": 0.0,
                        "cal_precision": float("nan"),
                        "cal_fpr": float("nan"),
                        "test_coverage": 0.0,
                        "test_precision": float("nan"),
                        "test_fpr": float("nan"),
                    })
                    rows.append(row)
                    continue
                best = int(ok[-1])  # max coverage satisfying risk on calibration.
                tau = float(s_sorted[best])
                acc_test = finite_test & (s_test >= tau)
                if acc_test.sum() == 0:
                    test_precision = float("nan")
                    test_cov = 0.0
                else:
                    test_precision = float(y_test[acc_test].mean())
                    test_cov = float(acc_test.mean())
                row.update({
                    "tau": tau,
                    "cal_coverage": float(coverage[best]),
                    "cal_precision": float(precision[best]),
                    "cal_fpr": float(fpr[best]),
                    "test_coverage": test_cov,
                    "test_precision": test_precision,
                    "test_fpr": float("nan") if not np.isfinite(test_precision) else 1.0 - test_precision,
                })
                rows.append(row)
    return rows


def _gap_summary_rows(df: pd.DataFrame) -> list[dict]:
    numeric = [
        "pred_pair_gap", "fqe_pair_gap", "rollout_adv", "pred_rollout_gap", "fqe_rollout_gap",
        "crossfit_pair_gap", "min_pred_crossfit_gap", "min_pred_fqe_crossfit_gap",
        "support_nll", "knn_sa_distance", "iql_q_disagreement", "fqe_disagreement", "crossfit_q_disagreement",
    ]
    if "pred_rollout_gap" not in df.columns and {"pred_pair_gap", "rollout_adv"}.issubset(df.columns):
        df = df.copy()
        df["pred_rollout_gap"] = df["pred_pair_gap"] - df["rollout_adv"]
    if "fqe_rollout_gap" not in df.columns and {"fqe_pair_gap", "rollout_adv"}.issubset(df.columns):
        df = df.copy()
        df["fqe_rollout_gap"] = df["fqe_pair_gap"] - df["rollout_adv"]
    rows = []
    for keys, g in df.groupby(["source", "env", "M"], dropna=False):
        source, env, M = keys
        row = {
            "source": source,
            "env": env,
            "M": int(M),
            "n": int(len(g)),
            "rollout_positive_rate": float(g["rollout_positive"].mean()),
            "rollout_fpi_rate": float(1.0 - g["rollout_positive"].mean()),
        }
        for col in numeric:
            if col in g.columns:
                row[f"{col}_mean"] = float(g[col].mean())
                row[f"{col}_std"] = float(g[col].std(ddof=1)) if len(g) > 1 else 0.0
        rows.append(row)
    return rows




def _table_text(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except Exception:
        return df.to_string(index=False)

def _write_report(out_dir: Path, align: pd.DataFrame, cal: pd.DataFrame, gap: pd.DataFrame) -> None:
    lines = []
    lines.append("# RASE Certificate Sprint Analysis\n")
    lines.append("This report evaluates candidate-level certificate scores against short-rollout labels. Positive means `rollout_adv > 0`.\n")
    if not align.empty:
        focus = align[(align["env"] == "ALL") if "env" in align.columns else slice(None)] if "env" in align.columns else align
        # Best scores by AUROC for each source/M.
        lines.append("## Best scores by source and M\n")
        for (source, M), g in focus.groupby(["source", "M"]):
            gg = g.sort_values("auroc_rollout_positive", ascending=False).head(5)
            lines.append(f"### source={source}, M={M}\n")
            lines.append(_table_text(gg[["score", "n", "rollout_positive_rate", "auroc_rollout_positive", "auprc_rollout_positive", "precision_at_cov_0.3"]]))
            lines.append("\n")
    if not cal.empty:
        lines.append("## Calibration split summary, target_fpr=0.35\n")
        g = cal[cal["target_fpr"] == 0.35].copy()
        if not g.empty:
            cols = [c for c in ["source", "env", "M", "score", "test_coverage", "test_precision", "test_fpr"] if c in g.columns]
            best = g.sort_values(["source", "env", "M", "test_fpr", "test_coverage"], ascending=[True, True, True, True, False]).groupby(["source", "env", "M"]).head(3)
            lines.append(_table_text(best[cols]))
            lines.append("\n")
    if not gap.empty:
        lines.append("## Rollout gap summary\n")
        cols = [c for c in ["source", "env", "M", "n", "rollout_positive_rate", "pred_pair_gap_mean", "fqe_pair_gap_mean", "rollout_adv_mean", "pred_rollout_gap_mean"] if c in gap.columns]
        lines.append(_table_text(gap[cols]))
        lines.append("\n")
    (out_dir / "RASE_certificate_sprint_report.md").write_text("\n".join(lines), encoding="utf-8")


def _make_all_env_rows(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d_all = d.copy()
    d_all["env"] = "ALL"
    return pd.concat([d, d_all], ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate RASE certificate-sprint rollout and crossfit results.")
    parser.add_argument("--out_dir", type=str, default="outputs/rase_phase0")
    parser.add_argument("--analysis_dir", type=str, default=None)
    parser.add_argument("--envs", type=str, default="halfcheetah-medium-replay-v2,hopper-medium-replay-v2,walker2d-medium-replay-v2")
    parser.add_argument("--seeds", type=str, default="41,42,43")
    parser.add_argument("--sources", type=str, default="bc,iql")
    parser.add_argument("--continuation_policy", type=str, default="iql")
    parser.add_argument("--coverages", type=str, default="0.2,0.3,0.5,0.7")
    parser.add_argument("--target_fprs", type=str, default="0.25,0.30,0.35,0.40")
    parser.add_argument("--calibration_seed", type=int, default=2026)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    analysis_dir = Path(args.analysis_dir) if args.analysis_dir else out_dir / "certificate_sprint_analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    envs = _parse_csv_list(args.envs)
    seeds = _parse_ints(args.seeds)
    sources = _parse_csv_list(args.sources)
    coverages = [float(x) for x in _parse_csv_list(args.coverages)]
    target_fprs = [float(x) for x in _parse_csv_list(args.target_fprs)]

    df = _load_pairs(out_dir, envs, seeds, sources, args.continuation_policy)
    df.to_csv(analysis_dir / "certificate_pairs_all.csv", index=False)

    df_group = _make_all_env_rows(df)
    align = pd.DataFrame(_alignment_rows(df_group, coverages, group_cols=["source", "env", "M"]))
    risk = pd.DataFrame(_risk_rows(df_group, coverages, group_cols=["source", "env", "M"]))
    cal = pd.DataFrame(_calibrated_split_rows(df_group, target_fprs, group_cols=["source", "env", "M"], seed=int(args.calibration_seed)))
    gap = pd.DataFrame(_gap_summary_rows(df_group))

    align.to_csv(analysis_dir / "certificate_score_alignment.csv", index=False)
    risk.to_csv(analysis_dir / "certificate_risk_coverage.csv", index=False)
    cal.to_csv(analysis_dir / "certificate_calibration_split.csv", index=False)
    gap.to_csv(analysis_dir / "certificate_gap_summary.csv", index=False)

    _write_report(analysis_dir, align, cal, gap)
    print(f"Saved certificate sprint analysis to {analysis_dir}")
    print(f"pairs={len(df)} alignment_rows={len(align)} calibration_rows={len(cal)}")


if __name__ == "__main__":
    main()
