from __future__ import annotations

from typing import Iterable

import numpy as np


def _as_np(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float64)


def safe_corr(x, y) -> float:
    x = _as_np(x)
    y = _as_np(y)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 2 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def auroc_from_scores(labels, scores) -> float:
    """AUROC without sklearn. labels=True means positive class.

    Equivalent to P(score_pos > score_neg) + 0.5 P(tie). Returns NaN if either
    class is absent.
    """
    y = np.asarray(labels).astype(bool)
    s = _as_np(scores)
    mask = np.isfinite(s)
    y = y[mask]
    s = s[mask]
    n_pos = int(y.sum())
    n_neg = int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(s, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1, dtype=np.float64)

    # Average ranks for ties.
    sorted_s = s[order]
    i = 0
    while i < len(sorted_s):
        j = i + 1
        while j < len(sorted_s) and sorted_s[j] == sorted_s[i]:
            j += 1
        if j - i > 1:
            avg = 0.5 * (i + 1 + j)
            ranks[order[i:j]] = avg
        i = j

    sum_pos_ranks = ranks[y].sum()
    auc = (sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def auprc_from_scores(labels, scores) -> float:
    """Average precision / AUPRC without sklearn."""
    y = np.asarray(labels).astype(bool)
    s = _as_np(scores)
    mask = np.isfinite(s)
    y = y[mask]
    s = s[mask]
    n_pos = int(y.sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    y_sorted = y[order]
    tp = np.cumsum(y_sorted)
    ranks = np.arange(1, len(y_sorted) + 1)
    precision = tp / ranks
    return float((precision * y_sorted).sum() / n_pos)


def threshold_at_coverage(scores, coverage: float) -> float:
    """Return score threshold accepting approximately the top `coverage` fraction."""
    s = _as_np(scores)
    s = s[np.isfinite(s)]
    if s.size == 0:
        return float("nan")
    coverage = float(np.clip(coverage, 0.0, 1.0))
    if coverage <= 0:
        return float("inf")
    if coverage >= 1:
        return float(np.min(s))
    return float(np.quantile(s, 1.0 - coverage, method="nearest"))


def precision_at_coverage(labels, scores, coverage: float) -> float:
    y = np.asarray(labels).astype(bool)
    s = _as_np(scores)
    tau = threshold_at_coverage(s, coverage)
    accepted = s >= tau
    if accepted.sum() == 0:
        return float("nan")
    return float(y[accepted].mean())


def fpr_at_coverage(labels, scores, coverage: float) -> float:
    p = precision_at_coverage(labels, scores, coverage)
    return float("nan") if not np.isfinite(p) else float(1.0 - p)


def risk_coverage_rows(labels, scores, coverages: Iterable[float], prefix: dict | None = None) -> list[dict]:
    rows = []
    prefix = dict(prefix or {})
    for cov in coverages:
        tau = threshold_at_coverage(scores, cov)
        accepted = _as_np(scores) >= tau
        if accepted.sum() == 0:
            precision = float("nan")
            realized_cov = 0.0
        else:
            y = np.asarray(labels).astype(bool)
            precision = float(y[accepted].mean())
            realized_cov = float(accepted.mean())
        row = dict(prefix)
        row.update({
            "target_coverage": float(cov),
            "tau": tau,
            "coverage": realized_cov,
            "precision": precision,
            "fpr": float("nan") if not np.isfinite(precision) else 1.0 - precision,
        })
        rows.append(row)
    return rows
