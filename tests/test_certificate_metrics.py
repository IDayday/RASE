from __future__ import annotations

import numpy as np
import pandas as pd

from rase.metrics import auroc_from_scores, precision_at_coverage
from run_certificate_analysis import _add_derived_scores, _score_specs


def main():
    labels = np.array([0, 0, 1, 1], dtype=bool)
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    assert abs(auroc_from_scores(labels, scores) - 1.0) < 1e-9
    assert abs(precision_at_coverage(labels, scores, 0.5) - 1.0) < 1e-9
    df = pd.DataFrame({
        "pred_pair_gap": [1.0, 2.0],
        "fqe_pair_gap": [0.5, -1.0],
        "crossfit_pair_gap": [0.2, 3.0],
        "support_nll": [2.0, 1.0],
    })
    d = _add_derived_scores(df)
    assert "min_pred_fqe_gap" in d.columns
    assert "min_pred_fqe_crossfit_gap" in d.columns
    names = [x[0] for x in _score_specs(d)]
    assert "pred_pair_gap" in names
    assert "min_pred_fqe_crossfit_gap" in names
    assert "neg_support_nll" in names
    print("certificate_metrics_tests_ok")


if __name__ == "__main__":
    main()
