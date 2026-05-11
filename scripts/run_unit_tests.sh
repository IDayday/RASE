#!/usr/bin/env bash
set -euo pipefail
PYTHONPATH=. python -m py_compile run_pipeline.py run_refresh_fqe.py run_proxy_alignment.py run_rollout_diagnostic.py run_crossfit_iql.py run_crossfit_verify_pairs.py run_certificate_analysis.py rase/*.py tests/*.py
PYTHONPATH=. python tests/test_core.py
PYTHONPATH=. python tests/test_certificate_metrics.py
