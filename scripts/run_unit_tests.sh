#!/usr/bin/env bash
set -euo pipefail
python -m py_compile run_pipeline.py rase/*.py tests/test_core.py
python tests/test_core.py
