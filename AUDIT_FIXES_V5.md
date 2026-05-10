# RASE v5 Code Audit Fixes

## Fixed error

`run_rollout_diagnostic.py` and `run_proxy_alignment.py` previously loaded `fqe_iql_ref.pt` directly. Older Phase-0 output directories contain a legacy single-Q FQE checkpoint without a `version` field, while the audited diagnostics require TwinQ/min-Q FQE (`version=2`). The old script therefore failed with:

```text
RuntimeError: FQE checkpoint is from an older incompatible version.
```

v5 fixes this by using `ensure_fqe_checkpoint()` in all diagnostic entry points. If the cached FQE checkpoint is legacy or incompatible, it is backed up and a compatible TwinQ/min-Q FQE evaluator is trained automatically.

## Legacy checkpoint compatibility

Early Phase-0 code used a clipped Gaussian policy. Later audited code uses tanh-squashed Gaussian policies. The network parameters have the same tensor shapes, so old IQL/BC checkpoints can load, but the action semantics differ. v5 adds:

```text
--policy_squash auto | tanh | clip
```

`auto` detects old `outputs/rase_phase0/...` checkpoints and selects `clip`, preserving legacy policy behavior for diagnostics. New runs use `policy_squash: tanh` by default.

## Python 3.9 compatibility

The runtime type alias in `rase/selection.py` used Python 3.10 `|` syntax outside annotations. v5 replaces it with `typing.Union`, so Python 3.9 environments import cleanly.

## Numerical / dependency robustness

`np.quantile(..., method="nearest")` is now guarded with a fallback to `interpolation="nearest"` for older NumPy.

## Updated scripts

Batch scripts now accept:

```bash
RASE_OUT_DIR=outputs/rase_phase0
RASE_POLICY_SQUASH=auto
```

so existing Phase-0 outputs can be reused without editing YAML.
