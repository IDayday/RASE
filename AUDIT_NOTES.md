# RASE Phase-0 Code Audit Notes

This version is an audited revision of `rase_phase0_preexp_v2_cachefix.zip`.

## Accuracy fixes

1. **False-positive label now uses a consistent pairwise baseline.**
   The earlier implementation compared `Q_IQL(s, a*) - V_IQL(s)` with
   `FQE(s, a*) - FQE(s, a_data)`. The revised implementation compares
   `Q_IQL(s, a*) - Q_IQL(s, a_data)` with `FQE(s, a*) - FQE(s, a_data)`.
   The old `Q-V` advantage is still reported as `iql_adv_vs_v_mean`.

2. **Candidate-pool sweep is nested.**
   For each state, the code samples `max(candidate_ms)` candidates once and uses
   prefixes for smaller `M`. This directly measures the effect of enlarging the
   same candidate pool and reduces Monte Carlo noise.

3. **FQE is now twin-Q/min-Q.**
   This makes the empirical proxy less optimistic on selected candidate actions
   and reports `fqe_disagreement_mean` as a diagnostic.

4. **Gaussian policies are tanh-squashed.**
   Sampling and log-probability now use a consistent tanh transform with the
   change-of-variables correction. This makes BC support-NLL diagnostics more
   meaningful than hard clipping.

5. **Evaluation state sampling is without replacement when possible.**
   This reduces duplicate-state noise in sweep metrics.

## Efficiency fixes

1. **Candidate evaluation cost drops from `sum(M)` to `max(M)` per state.**
   With `[1, 4, 16, 64, 256]`, candidate forward passes drop from 341 to 256
   per state, roughly a 25% reduction for the sweep stage.

2. **Candidate generation uses `GaussianPolicy.sample_n`.**
   This keeps sampling vectorized and avoids Python loops.

3. **Multi-GPU sweep uses one worker loop per GPU.**
   Each GPU immediately starts its next assigned job after finishing the previous
   job instead of waiting for an entire launch group to finish.

## Scope note

This package still implements the offline Phase-0 RASE diagnostic, not the full
PCAR offline-to-online algorithm. PCAR requires online replay, calibration buffer,
old-policy EMA, and supervised replacement updates; those are a separate training
pipeline.
