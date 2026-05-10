# RASE Phase-0 Results Audit

Input archive: `rase_phase0.tar.gz`.

Coverage: 4 environments × 3 seeds × 3 candidate sources × 5 candidate-pool sizes = 180 sweep rows. Evaluation states per row are recorded in `n`, generally 4096.

## Main empirical conclusion

The strongest Phase-0 signal is not that conditional FPI monotonically increases with candidate-pool size. It generally decreases. The stronger and more consistent signal is that the predicted-vs-FQE advantage gap grows as M increases, especially for BC/IQL candidate sources.

## M=1 to M=256 deltas by environment and source

| env                          | source   |   pred_adv_delta_mean |   fqe_adv_delta_mean |   gap_delta_mean |   pred_pos_delta_mean |   fpi_uncond_delta_mean |   fpi_cond_delta_mean |   support_delta_mean |
|:-----------------------------|:---------|----------------------:|---------------------:|-----------------:|----------------------:|------------------------:|----------------------:|---------------------:|
| antmaze-umaze-v2             | bc       |                0.0178 |               0.069  |          -0.0512 |                0.3095 |                  0.1036 |               -0.0685 |               1.8047 |
| antmaze-umaze-v2             | iql      |                0.0175 |               0.0877 |          -0.0702 |                0.3017 |                  0.1007 |               -0.069  |               1.9271 |
| antmaze-umaze-v2             | perturb  |                0.0047 |               0.0061 |          -0.0014 |                0.0924 |                  0.016  |               -0.0973 |               0.3037 |
| halfcheetah-medium-replay-v2 | bc       |                6.7894 |               3.5778 |           3.2117 |                0.399  |                  0.0155 |               -0.1691 |               1.0984 |
| halfcheetah-medium-replay-v2 | iql      |                5.4676 |               2.8636 |           2.604  |                0.291  |                 -0.0098 |               -0.1312 |              -2.6408 |
| halfcheetah-medium-replay-v2 | perturb  |                3.0155 |               1.2328 |           1.7827 |                0.2359 |                 -0.0815 |               -0.2888 |              -0.9701 |
| hopper-medium-replay-v2      | bc       |                6.3817 |               2.2824 |           4.0994 |                0.4822 |                  0.1341 |               -0.1014 |               1.9538 |
| hopper-medium-replay-v2      | iql      |                4.7469 |               1.7187 |           3.0282 |                0.2941 |                  0.0633 |               -0.0647 |               3.3804 |
| hopper-medium-replay-v2      | perturb  |                2.1409 |               0.6826 |           1.4583 |                0.1858 |                  0.0231 |               -0.1253 |               0.2678 |
| walker2d-medium-replay-v2    | bc       |                6.8924 |               3.0677 |           3.8247 |                0.5002 |                  0.1115 |               -0.134  |               1.5237 |
| walker2d-medium-replay-v2    | iql      |                5.9707 |               2.6024 |           3.3683 |                0.3964 |                  0.0758 |               -0.0929 |               2.7055 |
| walker2d-medium-replay-v2    | perturb  |                2.4368 |               0.9068 |           1.53   |                0.2076 |                  0.0004 |               -0.199  |              -0.6122 |

## Overall sweep means by source and M

| source   |   M |   pred_adv_mean |   pred_adv_std |   fqe_adv_mean |   fqe_adv_std |   pred_empirical_gap_mean |   pred_empirical_gap_std |   pred_positive_rate_mean |   pred_positive_rate_std |   fpi_unconditional_mean |   fpi_unconditional_std |   fpi_cond_mean |   fpi_cond_std |   support_nll_mean |   support_nll_std |
|:---------|----:|----------------:|---------------:|---------------:|--------------:|--------------------------:|-------------------------:|--------------------------:|-------------------------:|-------------------------:|------------------------:|----------------:|---------------:|-------------------:|------------------:|
| bc       |   1 |         -1.0226 |         0.9636 |        -0.0726 |        0.1062 |                   -0.95   |                   0.9189 |                    0.3509 |                   0.1257 |                   0.1556 |                  0.0544 |          0.4445 |         0.0238 |            -0.4829 |            1.2718 |
| bc       |   4 |          1.2504 |         1.1059 |         0.9045 |        0.5676 |                    0.3459 |                   0.718  |                    0.5563 |                   0.1614 |                   0.2175 |                  0.0574 |          0.3946 |         0.0296 |            -0.4919 |            1.3713 |
| bc       |  16 |          2.568  |         1.748  |         1.5151 |        0.941  |                    1.0529 |                   0.9274 |                    0.6764 |                   0.1703 |                   0.2354 |                  0.0479 |          0.3558 |         0.0479 |            -0.0561 |            1.446  |
| bc       |  64 |          3.421  |         2.2174 |         1.9028 |        1.2003 |                    1.5182 |                   1.1324 |                    0.7377 |                   0.1674 |                   0.2419 |                  0.0478 |          0.3347 |         0.0513 |             0.522  |            1.5348 |
| bc       | 256 |          3.9977 |         2.5392 |         2.1766 |        1.3525 |                    1.8211 |                   1.3059 |                    0.7736 |                   0.1627 |                   0.2468 |                  0.0463 |          0.3262 |         0.0572 |             1.1123 |            1.6091 |
| iql      |   1 |          0.4948 |         0.8425 |         0.7228 |        0.4575 |                   -0.228  |                   0.6851 |                    0.4592 |                   0.1734 |                   0.1831 |                  0.0635 |          0.4064 |         0.0391 |             1.431  |            2.2361 |
| iql      |   4 |          2.2907 |         1.5712 |         1.5021 |        0.9147 |                    0.7886 |                   0.8192 |                    0.6243 |                   0.1905 |                   0.221  |                  0.0544 |          0.3641 |         0.0471 |             0.8592 |            0.9935 |
| iql      |  16 |          3.3625 |         2.1503 |         1.99   |        1.2055 |                    1.3724 |                   1.0615 |                    0.7075 |                   0.1854 |                   0.233  |                  0.0472 |          0.3397 |         0.0554 |             1.2786 |            1.4439 |
| iql      |  64 |          4.0632 |         2.5515 |         2.3213 |        1.4066 |                    1.7419 |                   1.2708 |                    0.7532 |                   0.1764 |                   0.2369 |                  0.0457 |          0.3228 |         0.0549 |             2.0054 |            1.5266 |
| iql      | 256 |          4.5454 |         2.83   |         2.5409 |        1.54   |                    2.0046 |                   1.4178 |                    0.78   |                   0.168  |                   0.2406 |                  0.0428 |          0.317  |         0.059  |             2.7741 |            1.5878 |
| perturb  |   1 |         -1.0759 |         0.9625 |        -0.0666 |        0.064  |                   -1.0094 |                   0.9568 |                    0.3489 |                   0.1215 |                   0.1787 |                  0.068  |          0.5078 |         0.0203 |             1.8933 |            0.6076 |
| perturb  |   4 |         -0.3188 |         0.8542 |         0.2125 |        0.1343 |                   -0.5314 |                   0.8574 |                    0.4186 |                   0.1428 |                   0.1693 |                  0.0509 |          0.411  |         0.0316 |             1.1403 |            0.5324 |
| perturb  |  16 |          0.1732 |         0.8928 |         0.3962 |        0.2579 |                   -0.223  |                   0.826  |                    0.4651 |                   0.1535 |                   0.1605 |                  0.0407 |          0.3566 |         0.0557 |             1.1884 |            0.5723 |
| perturb  |  64 |          0.541  |         0.9881 |         0.5308 |        0.3458 |                    0.0102 |                   0.8364 |                    0.5017 |                   0.1599 |                   0.1645 |                  0.04   |          0.3412 |         0.0629 |             1.2982 |            0.593  |
| perturb  | 256 |          0.8236 |         1.0844 |         0.6405 |        0.4175 |                    0.1831 |                   0.8547 |                    0.5293 |                   0.1651 |                   0.1683 |                  0.0417 |          0.3303 |         0.0621 |             1.6406 |            0.5201 |

## M=256 best thresholds subject to coverage >= 0.2

| env                          | source   |   tau_mean |   coverage_mean |   precision_mean |   fpr_mean |   accepted_fqe_adv_mean |
|:-----------------------------|:---------|-----------:|----------------:|-----------------:|-----------:|------------------------:|
| antmaze-umaze-v2             | bc       |    -3.5    |          0.9978 |           0.5215 |     0.4785 |                  0.0827 |
| antmaze-umaze-v2             | iql      |    -2.3333 |          0.998  |           0.5258 |     0.4742 |                  0.1031 |
| antmaze-umaze-v2             | perturb  |    -0.8333 |          0.9771 |           0.5222 |     0.4778 |                  0.0055 |
| halfcheetah-medium-replay-v2 | bc       |     5      |          0.5087 |           0.7921 |     0.2079 |                  5.6353 |
| halfcheetah-medium-replay-v2 | iql      |     5      |          0.5254 |           0.8085 |     0.1915 |                  6.2689 |
| halfcheetah-medium-replay-v2 | perturb  |     2.6667 |          0.4692 |           0.7778 |     0.2222 |                  1.1328 |
| hopper-medium-replay-v2      | bc       |     5      |          0.3762 |           0.7017 |     0.2983 |                  4.4325 |
| hopper-medium-replay-v2      | iql      |     5      |          0.4219 |           0.7158 |     0.2842 |                  5.4168 |
| hopper-medium-replay-v2      | perturb  |     2      |          0.328  |           0.6448 |     0.3552 |                  0.7498 |
| walker2d-medium-replay-v2    | bc       |     5      |          0.38   |           0.7175 |     0.2825 |                  5.152  |
| walker2d-medium-replay-v2    | iql      |     5      |          0.4161 |           0.728  |     0.272  |                  5.7352 |
| walker2d-medium-replay-v2    | perturb  |    -0.3333 |          0.5015 |           0.6953 |     0.3047 |                  0.9107 |

## Interpretation notes

- Predicted advantage increases much faster than FQE advantage for BC and IQL proposals. This supports a selection-induced optimism / winner's-curse diagnosis.
- Conditional FPI, defined as `P(FQE_adv <= 0 | predicted_adv > 0)`, decreases with M because larger M also increases the mass of candidates that are genuinely FQE-positive. This metric alone should not be used as the main go/no-go criterion.
- Unconditional FPI, defined as `P(predicted_adv > 0 and FQE_adv <= 0)`, rises for BC and IQL because predicted-positive coverage expands substantially.
- AntMaze has very small advantage scale and weak risk-score separability. Fixed global thresholds are inappropriate there.
- RASE-style thresholding improves precision at reduced coverage on locomotion tasks, especially at M=64/256 and higher tau, but does not cleanly solve AntMaze.

## Files

- `all_sweep_rows.csv`: raw sweep CSV rows with environment/seed/source annotations.
- `sweep_aggregate_by_env_source_M.csv`: mean/std over seeds.
- `m1_to_m256_deltas_by_env_source.csv`: candidate-pool growth deltas.
- `risk_coverage_aggregate_by_source_M_tau.csv`: aggregated risk-coverage curves.
- `best_threshold_M256_coverage_ge_0p2_by_env_source.csv`: per-task threshold diagnostics.
- `plots/`: summary plots.