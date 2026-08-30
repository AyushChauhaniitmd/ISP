# Phase 1 Real CIFAR-10 Binary Results

Generated from `reports/phase1_v3_reference_corrected_run_index.json`.

This is a real-data CPU benchmark over CIFAR-10 classes 0 vs 1 using public average-pooled features, 12 simulated clients, one full-client deletion request, five seeds per cell, and the accountant-aligned `poisson_sum_fixed_public_normalizer_v2` DP mechanism.

Important: this is an exploratory benchmark result, not a state-of-the-art or certified-unlearning claim. The result is strong enough to guide the paper direction, but final claims still require stronger models, published baseline reproductions, calibrated attacks, and confirmatory seeds.

## Main Result

| Epsilon | Heterogeneity | Method | Test accuracy | JS to retrain | Forgotten MIA AUC | Retained MIA AUC | Client updates | Storage KiB |
|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 2.0 | 0.2 | DP-only/no action | 0.549 +/- 0.046 | 0.286 +/- 0.103 | 0.510 +/- 0.035 | 0.517 +/- 0.023 | 74.200 +/- 5.630 | 64.987 +/- 4.245 |
| 2.0 | 0.2 | Retained fine-tune | 0.585 +/- 0.037 | 0.287 +/- 0.090 | 0.508 +/- 0.034 | 0.516 +/- 0.023 | 22.600 +/- 1.140 | 0.0000 +/- 0.0000 |
| 2.0 | 0.2 | Cached-update reconstruction | 0.547 +/- 0.046 | 0.286 +/- 0.104 | 0.510 +/- 0.035 | 0.518 +/- 0.022 | 0.0000 +/- 0.0000 | 64.987 +/- 4.245 |
| 2.0 | 0.2 | Retrain target | 0.508 +/- 0.073 | 0.0000 +/- 0.0000 | 0.497 +/- 0.033 | 0.508 +/- 0.026 | 68.400 +/- 6.693 | 0.0000 +/- 0.0000 |
| 2.0 | 0.2 | Independent retrain | 0.519 +/- 0.128 | 0.304 +/- 0.065 | 0.503 +/- 0.019 | 0.478 +/- 0.012 | 63.600 +/- 2.702 | 0.0000 +/- 0.0000 |
| 2.0 | 0.75 | DP-only/no action | 0.547 +/- 0.049 | 0.286 +/- 0.108 | 0.530 +/- 0.032 | 0.527 +/- 0.025 | 74.200 +/- 5.630 | 64.987 +/- 4.245 |
| 2.0 | 0.75 | Retained fine-tune | 0.582 +/- 0.039 | 0.287 +/- 0.097 | 0.527 +/- 0.032 | 0.527 +/- 0.024 | 22.600 +/- 1.140 | 0.0000 +/- 0.0000 |
| 2.0 | 0.75 | Cached-update reconstruction | 0.543 +/- 0.051 | 0.287 +/- 0.109 | 0.529 +/- 0.034 | 0.527 +/- 0.025 | 0.0000 +/- 0.0000 | 64.987 +/- 4.245 |
| 2.0 | 0.75 | Retrain target | 0.508 +/- 0.069 | 0.0000 +/- 0.0000 | 0.496 +/- 0.029 | 0.496 +/- 0.037 | 68.400 +/- 6.693 | 0.0000 +/- 0.0000 |
| 2.0 | 0.75 | Independent retrain | 0.518 +/- 0.135 | 0.304 +/- 0.067 | 0.495 +/- 0.038 | 0.482 +/- 0.015 | 63.600 +/- 2.702 | 0.0000 +/- 0.0000 |
| 8.0 | 0.2 | DP-only/no action | 0.613 +/- 0.040 | 0.187 +/- 0.065 | 0.505 +/- 0.033 | 0.512 +/- 0.024 | 74.200 +/- 5.630 | 64.987 +/- 4.245 |
| 8.0 | 0.2 | Retained fine-tune | 0.649 +/- 0.023 | 0.179 +/- 0.047 | 0.497 +/- 0.035 | 0.506 +/- 0.024 | 22.600 +/- 1.140 | 0.0000 +/- 0.0000 |
| 8.0 | 0.2 | Cached-update reconstruction | 0.603 +/- 0.037 | 0.189 +/- 0.067 | 0.505 +/- 0.035 | 0.514 +/- 0.023 | 0.0000 +/- 0.0000 | 64.987 +/- 4.245 |
| 8.0 | 0.2 | Retrain target | 0.567 +/- 0.076 | 0.0000 +/- 0.0000 | 0.492 +/- 0.027 | 0.512 +/- 0.031 | 68.400 +/- 6.693 | 0.0000 +/- 0.0000 |
| 8.0 | 0.2 | Independent retrain | 0.570 +/- 0.111 | 0.223 +/- 0.043 | 0.500 +/- 0.0090 | 0.483 +/- 0.020 | 63.600 +/- 2.702 | 0.0000 +/- 0.0000 |
| 8.0 | 0.75 | DP-only/no action | 0.606 +/- 0.047 | 0.191 +/- 0.075 | 0.527 +/- 0.026 | 0.524 +/- 0.022 | 74.200 +/- 5.630 | 64.987 +/- 4.245 |
| 8.0 | 0.75 | Retained fine-tune | 0.645 +/- 0.025 | 0.178 +/- 0.053 | 0.514 +/- 0.034 | 0.521 +/- 0.019 | 22.600 +/- 1.140 | 0.0000 +/- 0.0000 |
| 8.0 | 0.75 | Cached-update reconstruction | 0.595 +/- 0.048 | 0.193 +/- 0.079 | 0.524 +/- 0.028 | 0.526 +/- 0.021 | 0.0000 +/- 0.0000 | 64.987 +/- 4.245 |
| 8.0 | 0.75 | Retrain target | 0.574 +/- 0.072 | 0.0000 +/- 0.0000 | 0.487 +/- 0.026 | 0.499 +/- 0.038 | 68.400 +/- 6.693 | 0.0000 +/- 0.0000 |
| 8.0 | 0.75 | Independent retrain | 0.567 +/- 0.124 | 0.224 +/- 0.045 | 0.491 +/- 0.039 | 0.485 +/- 0.013 | 63.600 +/- 2.702 | 0.0000 +/- 0.0000 |
| inf | 0.2 | DP-only/no action | 0.776 +/- 0.0095 | 0.0014 +/- 0.0007 | 0.471 +/- 0.015 | 0.501 +/- 0.020 | 74.200 +/- 5.630 | 55.940 +/- 4.245 |
| inf | 0.2 | Retained fine-tune | 0.782 +/- 0.011 | 0.0016 +/- 0.0006 | 0.472 +/- 0.014 | 0.502 +/- 0.020 | 22.600 +/- 1.140 | 0.0000 +/- 0.0000 |
| inf | 0.2 | Cached-update reconstruction | 0.777 +/- 0.010 | 0.0019 +/- 0.0009 | 0.466 +/- 0.016 | 0.506 +/- 0.012 | 0.0000 +/- 0.0000 | 55.940 +/- 4.245 |
| inf | 0.2 | Retrain target | 0.776 +/- 0.010 | 0.0000 +/- 0.0000 | 0.468 +/- 0.018 | 0.500 +/- 0.018 | 68.400 +/- 6.693 | 0.0000 +/- 0.0000 |
| inf | 0.2 | Independent retrain | 0.776 +/- 0.0078 | 0.0015 +/- 0.0007 | 0.471 +/- 0.015 | 0.502 +/- 0.017 | 63.600 +/- 2.702 | 0.0000 +/- 0.0000 |
| inf | 0.75 | DP-only/no action | 0.778 +/- 0.019 | 0.0017 +/- 0.0003 | 0.486 +/- 0.019 | 0.505 +/- 0.015 | 74.200 +/- 5.630 | 55.940 +/- 4.245 |
| inf | 0.75 | Retained fine-tune | 0.788 +/- 0.0075 | 0.0016 +/- 0.0003 | 0.487 +/- 0.018 | 0.503 +/- 0.016 | 22.600 +/- 1.140 | 0.0000 +/- 0.0000 |
| inf | 0.75 | Cached-update reconstruction | 0.757 +/- 0.043 | 0.0044 +/- 0.0038 | 0.476 +/- 0.013 | 0.506 +/- 0.013 | 0.0000 +/- 0.0000 | 55.940 +/- 4.245 |
| inf | 0.75 | Retrain target | 0.775 +/- 0.016 | 0.0000 +/- 0.0000 | 0.480 +/- 0.019 | 0.504 +/- 0.020 | 68.400 +/- 6.693 | 0.0000 +/- 0.0000 |
| inf | 0.75 | Independent retrain | 0.775 +/- 0.0097 | 0.0019 +/- 0.0007 | 0.486 +/- 0.020 | 0.507 +/- 0.014 | 63.600 +/- 2.702 | 0.0000 +/- 0.0000 |

## Redundancy Screen

| Epsilon | Heterogeneity | Baseline | Mean marginal JS benefit | Retrain-variability q90 | Screen interpretation |
|---:|---:|---|---:|---:|---|
| 2.0 | 0.2 | Retained fine-tune | -0.0010 | 0.366 | no practically material JS improvement detected by this screen |
| 2.0 | 0.2 | Cached-update reconstruction | 0.0001 | 0.366 | no practically material JS improvement detected by this screen |
| 2.0 | 0.75 | Retained fine-tune | -0.0010 | 0.370 | no practically material JS improvement detected by this screen |
| 2.0 | 0.75 | Cached-update reconstruction | -0.0005 | 0.370 | no practically material JS improvement detected by this screen |
| 8.0 | 0.2 | Retained fine-tune | 0.0084 | 0.259 | no practically material JS improvement detected by this screen |
| 8.0 | 0.2 | Cached-update reconstruction | -0.0023 | 0.259 | no practically material JS improvement detected by this screen |
| 8.0 | 0.75 | Retained fine-tune | 0.013 | 0.263 | no practically material JS improvement detected by this screen |
| 8.0 | 0.75 | Cached-update reconstruction | -0.0026 | 0.263 | no practically material JS improvement detected by this screen |
| inf | 0.2 | Retained fine-tune | -0.0001 | 0.0022 | no practically material JS improvement detected by this screen |
| inf | 0.2 | Cached-update reconstruction | -0.0005 | 0.0022 | no practically material JS improvement detected by this screen |
| inf | 0.75 | Retained fine-tune | 0.0001 | 0.0027 | no practically material JS improvement detected by this screen |
| inf | 0.75 | Cached-update reconstruction | -0.0027 | 0.0027 | no practically material JS improvement detected by this screen |

## Interpretation

- Across all six cells, this exploratory screen found no practically material JS-to-retrain improvement from explicit unlearning over DP-only/no-action.
- At epsilon 2 and 8, retrain-to-retrain variability is large because the DP noise dominates the small CPU model. That makes exact functional alignment a weak endpoint unless paired with stronger utility and attack evaluations.
- In the non-private control, all methods are already very close to the retrain target on test JS; this does not prove deletion, but it is a useful sanity check for the evaluator.
- Cached-update reconstruction is not a clean privacy post-processing method because it depends on sensitive per-client update history. It is reported as a historical-server-state baseline only.

## Reproducibility

- Run index: `reports\phase1_v3_reference_corrected_run_index.json`
- Redundancy JSON: `reports\phase1_v3_reference_corrected_grid_screen.json`
- Summary JSON: `reports\phase1_v3_reference_corrected_summary.json`
- Validation: every run in the index passed `scripts/validate_artifact.py`.
