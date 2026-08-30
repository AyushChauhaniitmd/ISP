# Exploratory redundancy-grid status

Dataset: **synthetic federation**. This report is an execution artifact, not a paper claim. It does not establish equivalence, unlearning success, or state of the art.

| Configuration | Seeds | DP-only JS to retrain | Fine-tune JS to retrain | MUB (DP-only minus fine-tune) | Retrain variability | Screen |
|---|---:|---:|---:|---:|---:|---|
| epsilon=2.0|heterogeneity=0.2 | 5 | 0.26765 ± 0.04836 | 0.23203 ± 0.03830 | 0.03562 [0.01746, 0.05937] | 0.23936 ± 0.08094 | no practically material JS improvement detected by this screen |
| epsilon=2.0|heterogeneity=0.85 | 5 | 0.26321 ± 0.05378 | 0.22463 ± 0.04849 | 0.03858 [0.02015, 0.05895] | 0.23642 ± 0.08189 | no practically material JS improvement detected by this screen |
| epsilon=8.0|heterogeneity=0.2 | 5 | 0.09199 ± 0.02201 | 0.06802 ± 0.01586 | 0.02396 [0.01100, 0.04581] | 0.07859 ± 0.03381 | no practically material JS improvement detected by this screen |
| epsilon=8.0|heterogeneity=0.85 | 5 | 0.09025 ± 0.02719 | 0.06322 ± 0.02040 | 0.02703 [0.01211, 0.04886] | 0.07704 ± 0.03314 | no practically material JS improvement detected by this screen |
| epsilon=inf|heterogeneity=0.2 | 5 | 0.00066 ± 0.00037 | 0.00077 ± 0.00016 | -0.00012 [-0.00031, 0.00007] | 0.00052 ± 0.00013 | no practically material JS improvement detected by this screen |
| epsilon=inf|heterogeneity=0.85 | 5 | 0.00114 ± 0.00060 | 0.00151 ± 0.00049 | -0.00037 [-0.00098, 0.00026] | 0.00102 ± 0.00115 | no practically material JS improvement detected by this screen |

## Interpretation boundary

The grid uses deterministic synthetic data and a retained-data fine-tuning baseline. It is useful for testing the experimental pipeline, request ledger, privacy accountant, and grouping logic. Its practical tolerance is derived from observed retrain variability for screening only; it is not a pre-registered multi-endpoint equivalence bound. Public datasets, published FU baselines, calibrated attacks, and confirmation runs are mandatory before any research claim.
