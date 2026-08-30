# Phase 0 execution status

This report confirms that the synthetic plumbing ran over the frozen five seeds. It is **not** a privacy-attack result, an equivalence test, or a state-of-the-art claim.

## Completed runs

| Seed | ε | DP-only JS to retrain | Fine-tune JS to retrain | Retrain-to-retrain JS | DP-only test accuracy | Fine-tune test accuracy | Retrain test accuracy |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 20260824 | 11.083 | 0.02691 | 0.02490 | 0.04852 | 0.848 | 0.857 | 0.840 |
| 20260825 | 11.083 | 0.06341 | 0.05383 | 0.06558 | 0.872 | 0.888 | 0.817 |
| 20260826 | 11.083 | 0.02870 | 0.02416 | 0.04363 | 0.808 | 0.840 | 0.860 |
| 20260827 | 11.083 | 0.03931 | 0.02412 | 0.03960 | 0.832 | 0.858 | 0.848 |
| 20260828 | 11.083 | 0.04500 | 0.02270 | 0.03051 | 0.763 | 0.827 | 0.825 |

## Mean ± sample standard deviation

| Metric | Mean | SD |
|---|---:|---:|
| epsilon | 11.08269 | 0.00000 |
| dp_js | 0.04067 | 0.01475 |
| finetune_js | 0.02994 | 0.01338 |
| retrain_variability_js | 0.04557 | 0.01300 |
| dp_accuracy | 0.82467 | 0.04137 |
| finetune_accuracy | 0.85400 | 0.02317 |
| target_accuracy | 0.83800 | 0.01746 |

## Baseline attack diagnostics

| Metric | Mean | SD |
|---|---:|---:|
| dp_forget_mia_auc | 0.46330 | 0.14722 |
| finetune_forget_mia_auc | 0.46519 | 0.14652 |
| dp_retain_mia_auc | 0.48355 | 0.04589 |
| finetune_retain_mia_auc | 0.49523 | 0.04397 |

## Interpretation boundary

These values establish that deterministic manifests, client-DP accounting, no-action DP, retained fine-tuning, independent target retrains, and held-out baseline attack diagnostics execute together. The fine-tuning baseline accesses raw retained data and therefore is not DP post-processing. Attack diagnostics are label-matched loss-threshold probes, not LiRA/A-LiRA or TC-UMIA. Retrain-to-retrain JS is a variability diagnostic, not yet a pre-registered equivalence band. No public-dataset result or secure-aggregation service has been run yet. Do not infer a redundancy boundary from this table.

## Next gate

Freeze member/non-member attack splits, add calibrated membership-inference sanity checks, and then move the same protocol to CIFAR-10. The independent target retrain gate is now implemented.
