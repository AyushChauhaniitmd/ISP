# Sequential request baseline status

This synthetic integration result validates stateful request handling for the retained-fine-tuning baseline. It is not a public-benchmark result or an unlearning/privacy claim.

| Request prefix | Request | Forgotten examples | DP-only JS to target | Stateful fine-tune JS to target | Retrain-to-retrain JS | Fine-tune incremental client updates |
|---:|---|---:|---:|---:|---:|---:|
| 1 | client 2, fraction 0.25 | 20 | 0.04369 | 0.03742 | 0.02470 | 20 |
| 2 | client 5, fraction 1.00 | 100 | 0.07943 | 0.07410 | 0.06190 | 24 |
| 3 | client 8, fraction 0.50 | 140 | 0.07033 | 0.05537 | 0.06023 | 20 |

## Interpretation boundary

The DP-only model is deliberately unchanged after each request. The stateful baseline accesses raw retained data after every request and therefore is not DP post-processing. Loss-based membership and tri-population outputs remain diagnostics, not LiRA/A-LiRA or TC-UMIA. The sample has one seed and synthetic data, so no threshold, equivalence, or retained-user leakage conclusion is permitted.

Raw artifact: `C:\Users\rinak\OneDrive\Desktop\isp\results\phase0_partial_sequential_client_dp_sequence_20260824T125133Z_seed20260901\sequence_metrics.json`.
