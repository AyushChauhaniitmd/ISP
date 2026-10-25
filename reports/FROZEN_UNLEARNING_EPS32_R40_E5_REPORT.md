# Frozen Unlearning Study: CIFAR-10, Client-Level DP

## Executive summary

We completed the frozen unlearning experiment for all three preselected seeds
(`20260918`, `20260919`, and `20260920`) using the same configuration. Each run
passed the project's artifact validator, and the aggregate analysis is saved in
[`frozen_unlearning_analysis.json`](../results/frozen_unlearning_eps32_r40_e5/frozen_unlearning_analysis.json).

Across the three seeds, leaving the differentially private model unchanged
achieved **42.25% test accuracy** (analysis 95% interval: 40.99%-43.51%).
Retained-data fine-tuning had the highest average test accuracy at **44.12%**
(43.11%-45.12%), but its per-seed unlearning classification varied. Cached
update reconstruction reduced mean forgotten-client accuracy, but did not
consistently improve overall utility. FedEraser was classified as
`UNLEARNING-HARMFUL` on all three seeds and had lower test accuracy.

The evidence does **not** support a definitive claim that explicit unlearning
is redundant or universally beneficial. DP-only was classified as redundant in
two seeds and inconclusive in one; explicit-method classifications also vary
by seed except for FedEraser. Membership-inference confidence intervals include
0.5 AUC for every evaluated method, so the attack results do not establish
reliable member/non-member discrimination.

## Experimental setup

| Item | Setting |
|---|---|
| Dataset | CIFAR-10, 10 classes |
| Model | `SmallGroupNormCNN` |
| Clients / examples | 100 clients / 200 examples per client |
| Client heterogeneity | Dirichlet alpha = 0.5 |
| Federated training | 40 rounds; 5 local epochs; batch size 64; client sampling rate 0.5 |
| Privacy | Central client-level DP; epsilon = 32.00000008; delta = 1e-5; clip norm = 1 |
| Deletion request | Delete client 2 |
| Retrain reference | Primary retrain plus four additional retrain models per run |
| Seeds | 20260918, 20260919, 20260920 |
| Runtime environment | Python 3.11.9; PyTorch 2.11.0+cu128; NVIDIA GeForce RTX 2050 |

The three selected run artifacts are:

- `multiclass_cnn_frozen_unlearning_eps32_r40_e5_20261012T190106Z_seed20260918`
- `multiclass_cnn_frozen_unlearning_eps32_r40_e5_20261012T201303Z_seed20260919`
- `multiclass_cnn_frozen_unlearning_eps32_r40_e5_20261012T203117Z_seed20260920`

## Results across seeds

Accuracy is shown as mean with the analysis JSON's 95% across-seed interval.
Forgotten-client and retained-client accuracy are means. MIA is the loss-based,
label-matched forgotten-versus-unseen attack; its AUC interval is also reported.
Lower test JS divergence means greater predictive alignment with the retrain
reference.

| Method | Test accuracy | Forgotten accuracy | Retained accuracy | Forgotten MIA AUC | Test JS to retrain |
|---|---:|---:|---:|---:|---:|
| DP-only / no action | 42.25% [40.99, 43.51] | 50.67% | 43.87% | 0.543 [0.487, 0.599] | 0.1203 |
| Retained-data fine-tuning | 44.12% [43.11, 45.12] | 52.17% | 46.33% | 0.537 [0.475, 0.599] | 0.1144 |
| Cached update reconstruction | 41.38% [39.69, 43.07] | 40.33% | 43.12% | 0.519 [0.460, 0.577] | 0.1234 |
| FedEraser | 37.05% [36.09, 38.01] | 42.00% | 37.97% | 0.530 [0.462, 0.599] | 0.1150 |
| Exact retrain reference | 42.31% [41.76, 42.86] | 44.83% | 44.10% | 0.504 [0.473, 0.534] | 0.0000 |

### Per-seed test accuracy

| Method | Seed 20260918 | Seed 20260919 | Seed 20260920 |
|---|---:|---:|---:|
| DP-only / no action | 43.37% | 42.23% | 41.15% |
| Retained-data fine-tuning | 44.82% | 44.41% | 43.12% |
| Cached update reconstruction | 41.95% | 42.50% | 39.68% |
| FedEraser | 36.15% | 37.83% | 37.18% |
| Exact retrain reference | 42.40% | 42.74% | 41.78% |

### Marginal Unlearning Benefit classifications

| Method | Seed 20260918 | Seed 20260919 | Seed 20260920 | Interpretation |
|---|---|---|---|---|
| DP-only / no action | Inconclusive | Redundant | Redundant | DP-only fell within the benchmark's retrain-variability criteria on two seeds, but not all. |
| Retained-data fine-tuning | Beneficial | Beneficial | Harmful | Classification is seed-sensitive; it is not a consistent win. |
| Cached update reconstruction | Beneficial | Harmful | Beneficial | Classification is seed-sensitive; the mean forgotten accuracy is lower, but mean test accuracy is also lower than DP-only. |
| FedEraser | Harmful | Harmful | Harmful | Consistent harm under this configuration; mean test accuracy is 5.20 percentage points below DP-only. |

The classifier uses retrain-variability bounds and MUB criteria. In particular,
it does not classify an algorithm solely from a lower forgotten accuracy or a
higher test accuracy. The current report therefore preserves the per-seed labels
instead of treating one label as a population-level conclusion.

## Interpretation

1. **This is a useful, non-collapsed DP setting.** The exact-retrain median
   validity gate passed in every run. DP-only test accuracy is more than four
   times the 10% CIFAR-10 chance level.
2. **DP-only is close to the retrain reference in average test utility.** The
   means are 42.25% and 42.31%, respectively. Its redundancy classification
   nevertheless varied for seed 20260918, so describe the finding as
   *suggestive*, not conclusive.
3. **Fine-tuning improves average test accuracy by 1.87 percentage points
   over DP-only**, and improves retained accuracy, but does not lower average
   forgotten-client accuracy in these runs (52.17% versus 50.67%). Its MUB
   classification changes to harmful on the third seed. This is not evidence
   of robust forgetting.
4. **Cached reconstruction has the lowest mean forgotten-client accuracy**
   among the explicit methods (40.33%), but averages 0.87 percentage points
   lower test accuracy than DP-only. It uses stored, sensitive client updates;
   its exposure and storage costs differ from DP-only.
5. **FedEraser underperforms on utility in this experiment.** Its mean test
   accuracy is 37.05%, versus 42.25% for DP-only, and its MUB label is harmful
   on every seed. The observed result is specific to this implementation and
   configuration, not a general statement about FedEraser.
6. **The MIA evidence is inconclusive.** Each method's analysis 95% interval
   for forgotten-versus-unseen AUC includes 0.5. Do not claim a reliable
   attack-based privacy improvement from these three seeds.

## Comparison with the earlier archived three-seed Colab study

The notebook contains an earlier analysis for the same nominal configuration.
Comparing its reported means with this completed local three-seed analysis:

| Method | Earlier archived mean | Current local mean | Difference |
|---|---:|---:|---:|
| DP-only / no action | 42.38% | 42.25% | -0.13 pp |
| Retained-data fine-tuning | 44.05% | 44.12% | +0.07 pp |
| Cached update reconstruction | 41.47% | 41.38% | -0.09 pp |
| FedEraser | 36.79% | 37.05% | +0.26 pp |
| Exact retrain reference | 41.73% | 42.31% | +0.58 pp |

The close means are a useful reproducibility check, not a paired statistical
test. The archived notebook used a prior Colab bundle snapshot, while the local
runs use the current workspace implementation; treat any implementation
differences as a caveat.

## Validity, reproducibility, and limitations

- All three current run directories passed `scripts/validate_artifact.py`.
- The analyzer selected exactly one latest run for each of the three requested
  seeds; it did not mix in older timestamped runs.
- `frozen_unlearning_analysis.json` contains the seed-level metrics, aggregate
  intervals, attack results, MUB values, and classifications.
- This is three seeds for one frozen setting, not a sweep over privacy budgets,
  heterogeneity, deletion size, or algorithms.
- Confidence intervals based on only three independent training seeds are
  necessarily imprecise. The attack intervals also reflect the benchmark's
  held-out attack evaluation and should not be confused with a proof of privacy.
- Retained-data fine-tuning and FedEraser calibration access retained client
  data after the request; cached reconstruction uses historical client updates.
  These are not equivalent exposure regimes or free DP post-processing.
- This benchmark is an exploratory CIFAR-10 federated study, not a
  state-of-the-art accuracy result, legal erasure guarantee, or certified
  unlearning result.

## Conclusion

For this setting, the most defensible conclusion is that client-level DP at
epsilon approximately 32 leaves a useful model whose average test accuracy is
close to exact retraining, and DP-only is classified as redundant for two of
three seeds. Explicit unlearning is not uniformly superior: fine-tuning has a
small average utility gain but variable MUB labels and no reduction in mean
forgotten accuracy; cached reconstruction lowers forgotten accuracy but has a
utility tradeoff; and FedEraser harms test utility consistently. The study is
promising but does not establish a general redundancy frontier or robust
membership-privacy improvement.
