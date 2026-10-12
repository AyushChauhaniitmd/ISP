# Multiclass CIFAR-10 Client-DP Utility Report

**Date:** 2026-09-18  
**Model:** `SmallGroupNormCNN`  
**Runtime:** Colab Tesla T4  
**Purpose:** Utility diagnosis before federated unlearning

## Executive Summary

The model and execution pipeline are working. The non-private control achieves 43.83%-46.00% accuracy across three seeds, with a mean of **44.82%**. This is well above the 10% CIFAR-10 chance level and confirms that the multiclass data path, CNN, federated loop, and T4 runtime are functioning.

The completed 60-round finite-DP sweep gives:

| Configuration | Mean accuracy | Status |
|---|---:|---|
| Non-private, 60 rounds, 3 seeds | **44.82%** | `VALID` |
| epsilon=32, 60 rounds, 3 seeds | **36.50%** | `WARNING` |
| epsilon=16, 60 rounds, 3 seeds | **28.79%** | `WARNING` |
| epsilon=8, 60 rounds, 3 seeds | **19.45%** | `INVALID` |

The best completed finite-DP point is epsilon=32 with 60 rounds. It retains substantial task signal but remains below the current `VALID >= 40%` threshold. The study is not yet ready for the main unlearning comparison, but it has now established a meaningful privacy-utility curve instead of universal chance-level behavior.

## 1. Setup

The reported runs use:

- CIFAR-10, 10 classes;
- 100 simulated clients;
- 200 samples per client;
- Dirichlet heterogeneity `alpha=0.5`;
- `SmallGroupNormCNN`;
- three local epochs;
- local batch size 64;
- participation rate `q=0.5`;
- client-level central DP with clipped client updates and server Gaussian noise;
- fixed public aggregation normalizer `q*N`;
- delta `1e-5`;
- accountant-calibrated actual epsilon values;
- three seeds for the completed 60-round and non-private studies.

Validity policy:

- `INVALID`: accuracy below 25%;
- `WARNING`: 25% to below 40%;
- `VALID`: at least 40%.

These are training-only results. No deletion, MUB, or unlearning conclusion is drawn from them.

## 2. Non-Private Control

| Seed | Accuracy | Status |
|---:|---:|---|
| 20260918 | 46.00% | `VALID` |
| 20260919 | 44.63% | `VALID` |
| 20260920 | 43.83% | `VALID` |
| **Mean** | **44.82%** | **`VALID`** |

The stable three-seed result shows that the CNN is learning meaningful multiclass structure without DP. The 44.82% result should not be compared with centralized SOTA CIFAR-10 accuracy: this is federated training with 20,000 training images, non-IID partitions, partial participation, limited rounds, and a compact CNN.

## 3. Completed 60-Round DP Sweep

| Epsilon | Seed 20260918 | Seed 20260919 | Seed 20260920 | Mean | Status pattern |
|---:|---:|---:|---:|---:|---|
| 8 | 17.82% | 19.17% | 21.37% | **19.45%** | `INVALID` for all seeds |
| 16 | 27.18% | 28.74% | 30.45% | **28.79%** | `WARNING` for all seeds |
| 32 | 35.45% | 35.79% | 38.26% | **36.50%** | `WARNING` for all seeds |

The accountant reported actual epsilon values of 8.0000, 16.0000, and 32.0000. This confirms that the sweep used the intended privacy points rather than relying on nominal noise settings.

## 4. Rounds Ablation

The rounds experiment uses:

```text
rounds = [40, 80, 120]
epsilons = [8, 16, 32]
seeds = [20260918, 20260919, 20260920]
```

The supplied output is complete for seed 20260918 and incomplete for seed 20260919. No seed 20260920 results were supplied. Therefore, this section is descriptive and not a final three-seed claim.

### Seed 20260918

The runner loops over rounds first and epsilon second:

| Rounds | epsilon=8 | epsilon=16 | epsilon=32 |
|---:|---:|---:|---:|
| 40 | 20.51% | 31.45% | **37.97%** |
| 80 | 18.48% | 26.63% | 35.88% |
| 120 | 12.60% | 19.78% | 32.50% |

### Partial seed 20260919 output

| Rounds | epsilon=8 | epsilon=16 | epsilon=32 |
|---:|---:|---:|---:|
| 40 | 22.09% | 29.28% | 33.90% |
| 80 | 19.67% | 27.19% | 34.53% |
| 120 | 17.48% | 25.46% | missing |

The missing 120-round epsilon=32 result for seed 20260919 and all seed 20260920 results must not be imputed.

## 5. Interpretation

### The CNN learns under finite DP

The finite-DP results are not all at chance:

- epsilon=32 produces 32.50%-38.26% in the displayed runs;
- epsilon=16 produces 27.18%-30.45% in the completed 60-round run;
- epsilon=8 remains weak and below the invalidity threshold.

This is a material improvement over the earlier Phase 7 private results near 10%.

### Utility decreases as privacy strengthens

At 60 rounds:

```text
epsilon=32: 36.50% mean
epsilon=16: 28.79% mean
epsilon=8:  19.45% mean
```

This is the expected privacy-utility tradeoff: smaller epsilon requires more DP noise and produces lower accuracy.

### More rounds are not automatically better

For seed 20260918, with total epsilon held fixed:

| Epsilon | 40 rounds | 80 rounds | 120 rounds |
|---:|---:|---:|---:|
| 8 | 20.51% | 18.48% | 12.60% |
| 16 | 31.45% | 26.63% | 19.78% |
| 32 | 37.97% | 35.88% | 32.50% |

Increasing rounds requires stronger per-round noise to preserve the same total epsilon. In this configuration, that reduced per-round signal-to-noise ratio outweighs the benefit of additional updates. This effect requires completion across all three seeds before it is treated as confirmed.

## 6. Best Current Configuration

Best displayed finite-DP point:

```text
40 rounds, epsilon=32, seed=20260918, accuracy=37.97%
```

Best completed three-seed configuration:

```text
60 rounds, epsilon=32, mean accuracy=36.50%
```

Neither passes the 40% `VALID` threshold, but both are substantially above chance and scientifically useful for utility diagnosis.

## 7. What Is Established

The current evidence supports these claims:

1. The Colab T4 execution path works.
2. The current CNN learns multiclass CIFAR-10 above chance.
3. Non-private training is stable across three seeds.
4. Finite client-level DP at epsilon=32 retains substantial task signal.
5. Epsilon=16 retains moderate but marginal utility.
6. Epsilon=8 is invalid under the current gate.
7. Accountant-calibrated epsilon matches each target.
8. More rounds are not beneficial in the displayed fixed-epsilon ablation.

## 8. What Is Not Established

These results do not establish:

- that DP-only performs unlearning;
- that explicit unlearning is redundant;
- that epsilon=32 is a universal utility threshold;
- that the rounds effect is stable across seeds;
- that the current CNN is optimal;
- that the current `WARNING` regime is sufficient for the final unlearning paper claim.

## 9. Recommended Next Step

First complete the missing rounds-ablation results, especially all seed 20260920 rows and the missing seed 20260919 row. Then compute means and standard deviations for every `(rounds, epsilon)` cell.

Based on the completed evidence, prioritize:

1. 40 or 60 rounds;
2. epsilon=32;
3. a controlled learning-rate or clipping-norm check if a `VALID` point above 40% is required;
4. GroupNorm ResNet-18 only if a higher utility ceiling is still needed.

Do not run the full unlearning comparison until the healthy configuration is frozen. The current results justify continued utility diagnosis, but not yet a confirmed redundancy frontier.

## Final Interim Conclusion

The earlier conclusion that the multiclass DP model simply fails at chance level is no longer correct for this configuration. The current CNN achieves meaningful finite-DP utility at epsilon=16 and especially epsilon=32, while the non-private control averages 44.82%.

The remaining limitation is that the current client-level-DP configuration produces a moderate-utility `WARNING` regime rather than a high-utility `VALID` regime. The model is not fundamentally broken; the remaining problem is the privacy/training signal-to-noise tradeoff. 

## Frozen Candidate for Unlearning

The subsequent three-seed local-epoch ablation selected the following frozen candidate:

```text
model        = SmallGroupNormCNN
epsilon      = 32
rounds       = 40
local_epochs = 5
seeds        = 20260918, 20260919, 20260920
```

Its accuracies were 43.03%, 41.28%, and 42.84%, with a mean of 42.38%. This is approximately 94.6% of the observed 44.82% non-private mean. The candidate is now suitable for the first unlearning comparison, while the final redundancy conclusion remains pending.
