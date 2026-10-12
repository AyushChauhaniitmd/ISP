# DP-ForgetBench
## When Is Explicit Client Unlearning Redundant?
### A privacy-aligned audit of federated learning

**Presentation audience:** FA / research review  
**Date:** 18 September 2026  
**Status:** Reproducible benchmark, validated utility gate, first frozen unlearning study

---

# 1. The Problem

A federated client may later request deletion.

There are three operational choices:

- **Do nothing:** release the existing client-level DP model unchanged (`M_DP`).
- **Unlearn:** apply an explicit federated unlearning method (`M_U`).
- **Retrain:** train from scratch without the deleted client (`M_R`).

Exact retraining is the clean reference, but it is expensive. The research question is:

> Under client-level differential privacy, does explicit unlearning provide meaningful benefit beyond leaving the DP model unchanged?

**Important boundary:** DP is not claimed to be exact deletion, legal deletion, or a certified unlearning method.

---

# 2. Why This Question Matters

The literature often compares unlearning methods against weak or single-run references.

Three risks follow:

1. A chance-level model can appear to have “forgotten” data.
2. One arbitrary retrain can be mistaken for ground truth.
3. A method can reduce forgotten accuracy while harming retained utility or privacy.

DP-ForgetBench addresses these risks with:

- a client-level privacy contract;
- independent retrain ensembles;
- validity gates before unlearning interpretation;
- forgotten, retained, and unseen privacy populations;
- utility, alignment, attack, and lifecycle-cost metrics.

---

# 3. What Is Novel Here?

This is **not** a claim that client-level DP or federated unlearning is new.

The intended novelty is an evaluation framework for discovering a **redundancy frontier**:

```text
(epsilon, heterogeneity, deletion size, request sequence)
                         |
                         v
       REDUNDANT / BENEFICIAL / HARMFUL / INCONCLUSIVE
```

Core contributions:

1. **Privacy-scope alignment:** client deletion is evaluated under client-level central DP.
2. **Retrain-variability reference:** independent retrains define natural stochastic variability.
3. **Marginal Unlearning Benefit:** explicit unlearning is compared directly with DP-only.
4. **Tri-population audit:** forgotten, retained, and unseen populations are separated.
5. **Lifecycle accounting:** storage, communication, runtime, and post-request data access are visible.
6. **Reproducible artifacts:** configs, manifests, ledgers, raw metrics, tests, and Colab/GPU workflows.

---

# 4. Privacy Contract

Primary mechanism:

1. Poisson-sample clients.
2. Train local client updates.
3. L2-clip each complete client update.
4. Sum clipped updates.
5. Add Gaussian noise at the server.
6. Divide by fixed public `q*N` normalization.
7. Account with the Poisson-sampled Gaussian RDP accountant.

**Adjacency:** add/remove one complete client dataset.  
**Released object:** final global model in the primary contract.  
**Delta:** `1e-5`.  
**Secure aggregation:** not implemented in the simulator.

Methods that re-query retained raw data or use sensitive historical updates are classified as separate exposure regimes, not free DP post-processing.

---

# 5. Experimental Logic

Before interpreting unlearning, the model must learn.

## Regime 1: Utility collapse

Accuracy near CIFAR-10 chance, approximately 10%.

Low attack signal here is vacuous: the model did not learn enough for deletion to matter.

## Regime 2: DP-dominated redundancy

The model is useful, but DP masks the deleted client's effect:

```text
M_DP is within normal M_R versus M_R' variability
```

## Regime 3: Unlearning-sensitive

The deleted client changes behavior beyond retraining variability, and explicit unlearning may provide material benefit.

**Validity gate:** `INVALID <25%`, `WARNING 25-40%`, `VALID >=40%`.

---

# 6. What We Built

## Phase 0-1: infrastructure and real-data pilot

- deterministic federated simulator;
- client-level DP and RDP ledger;
- deletion manifests with checksums;
- exact and independent retraining;
- fine-tuning and cached reconstruction;
- utility, JS, MIA, and cost diagnostics;
- corrected binary CIFAR-10 exploratory grid: 30 runs, five seeds per cell.

## Phase 2+: multiclass and validation

- fixed spatial-information loss in the CNN;
- real 10-class CIFAR-10 backend;
- GroupNorm CNN and GroupNorm ResNet path;
- retrain ensembles and validity gating;
- MUB and four-way classification;
- FedEraser-style baseline;
- positive convergence and positive-unlearning controls;
- T4/Colab reproducible execution.

---

# 7. The Utility Gate Was Necessary

Earlier private multiclass runs at ε=1-8 reached approximately 9.5-11.4% accuracy.

Those runs were correctly marked `INVALID`.

The 54.55% pilot was multiclass CIFAR-10, but its recorded epsilon was approximately **410.13**. It was a high-utility, practically non-private control, not evidence of strong privacy.

The current controlled configuration established a non-vacuous regime before unlearning.

---

# 8. Healthy Frozen Configuration

The selected candidate was frozen before the unlearning comparison:

```text
Model        SmallGroupNormCNN
Clients      100
Samples      200/client
Heterogeneity alpha=0.5
Privacy      epsilon ~= 32, delta=1e-5
Rounds       40
Local epochs 5
Participation q=0.5
Seeds        20260918, 20260919, 20260920
```

Non-private control mean: **44.82%**  
Frozen DP-only mean: **42.38%**  
Frozen DP utility relative to observed non-private mean: **94.6%**

Three-seed frozen DP-only accuracy:

| Seed | Accuracy |
|---:|---:|
| 20260918 | 43.03% |
| 20260919 | 41.28% |
| 20260920 | 42.84% |

---

# 9. First Unlearning Comparison

| Method | Mean test accuracy | Mean forgotten accuracy |
|---|---:|---:|
| DP-only / no action | 42.38% | 51.00% |
| Retained fine-tuning | **44.05%** | 50.83% |
| Cached reconstruction | 41.47% | 40.50% |
| FedEraser-style | 36.79% | 42.83% |
| Exact retrain reference | 41.73% | 43.50% |

Interpretation:

- Fine-tuning improves utility but does not consistently lower forgotten behavior.
- Cached reconstruction lowers forgotten accuracy but overshoots alignment and requires sensitive history.
- FedEraser loses approximately 5.6 percentage points versus DP-only.
- DP-only is only 0.65 percentage points above retrain in mean test accuracy.

Forgotten accuracy is not itself a privacy guarantee; it must be read with MIA and tri-population results.

---

# 10. Audited MIA Results

The attack audit now uses:

- label-matched member versus unseen populations;
- calibration/evaluation split separation;
- threshold selected on calibration data;
- held-out advantage evaluation;
- held-out AUC;
- bootstrap CIs;
- explicit population and score definitions.

Latest three-seed means:

| Method | Forgotten AUC | Forgotten advantage |
|---|---:|---:|
| DP-only | 0.535 | 0.062 |
| Fine-tuning | 0.526 | 0.037 |
| Cached reconstruction | 0.516 | -0.019 |
| FedEraser | 0.525 | 0.062 |
| Retrain | 0.516 | 0.053 |

The intervals are wide with only three seeds. These results do not demonstrate a statistically reliable privacy improvement for any explicit method over DP-only.

---

# 11. MUB and Cost Reading

| Method | Accuracy MUB | JS MUB | MIA MUB | Practical reading |
|---|---:|---:|---:|---|
| Fine-tuning | +0.0167 | +0.0061 | -0.0026 | Utility gain, no clear privacy gain |
| Cached reconstruction | -0.0092 | -0.0077 | -0.0060 | Lower attack score, worse alignment/utility |
| FedEraser | -0.0559 | -0.0027 | -0.0226 | Harmful utility tradeoff, no clear privacy gain |

Operational caveat:

- historical update storage is approximately 497 MB for the compact model/run;
- FedEraser communication is approximately 965 MB;
- DP-only requires no post-request action and should not be charged historical storage merely because other baselines collect shared history.

---

# 12. What the Current Evidence Says

The strongest defensible statement is:

> At ε≈32 with 40 rounds and 5 local epochs, DP-only is a strong baseline. Fine-tuning improves utility but not clearly forgotten-client privacy. Cached reconstruction reduces attack scores but moves away from the retraining reference and incurs historical-storage exposure. FedEraser substantially reduces utility without a demonstrated privacy advantage.

What we **do not** claim:

- DP perfectly deletes data.
- DP always makes unlearning unnecessary.
- A universal epsilon threshold.
- Certified or legal deletion.
- Final redundancy frontier confirmation.

Current classification: **evidence is suggestive but still inconclusive**.

---

# 13. Transparency and Limitations

- Current main study is compact CNN on CIFAR-10, not a universal FL benchmark.
- Only three independent seeds in the frozen study.
- Attack CIs are wide.
- The tri-population probe is a baseline logistic probe, not a TC-UMIA reproduction.
- Secure aggregation is not implemented.
- Fine-tuning and FedEraser access retained raw data after the request.
- Historical updates are sensitive server-side artifacts.
- Additional datasets and stronger baseline reproductions remain needed.

This transparency is part of the contribution: the benchmark is designed to prevent overclaiming from invalid or incomplete cells.

---

# 14. Next Research Stage

1. Aggregate the tri-population probe: accuracy, macro AUC, confusion matrix, and CIs.
2. Correct cost attribution for DP-only versus shared historical baselines.
3. Confirm transition or harmful cells with 10 seeds.
4. Add controlled heterogeneity and deletion-size strata.
5. Evaluate partial and sequential deletion requests.
6. Reproduce stronger published FU baselines under matched threat models.
7. Extend beyond CIFAR-10 to a natural client-partitioned dataset.

---

# 15. Closing Message

We are not claiming to have proved that DP makes unlearning unnecessary.

We have built and tested the machinery needed to ask the question correctly:

```text
client-level DP
+ exact retrain variability
+ explicit unlearning baselines
+ audited privacy populations
+ lifecycle cost
                         |
                         v
       empirical redundancy frontier
```

**Final research question:**

> When does client-level DP already suppress a client's influence enough that explicit unlearning provides no meaningful additional benefit—and when does it still fail to do so?

**One-line novelty statement:**

> We are not proposing another unlearning algorithm; we are systematically characterizing when client-level DP makes explicit federated unlearning redundant, beneficial, or harmful using retrain variability, privacy audits, heterogeneity, deletion scope, request sequences, and lifecycle cost.

---

# Appendix A. Detailed Result Values for Slide Notes

# Appendix B. Attack Metric Definitions

## Loss-based membership inference

The attack score is the **negative per-example cross-entropy loss**. Lower loss is more member-like, so higher negative loss is a stronger membership score.

Members and unseen non-members are label-matched, then split into calibration and held-out evaluation populations. The attack threshold is selected only on calibration data. AUC and the reported advantage are evaluated on the held-out split.

## AUC

Receiver Operating Characteristic Area Under the Curve. AUC measures how well the attack ranks members above non-members across all thresholds:

```text
AUC = 0.5  -> chance ranking
AUC = 1.0  -> perfect ranking
```

AUC is threshold-independent, but it can hide behavior at very low false-positive rates.

## Attack advantage

The reported advantage is:

```text
Advantage = held-out TPR - held-out FPR
```

The threshold is selected on the calibration split by maximizing `TPR-FPR`, then applied once to the held-out evaluation split. A positive value means the attack detects more members than false positives at that selected operating point.

## Low-FPR TPR

- `TPR @ 1% FPR`: fraction of members detected while the false-positive rate is at most 1%.
- `TPR @ 0.1% FPR`: stricter high-confidence operating point.

These metrics matter because an attack may have moderate aggregate AUC but be useful to an adversary only at a specific low false-positive operating point.

## Population definitions

- **Forgotten:** examples belonging to the deleted client.
- **Retained:** examples belonging to clients that remain after deletion.
- **Unseen:** held-out examples never used by any training client.

The loss MIA compares label-matched forgotten or retained members against label-matched unseen examples. The tri-population probe classifies forgotten, retained, and unseen populations from pre/post deletion loss-confidence features.

## Tri-population probe

The baseline probe uses four features per example:

1. pre-deletion loss;
2. post-deletion loss;
3. pre-deletion confidence;
4. post-deletion confidence.

A multinomial logistic regression is trained on a stratified 50/50 split and evaluated on the held-out half. It reports:

- probe accuracy;
- confusion matrix;
- macro one-vs-rest AUC;
- bootstrap 95% intervals for accuracy and macro AUC.

This is a transparent baseline probe, not a claim to reproduce TC-UMIA.

## How to interpret the attack table

Low AUC or advantage is not sufficient to claim successful unlearning. A chance-level model may also produce low attack signal. The correct order is:

```text
utility gate -> held-out attack metrics -> retained-set audit -> cost -> conclusion
```

The current frozen study has three independent seeds, so intervals are wide. The strongest current claim is that no explicit method has demonstrated a reliable forgotten-set privacy improvement over DP-only in this tested regime.

## Utility curve

| Privacy point | Mean accuracy | Interpretation |
|---|---:|---|
| Non-private | 44.82% | Stable learning control |
| epsilon=32 | 36.50% at 60 rounds; 42.38% frozen candidate | Best finite-DP utility regime tested |
| epsilon=16 | 28.79% at 60 rounds | Marginal utility |
| epsilon=8 | 19.45% at 60 rounds | Invalid under the predeclared gate |

## Local-epoch ablation

At epsilon=32 and 40 rounds:

| Local epochs | Mean accuracy |
|---:|---:|
| 2 | 32.12% |
| 4 | 40.15% |
| 5 | 42.38% |

The 2-to-5 epoch improvement is **10.26 percentage points**. This was the final controlled utility change before freezing the unlearning configuration.

## Frozen unlearning means

| Method | Test accuracy | Forgotten accuracy | Forgotten AUC | Forgotten advantage |
|---|---:|---:|---:|---:|
| DP-only | 42.38% | 51.00% | 0.535 | 0.062 |
| Fine-tuning | 44.05% | 50.83% | 0.526 | 0.037 |
| Cached reconstruction | 41.47% | 40.50% | 0.516 | -0.019 |
| FedEraser | 36.79% | 42.83% | 0.525 | 0.062 |
| Exact retrain | 41.73% | 43.50% | 0.516 | 0.053 |

## MUB means

| Method | Accuracy MUB | JS MUB | MIA MUB |
|---|---:|---:|---:|
| Fine-tuning | +0.0167 | +0.0061 | -0.0026 |
| Cached reconstruction | -0.0092 | -0.0077 | -0.0060 |
| FedEraser | -0.0559 | -0.0027 | -0.0226 |

## Cost notes

- DP-only: no post-request action; historical storage should be treated as 0 MB for the method itself.
- Fine-tuning: approximately 47.4 MB communication in the frozen runs; accesses retained raw data.
- Cached reconstruction: approximately 497 MB historical update storage; no post-request communication.
- FedEraser: approximately 497 MB historical storage and 965 MB communication; requires retained-client calibration.
- Exact retrain: approximately 482 MB communication in the recorded frozen runs; internal reference cost.

## Caveats to say aloud

- The confidence intervals are across three independent seeds, not a formal population-level guarantee.
- The attack is a calibrated baseline loss-MIA and tri-population probe, not a full TC-UMIA reproduction.
- The current result is a frozen-regime finding, not a universal epsilon threshold.
- The strongest current conclusion is “no reliable privacy benefit detected beyond DP-only,” not “DP makes unlearning unnecessary.”
