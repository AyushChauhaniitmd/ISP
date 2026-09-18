# DP-ForgetBench: A-to-Z Repository Report

**Report date:** 2026-09-18  
**Repository:** DP-ForgetBench  
**Working title:** *When is explicit client unlearning redundant? A privacy-aligned audit of differentially private federated learning*

## Executive Summary

DP-ForgetBench is a reproducible benchmark and audit framework for client deletion in federated learning (FL). Its central question is:

> Under client-level differential privacy (DP), when does explicit federated unlearning add a measurable benefit over leaving the released DP model unchanged?

The repository does **not** introduce a new unlearning algorithm. Its intended contribution is a privacy-aligned evaluation protocol that compares:

- **M_DP:** the client-level DP model left unchanged after a deletion request;
- **M_U:** an explicit unlearning baseline;
- **M_R:** exact retraining from scratch on retained clients; and
- independent retrains that measure ordinary retraining variability.

The strongest completed empirical result is the corrected Phase 1 real CIFAR-10 binary screen: across six cells and five seeds per cell, retained-data fine-tuning and cached reconstruction did not produce a practically material improvement in predictive Jensen-Shannon (JS) alignment to retraining over DP-only. The later multiclass Phase 7 screen is useful mainly as a scientific warning: private runs at epsilon 1-8 were near the 10% CIFAR-10 chance level and were correctly marked invalid for unlearning conclusions.

The repository contains positive controls showing that the model and pipeline can learn, plus a positive unlearning control showing that deletion-sensitive behavior can be detected. However, the broad claim that FedEraser is redundant throughout healthy private client-DP regimes is not yet fully established by the strongest validated artifacts. The 54.55% pilot that initially motivated that claim has a recorded epsilon of approximately 410.13 and is therefore practically non-private.

## A. Aim and Research Question

The project studies the relationship between two different guarantees:

1. **Differential privacy:** limits how much the released model changes when one protected unit changes.
2. **Machine unlearning:** attempts to make a model behave as if a requested data unit had never been used.

The protected unit in the primary study is a complete client dataset. The deletion unit is also a complete client, so the privacy and deletion scopes are aligned. The project asks whether DP noise can make a deletion effect smaller than normal retraining variability, and whether explicit unlearning is worth its additional compute, communication, storage, and privacy exposure.

The project deliberately rejects these stronger statements:

- DP perfectly deletes data.
- DP always makes unlearning unnecessary.
- DP-only is itself a certified unlearning algorithm.
- A universal epsilon threshold exists.
- A method that re-queries retained raw data is pure DP post-processing.

Primary references: [README.md](README.md), [RESEARCH_GRADE_PLAN.md](RESEARCH_GRADE_PLAN.md), [LIMITATIONS.md](LIMITATIONS.md), and [docs/THREAT_MODEL.md](docs/THREAT_MODEL.md).

## B. Scientific Positioning and Novelty

The repository correctly recognizes that “DP + FL + unlearning + membership inference” is not, by itself, novel. The proposed novelty is methodological and evaluative:

1. **Privacy-aligned redundancy frontier:** map when DP-only is close to retained-data retraining and when explicit unlearning changes the result, across privacy, heterogeneity, deletion size, and request sequence.
2. **Retrain-ensemble reference:** compare against a distribution of independent retrains rather than treating one arbitrary retrain as perfect ground truth.
3. **Multi-endpoint comparison:** jointly evaluate utility, predictive alignment, forgotten-set privacy, retained-set privacy, and lifecycle cost.
4. **Tri-population privacy audit:** distinguish forgotten, retained, and unseen populations instead of auditing only the deleted client.
5. **Privacy-scope mismatch audit:** keep client-level DP separate from record-level DP and explicitly classify post-request data access.
6. **Reproducible artifact model:** preserve configs, hashes, deletion manifests, privacy ledgers, raw metrics, tests, and result indexes.

This is potentially publishable as a benchmark/evaluation paper even without a new algorithm. The novelty is conditional on completing confirmatory experiments and stronger baseline/attack reproductions.

## C. Privacy Contract

The primary mechanism is central client-level DP:

1. Sample clients with Poisson participation.
2. Train each selected client locally.
3. Compute the client model update.
4. L2-clip each update to a public clipping norm.
5. Sum clipped updates.
6. Add Gaussian noise at the server.
7. Divide by a fixed public `q * N` normalizer, not the realized number of sampled clients.
8. Track privacy with an RDP accountant for the Poisson-sampled Gaussian mechanism.

The privacy ledger records the mechanism, epsilon, delta, sampling rate, rounds, clipping norm, noise multiplier, and release assumptions. Exact retraining uses a retained-client population normalizer and is treated as an internal reference rather than an additional public release.

Important contract boundaries:

- DP-SGD at the example level does not prove client-level removal.
- Secure aggregation is not implemented in the simulator.
- Retained fine-tuning and FedEraser calibration access retained raw data after the request and therefore are not DP post-processing of the final DP model.
- Cached reconstruction uses sensitive per-client historical updates and has a separate exposure model.

## D. Data and Experimental Scenarios

The repository contains several experimental tiers:

- **Phase 0:** synthetic federations for plumbing, determinism, deletion, and sequence behavior.
- **Phase 1:** real CIFAR-10 binary classification using public average-pooled features, 12 simulated clients, one full-client deletion, epsilon values 2, 8, and infinity, two heterogeneity settings, and five seeds per cell.
- **Phase 2 onward:** multiclass CIFAR-10 using the small GroupNorm CNN, Dirichlet client heterogeneity, retrain ensembles, DP sweeps, unlearning baselines, stress tests, and frontier artifacts.

The current empirical core remains CIFAR-10. FEMNIST, CIFAR-100, ResNet-18, and broader natural-FL validation remain planned rather than completed paper evidence.

Partitions and deletion manifests are generated deterministically and hashed with SHA256. The intended deletion strata include typical, influential, low-data, rare-class, partial, and sequential requests, although not all of these have a completed real-data grid.

## E. Model and Convergence Repair

The compact `SmallGroupNormCNN` has approximately 60,586 parameters and uses GroupNorm instead of BatchNorm to avoid client-local batch-statistics complications.

An early architecture used `AdaptiveAvgPool2d((1, 1))`, which destroyed most spatial information before classification. It was replaced by flattening the `64 x 8 x 8` feature map into the classifier. This was a root-cause repair, not a cosmetic tuning change.

The repository also fixed the real-image routing path, generalized JS divergence to multiclass softmax outputs, and corrected attack grouping that had assumed binary labels. The original architecture remains in the code as a regression comparison.

Relevant implementation: [src/dp_forgetbench/federated.py](src/dp_forgetbench/federated.py).

## F. Baseline Methods

### 1. Exact retrain, M_R

Retrain from scratch on the retained clients with the same training recipe. This is the operational gold-standard reference, not a claim of literal legal or certified deletion.

### 2. DP-only/no action, M_DP

Leave the full-data client-DP model unchanged after the deletion request. This is not called an unlearning algorithm. It tests whether the released DP model is already close to the retained-data reference.

### 3. Retained-data fine-tuning, M_FT

Start from the full model and train additional rounds on retained clients. It is cheap and useful as a baseline, but it accesses private retained data after deletion and needs a separate privacy analysis.

### 4. Cached reconstruction, M_CR

Remove the forgotten client's cached historical updates from the server-side trajectory without querying clients. It is fast but sensitive to trajectory drift and is not certified unlearning.

### 5. FedEraser-style reconstruction, M_FE

Use stored historical per-client updates, remove the forgotten client's contributions, and recalibrate retained-client update directions from the evolving unlearned model. The implementation preserves historical update magnitudes while using fresh retained-client directions. It requires historical storage and interactive retained-client calibration.

The method suite is implemented in [src/dp_forgetbench/federated.py](src/dp_forgetbench/federated.py) and orchestrated by [src/dp_forgetbench/run.py](src/dp_forgetbench/run.py).

## G. Evaluation Metrics

### Utility

- Test loss and accuracy.
- Forgotten-client loss and accuracy.
- Retained-data loss and accuracy.
- Per-client and distributional utility are supported by the design, though the main completed reports emphasize aggregate values.

### Functional alignment

Predictive JS divergence compares candidate outputs with the retrain target on test and forgotten data. The intended interpretation is distance to retained-data behavior, normalized against independent retrain variability where possible.

### Privacy

The implemented baseline probes report:

- Loss-based membership-inference AUC.
- Attack advantage, defined from the member/non-member separation.
- TPR at 1% and 0.1% false-positive rates.
- Forgotten, retained, and unseen population probes.
- A tri-population pre/post probe.

The repository includes an offline simplified LiRA module, but the standard grid reports should be treated as baseline loss/tri-population diagnostics, not as faithful LiRA/A-LiRA or TC-UMIA reproductions.

### Marginal Unlearning Benefit

For a candidate method `U`, MUB is intended to measure improvement over DP-only relative to retraining:

`MUB_j = distance_j(M_DP, M_R) - distance_j(M_U, M_R)`

Positive MUB means the explicit method moves closer to the retraining reference. The implementation reports accuracy, JS, and MIA variants and uses retrain-ensemble quantiles plus an equivalence margin for a four-way classifier:

- `REDUNDANT`
- `UNLEARNING-BENEFICIAL`
- `UNLEARNING-HARMFUL`
- `INCONCLUSIVE`

The implementation and the documentation should be read together: the current classifier uses practical thresholds and retrain bounds, but it is not a substitute for a preregistered multi-endpoint equivalence test.

### Cost

The run artifacts record runtime, client updates, communicated bytes, local examples, and persistent historical storage. This makes it possible to compare not only accuracy and privacy, but the complete request lifecycle.

## H. Phase-by-Phase Work Completed

### Phase 0: synthetic benchmark plumbing

Implemented deterministic synthetic data, client-level DP accounting, deletion manifests, exact and independent retraining, fine-tuning, cached reconstruction, initial MIA/JS diagnostics, and sequence state handling. These results are explicitly plumbing evidence, not final privacy claims.

### Phase 1: corrected binary CIFAR-10 benchmark

Completed 30 validated real-data runs: six cells, five seeds per cell, epsilon `{2, 8, infinity}`, heterogeneity `{0.2, 0.75}`, and one full-client deletion. The earlier Phase 1 v2 reference-normalizer error was identified and superseded by the corrected v3 artifacts.

### Phase 2: multiclass and convergence repair

Added the real-image backend, `SmallGroupNormCNN`, retrain ensembles, validity gating, MUB aggregation, multiclass attack fixes, and a DP hyperparameter sweep. The high-accuracy pilot exposed a privacy-accounting distinction: it learned well but had epsilon approximately 410.13.

### Phases 3-6: formalization and accounting

Formalized validity gates, retrain variability, MUB, redundancy classification, fixed public normalization, deletion-manifest integrity, artifact immutability, and DP accountant tests.

### Phase 7: multiclass factorial screen

Executed approximately 94 stored runs across 30 cells covering five epsilon regimes, three heterogeneity values, two deletion fractions, and three or four seeds depending on cell. Private epsilon 1-8 runs were near chance and marked invalid. The non-private screen was undertrained relative to later positive controls.

### Phase 8: privacy audit

Implemented forgotten, retained, and unseen population probes, AUC, advantage, low-FPR metrics, bootstrap support, and tri-population diagnostics. The attack suite is a strong baseline framework but not yet a faithful reproduction of every named published attack.

### Phase 9: unlearning baseline suite

Implemented exact retrain, DP-only, retained fine-tuning, cached reconstruction, and FedEraser-style calibration with explicit cost and privacy classifications.

### Phases 10-11: stress and frontier artifacts

Added sequential/influential-client stress-test artifacts, positive controls, utility-frontier summaries, and scripts for detecting transition cells. Confirmatory ten-seed transition-cell evidence is still a required research gate rather than a completed final result.

## I. Main Results

### Corrected Phase 1 result: strongest supported conclusion

The corrected binary CIFAR-10 screen found no practically material JS-to-retrain improvement from explicit unlearning over DP-only across all six cells.

Representative mean results:

| Epsilon | Heterogeneity | DP-only accuracy | Fine-tune accuracy | DP-only JS | Fine-tune JS | Fine-tune MUB-JS |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 0.2 | 0.549 | 0.585 | 0.286 | 0.287 | -0.0010 |
| 8 | 0.2 | 0.613 | 0.649 | 0.187 | 0.179 | +0.0084 |
| infinity | 0.2 | 0.776 | 0.782 | 0.0014 | 0.0016 | -0.0001 |
| infinity | 0.75 | 0.778 | 0.788 | 0.0017 | 0.0016 | +0.0001 |

The epsilon 8 fine-tuning accuracy gain is visible, but the primary alignment screen did not establish a material unlearning benefit. MIA AUCs were generally close to 0.5. The result is exploratory because the model is small, the task is binary, and the attack suite is baseline-level.

Source: [reports/PHASE1_REAL_CIFAR_RESULTS.md](reports/PHASE1_REAL_CIFAR_RESULTS.md).

### Phase 7 result: validity gate doing its job

Across the private multiclass screen, accuracy was approximately 9.5%-11.4%, near 10% random chance on CIFAR-10. The runs were marked `INVALID`, so their near-zero MUB values cannot prove redundancy. They show that low leakage or low deletion effect can be vacuous when the model has not learned useful signal.

The non-private screen averaged roughly 26.7% accuracy and was also below the later healthy positive controls. It is therefore useful as a diagnostic screen, not a final frontier result.

Source: [PHASE7_AUDIT.md](PHASE7_AUDIT.md) and [results/phase7_factorial/master_summary.csv](results/phase7_factorial/master_summary.csv).

### Positive controls

Later controls demonstrated that the architecture and data path can learn:

- Centralized CIFAR-10 training: 43.40% test accuracy.
- Standard non-private FedAvg: 44.35% test accuracy.
- Influential-client unlearning control: forgotten accuracy 82.5% before deletion, 68.0% after exact retraining, and 70.5% after retained fine-tuning.

These controls are important because they show that the benchmark can detect a meaningful deletion-sensitive effect when one exists.

### Pilot discrepancy

The 54.55% multiclass pilot used 100 clients, 100 rounds, five local epochs, full participation, and a noise multiplier of 0.5. Its privacy ledger reports epsilon 410.1266. It is therefore a high-utility, practically non-private control, not evidence of strong client-level DP at epsilon 1-8.

The later Phase 7 screen used 20 clients, 40 rounds, two local epochs, 50% participation, and much stronger noise for actual epsilon 1-8. The resulting utility collapse is explained by both the smaller signal pool and stronger effective noise, not by a failure of the architecture alone.

## J. Cost and Operational Findings

For the compact CNN, the repository reports approximately:

| Method | Persistent storage | Post-request rounds | Approximate runtime | Data/privacy exposure |
|---|---:|---:|---:|---|
| DP-only | 0 bytes | 0 | 0 seconds | Final-model client-DP ledger; pure post-processing |
| Retained fine-tuning | 0 bytes | 4 in the cost report | about 3.5 seconds | Re-queries retained raw data |
| Cached reconstruction | about 93.7 MB | 0 | about 0.1 seconds | Uses sensitive historical updates |
| FedEraser | about 93.7 MB | 30 calibration rounds in the cost report | about 7 seconds | Historical updates plus retained-client queries |
| Exact retrain | 0 bytes persistent history | 60 | about 28 seconds | Internal retained-data reference |

Historical storage scales with rounds, participating clients, and parameter count. The repository projects gigabyte-to-terabyte storage for larger models and federations. These are engineering projections, not independently reproduced production measurements.

The operational thesis is clear: when DP-only is already within the retrain variability band, explicit unlearning may offer little incremental value while adding storage, communication, compute, and privacy-accounting obligations.

## K. Quality Gates and Reproducibility

The pipeline records:

- Configuration hash.
- Seed and runtime metadata.
- Deletion manifest path and SHA256 checksum.
- Effective privacy configuration and reference privacy configuration.
- Privacy ledger.
- Utility, alignment, privacy, and cost metrics.
- Validity status.
- Method-specific privacy assumptions and limitations.

The tests cover DP accounting, data behavior, attacks, multiclass CNN behavior, LiRA plumbing, FedEraser behavior, statistics, privacy, and smoke execution. The project also includes no-overwrite/resume behavior and immutable run identifiers.

Primary commands:

```powershell
python -m pytest -q
python -m dp_forgetbench.run --config configs/phase0_toy.yaml --mode full
python scripts/validate_artifact.py results\<run-directory>
python scripts/check_grid_results.py
```

The corrected Phase 1 reproduction sequence is documented in [docs/RESEARCH_EXECUTION_STATUS.md](docs/RESEARCH_EXECUTION_STATUS.md).

## L. Limitations and Threats to Validity

1. Most real-data evidence is CIFAR-10; external validity to natural FL and larger models is unestablished.
2. The strongest corrected result is binary and uses a small linear model with pooled features.
3. The multiclass private factorial runs are utility-invalid.
4. The 54.55% pilot is not strongly private because epsilon is approximately 410.
5. The baseline attack suite is not yet a full calibrated LiRA/A-LiRA or TC-UMIA reproduction.
6. No secure aggregation service is implemented.
7. FedEraser and fine-tuning require separate privacy accounting after the request.
8. No complete ten-seed confirmation of frontier transition cells is established.
9. The raw repository contains inconsistent run-count language: the factorial design implies 90 runs at three seeds, while stored summaries report 94 runs and some documents use different private-run counts.
10. The current MUB classifier is useful operationally but is not a full preregistered equivalence/non-inferiority analysis across a controlled endpoint family.
11. Some documentation uses stronger “redundant under private DP” language than the invalid Phase 7 private models can support.

These limitations are not side notes; they determine the correct strength of the paper claim.

## M. What Has Been Achieved

The repository has achieved a substantial, tested benchmark prototype with:

- Correct client-level DP mechanism and ledger structure.
- Corrected convergence architecture.
- Deterministic partitions and deletion manifests.
- Exact and ensemble retraining references.
- Multiple explicit unlearning baselines.
- MUB and redundancy classification machinery.
- Utility, alignment, privacy, and lifecycle-cost evaluation.
- Positive convergence and positive unlearning controls.
- A corrected real-data exploratory result.
- Transparent validity gating that prevents chance-level models from being presented as successful unlearning.

## N. What Is Still Needed for a Strong Paper Claim

1. Run healthy multiclass client-DP frontier experiments, especially epsilon 8-32 with enough clients and training budget.
2. Freeze and reproduce at least two published unlearning baselines under the same threat model.
3. Run calibrated LiRA/A-LiRA and retained-set/tri-class attacks with frozen disjoint splits.
4. Add a natural client-partitioned dataset such as FEMNIST and, ideally, a second image setting.
5. Complete partial and sequential deletion studies on real data.
6. Use at least five exploratory seeds and ten confirmation seeds at transition or harmful-unlearning cells.
7. Apply preregistered equivalence and non-inferiority tests with endpoint multiplicity control.
8. Reconcile all run counts and documentation against raw result indexes.
9. Perform a clean-machine reproduction and final literature check.
10. Replace placeholder sections in [docs/PAPER_DRAFT.md](docs/PAPER_DRAFT.md) only after these gates pass.

## O. Final Scientific Conclusion

The most defensible current conclusion is:

> In the corrected exploratory binary CIFAR-10 client-deletion screen, explicit retained-data fine-tuning and cached reconstruction did not materially improve predictive JS alignment to retraining over leaving the client-level DP model unchanged.

A second conclusion is also well supported:

> Near-chance private models must not be called redundant merely because deletion metrics and attack signals are small; validity gating correctly identifies utility collapse as a separate regime.

The stronger frontier claim remains a hypothesis supported by the project design and preliminary artifacts, but not fully confirmed across healthy private multiclass regimes. The repository is therefore best described as a research-grade benchmark prototype with a credible exploratory result, not yet a completed universal claim about DP and federated unlearning.

## Evidence Index

- [README.md](README.md): project purpose, architecture, methods, and headline status.
- [RESEARCH_GRADE_PLAN.md](RESEARCH_GRADE_PLAN.md): research questions, novelty boundaries, protocol, hypotheses, and required evidence.
- [PROJECT_AUDIT.md](PROJECT_AUDIT.md): implementation and positive-control audit.
- [PHASE2_PROGRESS_REPORT.md](PHASE2_PROGRESS_REPORT.md): convergence repair, pilot, retrain ensemble, and MUB history.
- [PHASE7_AUDIT.md](PHASE7_AUDIT.md): multiclass factorial screening and pilot discrepancy diagnosis.
- [FEDERASER_AUDIT.md](FEDERASER_AUDIT.md): algorithm, privacy classification, storage, and cost analysis.
- [PRIVACY_AUDIT.md](PRIVACY_AUDIT.md): tri-population MIA design and reported privacy findings.
- [REDUNDANCY_FRONTIER.md](REDUNDANCY_FRONTIER.md): validity gates, regimes, MUB classifier, and normalized deletion effect.
- [LIMITATIONS.md](LIMITATIONS.md): scope and threats to validity.
- [docs/RESEARCH_EXECUTION_STATUS.md](docs/RESEARCH_EXECUTION_STATUS.md): earned claims versus research-gated work.
- [reports/PHASE1_REAL_CIFAR_RESULTS.md](reports/PHASE1_REAL_CIFAR_RESULTS.md): corrected real-data Phase 1 measurements.
- [src/dp_forgetbench/federated.py](src/dp_forgetbench/federated.py): models, training, DP aggregation, history, reconstruction, and FedEraser.
- [src/dp_forgetbench/evaluation.py](src/dp_forgetbench/evaluation.py): JS, MUB, evaluation metrics, attacks, and cost logging.
- [src/dp_forgetbench/run.py](src/dp_forgetbench/run.py): end-to-end experiment orchestration, validity gates, and artifact writing.
- [tests](tests): executable checks for accounting, data, attacks, models, unlearning, statistics, privacy, and smoke runs.