# Future Implementation Roadmap

## Purpose

This roadmap prioritizes the project's primary research objective:

> Estimate the setting-specific privacy-budget region in which DP-only is
> sufficiently close to retraining, and the region in which explicit machine
> unlearning provides a reliable, practically meaningful benefit.

This is an empirical frontier, not a universal epsilon threshold. Utility,
forgetting effectiveness, privacy-attack evidence, and operational cost are
measured to make that boundary interpretable; maximizing every method's metric
is not the goal. Do not make numbers look better by changing thresholds,
choosing favorable seeds, or weakening the privacy contract. Every comparison
must preserve a frozen protocol and report the achieved privacy guarantee.

## Current reference point

The completed frozen study uses CIFAR-10, 100 clients, 200 examples per client,
Dirichlet alpha 0.5, a `SmallGroupNormCNN`, 40 rounds, 5 local epochs, client
sampling rate 0.5, and client-level DP at epsilon approximately 32 and delta
1e-5. Three seeds were run.

| Method | Test accuracy | Forgotten-client accuracy | Forgotten-vs-unseen MIA AUC |
|---|---:|---:|---:|
| DP-only / no action | 42.25% | 50.67% | 0.543 |
| Retained-data fine-tuning | 44.12% | 52.17% | 0.537 |
| Cached update reconstruction | 41.38% | 40.33% | 0.519 |
| FedEraser | 37.05% | 42.00% | 0.530 |
| Exact retrain reference | 42.31% | 44.83% | 0.504 |

These values are starting points, not targets or guarantees. Attack confidence
intervals include 0.5 for all methods, and the MUB classifications vary by seed
except for FedEraser, which is harmful in all three. This single epsilon point
does not locate a boundary. The next work should estimate the epsilon response
curve under this fixed setup before investing in broad algorithm optimization.

## Principles for all follow-up work

1. **Freeze the comparison protocol first.** Keep the test set, client
   partitions, deletion request, seed list, metrics, and privacy target fixed
   while comparing a candidate with its baseline.
2. **Treat the ledger as authoritative.** Recompute epsilon for every change to
   sampling rate, round count, noise, clipping, or public model releases. Never
   call a result “epsilon 32” based only on a configured noise multiplier.
3. **Use the correct privacy unit.** The primary contract is client-level DP
   for a complete client dataset. Do not substitute example-level DP-SGD and
   describe it as equivalent.
4. **Report all seeds and failure cases.** Do not select the best seed, discard
   low-performing runs without a predefined exclusion rule, or tune on the
   final test set.
5. **Keep three goals distinct.** Higher test accuracy is utility; lower
   forgotten-client accuracy alone is not proof of forgetting; lower attack
   success is privacy evidence only when the attack is correctly calibrated
   and has sufficient power.
6. **Track exposure and cost.** Fine-tuning and FedEraser use retained raw
   data after the request; cached reconstruction relies on stored updates.
   Record access, communication, storage, and runtime along with utility.

## Primary objective — estimate the epsilon frontier

### What the boundary means

For a fixed dataset, model, client partition, deletion request, training
protocol, unlearning method, and evaluation protocol, define two operational
regions:

- **DP-only sufficient / redundancy candidate:** DP-only is practically
  equivalent to independent retraining on the predeclared primary endpoints,
  and the confidence interval rules out a practically meaningful improvement
  from explicit unlearning.
- **Unlearning-needed candidate:** DP-only is outside the predeclared
  retrain-equivalence range, while at least one explicit method moves it
  reliably into that range without violating the retained-utility constraint
  or introducing an unreported privacy/exposure regime.
- **Inconclusive:** utility validity fails, uncertainty is too wide, seeds
  disagree, attack sanity checks fail, or neither DP-only nor an explicit
  method meets the criteria.

The epsilon "boundary" may be a transition interval, be non-monotonic, or not
exist for a particular setting. Report the evidence and interval; do not force
the data into a single cutoff.

### Immediate study: matched coarse epsilon sweep

Use the completed frozen unlearning study as the protocol anchor. In the first
frontier sweep, hold all non-epsilon factors fixed:

| Factor | Fixed setting |
|---|---|
| Dataset and task | CIFAR-10, 10-class |
| Model | `SmallGroupNormCNN` |
| Federation | 100 clients, 200 examples/client, Dirichlet alpha 0.5 |
| Training | 40 rounds, 5 local epochs, batch size 64, client sampling rate 0.5 |
| Deletion | Full deletion of client 2, same deterministic partition policy |
| Evaluation | Same test/audit protocol and metric implementations |
| Initial epsilon screen | 8, 16, 24, 32, 48, and infinity (non-private control) |
| Seeds | Start with the same three seed values across epsilon; extend to at least five for confirmation |

These epsilon values are a **coarse screen**, not a prediction of the transition.
Recalibrate the Gaussian noise multiplier independently for each finite target
epsilon using the accountant and exact rounds/sampling schedule. Record both
target and achieved epsilon; only the ledger-computed value is authoritative.
For infinity, explicitly disable DP and label it as a non-private control.

For every epsilon and seed, evaluate:

1. DP-only/no action;
2. retained-data fine-tuning;
3. cached reconstruction and FedEraser where the deletion protocol supports
   them;
4. exact retraining plus independent retrain references.

Compare candidates against the retrain distribution using utility, predictive
alignment, forgetting and retained-client outcomes, calibrated attack
diagnostics, and lifecycle cost. Use matched seeds and the same client
partition/deletion manifest within each epsilon cell.

### Pre-register the decision rule before the sweep

Before inspecting the sweep's test results:

- Choose primary endpoints and practically meaningful equivalence margins.
  Use retrain-versus-independent-retrain variability to inform alignment
  margins; specify any separate retained-utility non-inferiority margin and
  justify it for the application.
- Use equivalence testing or confidence intervals against those margins.
  Failure to find a significant difference is **not** evidence of equivalence.
- Define a minimum model-utility gate. If a DP point is near chance or otherwise
  invalid, classify its unlearning conclusion as inconclusive rather than
  "redundant."
- Define attack positive/negative controls and declare privacy-attack results
  inconclusive if the attack lacks demonstrated power.
- Report per-seed results, paired differences, uncertainty, all achieved
  epsilons, and failures. Do not choose the best seed or tune on the test set.

### Refine only after screening

If neighboring epsilon cells produce different decisions, add intermediate
epsilon values in that interval and run at least 10 independent confirmation
seeds there if compute allows. If all sampled epsilon points have the same
decision, expand the coarse range rather than claiming a boundary. If the
response is non-monotonic, report multiple regions and investigate the cause.

The final claim must be scoped, for example: "under this dataset, model,
client partition, deletion size, and threat model, the transition was observed
between achieved epsilons X and Y." Do not call it a universal epsilon cutoff.

### Reuse and repair existing runners; do not duplicate them

The repository already has frontier-related tooling:

- `scripts/run_utility_frontier.py` performs an epsilon **utility-only** sweep;
  it does not run the full deletion/unlearning comparison, and its defaults
  (60 rounds, 3 local epochs) do not match this frozen 40-round, 5-epoch study.
- `scripts/run_phase7_factorial.py` runs epsilon, heterogeneity, and deletion
  factors, but its base config is
  `configs/phase2_cifar10_multiclass_cnn.yaml`, which currently uses 20 clients,
  40 rounds, and 2 local epochs. It is not the current frozen protocol.
- `scripts/run_phase11_frontier.py` confirms selected Phase 7 cells, but its
  frontier logic hardcodes epsilon 4 and 8 as special boundary values. It is
  not a general transition detector and inherits the Phase 7 base setup.

Therefore do **not** run these scripts unchanged for the new primary claim and
do not create a separate `run_frontier_grid.py` that duplicates their logic.
First adapt or factor the existing runner so the frozen 100-client protocol
can be configured and validated end-to-end. Add tests that assert each
generated cell differs only in target epsilon and seed for this first sweep,
uses the expected 100-client/40-round/5-epoch settings, computes an achieved
epsilon matching the target within a declared tolerance, writes the full
method/retrain artifacts, and can resume without mixing protocol versions.
Replace hardcoded frontier epsilon rules with a decision process based on
predeclared equivalence criteria and observed neighboring cells.

Only after the basic epsilon curve is understood should a second factorial
vary heterogeneity or deletion fraction. Varying model architecture, local
epochs, population size, or client sampling during the first sweep would
confound the epsilon effect and require a separately designed study.

## Priority 0 — strengthen the experiment before optimizing

### 0.1 Increase independent seeds

**Why:** Three seeds produce uncertain estimates and seed-dependent MUB labels.

**Implementation:**

- For the coarse screen, use at least three matched seeds per cell; extend
  transition cells to at least 10 independent training seeds.
- Use the same seed list for DP-only, every unlearning method, and retraining.
- Preserve the full per-seed records and aggregate only after every planned run
  completes. Do not launch a 72-cell grid with only three seeds before checking
  that a smaller matched epsilon screen is valid and affordable.
- Keep the current three-seed results as the pilot; do not silently replace
  them with a larger sample.

**Success check:** Report mean, standard deviation, interval estimates, and
per-seed paired differences for test accuracy, forgotten accuracy, JS
alignment, and MIA. State how many runs completed and why any are missing.

### 0.2 Improve statistical treatment of MUB and attack results

**Why:** Attack estimates have broad intervals; overlapping or near-chance
point estimates should not be presented as established privacy improvement.

**Implementation:**

- Retain held-out attack evaluation and calibrate the threshold on a distinct
  calibration split.
- Increase attack-set size using a documented, class-balanced sampling
  protocol; report set sizes and uncertainty.
- Add paired bootstrap intervals for per-seed differences between each
  candidate, DP-only, and retrain references.
- Keep training-seed uncertainty separate from within-run attack-sampling
  uncertainty.
- Report ROC AUC, advantage at the calibrated threshold, and low-FPR operating
  points together. Explicitly mark weakly powered attacks inconclusive.

**Success check:** A claimed MIA improvement must have consistent direction
across seeds and an uncertainty interval that supports that direction. An AUC
near 0.5 by itself is not evidence that forgetting occurred.

### 0.3 Freeze configurations and artifacts per run

**Why:** Reproducibility requires knowing exactly which code and config created
each result.

**Implementation:**

- Save the complete resolved config, code revision or source hash, package
  versions, device, data/partition checksums, and privacy ledger with each run.
- Ensure seed overrides are included in the resolved config and validation
  hash.
- Preserve old timestamped runs; never overwrite or aggregate mixed protocol
  versions without identifying them.
- Save aggregate reports only from validated run artifacts.

**Success check:** A clean run can be independently validated and traced to
its exact config, data split, code version, and computed privacy guarantee.

## Priority 1 — improve model utility under the same DP guarantee

### 1.1 Compare the existing CIFAR-sized GroupNorm ResNet-18

**Why:** The repository already includes a `groupnorm_resnet18` path. A
stronger feature extractor may improve the current 42% utility ceiling, but
could also amplify clipping/noise effects.

**Implementation:**

- Run a non-private control and a privacy-calibrated epsilon-32 utility pilot
  before any unlearning comparison.
- Match the client split, sample count, seeds, client sampling, total privacy
  budget, test protocol, and data augmentations across architectures.
- Recalibrate the noise multiplier for the exact number of releases/rounds.
- Measure GPU memory and runtime on the 4 GB RTX 2050; reduce batch size only
  if the change is recorded and controlled.

**Success check:** Promote the architecture only if its multi-seed mean test
accuracy improves over the CNN with uncertainty reported, it passes the
predefined validity gate, and the same client-level epsilon is verified.
Compare parameter count, runtime, and memory as well as accuracy.

### 1.2 Run a small, predeclared optimization study

**Why:** Learning rate, clipping threshold, local batch size, and local epoch
count jointly affect convergence and how often updates are clipped.

**Implementation:**

- Use a small grid for learning rate and clip norm, with a fixed privacy target
  and fixed client sampling.
- Tune using validation data or a separate tuning seed set; reserve final seeds
  for confirmation.
- Record clipping fractions/norm summaries per round, training loss, test loss,
  and divergence indicators.
- Compare 3 versus 5 local epochs only under correctly recalculated privacy
  accounting. Keep total rounds and privacy target fixed when interpreting the
  comparison.

**Success check:** A candidate must improve held-out multi-seed test utility
without exceeding the privacy budget or producing unstable client updates.
Do not choose a setting solely because it improves the unlearning result.

### 1.3 Test public data augmentation as a controlled factor

**Why:** CIFAR-10 utility may benefit from standard augmentations, but the
augmentation must be applied consistently and cannot use private/test
statistics.

**Implementation:**

- Compare a minimal, fixed public augmentation policy (for example,
  random crop with padding and horizontal flip) with the current preprocessing.
- Apply the same policy to all methods that train on the corresponding client
  data, including exact retrain references.
- Keep test preprocessing deterministic and unchanged.
- Record augmentation randomness as part of the run seed protocol.

**Success check:** Retain the augmentation only if it improves validation and
confirmation-seed utility consistently, with test data excluded from tuning.

### 1.4 Evaluate participation/client-count scaling as a new regime

**Why:** More participating clients can change aggregate signal and client-level
noise behavior, but changes the experiment and privacy accounting.

**Implementation:**

- Compare client sample rate and population size in a separate factorial
  experiment.
- Maintain the same stated client-level adjacency and fixed public
  normalization.
- Recompute epsilon for each point; do not reuse the current epsilon or noise
  multiplier.
- Keep total training examples, per-client allocation, and deletion fraction
  explicit so each comparison has a clear interpretation.

**Success check:** Compare privacy-utility curves rather than declaring a
single higher-accuracy point “better” when its epsilon or data regime differs.

## Priority 2 — improve forgetting without sacrificing retained utility

### 2.1 Diagnose why retained fine-tuning does not forget the deleted client

**Why:** Fine-tuning improves test accuracy, but mean forgotten-client accuracy
is higher than the DP-only value, and its classification changes by seed.

**Implementation:**

- Add per-round evaluation of test, forgotten-client, and retained-client
  loss/accuracy for analysis runs.
- Verify the deleted client is excluded from every post-request sampler and
  that no cached batches or optimizer state contain its records.
- Compare the fine-tuned model with exact retraining at both prediction and
  representation levels, not forgotten accuracy alone.
- Sweep a small number of predeclared unlearning rounds/learning rates, with
  retained utility and access cost reported.

**Success check:** A fine-tuning variant is promising only if forgetting
alignment improves over DP-only and retrain variability while retained/test
utility stays within a predefined tolerance. Include a separate privacy/access
description because this is not DP post-processing.

### 2.2 Add a retrain-alignment distillation baseline

**Why:** Retained-data fine-tuning optimizes the ordinary task and may preserve
the deleted client's behavior. A target model trained without that client
provides a direct functional reference.

**Implementation:**

- Use a retained-data retrain ensemble as a teacher/reference.
- Distill on a disjoint, permitted calibration set, comparing hard-label
  training with soft-target prediction matching.
- Explicitly state the calibration data source and whether it is public,
  retained-private, or synthetic.
- Keep the target retrain used for reporting separate from any teacher used
  for tuning to avoid evaluation leakage.

**Success check:** The candidate should reduce JS divergence to held-out
retrain references on forgotten and general test examples, while retaining
utility and not increasing measured attack success. Count data access,
communication, and training cost.

### 2.3 Improve cached reconstruction's accuracy trade-off

**Why:** Cached reconstruction has the lowest mean forgotten accuracy among
the explicit methods but lower overall utility than DP-only.

**Implementation:**

- Instrument the reconstruction to log per-round retained-update sums,
  removed-client contributions, noise terms, and normalization.
- Compare current reconstruction with exact retraining across all seeds to
  identify where error accumulates.
- Evaluate a bounded correction/calibration step using retained updates only;
  disclose if it introduces client participation or retained raw-data access.
- Do not tune reconstruction against the forgotten client's labels after the
  request.

**Success check:** Seek better test and retrain alignment without losing the
observed reduction in forgotten behavior; report historical update storage and
privacy exposure explicitly.

### 2.4 Diagnose FedEraser before changing the algorithm

**Why:** It is consistently harmful for this implementation/configuration and
has markedly lower test utility.

**Implementation:**

- Compare its reconstructed model with exact retraining per round and inspect
  update scaling, calibration ratio, retained-client selection, and noise
  handling.
- Add unit tests for update history completeness and calibration invariants.
- Establish a non-private positive control where historical-update unlearning
  should behave sensibly before evaluating it under DP.
- Change one algorithmic factor at a time and preserve a faithful baseline.

**Success check:** Do not claim improvement until the baseline implementation
passes its control tests and the modified method improves paired multi-seed
utility/alignment or forgetting metrics without hidden exposure changes.

## Priority 3 — make privacy evaluation more informative

### 3.1 Add calibrated shadow-model attack evaluation

**Why:** The current loss attack's AUC intervals are broad and include chance.
More attack power may distinguish genuine forgetting from a weak model or a
small evaluation sample.

**Implementation:**

- Add a shadow-model or LiRA-style evaluation only with a documented,
  computationally feasible protocol.
- Separate attack training, threshold calibration, and final evaluation data.
- Include positive controls where membership should be distinguishable and
  negative controls where it should not.
- Report attack calibration, sample counts, confidence intervals, and runtime.

**Success check:** Attack conclusions must be supported by positive/negative
controls and must not be described as formal DP proof or certified unlearning.

### 3.2 Add client-level privacy stress cases

**Why:** Deleting one selected client may not represent difficult deletion
requests.

**Implementation:**

- Predeclare deleted-client selection strategies: random, high-data-volume,
  high-heterogeneity, and multiple-client deletion.
- Stratify results by client label composition and sample count.
- For each deletion size, update the retained-data retrain protocol and
  record the exact privacy unit and deletion fraction.
- Do not present selected hard/easy clients as representative without the
  corresponding sampling design.

**Success check:** Report distributions across deletion requests and explain
when the method's benefit or harm depends on client composition.

## Priority 4 — reporting and execution quality

### 4.1 Produce a metric dashboard with trade-offs

For each method and seed, show:

- test and retained accuracy/loss;
- forgotten-client accuracy/loss;
- JS divergence to independent retrain references;
- MIA AUC, calibrated advantage, low-FPR performance, and intervals;
- MUB components and per-seed classification;
- runtime, communication, storage, client participation, and raw-data access.

Avoid reducing performance to a single score. A method that lowers forgotten
accuracy while severely harming retained utility must make that trade-off
visible.

### 4.2 Add experiment-budget estimation

Estimate GPU hours, model-training count, disk use, and peak VRAM before a
multi-seed sweep. Use checkpoint/resume only if checkpoints preserve enough
state for faithful continuation and are recorded in the artifact metadata.

### 4.3 Keep the result bundle reproducible

Create one command that:

1. validates every run artifact;
2. verifies one run per requested seed and matching frozen configs;
3. regenerates the aggregate JSON;
4. produces the report tables from that JSON;
5. fails if any seed is missing or any result is invalid.

## Suggested order of work

1. Write and freeze the operational equivalence, utility, and inconclusive
   criteria for the epsilon-boundary claim.
2. Adapt the existing runner to the frozen 100-client/40-round/5-epoch study;
   verify generated configs, noise calibration, ledgers, validation, and
   resumability with a smoke test.
3. Run the matched coarse epsilon screen using the same seeds and all required
   methods/retrain references.
4. Inspect the utility gate, attack controls, per-seed comparisons, and
   transition candidates; do not infer a boundary from accuracy alone.
5. Refine the epsilon interval and increase seeds around any candidate
   transition; report a transition interval or no-boundary result honestly.
6. After this primary curve is understood, study heterogeneity/deletion size
   as separate factors and only then investigate model/optimizer changes if
   they are needed to make the frontier estimable.
7. Improve attack protocols and algorithm-specific weaknesses as targeted
   follow-up work, not as a substitute for the epsilon sweep.
8. Regenerate the report and figures from validated, protocol-matched
   artifacts.

## Decision rule for calling a metric improved

Call a result an improvement only when all of the following are true:

- the comparison uses the same privacy target, data protocol, evaluation split,
  and deletion request design;
- the run passes artifact and validity checks;
- the improvement is visible across independent confirmation seeds, not only
  in a selected seed;
- uncertainty is reported and the direction is not explained by a changed
  threat model or exposure regime;
- relevant trade-offs (test utility, forgetting alignment, attack success,
  runtime, communication, storage, and data access) are disclosed.

If these conditions are not met, report the result as exploratory or
inconclusive rather than as a metric improvement.
