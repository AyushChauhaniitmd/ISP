# DP-ForgetBench: a research-grade plan

## Working title

**When is explicit client unlearning redundant? A privacy-aligned audit of differentially private federated learning**

Do not promise a threshold or “SOTA” result in the title, abstract, or proposal. The contribution is the *pre-registered test and benchmark* that can discover either a redundancy region, a non-redundancy result, or a harmful-unlearning region.

## Executive decision

The project should be a rigorous empirical systems-and-privacy paper, not a claim that “DP-SGD already unlearns data.” The corrected core question is:

> Under **client-level** differential privacy, which combinations of privacy budget, data heterogeneity, deletion size, and request sequence make a post-hoc federated unlearning method measurably closer to retraining than leaving the released DP model unchanged—and is that gain worth its additional privacy, compute, communication, and storage cost?

The unit of privacy and deletion must match. A conventional per-example DP-SGD guarantee is appropriate for record deletion; it does **not** by itself establish a client-removal guarantee. The primary study therefore uses central/user-level DP: clip each participating client update, add Gaussian noise to the aggregate at the server, account for client sampling, and protect the released global model. A second, explicitly labelled record-level track may compare the two scopes, but must never mix them.

## What is genuinely publishable here

Existing work already covers federated unlearning, DP-aware federated unlearning, certified client removal, and retained-set membership leakage. Therefore “DP + FL + unlearning + MIA” is not novel.

The defensible contribution package is:

1. **Privacy-aligned redundancy frontier.** Estimate, with uncertainty, where no-unlearning DP models are equivalent to a retraining target and where explicit unlearning adds material benefit. The output is a map over `(epsilon, heterogeneity, deletion granularity, request count)`, not an assumed threshold.
2. **Deletion-scope mismatch audit.** Directly contrast record-level and client-level DP under both record and client requests. This closes the common but serious mismatch between a DP claim and a deletion claim.
3. **Tri-population, time-aware privacy audit.** Evaluate forgotten, retained, and unseen populations using pre/post-deletion attacks; quantify whether an unlearning method protects the requester while increasing retained-client leakage.
4. **Sequential-request and cost-aware evaluation.** Measure degradation and total lifecycle cost over repeated deletions rather than reporting one convenient deletion event.
5. **An open, reproducible benchmark.** Release fixed partitions, deletion manifests, target retrains, privacy ledgers, attack code, configuration files, and complete raw results.

This is research-grade even if it introduces no new unlearning algorithm. If Phase 3 reveals a real failure mode, implement a narrowly motivated method; do not invent one in advance only to claim novelty.

## Claims that are allowed and claims that are not

Allowed after evidence:

- “For the evaluated setting, DP-only was statistically equivalent/non-equivalent to the retraining reference under the pre-registered tolerance.”
- “Explicit unlearning improved/harmed these metrics under the stated threat model.”
- “The boundary shifted with client heterogeneity or request size.”

Never claim:

- that DP deletes data, produces exact unlearning, or satisfies a legal deletion request by itself;
- a universal epsilon threshold;
- client-level privacy from an example-level accountant;
- “SOTA” without a named benchmark, frozen code/data version, and a like-for-like comparison;
- a DP guarantee after an unlearning step that accesses retained raw data or unprotected update history without accounting for that new access/release.

## Precise research questions and falsifiable hypotheses

| ID | Research question | Falsifiable hypothesis |
|---|---|---|
| RQ1 | Does client-level DP reduce the gap between a full-data model and retraining without a removed client? | Stronger client DP reduces the gap on attack and functional metrics, but does not necessarily eliminate it. |
| RQ2 | Does explicit unlearning add benefit beyond DP? | Its benefit is largest at weak privacy, large deletion size, and highly non-IID clients. |
| RQ3 | Does the result depend on the protected unit? | Record-level DP can appear adequate for record requests while client-level removal remains distinguishable. |
| RQ4 | Does unlearning harm non-requesting users? | Some approximate methods increase retained-set membership/label leakage relative to DP-only and retraining. |
| RQ5 | Are one-shot conclusions stable under a request stream? | Approximation and retained-set risk accumulate or change under sequential deletion. |
| RQ6 | Is an unlearning method operationally worthwhile? | Some strata have negligible incremental benefit per unit lifecycle cost. |

The null result is valuable: no robust redundancy frontier is still an important answer.

## Formal setup

Let `D = {D_1, …, D_n}` be client datasets and let a request remove `F` (one record subset, one whole client, or a sequence). For every configuration and seed train:

- `M_full`: model trained on all eligible data;
- `M_DP`: client-DP model trained on all eligible data and left unchanged after the request;
- `M_U`: model obtained by applying an explicit unlearning method after the request;
- `M_R`: **gold-standard retrain** on `D \ F`, using the same training recipe and privacy scope;
- `M_R'`: an independent retrain on `D \ F` with a different random seed.

`M_R` versus `M_R'` establishes the irreducible retraining-variability band. A method must be compared to that band, not required to have zero parameter distance to one arbitrary retrain.

For each endpoint `j`, define a signed improvement of unlearning over DP-only, relative to retraining:

`MUB_j = distance_j(M_DP, M_R) - distance_j(M_U, M_R)`.

Positive values help. Do **not** multiply accuracy, attacks, and runtime into one unvalidated score. Report their Pareto frontier. A configuration is an empirical **redundancy candidate** only if DP-only is within the pre-declared retrain-variability equivalence band for all primary endpoints and the confidence interval excludes a practically important `MUB`.

Set practical tolerances before seeing test results. Prefer data-driven tolerances based on the 90th/95th percentile of `M_R`–`M_R'` variability, with a separately declared maximum utility loss (for example, 1 percentage point only if justified by application requirements). Use two-one-sided equivalence tests and bootstrap confidence intervals; never conclude “no difference” from a non-significant p-value.

## Threat model and privacy contract (mandatory)

### Primary privacy model: client/user-level central DP

- **Adjacency:** datasets differ by all records of one client.
- **Mechanism:** sample clients, L2-clip each client update, securely aggregate clipped updates, add Gaussian noise at the server, and use an RDP/PRV accountant matched to the sampling mechanism.
- **Output protected:** every released global checkpoint and the final model. If multiple releases/checkpoints are public, compose them in the privacy ledger.
- **Trust assumption:** the server sees only an aggregate when secure aggregation is enabled. State whether the server is trusted for DP noise generation; if not, use distributed noise generation or state that protection is only against external/other-client observers.
- **Privacy reporting per run:** achieved `(epsilon, delta)`, sampling model and rate, number of rounds, noise multiplier, clipping norm, accountant and version, all released artifacts, and composition.

### Secondary track: record-level DP

Use only for record-removal experiments. It can use per-example DP-SGD and an Opacus accountant, but results must be labelled *record-level DP*. It is a scope-mismatch control, not a substitute for the primary client-removal study.

### Unlearning privacy rule

Post-processing of `M_DP` preserves DP only if the unlearning procedure consumes no additional private information. Methods that use raw retained data, private historical client updates, or new model evaluations must receive their own privacy ledger/DP mechanism or be classified as a separate, non-comparable exposure regime. Secure aggregation helps protect updates in transit; it is not a DP guarantee.

## Methods and baselines

Use public implementations or reimplement only after reproducing a small published reference. Freeze the algorithm list before the final benchmark.

| Category | Include | Purpose |
|---|---|---|
| Reference | Exact DP retraining (`M_R`) | Required gold standard; report total cost. |
| No action | `M_full` / `M_DP` unchanged after request | Tests whether DP alone is adequate. |
| Cheap baseline | Retain-data fine-tuning and a simple negative-gradient/gradient-ascent control | Prevents weak-baseline claims. |
| FU historical reconstruction | FedEraser-style method | Established client-level FU comparator. |
| DP-relevant FU | FedRecover-style method, if source and license are reproducible | Direct DP/FU comparator. |
| Design-for-unlearning | SISA/sharding or FedShard-style baseline | Tests whether up-front structure beats post-hoc repair. |
| Privacy-aware FU | Starfish/certified client removal only if its system assumptions can be faithfully reproduced | Strong comparator; do not compare an insecure simulation to a 2PC system only on time. |

Every baseline gets the same data partitions, client schedule where feasible, stopping rule, validation budget, model capacity, and hyperparameter-search budget. State storage of historical updates, which is often omitted but central to FU cost and privacy.

## Datasets, models, and federated scenarios

Do not start with a giant model. Begin with a setting where retraining, shadow models, and repeated seeds are affordable.

| Tier | Dataset/model | Why | Use in paper |
|---|---|---|---|
| Pilot | CIFAR-10 / small GroupNorm CNN | fast debugging and attack calibration | not a final standalone claim |
| Core controlled | CIFAR-10 and CIFAR-100 / ResNet-18 with GroupNorm | controlled Dirichlet heterogeneity and class-skew deletion | main ablations |
| Core natural FL | FEMNIST (LEAF split) / compact CNN | naturally user-partitioned, client-level deletion | external validity |
| Stress, only if resources permit | DomainNet or TinyImageNet / ResNet-18 | stronger distribution shift/scale | final robustness result |

Avoid BatchNorm in the primary model because client-local batch statistics complicate reproducibility and privacy interpretation. Use 50–200 eligible clients, a fixed client-participation rate (for example 10%), 1–5 local epochs, and FedAvg as the initial aggregator. Tune rounds and learning-rate schedules for each privacy regime fairly; do not deliberately undertrain DP methods.

### Partitions and deletion manifests

- IID control plus Dirichlet class-skew `alpha ∈ {1.0, 0.3, 0.1}`. Record realized per-client label entropy and sample counts; alpha alone is insufficient.
- Deletion unit: record subset `{1%, 10%, 25%, 50%}`, full client, and a rare-class/dominant-client stress request.
- Sequence: one deletion; then a fixed four-client sequence chosen before training. Re-evaluate after every request.
- Select deletion clients by pre-registered strata: typical, highly influential, low-data, and rare-class. Do not choose only dramatic examples after seeing results.
- Publish hashable CSV manifests with split, client, request order, and seed.

## Evaluation: primary endpoints

All attacks must use disjoint attack-training, calibration, and evaluation members/non-members. Match class, client size, and difficulty across populations; otherwise AUC can measure distribution shift rather than membership.

| Dimension | Primary metrics | Required comparison |
|---|---|---|
| Retained utility | test accuracy, NLL, ECE, macro-F1; per-client distribution | `M_DP`, `M_U`, `M_R` |
| Functional alignment | per-example JS divergence/logit distance to retrain; prediction disagreement; representation distance | method versus `M_R`, normalized by `M_R`–`M_R'` |
| Forget privacy | calibrated LiRA/A-LiRA-style attack AUC and advantage for forgotten vs matched unseen data | `M_DP` vs `M_U` vs `M_R` |
| Retained privacy | same attack on retained members vs matched unseen; report AUC, advantage, TPR at FPR 1% and 0.1% | all methods |
| Tri-class audit | TC-UMIA or faithful equivalent distinguishing forget/retain/unseen from pre/post outputs | confusion matrix and per-class AUC/accuracy |
| FU-specific exposure | label-inference/trajectory attack only when server updates or deltas are actually exposed in the declared threat model | clearly segregated white-box result |
| Efficiency | wall time, GPU-hours/energy where available, post-request latency, communication bytes, persistent storage, peak memory | include original-training and request-lifecycle totals |
| Fairness/robustness | worst-client and quartile utility/privacy, client-size and class-skew strata | no aggregate-only conclusion |

Report privacy attack **advantage** as well as AUC. AUC close to 0.5 is only meaningful with calibrated attacks, confidence intervals, matched reference populations, and a demonstrated ability of the attack pipeline to detect leakage in non-private controls.

### Quality gates for every experimental run

1. Partition checksum, deletion-manifest checksum, and code commit are logged.
2. Privacy accountant passes a unit test on known toy values; achieved epsilon is stored, not inferred from noise alone.
3. DP train utility reaches a pre-specified viability floor relative to the corresponding non-private run. If it fails, tune before comparing unlearning.
4. The attack pipeline separates `M_full` members from unseen data above a pre-specified sanity threshold, or the attack result is declared inconclusive.
5. Two independent target retrains define variability before interpreting alignment.
6. No result is presented from fewer than five independent training seeds; use ten for the final main table if compute allows.

## Experimental program that is feasible

The full Cartesian grid in the draft would require thousands of runs and is not defensible if underpowered. Execute this staged design; advance only when the preceding gate passes.

### Phase 0 — specification and feasibility (1–2 weeks)

- Write a one-page protocol, threat model, hypotheses, endpoints, and decision rule before final experiments.
- Build deterministic data partitioning, a FedAvg non-DP reference, deletion manifests, checkpoint policy, and exact retrain.
- Implement unit tests for client clipping, Gaussian noise, accountant, deletion exclusion, and target-retrain reproducibility.
- Deliverable: a reproducible no-DP one-client deletion result over five seeds.

### Phase 1 — privacy-aligned pilot (2–3 weeks)

One dataset (CIFAR-10), two heterogeneity levels (`alpha={1.0,0.1}`), three privacy points (`epsilon≈{2,8,infinity}`), full-client deletion, and two methods (no action + one FU method), five seeds.

Gate: achieved epsilon is correct, DP model is useful, retrain variability is measurable, attacks pass sanity checks, and one unlearning method runs end-to-end. This is a debugging study, not an SOTA claim.

### Phase 2 — core factorial study (4–6 weeks)

Core configurations: two datasets (CIFAR-100 and FEMNIST), `alpha={1.0,0.3,0.1}` where synthetic partitioning applies, `epsilon≈{1,2,4,8,infinity}`, full-client deletion and 25%-partial deletion, no-action + two FU families + retrain, five seeds. Use a fractional design for sequential deletion: choose the privacy points `{2,8,infinity}` and heterogeneity extremes `{1.0,0.1}`.

This first estimates the frontier, tests scope alignment, and finds which cells warrant more seeds. Do not expand all dimensions blindly.

### Phase 3 — confirmation and extension (3–5 weeks)

- Re-run only the frontier transition cells and any harmful-unlearning cell with ten seeds.
- Add tri-class and retained-set attacks, sequential deletion, complete cost accounting, and the record-vs-client DP control.
- If a repeatable gap remains, develop **one** constrained method: a DP-ledgered unlearning update that optimizes closeness to retraining while enforcing a retained-privacy/utility constraint. Compare it only to the frozen Phase-2 methods.

### Phase 4 — artifact and paper (2–3 weeks)

- Run clean-room reproduction from a new machine/container.
- Publish configs, raw metric tables, confidence intervals, privacy ledgers, attack calibration, checkpoints only where consent/licensing permits, and a model card.
- Write the paper around results, including negative findings and limitations.

## Compute budget and run accounting

Before Phase 2, benchmark a single seed and estimate cost:

`total GPU-hours = conditions × seeds × (full train + target retrain + unlearning/attack cost)`.

Attack shadow models can dominate training cost. Cache only artifacts permitted by the threat model, quote both incremental unlearning time and total lifecycle time, and record checkpoint storage. If budget is limited, reduce datasets and method families first—not seeds, target retrains, or attack calibration.

## Figures and tables that make the paper

1. **Privacy-scope diagram:** record adjacency vs client adjacency; where clipping/noise/accounting occur.
2. **Central result:** frontier heat map of practical `MUB`/equivalence verdict over epsilon × heterogeneity, with one panel per deletion size.
3. **Pareto plots:** retained utility, alignment-to-retrain, forget leakage, retained leakage, and lifecycle cost.
4. **Sequential timeline:** metrics and cumulative cost after each request.
5. **Tri-class confusion/ROC:** forget, retain, unseen after deletion.
6. **Scope-mismatch table:** record-DP vs client-DP under each request granularity.
7. **Reproducibility table:** all achieved epsilons, deltas, clipping/noise, rounds, client participation, seeds, CIs, and hardware.

## Repository and artifact layout

```text
dp-forgetbench/
  configs/                 # immutable YAML experiment configs
  data/                    # download scripts; never bundled restricted data
  partitions/              # versioned split and deletion manifests
  src/
    fl/                    # FedAvg, client sampling, secure-aggregation interface
    privacy/               # client DP, accountant adapter, privacy ledger
    unlearning/            # each baseline behind one interface
    attacks/               # LiRA/A-LiRA, tri-class, calibration
    evaluation/            # utility, alignment, cost, statistics
  tests/                   # privacy, deletion, determinism, accounting tests
  scripts/                 # train, retrain, unlearn, attack, aggregate
  results/                 # raw immutable per-run JSON/Parquet
  reports/                 # generated figures and tables only
  environment/             # locked environment/container
  docs/                    # threat model, preregistration, model card
```

Use a configuration hash, Git commit hash, seed, hardware description, package lockfile, and dataset/partition checksum in every result record. CI should run a toy FL test, a privacy-accountant test, a deletion-exclusion test, and an end-to-end smoke test.

## Statistical analysis plan

- Independent unit: training seed; client-level observations are nested, not independent replicates of a seed.
- Report mean, 95% bootstrap CI, paired effect sizes versus no-action DP, and raw per-seed values.
- Use hierarchical/mixed-effects models for the factorial analysis, with fixed effects for DP regime, heterogeneity, deletion size, request index, and method; random effect for seed/dataset as appropriate.
- Control false-discovery rate across the pre-registered endpoint family. Treat exploratory analyses as exploratory.
- Use equivalence tests for redundancy and non-inferiority tests for retained utility. Publish null/negative results.
- Select hyperparameters on validation clients/data that are not used for attack evaluation; freeze them before test runs.

## Reviewer's checklist / failure modes to pre-empt

- **“Your DP is record-level, but the request is client-level.”** Primary design uses client-level adjacency and accounting.
- **“No unlearning is not a deletion baseline.”** It is not claimed as deletion; it is a DP-only comparison against retraining.
- **“You compared to one retrain.”** Use independent target retrains and equivalence bands.
- **“MIA is weak or distribution-confounded.”** Match populations, calibrate attacks, and show non-private detection sanity checks.
- **“DP+U privacy is not accounted.”** Maintain a release/access ledger and separate post-processing-only from data-accessing U.
- **“Methods were tuned unequally.”** Equalize and document tuning budget.
- **“Sequential requests were ignored.”** Include a pre-registered request stream and cumulative cost.
- **“SOTA is cherry-picked.”** Use named reproducible baselines and release raw outcomes; state benchmark-specific claims only.

## Success criteria

Minimum publishable result: two datasets, two privacy scopes clearly separated, exact retraining, at least two FU families, calibrated forget/retain attacks, five seeds, and a reproducible redundancy analysis.

Strong conference-level result: the above plus ten-seed confirmation at transition cells, sequential requests, tri-class attack, full lifecycle accounting, and a clear finding that changes the practitioner decision (unlearning needed / redundant / harmful in specified regions).

“SOTA” is not a project plan or a guarantee. It is a benchmark-specific outcome verified only after the final literature cutoff and fair reruns of all public competing methods. The target should be **credible state-of-the-art evaluation quality**, not an unsupported leaderboard claim.

## Key literature and implementation sources (verified 2026-08-24)

- Gu, He, and Chen, [Auditing Approximate Machine Unlearning for Differentially Private Models](https://arxiv.org/abs/2508.18671): motivates the retained-set audit for DP models.
- Fu et al., [Revisiting Privacy Leakage in Machine Unlearning: Membership Inference Beyond the Forgotten Set](https://arxiv.org/abs/2605.01129): introduces a tri-class forget/retain/unseen MIA and reports retained-set leakage risks.
- Liu et al., [Privacy-Preserving Federated Unlearning with Certified Client Removal](https://arxiv.org/abs/2404.09724): a client-removal/privacy-aware FU comparator with specific system assumptions.
- Zhao et al., [Exploring Federated Unlearning: Analysis, Comparison, and Insights](https://arxiv.org/abs/2310.19218): OpenFederatedUnlearning benchmark and FU evaluation context.
- Cadet et al., [Deep Unlearn: Benchmarking Machine Unlearning](https://arxiv.org/abs/2410.01276): demonstrates why robust baselines, attack evaluation, and repeated initializations matter.
- [Flower central/user-level differential privacy documentation](https://flower.ai/docs/framework/explanation-differential-privacy.html): describes server-side central DP aimed at protecting each client’s data; [Flower DP plus secure-aggregation example](https://flower.ai/docs/examples/fl-dp-sa.html).
- [Opacus DP-SGD documentation](https://opacus.ai/docs/faq): describes example-level DP-SGD and `(epsilon, delta)` accounting; use this only for the separate record-level track.

