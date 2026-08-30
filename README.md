# DP-ForgetBench

**When is explicit client unlearning redundant? A privacy-aligned audit of differentially private federated learning.**

This repository builds a reproducible benchmark for client deletion in federated learning. It does not claim that DP deletes data, and it does not inflate exploratory numbers into SOTA. The goal is sharper: compare DP-only release, explicit unlearning baselines, and retraining references under a privacy contract whose adjacency matches the deletion request.

## Current Status

| Area | Status | Evidence now available | Next gate |
|---|---|---|---|
| Research protocol | Complete | [research plan](RESEARCH_GRADE_PLAN.md), [Phase 0 protocol](docs/PHASE_0_PROTOCOL.md), [threat model](docs/THREAT_MODEL.md) | freeze preregistered final endpoints |
| PDF proposal | Complete | [RESEARCH_GRADE_PLAN.pdf](reports/RESEARCH_GRADE_PLAN.pdf) | regenerate after major plan changes |
| Deterministic federation | Complete | synthetic plus real CIFAR-10 binary adapters; versioned deletion manifests | multi-class CIFAR-10/CIFAR-100/FEMNIST |
| FedAvg reference | Complete | client sampling, local training, aggregation, exact retrain, independent retrain | reproduce published FL baseline settings |
| Client-level central DP | Complete for simulator | clipping, server Gaussian noise, Poisson RDP ledger, fixed public normalizer | secure aggregation and independent accountant audit |
| Explicit unlearning | Baselines partial | retained-data fine-tuning; cached-update direct reconstruction | faithful FedEraser/FedRecover/modern FU reproduction |
| Evaluation | Partial but runnable | utility, deleted/retained loss, JS alignment, lifecycle cost, label-matched MIA probes, tri-population probe | LiRA/A-LiRA and faithful TC-UMIA style attacks |
| Real-data results | Exploratory complete | [PHASE1_REAL_CIFAR_RESULTS.md](reports/PHASE1_REAL_CIFAR_RESULTS.md), 30 validated CIFAR runs | stronger models, stronger baselines, confirmatory seeds |

## Real CIFAR Result Snapshot

The corrected Phase 1 grid used CIFAR-10 classes 0 vs 1, public average-pooled features, 12 clients, one full-client deletion, epsilons `{2, 8, infinity}`, heterogeneity `{0.2, 0.75}`, and 5 seeds per cell.

Main exploratory result: across all six cells, explicit unlearning baselines did **not** show a practically material JS-to-retrain improvement over DP-only/no-action under the current screen. This is a useful research signal, not a final SOTA claim.

Artifacts:

- run index: [phase1_v3_reference_corrected_run_index.json](reports/phase1_v3_reference_corrected_run_index.json)
- grouped screen: [phase1_v3_reference_corrected_grid_screen.json](reports/phase1_v3_reference_corrected_grid_screen.json)
- readable report: [PHASE1_REAL_CIFAR_RESULTS.md](reports/PHASE1_REAL_CIFAR_RESULTS.md)
- summary JSON: [phase1_v3_reference_corrected_summary.json](reports/phase1_v3_reference_corrected_summary.json)

## Privacy Rule

The primary experiment is **client-level central DP**: whole client updates are clipped, Gaussian noise is added at the server, and the accountant records client-sampling privacy loss. Example-level DP-SGD belongs only to a separate record-deletion control. Do not present one as the other.

The current mechanism is `poisson_sum_fixed_public_normalizer_v2`: sampled clipped updates are summed, noise is added to the sum, and the result is divided by fixed public `q * N`, never by the realized number of sampled clients. The released model ledger uses the original client population; retraining references use the retained-client population and record that separately in metadata.

## Quick Start

Python 3.10-3.13 is supported.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest -q
```

Run one deterministic smoke experiment:

```powershell
python -m dp_forgetbench.run --config configs/phase0_toy.yaml --mode full
```

Run the corrected real-data grid:

```powershell
python scripts/generate_phase1_grid.py --spec configs/phase1_grid_spec_v2.yaml --output configs/generated/phase1_v2
python scripts/run_generated_grid.py --index configs/generated/phase1_v2/INDEX.txt --run-index reports/phase1_v3_reference_corrected_run_index.json --execute
python scripts/analyze_grid.py --run-index reports/phase1_v3_reference_corrected_run_index.json --output reports/phase1_v3_reference_corrected_grid_screen.json
python scripts/report_phase1_real.py --run-index reports/phase1_v3_reference_corrected_run_index.json --screen reports/phase1_v3_reference_corrected_grid_screen.json --markdown reports/PHASE1_REAL_CIFAR_RESULTS.md --json reports/phase1_v3_reference_corrected_summary.json
```

Validate a run before using it in a report:

```powershell
python scripts/validate_artifact.py results\<run-dir>
```

Regenerate PDFs:

```powershell
python scripts/build_pdf.py RESEARCH_GRADE_PLAN.md reports/RESEARCH_GRADE_PLAN.pdf
python scripts/build_pdf.py reports/PHASE1_REAL_CIFAR_RESULTS.md reports/PHASE1_REAL_CIFAR_RESULTS.pdf
python scripts/build_pdf.py README.md reports/README.pdf
```

## What Each Run Writes

Each result directory under `results/` contains:

- `metrics.json`: utility, deleted/retained metrics, JS alignment, attack diagnostics, and cost.
- `privacy_ledger.json`: DP mechanism, epsilon, delta, sampling rate, release count, and assumptions.
- `run_metadata.json`: config hash, runtime details, deletion manifest checksum, method privacy classes, and reference privacy config.

## What Is Still Needed For A Paper-Level Claim

- stronger CIFAR-10/CIFAR-100/FEMNIST adapters and at least one GPU-scale model;
- faithful published FU baselines, starting with FedEraser-style and FedRecover-style reproductions;
- calibrated LiRA/A-LiRA membership inference plus stronger forget/retain probes;
- preregistered multi-endpoint equivalence or non-inferiority tests;
- >=10 confirmation seeds on transition cells;
- a final 2026 literature check immediately before submission.

## Layout

```text
configs/                 Immutable experiment configurations
docs/                    Protocol and threat-model documentation
partitions/              Versioned deletion manifests generated by runs
reports/                 PDFs, result summaries, validity notices
results/                 Git-ignored raw experiment outputs
scripts/                 Grid, validation, analysis, and PDF helpers
src/dp_forgetbench/      Benchmark implementation
tests/                   Determinism, privacy, deletion, and smoke tests
```

Read [RESEARCH_GRADE_PLAN.md](RESEARCH_GRADE_PLAN.md) before changing experimental scope.
