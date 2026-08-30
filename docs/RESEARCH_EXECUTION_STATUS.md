# Research execution dashboard

Last updated: 2026-08-25

## Completed with executable evidence

- Research plan, research questions, novelty boundaries, and privacy scope are documented.
- Proposal PDF is generated and validates as a PDF file.
- CIFAR-10 archive was verified and extracted locally; the real-data adapter runs without downloading.
- Client-level central-DP simulator, RDP ledger, deterministic deletion manifests, exact retraining, independent target retraining, retained-data fine-tuning, and cached-update reconstruction are implemented.
- Utility, functional JS alignment, lifecycle cost, label-matched loss MIA, retained MIA, and held-out tri-population baseline probes are recorded per run.
- Accountant-aligned v2 DP mechanism is implemented: clipped update sum, server Gaussian noise, fixed public `q * N` normalizer.
- Retained-data retrain references now use and record a separate retained-client population normalizer.
- Corrected Phase 1 real CIFAR-10 binary grid is complete: 30 validated runs, five seeds per cell, epsilon `{2, 8, infinity}`, heterogeneity `{0.2, 0.75}`.
- Real-data summary artifacts are available in [PHASE1_REAL_CIFAR_RESULTS.md](../reports/PHASE1_REAL_CIFAR_RESULTS.md), [phase1_v3_reference_corrected_grid_screen.json](../reports/phase1_v3_reference_corrected_grid_screen.json), and [phase1_v3_reference_corrected_summary.json](../reports/phase1_v3_reference_corrected_summary.json).
- Automated test suite passes: 8 tests.

## Superseded evidence

- Pre-fix synthetic batches executed before the DP mechanism audit are retained only as debugging artifacts; see [result validity notice](../reports/RESULT_VALIDITY_NOTICE.md).
- The first `phase1_v2` real-data grid used the corrected v2 released-model DP mechanism, but the retrain reference reused the original full-client population normalizer. It is superseded by `phase1_v3_reference_corrected_run_index.json` and must not be used for paper results.

## In progress / research-gated

| Work item | Why it is not yet a paper result | Completion condition |
|---|---|---|
| Stronger CIFAR/FEMNIST models | Current real grid is CPU-feasible binary CIFAR with pooled features | GPU-scale CNN/ResNet adapter with validated DP accounting and comparable retrain references |
| Published FU baselines | Current explicit methods are labelled baselines, not faithful published reproductions | Reproduce FedEraser/FedRecover-style baselines under a documented threat model |
| LiRA/A-LiRA and TC-UMIA | Current attacks are baseline diagnostics only | Frozen attack splits, shadow-model calibration, held-out attack evaluation |
| Sequential and partial requests | Runner supports sequence mode, but real-data grid has only one full-client deletion | Pre-register and run fractional/sequential real-data grid |
| Confirmatory redundancy frontier | Current result is an exploratory screen over five seeds | >=10 confirmation seeds on transition cells and locked multi-endpoint tests |

## Claim status

No SOTA, certified-unlearning, legal-deletion, or universal-DP-redundancy claim has been earned. The current earned result is narrower and stronger: in the corrected exploratory CIFAR-10 binary client-deletion grid, explicit unlearning baselines did not materially improve JS-to-retrain over DP-only/no-action under the registered screen. Paper-level claims still require stronger baselines, calibrated attacks, stronger models, confirmation seeds, and a final literature check immediately before submission.

## Reproduction commands

```powershell
python -m pytest -q
python scripts/generate_phase1_grid.py --spec configs/phase1_grid_spec_v2.yaml --output configs/generated/phase1_v2
python scripts/run_generated_grid.py --index configs/generated/phase1_v2/INDEX.txt --run-index reports/phase1_v3_reference_corrected_run_index.json --execute
python scripts/analyze_grid.py --run-index reports/phase1_v3_reference_corrected_run_index.json --output reports/phase1_v3_reference_corrected_grid_screen.json
python scripts/report_phase1_real.py --run-index reports/phase1_v3_reference_corrected_run_index.json --screen reports/phase1_v3_reference_corrected_grid_screen.json --markdown reports/PHASE1_REAL_CIFAR_RESULTS.md --json reports/phase1_v3_reference_corrected_summary.json
```
