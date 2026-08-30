# Execution guide and evidence gates

## 1. Freeze privacy budgets before execution

```powershell
python scripts/calibrate_noise.py --epsilon 1 2 4 8 --sample-rate 0.5 --rounds 12 --delta 1e-5
python scripts/generate_phase1_grid.py --spec configs/phase1_grid_spec_v2.yaml --output configs/generated/phase1_v2
```

Review `configs/generated/phase1_v2/INDEX.txt` before execution, preserve the generated YAMLs, and do not change hyperparameters after seeing outcomes.

## 2. Run a single real-data smoke job first

```powershell
python scripts/run_generated_grid.py --index configs/generated/phase1_v2/INDEX.txt --limit 1 --execute
python scripts/validate_artifact.py results/<run-directory>
```

The CIFAR-10 archive can live at `data/cifar-10-python.tar.gz`; the adapter will use the extracted `data/cifar-10-batches-py` directory without downloading.

## 3. Execute the corrected frozen grid

```powershell
python scripts/run_generated_grid.py --index configs/generated/phase1_v2/INDEX.txt --run-index reports/phase1_v3_reference_corrected_run_index.json --execute
python scripts/analyze_grid.py --run-index reports/phase1_v3_reference_corrected_run_index.json --output reports/phase1_v3_reference_corrected_grid_screen.json
python scripts/report_phase1_real.py --run-index reports/phase1_v3_reference_corrected_run_index.json --screen reports/phase1_v3_reference_corrected_grid_screen.json --markdown reports/PHASE1_REAL_CIFAR_RESULTS.md --json reports/phase1_v3_reference_corrected_summary.json
```

Validate every raw artifact before aggregating or plotting it:

```powershell
$idx = Get-Content reports/phase1_v3_reference_corrected_run_index.json | ConvertFrom-Json
foreach ($r in $idx.runs) { python scripts/validate_artifact.py $r }
```

## 4. Interpret cautiously

The screen is exploratory. It cannot certify equivalence, unlearning, legal deletion, or SOTA. Confirmation requires locked tolerances, public baseline reproductions, stronger attacks, multi-class/natural-FL datasets, and the Phase 2/3 protocol in the research plan.

For a multi-cell frozen grid, use `scripts/analyze_grid.py` and `scripts/report_phase1_real.py` rather than pooling incompatible epsilon/heterogeneity cells.

## Current resource observation

This environment reports `torch 2.9.0+cpu` and no CUDA device. The current completed real-data grid is intentionally CPU-feasible: CIFAR-10 binary classification with public average-pooled features. GPU-scale CNN/ResNet runs are still a separate paper-level gate.
