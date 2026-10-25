# GPU Execution Guide

## Environment

From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Verify the environment:

```powershell
python -c "import torch, torchvision; print(torch.__version__); print(torchvision.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

The sweep automatically uses CUDA when PyTorch exposes it. The current local machine may run only the smoke tests; use a CUDA workstation or Colab T4 for the real sweep.

## Smoke test

```powershell
python -m pytest -q tests/test_multiclass_cnn.py
python scripts/calibrate_noise.py --epsilon 8 12 16 20 24 32 --sample-rate 0.5 --rounds 60 --delta 1e-5
```

## Utility sweep

```powershell
python scripts/run_multiclass_utility_sweep.py --config configs/multiclass_resnet18_utility_sweep.yaml
```

Outputs:

```text
results/utility_sweep_resnet18/utility_sweep.csv
results/utility_sweep_resnet18/utility_sweep.json
```

This command is intentionally training-only. It does not create deletion manifests or run unlearning.

## Controlled ablations

Change one family at a time in a copied config:

```yaml
federated:
  rounds: 40   # then 80, then 120
```

Then freeze rounds before varying `local_epochs`, followed by client count and participation. Preserve the actual ledger epsilon for every run.

## After a healthy point

Do not tune the configuration using unlearning outcomes. Copy the selected config to a frozen configuration, record its commit and checksum, and only then run the full unlearning comparison.

## Add confirmatory seeds to the frozen multiclass study

From the repository root, use the batch runner to add the seven planned seeds
(`20260921`–`20260927`) to the existing three-seed study:

```powershell
python scripts\run_frozen_seed_batch.py --dry-run
python scripts\run_frozen_seed_batch.py
```

The runner checks that the frozen configuration still matches the study,
executes seeds sequentially to avoid competing for GPU resources, skips
completed matching seeds on rerun, and writes the combined summary to
`results\frozen_unlearning_eps32_r40_e5\frozen_unlearning_summary.json`.
Use `--seeds` to specify another non-overlapping seed batch. The five-member
retrain ensemble remains enabled for every seed; reducing it would change the
study protocol rather than merely speed up execution.