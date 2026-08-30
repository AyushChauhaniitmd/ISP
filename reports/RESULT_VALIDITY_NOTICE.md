# Result Validity Notice

Date: 2026-08-25

## Correction 1: DP mechanism/accountant alignment

The original simulator averaged selected client updates using the realized number of sampled clients while the ledger used the RDP accountant for a Poisson-sampled Gaussian sum mechanism. Those are not the same mechanism.

The implementation now uses `poisson_sum_fixed_public_normalizer_v2`: it sums L2-clipped updates, adds Gaussian noise to the sum, and divides only by fixed public `q * N`. The RDP accountant matches this mechanism.

All result directories whose `privacy_ledger.json` lacks `mechanism_version: poisson_sum_fixed_public_normalizer_v2` are pre-fix engineering artifacts. They must not be used in a privacy, unlearning, redundancy, or paper result.

## Correction 2: retained-reference population

The first completed real-data `phase1_v2` grid used the corrected released-model DP mechanism, but the retrain reference still reused the original full-client population normalizer after a full-client deletion. That made the reference target less clean than the intended D-minus-F retrain.

The corrected runner now records two privacy configs:

- `effective_privacy_config`: the public released full-model mechanism, with original client population.
- `reference_privacy_config`: the internal retained-data retrain target, with retained-client population.

The valid corrected real-data grid is indexed at `reports/phase1_v3_reference_corrected_run_index.json`.

## Required replacement evidence

1. Rerun every reported cell with the v2 mechanism and corrected retrain-reference population.
2. Validate each output with `python scripts/validate_artifact.py <run-dir>`.
3. Aggregate only validated runs from the corrected index.
4. Treat results as exploratory until public benchmark-scale models, faithful baselines, calibrated attacks, and confirmation runs are complete.
