# Phase 1 v2 Grid Superseded

Date: 2026-08-25

The run index `reports/phase1_v2_run_index.json` is superseded.

Reason: those runs used the accountant-aligned released-model DP mechanism, but the internal retrain reference reused the original full-client population normalizer after a full-client deletion. The released DP ledger is still useful for debugging, but the grid should not be used for paper results or final benchmark tables.

Replacement: use `reports/phase1_v3_reference_corrected_run_index.json`, whose run metadata records:

- `effective_privacy_config.population_size = 12` for the released full model;
- `reference_privacy_config.population_size = 11` for the retained-client retrain target.

All runs in the replacement index passed `scripts/validate_artifact.py`.
