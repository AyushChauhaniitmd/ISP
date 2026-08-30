# Generated configuration validity

- `phase1/` and `phase0_synthetic/` were generated before the v2 accountant-mechanism correction. They remain as provenance artifacts and are **not eligible for result claims**.
- Generate new accountant-aligned configurations from `configs/phase1_grid_spec_v2.yaml` into `configs/generated/phase1_v2/`.
- Never modify a generated configuration after execution. Its SHA-256 is stored in every result directory.

