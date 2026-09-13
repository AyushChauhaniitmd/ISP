# CHANGELOG — Phase 2 scaffolding session

Applied by request after the last meeting/status review. Summary for your own
verification before you treat any of this as validated — nothing here has been
run against real (downloaded) CIFAR-10 or on GPU; only mocked-data smoke tests.

## New capability

- **Multi-class CIFAR-10 data backend**: `data.py::make_cifar10_multiclass_federation`.
  Real 10-class images kept as `(N, 3, 32, 32)` tensors (no flatten/pool), so a
  convolutional model can train on them. Client heterogeneity uses a symmetric
  Dirichlet(alpha) split over the 10 classes (standard non-IID-FL convention),
  replacing the binary backend's flip-probability knob.
- **CNN model now reachable**: `SmallGroupNormCNN` already existed in
  `federated.py` but nothing routed real image data to it. Fixed via a new
  `infer_n_features` helper so `run.py` no longer assumes `x.shape[1]` is a
  flat feature count (it isn't, for image tensors).
- **Simplified offline LiRA**: `src/dp_forgetbench/lira.py` +
  `scripts/run_lira_confirmation.py`. A genuine shadow-model likelihood-ratio
  attack (Carlini et al. 2022, offline/simplified variant — no cross-example
  variance pooling, documented in the module docstring). Deliberately kept
  *outside* the main `run_experiment` pipeline: it's expensive (trains N shadow
  models) and is meant as a targeted confirmation check against a completed
  run, not a per-seed grid metric. It re-derives the exact target model from
  the run's saved config + seed (model weights are not persisted to disk by
  `run.py`, and this was the lower-risk option vs. changing what gets saved).
- **Phase 2 configs + grid generator**: `configs/phase2_cifar10_multiclass_cnn.yaml`
  (+ nonprivate twin), `configs/phase2_grid_spec.yaml`,
  `scripts/generate_phase2_grid.py`. Mirrors the existing validated Phase 1 v2
  pattern; does not modify `generate_phase1_grid.py`.

## Bug fixes found while generalizing existing code

- `attacks.py::_balanced_binary_groups` hardcoded `for label in (0.0, 1.0)`.
  On a 10-class task this would have silently matched zero examples for every
  label outside {0,1} and raised (or worse, silently starved the attack of
  data if any 0/1-labeled examples happened to exist). Now derives the label
  set from the data itself. Covered by a regression test
  (`test_loss_membership_probe_handles_ten_classes`).
- `evaluation.py::js_divergence_to_target` assumed a single sigmoid logit.
  Generalized to accept softmax multi-class logits too, sharing one code path.
- `run.py` computed `n_features = x.shape[1]`, which is meaningless (and
  wrong) for `(N, C, H, W)` image tensors. Replaced with `infer_n_features`,
  which returns `None` for image data (the CNN branch of `make_model` ignores
  the argument anyway).

## What is NOT done — do not skip this

- **No real training has been run.** Everything above was validated with
  `unittest.mock.patch` standing in for `torchvision.datasets.CIFAR10` (tiny
  random images, a few seconds per test). This proves the code paths execute
  and produce well-formed output — it says nothing about whether the CNN
  actually learns anything useful on real CIFAR-10, whether 30 rounds/the
  given learning rate are remotely sensible, or whether the "DP makes
  unlearning redundant" signal survives with a real non-linear model.
- **LiRA has not been run with a real shadow-model count.** Tests use 3-4
  shadow models to check plumbing only. A meaningful confirmation run needs
  >=32 (see the run_lira_confirmation.py docstring) and will take real time.
- Phase 2 config hyperparameters (`rounds: 30`, `learning_rate: 0.02`,
  `clip_norm: 1.0`) are untuned starting points, not the result of any search.
- FEMNIST and ResNet-18 (also on your roadmap slide) are still not started —
  this session only added CIFAR-10 multi-class + the small CNN already sitting
  unused in the codebase.

## Test status

`pytest -q` → 13 passed (8 pre-existing + 5 new: 4 in
`tests/test_multiclass_cnn.py`, 1 in `tests/test_lira.py`). No regressions in
existing Phase 0/1 tests.
