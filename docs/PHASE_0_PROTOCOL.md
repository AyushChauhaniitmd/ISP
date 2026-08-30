# Phase 0 protocol: deterministic plumbing and privacy-scope validation

## Goal

Demonstrate a reproducible end-to-end client-deletion experiment, not a publishable result. The deliverable is five independent seeds with valid manifests, privacy ledgers, target retrains, and baseline metrics.

## Frozen unit and threat model

- Protected/deleted unit: one complete client dataset.
- Adjacency: add/remove one client dataset.
- Simulator: central DP, Poisson client sampling, client-update L2 clipping, server Gaussian noise.
- Observer protected against: a black-box observer of the final released global model.
- Not provided: production secure aggregation, protection from a server that observes unclipped updates, or a legal deletion guarantee.
- Releases: final checkpoint only. Releasing intermediate checkpoints requires composition and a revised ledger.

## Required checks before interpreting numbers

1. `pytest` passes.
2. The same seed produces the same partition manifest checksum and metrics within numerical tolerance.
3. The forgotten client's samples are absent from the target retrain and retained fine-tuning client set.
4. The ledger records `(epsilon, delta)`, clipping, noise, sampling, and release count.
5. Two independent target retrains establish target variability before any frontier/equivalence claim.

## Phase 0 decision

Advance only when all five runs complete, raw result directories are preserved, and the achieved epsilon is plausibly useful for the selected utility regime. Do not tune against the test set or run attacks until fixed member/non-member splits are implemented.

## Cumulative/partial request control

`configs/phase0_partial_sequential.yaml` is an executable integration control. It creates a deterministic manifest for a partial-client request followed by two further requests and compares the resulting cumulative target to the same baselines. It validates request bookkeeping and final-state evaluation; it does **not** yet model a stateful production unlearning service that updates its method after every request.

Run stateful sequential baseline evaluation with:

```powershell
python -m dp_forgetbench.run --config configs/phase0_partial_sequential.yaml --mode sequence
```

This runner now persists a retained-fine-tuning baseline across each request prefix and emits `sequence_metrics.json`. It is still not a historical-reconstruction or certified FU method.
