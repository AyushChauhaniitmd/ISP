# Phase 1 CIFAR-10 CPU pilot protocol

## Purpose

Exercise the full privacy/deletion/attack pipeline on real public data before paying for a multi-class, multi-method study. This is a binary (airplane vs automobile), linear-model CIFAR-10 pilot to remain tractable on the detected CPU-only hardware. Images receive fixed public 4×4 average pooling (192 dimensions); this preprocessing has no fitted private parameters and is frozen before grid generation.

## Frozen conditions

- 12 clients, 100 examples per client, class-skewed deterministic assignment.
- One complete client deletion, client-level central DP, Poisson client sampling.
- DP-only/no action, retained-data fine-tuning baseline, two target retrains.
- Held-out CIFAR training examples are used as unseen audit data; they are not used for training.
- Attack outputs are loss-threshold probes and a held-out pre/post tri-population probe.

## Interpretation

This pilot may validate implementation and reveal failures. It cannot establish SOTA, a universal threshold, multi-class robustness, or a competitive federated-unlearning result. Promotion to a paper figure requires the Phase 2 public-data grid, published FU baselines, LiRA/A-LiRA, and at least five seeds.
