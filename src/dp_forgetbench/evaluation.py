"""Phase 0 evaluation: utility, functional alignment, and transparent cost."""

from __future__ import annotations

import torch

from .attacks import loss_membership_probe, tri_population_probe
from .federated import BinaryLinearModel, TrainCost, evaluate_loss_accuracy, flatten_logits


def js_divergence_to_target(candidate: BinaryLinearModel, target: BinaryLinearModel, x: torch.Tensor) -> float:
    """Mean Jensen-Shannon divergence between binary predictive distributions."""
    p = torch.sigmoid(flatten_logits(candidate, x)).clamp(1e-7, 1 - 1e-7)
    q = torch.sigmoid(flatten_logits(target, x)).clamp(1e-7, 1 - 1e-7)
    p_dist = torch.stack([p, 1 - p], dim=1)
    q_dist = torch.stack([q, 1 - q], dim=1)
    midpoint = 0.5 * (p_dist + q_dist)
    kl_p = (p_dist * (p_dist.log() - midpoint.log())).sum(dim=1)
    kl_q = (q_dist * (q_dist.log() - midpoint.log())).sum(dim=1)
    return float((0.5 * (kl_p + kl_q)).mean().item())


def evaluate_against_target(
    *,
    candidate: BinaryLinearModel,
    target: BinaryLinearModel,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    forgotten_x: torch.Tensor,
    forgotten_y: torch.Tensor,
    retained_x: torch.Tensor,
    retained_y: torch.Tensor,
    unseen_x: torch.Tensor,
    unseen_y: torch.Tensor,
    pre_deletion_model: BinaryLinearModel,
    attack_seed: int,
    cost: TrainCost,
) -> dict:
    return {
        "test": evaluate_loss_accuracy(candidate, test_x, test_y),
        "forgotten_client": evaluate_loss_accuracy(candidate, forgotten_x, forgotten_y),
        "retained_data": evaluate_loss_accuracy(candidate, retained_x, retained_y),
        "alignment_to_retrain": {
            "test_js_divergence": js_divergence_to_target(candidate, target, test_x),
            "forgotten_js_divergence": js_divergence_to_target(candidate, target, forgotten_x),
        },
        "attack_diagnostics": {
            "forgotten_vs_unseen_loss_mia": loss_membership_probe(candidate, forgotten_x, forgotten_y, unseen_x, unseen_y),
            "retained_vs_unseen_loss_mia": loss_membership_probe(candidate, retained_x, retained_y, unseen_x, unseen_y),
            "tri_population_pre_post_probe": tri_population_probe(
                pre_deletion_model,
                candidate,
                {"forget": (forgotten_x, forgotten_y), "retain": (retained_x, retained_y), "unseen": (unseen_x, unseen_y)},
                seed=attack_seed,
            ),
        },
        "cost": cost.as_dict(),
    }
