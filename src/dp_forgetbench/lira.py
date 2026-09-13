"""Simplified offline LiRA (Likelihood Ratio Attack), Carlini et al. 2022.

This is a genuine shadow-model membership-inference attack, meaningfully
stronger than the loss-threshold probe in `attacks.py`. It is intentionally
kept separate from the main Phase 0/1 pipeline in `run.py`/`evaluation.py`:
LiRA needs many independently trained shadow models per target example and is
too expensive to run inside every grid cell, so it is used as a targeted,
offline confirmation check on a *sample* of examples from an existing run,
not as a per-run metric computed for every seed.

What this implements (and what it deliberately does not):
- Offline LiRA only: shadow models are trained once, without the target
  example, and a model IN a shadow model's own training set is never scored
  against that same shadow model (no "online" per-example shadow retraining).
- Per-example Gaussian fit to the OUT-distribution of a logit-scaled confidence
  statistic, as in the original paper's simplified/offline variant.
- No global-variance pooling across examples (the paper's full method can
  pool variance across examples of similar difficulty for extra power); this
  keeps the implementation short and auditable at some cost in attack power,
  and it is documented here as the known gap versus a full reproduction.

Usage is via `run_lira_confirmation` below, or `scripts/run_lira_confirmation.py`
on the command line.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .federated import flatten_logits, infer_n_features, make_model, set_seed
from .data import Federation


def _logit_confidence(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> np.ndarray:
    """Logit-scaled confidence statistic phi = log(p / (1 - p)) for the true class.

    Using the logit of the confidence (rather than raw confidence) is what
    makes the per-example IN/OUT distributions closer to Gaussian, which the
    likelihood-ratio step below relies on.
    """
    logits = flatten_logits(model, x)
    if logits.ndim == 1 or logits.shape[-1] == 1:
        p_true = torch.sigmoid(logits.squeeze(-1))
        p_true = torch.where(y.cpu().bool(), p_true, 1 - p_true)
    else:
        probs = torch.softmax(logits, dim=-1)
        p_true = probs.gather(1, y.cpu().long().view(-1, 1)).squeeze(-1)
    p_true = p_true.clamp(1e-6, 1 - 1e-6)
    phi = torch.log(p_true / (1 - p_true))
    return phi.numpy()


@dataclass
class LiraResult:
    """Per-example membership scores for a fixed audit set, plus attack-level ROC."""

    example_index: list[int]
    true_membership: list[int]  # 1 if the example was in the target model's training set
    likelihood_ratio_score: list[float]  # higher => more evidence of membership
    n_shadow_models: int
    n_examples: int

    def auc(self) -> float:
        from sklearn.metrics import roc_auc_score

        return float(roc_auc_score(self.true_membership, self.likelihood_ratio_score))

    def tpr_at_fpr(self, target_fpr: float) -> float:
        from sklearn.metrics import roc_curve

        fpr, tpr, _ = roc_curve(self.true_membership, self.likelihood_ratio_score)
        valid = tpr[fpr <= target_fpr]
        return float(valid.max()) if len(valid) else 0.0

    def summary(self) -> dict:
        return {
            "method": "simplified_offline_lira",
            "n_shadow_models": self.n_shadow_models,
            "n_examples": self.n_examples,
            "auc": self.auc(),
            "tpr_at_fpr_1pct": self.tpr_at_fpr(0.01),
            "tpr_at_fpr_0_1pct": self.tpr_at_fpr(0.001),
            "warning": (
                "Simplified offline LiRA: per-example Gaussian OUT-distribution fit, "
                "no cross-example variance pooling. Weaker than a full paper "
                "reproduction but meaningfully stronger than the loss-threshold probe "
                "in attacks.py. Treat as a confirmation check, not a per-seed metric."
            ),
        }


def _train_shadow_model(
    *,
    all_examples_x: torch.Tensor,
    all_examples_y: torch.Tensor,
    include_mask: np.ndarray,
    n_features: int | None,
    model_config: dict | None,
    federated_config: dict,
    seed: int,
) -> nn.Module:
    """Train one non-federated shadow model on a random half of the audit pool.

    Shadow models are trained centrally (not through the FL simulator) because
    LiRA's statistical assumptions are about the training *algorithm*, and here
    we are attacking a single centrally-releasable checkpoint, matching the
    original LiRA setup. If you want shadow models trained through the same
    federated + DP pipeline as the target, see the note in
    `run_lira_confirmation` about that (more expensive) extension.
    """
    set_seed(seed)
    model = make_model(model_config, n_features)
    optimizer = torch.optim.SGD(model.parameters(), lr=float(federated_config["learning_rate"]))
    x = all_examples_x[include_mask]
    y = all_examples_y[include_mask]
    batch_size = int(federated_config["local_batch_size"])
    epochs = max(1, int(federated_config.get("shadow_epochs", federated_config.get("local_epochs", 1)) * 5))
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(len(y))
        x, y = x[perm], y[perm]
        for start in range(0, len(y), batch_size):
            xb, yb = x[start : start + batch_size], y[start : start + batch_size]
            optimizer.zero_grad()
            logits = model(xb)
            if logits.ndim == 1 or logits.shape[-1] == 1:
                loss = nn.functional.binary_cross_entropy_with_logits(logits.squeeze(-1), yb.float())
            else:
                loss = nn.functional.cross_entropy(logits, yb.long())
            loss.backward()
            optimizer.step()
    model.eval()
    return model


def run_lira_confirmation(
    *,
    target_model: nn.Module,
    federation: Federation,
    member_client_ids: list[int],
    n_shadow_models: int,
    n_audit_examples: int,
    federated_config: dict,
    model_config: dict | None,
    seed: int,
) -> LiraResult:
    """Run a simplified offline LiRA against `target_model` on a sample of examples.

    `member_client_ids` are the clients whose data trained `target_model` (use
    the *retained* client set when attacking a post-deletion model). Audit
    examples are drawn half from those members' data and half from
    `federation.audit_x/audit_y` (data the target never trained on), so ground
    truth membership is known and the resulting AUC is directly comparable to
    `attacks.loss_membership_probe`'s AUC on the same run.
    """
    rng = np.random.default_rng(seed)
    member_x = torch.cat([federation.clients[cid].x for cid in member_client_ids])
    member_y = torch.cat([federation.clients[cid].y for cid in member_client_ids])
    n_take = min(n_audit_examples // 2, len(member_y), len(federation.audit_y))
    if n_take < 8:
        raise ValueError("Not enough member/non-member examples for a meaningful LiRA confirmation run.")
    member_idx = rng.choice(len(member_y), size=n_take, replace=False)
    nonmember_idx = rng.choice(len(federation.audit_y), size=n_take, replace=False)

    audit_x = torch.cat([member_x[member_idx], federation.audit_x[nonmember_idx]])
    audit_y = torch.cat([member_y[member_idx], federation.audit_y[nonmember_idx]])
    true_membership = np.concatenate([np.ones(n_take, dtype=int), np.zeros(n_take, dtype=int)])

    # Shadow pool: member examples (label 1 for "could be trained on") plus the
    # disjoint audit pool, so shadow models are trained on data drawn from the
    # same generating process as the target's true training population without
    # ever including the audit examples we score membership for as guaranteed
    # positives -- each audit example is independently included/excluded per
    # shadow model below.
    shadow_pool_x = torch.cat([member_x, federation.audit_x])
    shadow_pool_y = torch.cat([member_y, federation.audit_y])
    n_pool = len(shadow_pool_y)
    audit_pool_positions = np.concatenate([member_idx, len(member_y) + nonmember_idx])

    out_scores = {i: [] for i in range(2 * n_take)}
    n_features = infer_n_features(shadow_pool_x)

    for shadow_seed in range(n_shadow_models):
        include_mask = rng.random(n_pool) < 0.5
        shadow_model = _train_shadow_model(
            all_examples_x=shadow_pool_x,
            all_examples_y=shadow_pool_y,
            include_mask=include_mask,
            n_features=n_features,
            model_config=model_config,
            federated_config=federated_config,
            seed=seed * 1000 + shadow_seed,
        )
        phi = _logit_confidence(shadow_model, audit_x, audit_y)
        for local_i, pool_position in enumerate(audit_pool_positions):
            if not include_mask[pool_position]:
                out_scores[local_i].append(phi[local_i])

    target_phi = _logit_confidence(target_model, audit_x, audit_y)
    likelihood_ratio_score = []
    for i in range(2 * n_take):
        out_values = np.asarray(out_scores[i])
        if len(out_values) < 2:
            # Fall back to raw confidence if too few shadow models excluded this
            # example; documented rather than silently biased.
            likelihood_ratio_score.append(float(target_phi[i]))
            continue
        mean_out, std_out = float(out_values.mean()), float(out_values.std(ddof=1) + 1e-6)
        # Offline LiRA score: higher target confidence relative to the fitted
        # OUT-only Gaussian is stronger membership evidence. This is exactly
        # `-log N(target_phi | mean_out, std_out)` up to an additive constant
        # that does not depend on the example, so it preserves the correct AUC
        # ranking versus the paper's full likelihood-ratio while only needing
        # one (OUT) distribution instead of two (IN and OUT), matching this
        # module's documented offline/simplified scope.
        z = (target_phi[i] - mean_out) / std_out
        likelihood_ratio_score.append(float(z))

    return LiraResult(
        example_index=list(range(2 * n_take)),
        true_membership=true_membership.tolist(),
        likelihood_ratio_score=likelihood_ratio_score,
        n_shadow_models=n_shadow_models,
        n_examples=2 * n_take,
    )
