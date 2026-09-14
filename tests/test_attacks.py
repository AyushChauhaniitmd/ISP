"""Tests for calibrated membership inference attacks with bootstrap CIs."""

import numpy as np
import pytest
import torch

from dp_forgetbench.attacks import _bootstrap_roc_metrics, loss_membership_probe
from dp_forgetbench.federated import BinaryLinearModel


def test_bootstrap_roc_metrics_monotonic_and_bounded():
    labels = np.array([1] * 50 + [0] * 50)
    # Perfect separation scores
    scores = np.array([10.0] * 50 + [-10.0] * 50)
    res = _bootstrap_roc_metrics(labels, scores, n_bootstraps=50, seed=123)
    assert "auc_ci_95" in res
    assert "advantage_ci_95" in res
    assert "tpr_at_fpr_1pct_ci_95" in res
    assert "tpr_at_fpr_0_1pct_ci_95" in res

    # AUC should be 1.0
    assert res["auc_ci_95"][0] == 1.0
    assert res["auc_ci_95"][1] == 1.0
    assert res["advantage_ci_95"][0] == 1.0


def test_loss_membership_probe_outputs_ci_and_rates():
    torch.manual_seed(42)
    model = BinaryLinearModel(n_features=4)
    # Synthetic members and nonmembers
    mx = torch.randn(20, 4)
    my = torch.randint(0, 2, (20,)).float()
    ux = torch.randn(20, 4)
    uy = torch.randint(0, 2, (20,)).float()

    probe_res = loss_membership_probe(model, mx, my, ux, uy, n_bootstraps=20, seed=42)
    assert "auc" in probe_res
    assert "auc_ci_95" in probe_res
    assert "advantage" in probe_res
    assert "advantage_ci_95" in probe_res
    assert "tpr_at_fpr_1pct" in probe_res
    assert "tpr_at_fpr_1pct_ci_95" in probe_res
    assert "tpr_at_fpr_0_1pct" in probe_res
    assert "tpr_at_fpr_0_1pct_ci_95" in probe_res
    assert probe_res["members"] > 0
    assert probe_res["nonmembers"] > 0
