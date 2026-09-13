"""Fast unit test for the simplified offline LiRA implementation.

Uses tiny synthetic data and very few shadow models purely to check the code
path runs and returns well-formed output -- not to check attack power. A real
confirmation run needs >=32 shadow models (see scripts/run_lira_confirmation.py).
"""

from __future__ import annotations

import torch

from dp_forgetbench.data import ClientDataset, Federation
from dp_forgetbench.lira import run_lira_confirmation


def _tiny_federation(seed: int) -> Federation:
    torch.manual_seed(seed)
    clients = {}
    for client_id in range(4):
        clients[client_id] = ClientDataset(
            client_id=client_id,
            x=torch.randn(20, 6),
            y=torch.randint(0, 2, (20,)).float(),
        )
    return Federation(
        clients=clients,
        test_x=torch.randn(30, 6),
        test_y=torch.randint(0, 2, (30,)).float(),
        audit_x=torch.randn(30, 6),
        audit_y=torch.randint(0, 2, (30,)).float(),
    )


def test_run_lira_confirmation_returns_well_formed_result() -> None:
    from dp_forgetbench.federated import train_federated

    federation = _tiny_federation(seed=1)
    federated_config = {
        "rounds": 2,
        "local_epochs": 1,
        "local_batch_size": 8,
        "learning_rate": 0.1,
        "client_sample_rate": 1.0,
    }
    target_model, _cost = train_federated(
        clients=federation.clients,
        n_features=6,
        federated_config=federated_config,
        privacy_config={"enabled": False},
        seed=1,
    )
    result = run_lira_confirmation(
        target_model=target_model,
        federation=federation,
        member_client_ids=list(federation.clients),
        n_shadow_models=3,
        n_audit_examples=16,
        federated_config=federated_config,
        model_config=None,
        seed=2,
    )
    assert result.n_shadow_models == 3
    assert result.n_examples == 16
    assert len(result.likelihood_ratio_score) == 16
    assert set(result.true_membership) <= {0, 1}
    summary = result.summary()
    assert summary["method"] == "simplified_offline_lira"
    assert 0.0 <= summary["auc"] <= 1.0
