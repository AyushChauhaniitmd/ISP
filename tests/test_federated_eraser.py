"""Tests for faithful FedEraser implementation."""

import pytest
import torch

from dp_forgetbench.data import ClientDataset
from dp_forgetbench.federated import (
    FederatedHistory,
    RoundHistory,
    federated_eraser,
    l2_norm,
    make_model,
    train_federated,
)


def test_federated_eraser_execution_and_calibration():
    # Setup small synthetic clients
    torch.manual_seed(42)
    n_features = 4
    c0 = ClientDataset(client_id=0, x=torch.randn(10, n_features), y=torch.randint(0, 2, (10,)).float())
    c1 = ClientDataset(client_id=1, x=torch.randn(10, n_features), y=torch.randint(0, 2, (10,)).float())
    c2 = ClientDataset(client_id=2, x=torch.randn(10, n_features), y=torch.randint(0, 2, (10,)).float())
    clients = {0: c0, 1: c1, 2: c2}

    fed_config = {
        "rounds": 3,
        "local_epochs": 1,
        "local_batch_size": 5,
        "learning_rate": 0.05,
        "client_sample_rate": 1.0,
    }
    priv_config = {
        "enabled": True,
        "population_size": 3,
        "clip_norm": 1.0,
        "noise_multiplier": 0.5,
        "delta": 1e-5,
    }

    # Train initial model with history tracking
    history = FederatedHistory()
    full_model, full_cost = train_federated(
        clients=clients,
        n_features=n_features,
        federated_config=fed_config,
        privacy_config=priv_config,
        seed=42,
        history=history,
    )

    assert len(history.rounds) == 3
    assert history.storage_bytes() > 0

    # Unlearn client 2 using FedEraser
    retained_clients = {0: c0, 1: c1}
    unlearned_model, unlearn_cost = federated_eraser(
        history=history,
        clients=retained_clients,
        forgotten_client_ids={2},
        n_features=n_features,
        federated_config=fed_config,
        privacy_config=priv_config,
        calibration_ratio=0.5,
    )

    assert unlearned_model is not None
    assert unlearn_cost.rounds == 3
    assert unlearn_cost.client_updates > 0
    assert unlearn_cost.communicated_bytes > 0
    assert unlearn_cost.persistent_storage_bytes > 0
    assert unlearn_cost.runtime_seconds >= 0.0
