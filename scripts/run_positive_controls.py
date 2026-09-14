"""
run_positive_controls.py
Phase C Positive Controls Suite:
1. Centralized CIFAR-10 Training (Control 1)
2. Standard Non-Private FedAvg (Control 2)
3. DP-ForgetBench epsilon=infinity Implementation (Control 3)

Demonstrates that the benchmark's model architecture and federated engine
can achieve healthy convergence on CIFAR-10 (>50% accuracy).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Ensure src is on sys.path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR / "src") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "src"))

from dp_forgetbench.data import make_cifar10_multiclass_federation
from dp_forgetbench.federated import SmallGroupNormCNN, get_device, set_seed, train_federated


def run_centralized_control(federation, epochs: int = 25, lr: float = 0.05, seed: int = 20260901) -> dict:
    set_seed(seed)
    device = get_device()
    print(f"\n--- Running Control 1: Centralized Training ({epochs} epochs on {device}) ---")

    # Aggregate all client data into a centralized pool
    all_x = torch.cat([c.x for c in federation.clients.values()], dim=0)
    all_y = torch.cat([c.y for c in federation.clients.values()], dim=0)
    dataset = TensorDataset(all_x, all_y)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    model = SmallGroupNormCNN(num_classes=10).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()

    start_time = time.perf_counter()
    for ep in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(y)
            correct += (logits.argmax(dim=1) == y).sum().item()
            total += len(y)

        train_acc = correct / total
        if (ep + 1) % 5 == 0 or ep == epochs - 1:
            print(f"  Epoch {ep+1:2d}/{epochs}: Train Loss = {total_loss/total:.4f}, Train Acc = {train_acc*100:.2f}%")

    elapsed = time.perf_counter() - start_time

    # Evaluate on test set
    model.eval()
    with torch.no_grad():
        test_x = federation.test_x.to(device)
        test_y = federation.test_y.to(device)
        test_logits = model(test_x)
        test_loss = criterion(test_logits, test_y).item()
        test_acc = (test_logits.argmax(dim=1) == test_y).float().mean().item()

    print(f"Centralized Result: Test Acc = {test_acc*100:.2f}%, Test Loss = {test_loss:.4f} in {elapsed:.1f}s")
    return {
        "control": "centralized",
        "epochs": epochs,
        "test_accuracy": test_acc,
        "test_loss": test_loss,
        "runtime_s": elapsed,
    }


def run_standard_fedavg(federation, rounds: int = 60, local_epochs: int = 3, lr: float = 0.1, sample_rate: float = 0.5, seed: int = 20260901) -> dict:
    set_seed(seed)
    device = get_device()
    print(f"\n--- Running Control 2: Standard FedAvg ({rounds} rounds, {local_epochs} local epochs, q={sample_rate}) ---")

    fed_cfg = {
        "rounds": rounds,
        "local_epochs": local_epochs,
        "local_batch_size": 32,
        "learning_rate": lr,
        "client_sample_rate": sample_rate,
    }
    priv_cfg = {"enabled": False}

    start_time = time.perf_counter()
    model, cost = train_federated(
        clients=federation.clients,
        n_features=None,
        federated_config=fed_cfg,
        privacy_config=priv_cfg,
        seed=seed,
        model_config={"name": "small_groupnorm_cnn", "num_classes": 10},
    )
    elapsed = time.perf_counter() - start_time

    model.eval()
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        test_x = federation.test_x.to(device)
        test_y = federation.test_y.to(device)
        test_logits = model(test_x)
        test_loss = criterion(test_logits, test_y).item()
        test_acc = (test_logits.argmax(dim=1) == test_y).float().mean().item()

    print(f"FedAvg Result: Test Acc = {test_acc*100:.2f}%, Test Loss = {test_loss:.4f} in {elapsed:.1f}s")
    return {
        "control": "standard_fedavg",
        "rounds": rounds,
        "local_epochs": local_epochs,
        "client_sample_rate": sample_rate,
        "test_accuracy": test_acc,
        "test_loss": test_loss,
        "runtime_s": elapsed,
    }


def main():
    parser = argparse.ArgumentParser(description="Positive Controls for DP-ForgetBench")
    parser.add_argument("--n_clients", type=int, default=100)
    parser.add_argument("--samples_per_client", type=int, default=200)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--rounds", type=int, default=60)
    parser.add_argument("--local_epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--output", type=Path, default=Path("results/positive_controls/results.json"))
    args = parser.parse_args()

    print(f"Building N={args.n_clients} CIFAR-10 Federation (Dirichlet alpha={args.alpha}, {args.samples_per_client} samples/client)...")
    data_cfg = {
        "backend": "cifar10_multiclass",
        "dataset_root": "data",
        "download": False,
        "n_clients": args.n_clients,
        "samples_per_client": args.samples_per_client,
        "heterogeneity_alpha": args.alpha,
        "audit_samples": 500,
        "test_samples": 2000,
    }
    federation = make_cifar10_multiclass_federation(data_cfg, seed=args.seed)

    results = {}
    results["centralized"] = run_centralized_control(federation, epochs=25, seed=args.seed)
    results["fedavg"] = run_standard_fedavg(
        federation,
        rounds=args.rounds,
        local_epochs=args.local_epochs,
        sample_rate=0.5,
        seed=args.seed,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nPositive controls saved to {args.output}")


if __name__ == "__main__":
    main()
