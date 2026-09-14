"""
run_utility_frontier.py
Phase E: Utility Frontier Discovery

Performs a focused epsilon sweep to map the exact boundary between:
A. Healthy training regime (test accuracy >= 40%) -> VALID
B. Marginal training regime (25% <= test accuracy < 40%) -> WARNING
C. Utility collapse regime (test accuracy < 25%) -> INVALID_TRAINING

Under client-level DP with N=100 clients, q=0.5, T=60 rounds, E=3 local epochs.
Records test accuracy, loss, NLL, ECE, retrain accuracy, and retrain difference.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR / "src") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "src"))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from dp_forgetbench.data import make_cifar10_multiclass_federation
from dp_forgetbench.federated import (
    SmallGroupNormCNN,
    get_device,
    set_seed,
    train_federated,
)
from dp_forgetbench.privacy import poisson_gaussian_epsilon
from scripts.run_phase7_factorial import solve_noise_multiplier


def compute_ece_and_nll(logits: torch.Tensor, labels: torch.Tensor, n_bins: int = 15) -> tuple[float, float]:
    """Compute Negative Log-Likelihood and Expected Calibration Error."""
    probs = torch.softmax(logits, dim=1)
    confidences, predictions = torch.max(probs, dim=1)
    accuracies = predictions.eq(labels)
    nll = nn.functional.cross_entropy(logits, labels).item()

    bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=logits.device)
    ece = 0.0
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.float().mean().item()
        if prop_in_bin > 0:
            acc_in_bin = accuracies[in_bin].float().mean().item()
            conf_in_bin = confidences[in_bin].mean().item()
            ece += abs(conf_in_bin - acc_in_bin) * prop_in_bin
    return nll, ece


def evaluate_model_utility(model: nn.Module, test_x: torch.Tensor, test_y: torch.Tensor, device: torch.device) -> dict:
    model.eval()
    with torch.no_grad():
        x = test_x.to(device)
        y = test_y.to(device)
        logits = model(x)
        acc = (logits.argmax(dim=1) == y).float().mean().item()
        nll, ece = compute_ece_and_nll(logits, y)
    return {"accuracy": acc, "nll": nll, "ece": ece}


def run_utility_sweep(
    epsilons: list[float] = [float("inf"), 32.0, 16.0, 12.0, 8.0, 4.0],
    seeds: list[int] = [20260901, 20260902, 20260903],
    rounds: int = 60,
    local_epochs: int = 3,
    sample_rate: float = 0.5,
    n_clients: int = 100,
    samples_per_client: int = 200,
    alpha: float = 0.5,
    output_dir: Path = Path("results/utility_frontier"),
) -> pd.DataFrame:
    device = get_device()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("================================================================================")
    print("                      PHASE E: UTILITY FRONTIER DISCOVERY                       ")
    print(f"Federation: N={n_clients}, q={sample_rate}, T={rounds}, E={local_epochs}, alpha={alpha}")
    print(f"Target Epsilons: {epsilons}")
    print(f"Seeds: {seeds}")
    print("================================================================================")

    records = []

    for eps in epsilons:
        eps_str = "inf" if math.isinf(eps) else f"{eps:g}"
        is_private = not math.isinf(eps)
        sigma = 0.0 if not is_private else solve_noise_multiplier(eps, sample_rate, rounds, delta=1e-5)

        print(f"\n--- Testing Epsilon = {eps_str} (sigma = {sigma:.4f}) ---")

        for seed in seeds:
            set_seed(seed)
            data_cfg = {
                "backend": "cifar10_multiclass",
                "dataset_root": "data",
                "download": False,
                "n_clients": n_clients,
                "samples_per_client": samples_per_client,
                "heterogeneity_alpha": alpha,
                "audit_samples": 500,
                "test_samples": 2000,
            }
            federation = make_cifar10_multiclass_federation(data_cfg, seed=seed)

            fed_cfg = {
                "rounds": rounds,
                "local_epochs": local_epochs,
                "local_batch_size": 32,
                "learning_rate": 0.1,
                "client_sample_rate": sample_rate,
            }
            priv_cfg = {
                "enabled": is_private,
                "population_size": n_clients,
                "delta": 1e-5,
                "clip_norm": 1.0,
                "noise_multiplier": sigma,
                "sampling": "poisson",
            }
            model_cfg = {"name": "small_groupnorm_cnn", "num_classes": 10}

            # 1. Train DP Full Model
            t0 = time.perf_counter()
            dp_model, _ = train_federated(
                clients=federation.clients,
                n_features=None,
                federated_config=fed_cfg,
                privacy_config=priv_cfg,
                seed=seed,
                model_config=model_cfg,
            )
            dp_time = time.perf_counter() - t0

            # 2. Train Retrain Reference (leave out client 2)
            retained_clients = {cid: c for cid, c in federation.clients.items() if cid != 2}
            retrain_priv_cfg = copy.deepcopy(priv_cfg)
            if is_private:
                retrain_priv_cfg["population_size"] = n_clients - 1
                retrain_priv_cfg["noise_multiplier"] = solve_noise_multiplier(eps, sample_rate, rounds, delta=1e-5)

            retrain_model, _ = train_federated(
                clients=retained_clients,
                n_features=None,
                federated_config=fed_cfg,
                privacy_config=retrain_priv_cfg,
                seed=seed + 10_000,
                model_config=model_cfg,
            )

            # Evaluate utility
            dp_util = evaluate_model_utility(dp_model, federation.test_x, federation.test_y, device)
            retrain_util = evaluate_model_utility(retrain_model, federation.test_x, federation.test_y, device)

            # Assign validity status
            if dp_util["accuracy"] >= 0.40:
                validity = "VALID"
                regime = "HEALTHY"
            elif dp_util["accuracy"] >= 0.25:
                validity = "WARNING"
                regime = "MARGINAL"
            else:
                validity = "INVALID"
                regime = "COLLAPSE"

            actual_eps = (
                float("inf")
                if not is_private
                else poisson_gaussian_epsilon(
                    sample_rate=sample_rate, noise_multiplier=sigma, rounds=rounds, delta=1e-5
                )
            )

            record = {
                "epsilon_target": eps,
                "epsilon_actual": actual_eps,
                "sigma": sigma,
                "seed": seed,
                "dp_accuracy": dp_util["accuracy"],
                "dp_nll": dp_util["nll"],
                "dp_ece": dp_util["ece"],
                "retrain_accuracy": retrain_util["accuracy"],
                "retrain_nll": retrain_util["nll"],
                "retrain_ece": retrain_util["ece"],
                "accuracy_diff": dp_util["accuracy"] - retrain_util["accuracy"],
                "validity_status": validity,
                "regime": regime,
                "runtime_s": round(dp_time, 2),
            }
            records.append(record)
            print(f"  Seed {seed}: DP Acc = {dp_util['accuracy']*100:.2f}%, Retrain Acc = {retrain_util['accuracy']*100:.2f}%, ECE = {dp_util['ece']:.3f} -> [{regime}] ({validity})")

    df = pd.DataFrame(records)
    csv_file = output_dir / "frontier_summary.csv"
    json_file = output_dir / "frontier_summary.json"
    df.to_csv(csv_file, index=False)
    with open(json_file, "w") as f:
        json.dump(records, f, indent=2)

    # Print summary table grouped by epsilon
    print("\n================================================================================")
    print("                       UTILITY FRONTIER SUMMARY TABLE                           ")
    print("================================================================================")
    summary = df.groupby("epsilon_target").agg({
        "dp_accuracy": ["mean", "std"],
        "retrain_accuracy": ["mean", "std"],
        "dp_nll": "mean",
        "dp_ece": "mean",
        "regime": lambda x: x.mode()[0],
        "validity_status": lambda x: x.mode()[0],
    }).reset_index()
    print(summary.to_string())

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Utility Frontier Discovery")
    parser.add_argument("--epsilons", type=float, nargs="+", default=[float("inf"), 32.0, 16.0, 12.0, 8.0, 4.0])
    parser.add_argument("--seeds", type=int, nargs="+", default=[20260901, 20260902, 20260903])
    args = parser.parse_args()
    run_utility_sweep(epsilons=args.epsilons, seeds=args.seeds)
