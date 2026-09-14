"""
run_unlearning_control.py
Phase D: Positive Unlearning Control Suite

Establishes a deliberately deletion-sensitive regime where unlearning SHOULD matter:
- epsilon = infinity (non-private control, high signal)
- strong heterogeneity (alpha = 0.1, label-skewed Dirichlet)
- full deletion of an influential client (identified by class uniqueness/loss impact)
- comprehensive evaluation:
  * Full Model (DP-only / No Action)
  * Exact Retrain Ensemble (Target reference)
  * Retained Fine-Tuning
  * Cached Reconstruction
  * Faithful FedEraser

Demonstrates that DP-ForgetBench can detect genuine unlearning and measure
clear separation between DP-only and unlearned models.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn as nn

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR / "src") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "src"))

from dp_forgetbench.data import make_cifar10_multiclass_federation
from dp_forgetbench.federated import (
    SmallGroupNormCNN,
    TrainCost,
    get_device,
    set_seed,
    train_federated,
    reconstruct_from_cached_updates,
    federated_eraser,
    finetune_retained,
    FederatedHistory,
)
from dp_forgetbench.evaluation import evaluate_against_target, compute_method_mub


def identify_influential_client(federation) -> int:
    """Find the client with the most distinct class distribution / highest label concentration."""
    client_entropy = {}
    for cid, c in federation.clients.items():
        counts = torch.bincount(c.y, minlength=10).float()
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        entropy = -(probs * torch.log(probs)).sum().item()
        client_entropy[cid] = entropy

    # Client with lowest label entropy has the highest class concentration (most specialized)
    influential_id = min(client_entropy, key=client_entropy.get)
    print(f"Identified influential client {influential_id} (label entropy = {client_entropy[influential_id]:.3f})")
    return influential_id


def run_positive_unlearning_control(
    n_clients: int = 50,
    samples_per_client: int = 200,
    alpha: float = 0.1,
    rounds: int = 40,
    local_epochs: int = 3,
    lr: float = 0.1,
    seed: int = 20260901,
    output_dir: Path = Path("results/positive_unlearning_control"),
) -> dict:
    set_seed(seed)
    device = get_device()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n================================================================================")
    print(f"            PHASE D: POSITIVE UNLEARNING CONTROL (eps=inf, alpha={alpha})         ")
    print(f"================================================================================")

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

    # Pick an influential client to delete
    del_client_id = identify_influential_client(federation)

    # Partition into retained and forgotten
    retained_clients = {cid: c for cid, c in federation.clients.items() if cid != del_client_id}
    forgotten_client = federation.clients[del_client_id]

    retained_x = torch.cat([c.x for c in retained_clients.values()])
    retained_y = torch.cat([c.y for c in retained_clients.values()])

    fed_cfg = {
        "rounds": rounds,
        "local_epochs": local_epochs,
        "local_batch_size": 32,
        "learning_rate": lr,
        "client_sample_rate": 0.5,
    }
    priv_cfg = {"enabled": False}
    model_cfg = {"name": "small_groupnorm_cnn", "num_classes": 10}

    # 1. Train Full Pre-Deletion Model (with historical update caching for FedEraser)
    print(f"\n1. Training Full Pre-Deletion Model ({rounds} rounds)...")
    history = FederatedHistory()
    full_model, full_cost = train_federated(
        clients=federation.clients,
        n_features=None,
        federated_config=fed_cfg,
        privacy_config=priv_cfg,
        seed=seed,
        history=history,
        model_config=model_cfg,
    )

    # 2. Train Target Retrain Reference (Ensemble size = 3)
    print("2. Training Retrain Reference Ensemble on Retained Clients...")
    target_models = []
    target_costs = []
    for i in range(3):
        m, c = train_federated(
            clients=retained_clients,
            n_features=None,
            federated_config=fed_cfg,
            privacy_config=priv_cfg,
            seed=seed + 10_000 + i,
            model_config=model_cfg,
        )
        target_models.append(m)
        target_costs.append(c)
    primary_target = target_models[0]

    # 3. Retained Fine-Tuning Baseline
    print("3. Executing Retained Fine-Tuning Baseline (4 rounds)...")
    ft_model, ft_cost = finetune_retained(
        full_model=full_model,
        clients=retained_clients,
        n_features=None,
        federated_config=fed_cfg,
        unlearning_rounds=4,
        seed=seed + 20_000,
        model_config=model_cfg,
    )

    # 4. Cached Update Reconstruction Baseline
    print("4. Executing Cached Update Reconstruction Baseline...")
    cached_model, cached_cost = reconstruct_from_cached_updates(
        history=history,
        forgotten_client_ids={del_client_id},
        n_features=None,
        federated_config=fed_cfg,
        privacy_config=priv_cfg,
        model_config=model_cfg,
    )

    # 5. Faithful FedEraser Baseline
    print("5. Executing Faithful FedEraser Baseline (with calibration training)...")
    eraser_model, eraser_cost = federated_eraser(
        history=history,
        clients=retained_clients,
        forgotten_client_ids={del_client_id},
        n_features=None,
        federated_config=fed_cfg,
        privacy_config=priv_cfg,
        calibration_ratio=0.5,
        model_config=model_cfg,
    )

    # Comprehensive Evaluation Suite
    print("\nEvaluating all baselines against retrain target and running Tri-Population MIAs...")
    method_models = {
        "dp_only_no_action": (full_model, full_cost),
        "target_retrain": (primary_target, target_costs[0]),
        "retained_finetune": (ft_model, ft_cost),
        "cached_reconstruction": (cached_model, cached_cost),
        "federated_eraser": (eraser_model, eraser_cost),
    }

    metrics = {}
    for name, (model, cost) in method_models.items():
        metrics[name] = evaluate_against_target(
            candidate=model,
            target=primary_target,
            test_x=federation.test_x,
            test_y=federation.test_y,
            forgotten_x=forgotten_client.x,
            forgotten_y=forgotten_client.y,
            retained_x=retained_x,
            retained_y=retained_y,
            unseen_x=federation.audit_x,
            unseen_y=federation.audit_y,
            pre_deletion_model=full_model,
            attack_seed=seed + 30_000,
            cost=cost,
        )

    # Evaluate ensemble targets
    all_retrain_evals = [metrics["target_retrain"]]
    for m, c in zip(target_models[1:], target_costs[1:]):
        eval_dict = evaluate_against_target(
            candidate=m,
            target=primary_target,
            test_x=federation.test_x,
            test_y=federation.test_y,
            forgotten_x=forgotten_client.x,
            forgotten_y=forgotten_client.y,
            retained_x=retained_x,
            retained_y=retained_y,
            unseen_x=federation.audit_x,
            unseen_y=federation.audit_y,
            pre_deletion_model=full_model,
            attack_seed=seed + 30_000,
            cost=c,
        )
        all_retrain_evals.append(eval_dict)

    # Compute MUB relative to retrain ensemble
    mub_results = {}
    for name in ["retained_finetune", "cached_reconstruction", "federated_eraser"]:
        mub_results[name] = compute_method_mub(
            candidate_metrics=metrics[name],
            dp_metrics=metrics["dp_only_no_action"],
            all_retrain_metrics=all_retrain_evals,
        )

    print("\n=== POSITIVE UNLEARNING CONTROL RESULTS ===")
    print(f"{'Method':<25} | {'Test Acc':<10} | {'Forgot Acc':<10} | {'MIA AUC':<10} | {'MIA Adv':<10} | {'Classification':<22}")
    print("-" * 95)
    for name in method_models:
        m = metrics[name]
        acc = m["test"]["accuracy"] * 100
        f_acc = m["forgotten_client"]["accuracy"] * 100
        mia = m["attack_diagnostics"]["forgotten_vs_unseen_loss_mia"]
        auc = mia["auc"]
        adv = mia["advantage"]
        cls_str = mub_results.get(name, {}).get("classification", "TARGET / BASE")
        print(f"{name:<25} | {acc:>6.2f}%    | {f_acc:>6.2f}%    | {auc:>8.3f}   | {adv:>8.3f}   | {cls_str:<22}")

    out_file = output_dir / "positive_unlearning_results.json"
    with open(out_file, "w") as f:
        json.dump({"metrics": metrics, "mub": mub_results}, f, indent=2)
    print(f"\nSaved results to {out_file}")
    return {"metrics": metrics, "mub": mub_results}


if __name__ == "__main__":
    run_positive_unlearning_control()
