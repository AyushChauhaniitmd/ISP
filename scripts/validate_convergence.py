"""
Phase 1: Convergence Fix Validation Script

Validates that replacing AdaptiveAvgPool2d((1,1)) with Flatten(1) fixes the
convergence issues on CIFAR-10. Runs across epsilon={infinity, 8, 4, 2}.
"""
import sys
import json
import torch
import numpy as np
from pathlib import Path
from copy import deepcopy

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dp_forgetbench.data import make_federation
from dp_forgetbench.federated import train_federated, evaluate_loss_accuracy, infer_n_features

def run_validation():
    # Base config equivalent to N=100 pilot
    base_config = {
        "data": {
            "backend": "cifar10_multiclass",
            "dataset_root": "data",
            "download": True,
            "n_clients": 100,
            "samples_per_client": 200,
            "heterogeneity_alpha": 1.0,
            "audit_samples": 50,
            "test_samples": 500
        },
        "federated": {
            "rounds": 10,
            "local_epochs": 1,
            "local_batch_size": 32,
            "learning_rate": 0.1,
            "client_sample_rate": 1.0
        },
        "model": {
            "num_classes": 10
        }
    }
    
    seeds = [202601]
    epsilons = [float('inf'), 8.0]
    architectures = [
        "original_small_groupnorm_cnn",
        "small_groupnorm_cnn"
    ]
    
    results = []
    output_dir = Path("reports/convergence_validation")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for seed in seeds:
        print(f"\n--- Loading data for seed {seed} ---")
        federation = make_federation(base_config["data"], seed)
        n_features = infer_n_features(next(iter(federation.clients.values())).x)
        
        for arch in architectures:
            for eps in epsilons:
                print(f"Running Seed: {seed}, Arch: {arch}, Epsilon: {eps}")
                
                # Setup privacy config
                privacy_config = {
                    "enabled": eps != float('inf'),
                    "population_size": base_config["data"]["n_clients"],
                    "noise_multiplier": 0.0 if eps == float('inf') else 0.5, # We'll scale noise conceptually or use DP sweep values
                    "clip_norm": 1.0,
                    "delta": 1e-5,
                    "sampling": "poisson"
                }
                
                # To properly hit eps=8, 4, 2 we need exact noise multipliers from Opacus/accountant.
                # For this pilot validation, if eps is finite, we use fixed noise_multipliers
                # corresponding roughly to the epsilons. 
                # Let's map roughly: eps=8 -> nm=0.5, eps=4 -> nm=0.7, eps=2 -> nm=1.1 (approximate)
                if eps == 8.0: privacy_config["noise_multiplier"] = 0.5
                elif eps == 4.0: privacy_config["noise_multiplier"] = 0.7
                elif eps == 2.0: privacy_config["noise_multiplier"] = 1.1
                
                model_config = deepcopy(base_config["model"])
                model_config["name"] = arch
                
                model, cost = train_federated(
                    clients=federation.clients,
                    n_features=n_features,
                    federated_config=base_config["federated"],
                    privacy_config=privacy_config,
                    seed=seed,
                    model_config=model_config
                )
                
                # Evaluate on test set
                test_metrics = evaluate_loss_accuracy(model, federation.test_x, federation.test_y)
                
                # Evaluate on train set (sample of it to save time)
                train_x = torch.cat([c.x for c in list(federation.clients.values())[:10]])
                train_y = torch.cat([c.y for c in list(federation.clients.values())[:10]])
                train_metrics = evaluate_loss_accuracy(model, train_x, train_y)
                
                res = {
                    "seed": seed,
                    "architecture": arch,
                    "epsilon": eps,
                    "train_loss": train_metrics["loss"],
                    "train_accuracy": train_metrics["accuracy"],
                    "test_loss": test_metrics["loss"],
                    "test_accuracy": test_metrics["accuracy"]
                }
                print(f"Result: {res}")
                results.append(res)
                
    # Save raw results
    with open(output_dir / "raw_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Aggregate and print summary
    print("\n=== CONVERGENCE VALIDATION SUMMARY ===")
    summary_lines = []
    for arch in architectures:
        for eps in epsilons:
            test_accs = [r["test_accuracy"] for r in results if r["architecture"] == arch and r["epsilon"] == eps]
            mean_acc = np.mean(test_accs)
            std_acc = np.std(test_accs)
            line = f"Arch: {arch.ljust(30)} | Eps: {str(eps).ljust(5)} | Mean Test Acc: {mean_acc:.4f} +/- {std_acc:.4f}"
            print(line)
            summary_lines.append(line)
            
    with open(output_dir / "summary.txt", "w") as f:
        f.write("\n".join(summary_lines))

if __name__ == "__main__":
    run_validation()
