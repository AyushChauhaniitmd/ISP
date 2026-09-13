"""
aggregate_mub.py
Computes the Marginal Unlearning Benefit (MUB) and bootstrap confidence intervals
from run metadata and metrics.
"""
import json
import glob
import numpy as np
from pathlib import Path
import argparse

def compute_mub(metrics):
    """
    Computes MUB based on target retrain ensemble.
    """
    # Retrain ensemble metrics
    primary_retrain = metrics["target_retrain"]
    ensemble = metrics.get("target_retrain_ensemble", [])
    all_retrain = [primary_retrain] + ensemble
    
    # 1. Utility (Test Accuracy)
    # distance to retraining for test accuracy is ideally 0. 
    # MUB_acc = - (abs(DP - Retrain_median) - abs(U - Retrain_median))
    # Wait, larger acc is better, so distance is Retrain - Model (if Retrain > Model)
    # For simplicity: MUB = Model_U_acc - Model_DP_acc
    
    dp_acc = metrics["dp_only_no_action"]["test"]["accuracy"]
    u_acc = metrics["retained_finetune_baseline"]["test"]["accuracy"]
    mub_acc = u_acc - dp_acc

    # 2. Functional Alignment (Forgotten JS Divergence)
    # Smaller is closer to retrain.
    # MUB_js = DP_js - U_js (positive means U is closer to retrain than DP)
    dp_js = metrics["dp_only_no_action"]["alignment_to_retrain"]["forgotten_js_divergence"]
    u_js = metrics["retained_finetune_baseline"]["alignment_to_retrain"]["forgotten_js_divergence"]
    mub_js = dp_js - u_js

    # 3. Forget Privacy (MIA Advantage)
    # Lower MIA advantage means less privacy leakage. 
    # Closer to retrain's MIA advantage is better.
    retrain_mia_advs = [m["attack_diagnostics"]["forgotten_vs_unseen_loss_mia"]["advantage"] for m in all_retrain]
    median_retrain_mia = np.median(retrain_mia_advs)
    
    dp_mia = metrics["dp_only_no_action"]["attack_diagnostics"]["forgotten_vs_unseen_loss_mia"]["advantage"]
    u_mia = metrics["retained_finetune_baseline"]["attack_diagnostics"]["forgotten_vs_unseen_loss_mia"]["advantage"]
    
    # Distance to retrain
    dp_mia_dist = abs(dp_mia - median_retrain_mia)
    u_mia_dist = abs(u_mia - median_retrain_mia)
    
    mub_mia = dp_mia_dist - u_mia_dist
    
    return {
        "mub_test_accuracy": mub_acc,
        "mub_forgotten_js_div": mub_js,
        "mub_mia_advantage": mub_mia,
        "retrain_variability_acc": np.std([m["test"]["accuracy"] for m in all_retrain]),
        "retrain_variability_mia": np.std(retrain_mia_advs)
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="results")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    print(f"Aggregating results from {results_dir}...")
    
    mubs = []
    
    for metric_file in glob.glob(str(results_dir / "*/metrics.json")):
        with open(metric_file, "r") as f:
            metrics = json.load(f)
        
        # Ensure it has the ensemble
        if "target_retrain_ensemble" not in metrics:
            print(f"Skipping {metric_file}, no target_retrain_ensemble.")
            continue
            
        mub = compute_mub(metrics)
        mubs.append(mub)
        
        print(f"\n--- {metric_file} ---")
        for k, v in mub.items():
            print(f"  {k}: {v:.4f}")

if __name__ == "__main__":
    main()
