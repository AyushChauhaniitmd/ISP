"""
aggregate_mub.py
Computes the Marginal Unlearning Benefit (MUB) with 95% Bootstrap Confidence Intervals,
retrain variability, and 4-way redundancy classification.
"""

import json
import glob
import numpy as np
from pathlib import Path
import argparse

def compute_bootstrap_ci(data, stat_func, n_bootstraps=1000, ci_level=0.95):
    """Computes bootstrap confidence intervals for a given statistic."""
    data = np.array(data)
    n = len(data)
    if n == 0:
        return 0.0, (0.0, 0.0)
    
    boot_stats = []
    for _ in range(n_bootstraps):
        sample = np.random.choice(data, size=n, replace=True)
        boot_stats.append(stat_func(sample))
        
    lower_percentile = (1 - ci_level) / 2 * 100
    upper_percentile = (1 + ci_level) / 2 * 100
    
    lower_bound = np.percentile(boot_stats, lower_percentile)
    upper_bound = np.percentile(boot_stats, upper_percentile)
    return stat_func(data), (lower_bound, upper_bound)

def compute_mub(metrics):
    """
    Computes MUB based on target retrain ensemble.
    """
    primary_retrain = metrics["target_retrain"]
    ensemble = metrics.get("target_retrain_ensemble", [])
    all_retrain = [primary_retrain] + ensemble
    
    # Extract data points
    dp_acc = metrics["dp_only_no_action"]["test"]["accuracy"]
    u_acc = metrics["retained_finetune_baseline"]["test"]["accuracy"]
    retrain_accs = [m["test"]["accuracy"] for m in all_retrain]
    
    dp_js = metrics["dp_only_no_action"]["alignment_to_retrain"]["forgotten_js_divergence"]
    u_js = metrics["retained_finetune_baseline"]["alignment_to_retrain"]["forgotten_js_divergence"]
    retrain_js = [m["alignment_to_retrain"]["forgotten_js_divergence"] for m in all_retrain]
    
    dp_mia = metrics["dp_only_no_action"]["attack_diagnostics"]["forgotten_vs_unseen_loss_mia"]["advantage"]
    u_mia = metrics["retained_finetune_baseline"]["attack_diagnostics"]["forgotten_vs_unseen_loss_mia"]["advantage"]
    retrain_mia_advs = [m["attack_diagnostics"]["forgotten_vs_unseen_loss_mia"]["advantage"] for m in all_retrain]

    # Retrain Variabilities (pairwise distances conceptually, or simply distribution quantiles)
    retrain_acc_median = np.median(retrain_accs)
    retrain_acc_q90 = np.percentile(retrain_accs, 90)
    
    retrain_js_median = np.median(retrain_js)
    retrain_js_q90 = np.percentile(retrain_js, 90)
    
    retrain_mia_median = np.median(retrain_mia_advs)
    retrain_mia_q90 = np.percentile(retrain_mia_advs, 90)
    
    # 1. Utility MUB: U_acc - DP_acc
    # (Bootstrapping relies on retrain distribution if we were doing distance, but this is direct U vs DP. 
    # For a single run, U and DP are fixed points. The uncertainty comes from comparing to retrain variability.)
    mub_acc = u_acc - dp_acc
    
    # 2. Alignment MUB: U should be closer to retrain than DP is.
    # MUB_js = DP_js - U_js (positive means U is closer/better)
    mub_js = dp_js - u_js
    
    # 3. Privacy MUB: U should have MIA closer to retrain.
    mub_mia = abs(dp_mia - retrain_mia_median) - abs(u_mia - retrain_mia_median)
    
    # Let's bootstrap the Retrain Median to get CI for MUB_mia
    def mub_mia_stat(sample_retrain_mia):
        med = np.median(sample_retrain_mia)
        return abs(dp_mia - med) - abs(u_mia - med)
        
    mub_mia_point, mub_mia_ci = compute_bootstrap_ci(retrain_mia_advs, mub_mia_stat)
    
    # 4-way Classification
    # To be redundant, DP must be within retrain variability.
    # Let's say retrain variability bound is [Q10, Q90].
    dp_within_acc_var = np.percentile(retrain_accs, 10) <= dp_acc <= np.percentile(retrain_accs, 90)
    dp_within_js_var = dp_js <= np.percentile(retrain_js, 90) # lower JS is better
    dp_within_mia_var = np.percentile(retrain_mia_advs, 10) <= dp_mia <= np.percentile(retrain_mia_advs, 90)
    
    dp_is_redundant_base = dp_within_acc_var and dp_within_js_var and dp_within_mia_var
    
    # Check if U adds benefit
    # If MUB > 0.05 for acc or MIA, it's beneficial. 
    # (Just an example threshold; would normally be preregistered equivalence margin)
    equivalence_margin = 0.02 
    
    unlearning_adds_benefit = mub_acc > equivalence_margin or mub_mia_point > equivalence_margin
    unlearning_is_harmful = mub_acc < -equivalence_margin or mub_mia_point < -equivalence_margin
    
    if dp_is_redundant_base and not unlearning_adds_benefit and not unlearning_is_harmful:
        classification = "REDUNDANT"
    elif unlearning_adds_benefit:
        classification = "UNLEARNING-BENEFICIAL"
    elif unlearning_is_harmful:
        classification = "UNLEARNING-HARMFUL"
    else:
        classification = "INCONCLUSIVE"
        
    return {
        "mub_test_accuracy": mub_acc,
        "mub_forgotten_js_div": mub_js,
        "mub_mia_advantage": mub_mia_point,
        "mub_mia_ci_95": list(mub_mia_ci),
        "retrain_variability_acc_q90": retrain_acc_q90,
        "retrain_variability_mia_q90": retrain_mia_q90,
        "classification": classification
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
        
        if "target_retrain_ensemble" not in metrics:
            print(f"Skipping {metric_file}, no target_retrain_ensemble.")
            continue
            
        mub = compute_mub(metrics)
        mubs.append(mub)
        
        print(f"\n--- {metric_file} ---")
        for k, v in mub.items():
            print(f"  {k}: {v}")

if __name__ == "__main__":
    main()
