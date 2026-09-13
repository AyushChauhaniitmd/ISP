"""
run_dp_sweep.py
Runs a hyperparameter sweep over DP-FL settings to find a configuration that
reaches the 40% accuracy threshold.
"""
import yaml
import subprocess
import json
import glob
from pathlib import Path
import os
import shutil

BASE_CONFIG = "configs/phase2_dp_sweep.yaml"
TEMP_CONFIG = "configs/temp_sweep.yaml"
RESULTS_DIR = Path("results")

def get_latest_result_dir():
    dirs = [d for d in RESULTS_DIR.iterdir() if d.is_dir() and d.name.startswith("phase2_cifar10_multiclass_cnn")]
    if not dirs:
        return None
    dirs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
    return dirs[0]

def main():
    # Sweep parameters
    learning_rates = [0.1, 0.5, 1.0]
    noise_multipliers = [0.1, 0.5]
    client_sample_rates = [1.0]

    with open(BASE_CONFIG, "r") as f:
        base_config = yaml.safe_load(f)

    results = []

    for lr in learning_rates:
        for nm in noise_multipliers:
            for csr in client_sample_rates:
                print(f"\n--- Testing LR={lr}, Noise={nm}, SampleRate={csr} ---")
                
                # Update config
                cfg = base_config.copy()
                cfg["federated"]["learning_rate"] = lr
                cfg["federated"]["client_sample_rate"] = csr
                cfg["privacy"]["noise_multiplier"] = nm
                
                with open(TEMP_CONFIG, "w") as f:
                    yaml.dump(cfg, f)
                
                # Ensure PYTHONPATH and CUBLAS are set
                env = os.environ.copy()
                env["PYTHONPATH"] = "src"
                env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
                
                # Run the experiment
                print("Running DP-FL...")
                subprocess.run(["python", "-m", "dp_forgetbench.run", "--config", TEMP_CONFIG], env=env, check=True)
                
                # Get the result
                latest_dir = get_latest_result_dir()
                
                with open(latest_dir / "metrics.json", "r") as f:
                    metrics = json.load(f)
                with open(latest_dir / "run_metadata.json", "r") as f:
                    metadata = json.load(f)
                
                dp_acc = metrics["dp_only_no_action"]["test"]["accuracy"]
                retrain_acc = metrics["target_retrain"]["test"]["accuracy"]
                validity = metadata["validity_status"]
                
                print(f"Result: DP Acc = {dp_acc:.4f}, Retrain Acc = {retrain_acc:.4f}, Status = {validity}")
                
                results.append({
                    "lr": lr,
                    "nm": nm,
                    "csr": csr,
                    "dp_acc": dp_acc,
                    "retrain_acc": retrain_acc,
                    "validity": validity,
                    "dir": str(latest_dir)
                })
                
                # If we hit a valid configuration, we can stop the sweep, but let's 
                # continue to see if we can find one with higher noise
                if validity == "VALID" and dp_acc > 0.40:
                    print("Found a VALID configuration!")

    print("\n\n--- SWEEP SUMMARY ---")
    for r in results:
        print(f"LR={r['lr']}, Noise={r['nm']}, CSR={r['csr']} => DP={r['dp_acc']:.4f}, Retrain={r['retrain_acc']:.4f}, {r['validity']}")

if __name__ == "__main__":
    main()
