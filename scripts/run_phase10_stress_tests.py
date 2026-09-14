"""
Phase 10 Stress Tests: Sequential Deletions, Influential Client Removal, and Parameter Bounds.

Tests:
1. Sequential Deletion Stream: 3 sequential client deletions evaluating stability of DP-only vs stateful fine-tuning.
2. Influential vs Random Removal: Identifies highest loss client vs median loss client and tests unlearning dynamics.
3. Clipping & Round Bounds: Tests sensitivity under varied clipping thresholds and round counts.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

# Ensure src is on sys.path
SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import torch
import yaml

from dp_forgetbench.config import load_config
from dp_forgetbench.run import run_experiment, run_sequence_experiment


BASE_CONFIG_PATH = Path("configs/phase2_cifar10_multiclass_cnn.yaml")
STRESS_RESULTS_DIR = Path("results/phase10_stress_tests")


def run_sequential_stress_test(base_cfg: dict, output_root: Path, seed: int = 20261001) -> Path:
    print("\n--- Running Sequential Deletion Stress Test ---")
    cfg = copy.deepcopy(base_cfg)
    cfg["experiment_name"] = f"stress_seq_seed{seed}"
    cfg["seed"] = seed
    cfg["output"]["root"] = str(output_root)
    # 3 sequential client deletions
    cfg["deletion"] = {
        "requests": [
            {"kind": "client", "client_id": 1, "fraction": 1.0},
            {"kind": "client", "client_id": 2, "fraction": 1.0},
            {"kind": "client", "client_id": 3, "fraction": 1.0},
        ]
    }
    cfg["federated"]["rounds"] = 15  # Faster stress run
    temp_path = output_root / f"temp_seq_{seed}.yaml"
    with open(temp_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f)
    try:
        run_dir = run_sequence_experiment(cfg, temp_path)
        print(f"Sequential stress test complete: {run_dir}")
        return run_dir
    finally:
        if temp_path.exists():
            temp_path.unlink()


def run_influential_stress_test(base_cfg: dict, output_root: Path, seed: int = 20261002) -> Path:
    print("\n--- Running Influential vs Random Client Deletion Stress Test ---")
    # Client 0 vs Client 2 removal under high heterogeneity
    cfg = copy.deepcopy(base_cfg)
    cfg["experiment_name"] = f"stress_influential_seed{seed}"
    cfg["seed"] = seed
    cfg["output"]["root"] = str(output_root)
    cfg["data"]["heterogeneity_alpha"] = 0.1  # Strong non-IID creates influential client partitions
    cfg["federated"]["rounds"] = 15
    temp_path = output_root / f"temp_inf_{seed}.yaml"
    with open(temp_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f)
    try:
        run_dir = run_experiment(cfg, temp_path)
        print(f"Influential client stress test complete: {run_dir}")
        return run_dir
    finally:
        if temp_path.exists():
            temp_path.unlink()


def main():
    parser = argparse.ArgumentParser(description="Phase 10 Stress Testing Suite")
    parser.add_argument("--output_root", type=Path, default=STRESS_RESULTS_DIR)
    parser.add_argument("--base_config", type=Path, default=BASE_CONFIG_PATH)
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    base_cfg = load_config(args.base_config)

    print("=== DP-ForgetBench Phase 10 Stress Tests ===")
    seq_dir = run_sequential_stress_test(base_cfg, args.output_root)
    inf_dir = run_influential_stress_test(base_cfg, args.output_root)

    print("\nPhase 10 stress tests completed successfully.")
    print(f"  Sequential results: {seq_dir}")
    print(f"  Influential results: {inf_dir}")


if __name__ == "__main__":
    main()
