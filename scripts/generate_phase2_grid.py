"""Generate immutable Phase 2 experiment YAML files with calibrated epsilon.

Mirrors generate_phase1_grid.py, but sweeps `data.heterogeneity_alpha` (the
Dirichlet concentration used by the cifar10_multiclass backend) instead of the
binary backend's `data.heterogeneity` flip-probability knob. Kept as a separate
script rather than branching inside generate_phase1_grid.py so the Phase 1
generator that produced the already-validated Phase 1 results stays untouched.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from calibrate_noise import calibrate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=Path("configs/phase2_grid_spec.yaml"))
    parser.add_argument("--output", type=Path, default=Path("configs/generated/phase2"))
    args = parser.parse_args()
    spec = yaml.safe_load(args.spec.read_text(encoding="utf-8"))
    base_path = Path(spec["base_config"])
    base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)
    generated = []
    for target_epsilon in spec["target_epsilons"]:
        finite = not (isinstance(target_epsilon, str) and target_epsilon.lower() == "inf")
        if finite:
            noise, achieved = calibrate(
                float(target_epsilon),
                float(base["federated"]["client_sample_rate"]),
                int(base["federated"]["rounds"]),
                float(base["privacy"]["delta"]),
            )
            privacy = {**base["privacy"], "enabled": True, "noise_multiplier": float(noise), "target_epsilon": float(target_epsilon), "calibrated_epsilon": float(achieved)}
            epsilon_name = f"eps{target_epsilon:g}"
        else:
            privacy = {**base["privacy"], "enabled": False, "target_epsilon": "inf", "calibrated_epsilon": None}
            epsilon_name = "epsinf"
        for alpha in spec["heterogeneity"]:
            for client_id in spec["deletion_client_ids"]:
                for seed in spec["seeds"]:
                    config = yaml.safe_load(yaml.safe_dump(base))
                    config["seed"] = int(seed)
                    config["data"]["heterogeneity_alpha"] = float(alpha)
                    config["deletion"]["client_id"] = int(client_id)
                    config["privacy"] = privacy
                    config["experiment_name"] = f"{base['experiment_name']}_{epsilon_name}_a{alpha:g}_delete{client_id}_seed{seed}"
                    name = f"{epsilon_name}_a{alpha:g}_delete{client_id}_seed{seed}.yaml"
                    path = args.output / name
                    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
                    generated.append(path)
    index = args.output / "INDEX.txt"
    index.write_text("\n".join(str(path) for path in generated) + "\n", encoding="utf-8")
    print(f"Generated {len(generated)} frozen configs in {args.output}")


if __name__ == "__main__":
    main()
