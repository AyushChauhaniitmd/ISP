"""Group a completed run index by frozen epsilon/heterogeneity config values."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import yaml

from dp_forgetbench.statistics import redundancy_screen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    groups = defaultdict(lambda: {"dp": [], "finetune": [], "cached": [], "variability": [], "runs": []})
    for run_text in json.loads(args.run_index.read_text(encoding="utf-8"))["runs"]:
        run_dir = Path(run_text)
        metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
        config = yaml.safe_load(Path(metadata["config_path"]).read_text(encoding="utf-8"))
        epsilon = config["privacy"].get("target_epsilon", "configured") if config["privacy"]["enabled"] else "inf"
        key = f"epsilon={epsilon}|heterogeneity={config['data'].get('heterogeneity')}"
        metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        group = groups[key]
        group["dp"].append(metrics["dp_only_no_action"]["alignment_to_retrain"]["test_js_divergence"])
        group["finetune"].append(metrics["retained_finetune_baseline"]["alignment_to_retrain"]["test_js_divergence"])
        if "cached_update_reconstruction_baseline" in metrics:
            group["cached"].append(metrics["cached_update_reconstruction_baseline"]["alignment_to_retrain"]["test_js_divergence"])
        group["variability"].append(metrics["target_retrain_independent"]["alignment_to_retrain"]["test_js_divergence"])
        group["runs"].append(str(run_dir))
    result = {}
    for key, value in groups.items():
        item = {
            "retained_finetune": redundancy_screen(value["dp"], value["finetune"], value["variability"]),
            "run_directories": value["runs"],
        }
        if value["cached"]:
            item["cached_update_reconstruction"] = redundancy_screen(value["dp"], value["cached"], value["variability"])
        result[key] = item
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
