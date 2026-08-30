"""Aggregate a run index into a transparent exploratory redundancy screen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dp_forgetbench.statistics import redundancy_screen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run_dirs = [Path(path) for path in json.loads(args.run_index.read_text(encoding="utf-8"))["runs"]]
    dp, unlearning, variability = [], [], []
    for run_dir in run_dirs:
        metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        dp.append(metrics["dp_only_no_action"]["alignment_to_retrain"]["test_js_divergence"])
        unlearning.append(metrics["retained_finetune_baseline"]["alignment_to_retrain"]["test_js_divergence"])
        variability.append(metrics["target_retrain_independent"]["alignment_to_retrain"]["test_js_divergence"])
    result = redundancy_screen(dp, unlearning, variability)
    result["run_directories"] = [str(path) for path in run_dirs]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

