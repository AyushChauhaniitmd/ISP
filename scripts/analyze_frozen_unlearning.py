"""Aggregate attack, MUB, utility, and cost results from frozen runs."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
import math
import re


METHODS = (
    "dp_only_no_action",
    "retained_finetune_baseline",
    "cached_update_reconstruction_baseline",
    "federated_eraser_baseline",
    "target_retrain",
)


def _mean(values: list[float]) -> float | None:
    return round(statistics.mean(values), 6) if values else None


def _ci(values: list[float]) -> list[float] | None:
    if not values:
        return None
    if len(values) == 1:
        return [round(values[0], 6), round(values[0], 6)]
    mean = statistics.mean(values)
    standard_error = statistics.stdev(values) / math.sqrt(len(values))
    return [round(mean - 1.96 * standard_error, 6), round(mean + 1.96 * standard_error, 6)]


def _metric(metrics: dict, method: str, path: tuple[str, ...]) -> float | None:
    value = metrics.get(method)
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return float(value) if isinstance(value, (int, float)) else None


def analyze(root: Path) -> dict:
    candidates: dict[int, tuple[str, Path]] = {}
    pattern = re.compile(r"_seed(?P<seed>\d+)$")
    for run_dir in sorted(root.iterdir()):
        metrics_path = run_dir / "metrics.json"
        metadata_path = run_dir / "run_metadata.json"
        if not run_dir.is_dir() or not metrics_path.exists():
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
        seed = metadata.get("seed")
        match = pattern.search(run_dir.name)
        if seed is None and match:
            seed = int(match.group("seed"))
        if seed is None:
            raise ValueError(f"Could not determine seed for {run_dir}")
        timestamp_match = re.search(r"_(\d{8}T\d{6}Z)_seed", run_dir.name)
        timestamp = timestamp_match.group(1) if timestamp_match else str(run_dir.stat().st_mtime_ns)
        previous = candidates.get(int(seed))
        if previous is None or timestamp > previous[0]:
            candidates[int(seed)] = (timestamp, run_dir)

    rows: list[dict] = []
    selected_runs: list[str] = []
    for seed, (_, run_dir) in sorted(candidates.items()):
        rows.append(json.loads((run_dir / "metrics.json").read_text(encoding="utf-8")))
        selected_runs.append(run_dir.name)
    if not rows:
        raise FileNotFoundError(f"No metrics.json files found under {root}")

    summary: dict[str, dict] = {}
    paths = {
        "test_accuracy": ("utility", "test_accuracy"),
        "forgotten_accuracy": ("utility", "forgotten_accuracy"),
        "retained_accuracy": ("utility", "retained_accuracy"),
        "forgotten_mia_auc": ("forgotten_privacy", "auc"),
        "forgotten_mia_advantage": ("forgotten_privacy", "advantage"),
        "retained_mia_auc": ("retained_privacy", "auc"),
        "retained_mia_advantage": ("retained_privacy", "advantage"),
        "forgotten_js": ("alignment_to_retrain", "forgotten_js_divergence"),
        "test_js": ("alignment_to_retrain", "test_js_divergence"),
        "mub_accuracy": ("mub", "mub_test_accuracy"),
        "mub_js": ("mub", "mub_forgotten_js_div"),
        "mub_mia": ("mub", "mub_mia_advantage"),
        "runtime_seconds": ("cost", "runtime_seconds"),
        "persistent_storage_bytes": ("cost", "persistent_storage_bytes"),
        "communicated_bytes": ("cost", "communicated_bytes"),
    }
    for method in METHODS:
        summary[method] = {}
        for label, path in paths.items():
            values = [value for row in rows if (value := _metric(row, method, path)) is not None]
            summary[method][label] = {"mean": _mean(values), "ci_95_across_seeds": _ci(values), "per_seed": values}
        classifications = [row.get(method, {}).get("mub", {}).get("classification") for row in rows]
        summary[method]["classifications"] = [value for value in classifications if value]
    return {
        "run_count": len(rows),
        "selected_seeds": sorted(candidates),
        "selected_runs": selected_runs,
        "deduplication": "latest timestamped run selected per seed",
        "attack_protocol": {
            "loss_mia": "label-matched member versus unseen loss attack; threshold selected on calibration split and evaluated on disjoint held-out split",
            "advantage": "held-out TPR minus held-out FPR at the calibration-selected threshold",
            "auc": "ROC AUC on held-out scores; threshold-independent",
            "tri_population": "forget/retain/unseen logistic probe using pre/post loss-confidence features and stratified 50/50 train/evaluation split",
        },
        "methods": summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    result = analyze(args.root)
    if args.output:
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()