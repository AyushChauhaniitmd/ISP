"""Run the frozen five-seed Phase 0 protocol and create an honest status summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from dp_forgetbench.config import load_config
from dp_forgetbench.run import run_experiment


PHASE0_SEEDS = [20260824, 20260825, 20260826, 20260827, 20260828]


def metric_at(payload: dict, *keys: str) -> float:
    current = payload
    for key in keys:
        current = current[key]
    return float(current)


def summarize(run_directories: list[Path], output_path: Path) -> None:
    rows = []
    for run_dir in run_directories:
        metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        ledger = json.loads((run_dir / "privacy_ledger.json").read_text(encoding="utf-8"))
        metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
        rows.append({
            "seed": metadata["seed"],
            "epsilon": ledger["epsilon"],
            "dp_js": metric_at(metrics, "dp_only_no_action", "alignment_to_retrain", "test_js_divergence"),
            "finetune_js": metric_at(metrics, "retained_finetune_baseline", "alignment_to_retrain", "test_js_divergence"),
            "dp_accuracy": metric_at(metrics, "dp_only_no_action", "test", "accuracy"),
            "finetune_accuracy": metric_at(metrics, "retained_finetune_baseline", "test", "accuracy"),
            "target_accuracy": metric_at(metrics, "target_retrain", "test", "accuracy"),
            "retrain_variability_js": metric_at(metrics, "target_retrain_independent", "alignment_to_retrain", "test_js_divergence"),
            "dp_forget_mia_auc": metric_at(metrics, "dp_only_no_action", "attack_diagnostics", "forgotten_vs_unseen_loss_mia", "auc"),
            "finetune_forget_mia_auc": metric_at(metrics, "retained_finetune_baseline", "attack_diagnostics", "forgotten_vs_unseen_loss_mia", "auc"),
            "dp_retain_mia_auc": metric_at(metrics, "dp_only_no_action", "attack_diagnostics", "retained_vs_unseen_loss_mia", "auc"),
            "finetune_retain_mia_auc": metric_at(metrics, "retained_finetune_baseline", "attack_diagnostics", "retained_vs_unseen_loss_mia", "auc"),
        })
    means = {key: float(np.mean([row[key] for row in rows])) for key in rows[0] if key != "seed"}
    stds = {key: float(np.std([row[key] for row in rows], ddof=1)) for key in means}
    lines = [
        "# Phase 0 execution status",
        "",
        "This report confirms that the synthetic plumbing ran over the frozen five seeds. It is **not** a privacy-attack result, an equivalence test, or a state-of-the-art claim.",
        "",
        "## Completed runs",
        "",
        "| Seed | ε | DP-only JS to retrain | Fine-tune JS to retrain | Retrain-to-retrain JS | DP-only test accuracy | Fine-tune test accuracy | Retrain test accuracy |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['seed']} | {row['epsilon']:.3f} | {row['dp_js']:.5f} | {row['finetune_js']:.5f} | {row['retrain_variability_js']:.5f} | "
            f"{row['dp_accuracy']:.3f} | {row['finetune_accuracy']:.3f} | {row['target_accuracy']:.3f} |"
        )
    lines.extend([
        "",
        "## Mean ± sample standard deviation",
        "",
        "| Metric | Mean | SD |",
        "|---|---:|---:|",
    ])
    for key in ["epsilon", "dp_js", "finetune_js", "retrain_variability_js", "dp_accuracy", "finetune_accuracy", "target_accuracy"]:
        lines.append(f"| {key} | {means[key]:.5f} | {stds[key]:.5f} |")
    lines.extend([
        "",
        "## Baseline attack diagnostics",
        "",
        "| Metric | Mean | SD |",
        "|---|---:|---:|",
    ])
    for key in ["dp_forget_mia_auc", "finetune_forget_mia_auc", "dp_retain_mia_auc", "finetune_retain_mia_auc"]:
        lines.append(f"| {key} | {means[key]:.5f} | {stds[key]:.5f} |")
    lines.extend([
        "",
        "## Interpretation boundary",
        "",
        "These values establish that deterministic manifests, client-DP accounting, no-action DP, retained fine-tuning, independent target retrains, and held-out baseline attack diagnostics execute together. The fine-tuning baseline accesses raw retained data and therefore is not DP post-processing. Attack diagnostics are label-matched loss-threshold probes, not LiRA/A-LiRA or TC-UMIA. Retrain-to-retrain JS is a variability diagnostic, not yet a pre-registered equivalence band. No public-dataset result or secure-aggregation service has been run yet. Do not infer a redundancy boundary from this table.",
        "",
        "## Next gate",
        "",
        "Freeze member/non-member attack splits, add calibrated membership-inference sanity checks, and then move the same protocol to CIFAR-10. The independent target retrain gate is now implemented.",
    ])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/phase0_toy.yaml"))
    parser.add_argument("--report", type=Path, default=Path("reports/PHASE_0_STATUS.md"))
    parser.add_argument("--run-index", type=Path, default=Path("reports/phase0_run_index.json"))
    parser.add_argument("--summarize-only", action="store_true", help="Regenerate the report from an existing run index without executing new runs.")
    args = parser.parse_args()
    if args.summarize_only:
        run_directories = [Path(path) for path in json.loads(args.run_index.read_text(encoding="utf-8"))["runs"]]
    else:
        run_directories = []
        for seed in PHASE0_SEEDS:
            config = load_config(args.config)
            config["seed"] = seed
            run_directories.append(run_experiment(config, args.config))
            print(f"Completed seed {seed}: {run_directories[-1]}")
    summarize(run_directories, args.report)
    if not args.summarize_only:
        args.run_index.parent.mkdir(parents=True, exist_ok=True)
        args.run_index.write_text(json.dumps({"runs": [str(path) for path in run_directories]}, indent=2), encoding="utf-8")
    print(f"Wrote status report: {args.report}")


if __name__ == "__main__":
    main()
