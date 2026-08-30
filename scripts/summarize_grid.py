"""Render the grouped exploratory grid screen as a bounded Markdown report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-label", default="synthetic federation")
    args = parser.parse_args()
    screen = json.loads(args.screen.read_text(encoding="utf-8"))
    lines = [
        "# Exploratory redundancy-grid status",
        "",
        f"Dataset: **{args.dataset_label}**. This report is an execution artifact, not a paper claim. It does not establish equivalence, unlearning success, or state of the art.",
        "",
        "| Configuration | Seeds | DP-only JS to retrain | Fine-tune JS to retrain | MUB (DP-only minus fine-tune) | Retrain variability | Screen |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for key, result in sorted(screen.items()):
        method_result = result.get("retained_finetune", result)
        dp = method_result["dp_only_distance"]
        unlearning = method_result["unlearning_distance"]
        benefit = method_result["marginal_unlearning_benefit"]
        variability = method_result["retrain_variability"]
        lines.append(
            f"| {key} | {dp['n_seeds']} | {dp['mean']:.5f} ± {dp['sample_sd']:.5f} | "
            f"{unlearning['mean']:.5f} ± {unlearning['sample_sd']:.5f} | "
            f"{benefit['mean']:.5f} [{benefit['ci95_low']:.5f}, {benefit['ci95_high']:.5f}] | "
            f"{variability['mean']:.5f} ± {variability['sample_sd']:.5f} | {method_result['screen_interpretation']} |"
        )
    lines.extend([
        "",
        "## Interpretation boundary",
        "",
        "The grid uses deterministic synthetic data and a retained-data fine-tuning baseline. It is useful for testing the experimental pipeline, request ledger, privacy accountant, and grouping logic. Its practical tolerance is derived from observed retrain variability for screening only; it is not a pre-registered multi-endpoint equivalence bound. Public datasets, published FU baselines, calibrated attacks, and confirmation runs are mandatory before any research claim.",
    ])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
