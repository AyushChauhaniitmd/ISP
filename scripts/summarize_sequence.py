"""Create a compact, bounded report from a stateful sequential baseline run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="Directory containing sequence_metrics.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    steps = json.loads((args.run / "sequence_metrics.json").read_text(encoding="utf-8"))["steps"]
    lines = [
        "# Sequential request baseline status",
        "",
        "This synthetic integration result validates stateful request handling for the retained-fine-tuning baseline. It is not a public-benchmark result or an unlearning/privacy claim.",
        "",
        "| Request prefix | Request | Forgotten examples | DP-only JS to target | Stateful fine-tune JS to target | Retrain-to-retrain JS | Fine-tune incremental client updates |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for step in steps:
        request = step["request"]
        dp = step["dp_only_no_action"]
        ft = step["stateful_retained_finetune"]
        target_2 = step["target_retrain_independent"]
        forgotten_n = dp["attack_diagnostics"]["forgotten_vs_unseen_loss_mia"]["members"]
        lines.append(
            f"| {step['request_index']} | client {request['client_id']}, fraction {request['fraction']:.2f} | {forgotten_n} | "
            f"{dp['alignment_to_retrain']['test_js_divergence']:.5f} | "
            f"{ft['alignment_to_retrain']['test_js_divergence']:.5f} | "
            f"{target_2['alignment_to_retrain']['test_js_divergence']:.5f} | "
            f"{ft['cost']['client_updates']} |"
        )
    lines.extend([
        "",
        "## Interpretation boundary",
        "",
        "The DP-only model is deliberately unchanged after each request. The stateful baseline accesses raw retained data after every request and therefore is not DP post-processing. Loss-based membership and tri-population outputs remain diagnostics, not LiRA/A-LiRA or TC-UMIA. The sample has one seed and synthetic data, so no threshold, equivalence, or retained-user leakage conclusion is permitted.",
        "",
        f"Raw artifact: `{args.run / 'sequence_metrics.json'}`.",
    ])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()

