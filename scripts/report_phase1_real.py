"""Create a readable Phase 1 real-data report from a completed run index."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml


METHODS = {
    "dp_only_no_action": "DP-only/no action",
    "retained_finetune_baseline": "Retained fine-tune",
    "cached_update_reconstruction_baseline": "Cached-update reconstruction",
    "target_retrain": "Retrain target",
    "target_retrain_independent": "Independent retrain",
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _nested(payload: dict, *keys: str) -> float:
    value = payload
    for key in keys:
        value = value[key]
    return float(value)


def _mean_sd(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(array.mean()),
        "sample_sd": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "n": int(len(array)),
    }


def _fmt(value: float) -> str:
    if abs(value) < 0.01:
        return f"{value:.4f}"
    return f"{value:.3f}"


def _fmt_mean_sd(summary: dict[str, float | int]) -> str:
    return f"{_fmt(float(summary['mean']))} +/- {_fmt(float(summary['sample_sd']))}"


def _epsilon_sort_key(text: str) -> tuple[int, float]:
    return (1, float("inf")) if text == "inf" else (0, float(text))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-index", type=Path, required=True)
    parser.add_argument("--screen", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()

    screen = read_json(args.screen)
    groups: dict[tuple[str, str], dict] = defaultdict(lambda: {"methods": defaultdict(lambda: defaultdict(list)), "runs": []})
    for run_text in read_json(args.run_index)["runs"]:
        run_dir = Path(run_text)
        metadata = read_json(run_dir / "run_metadata.json")
        config = yaml.safe_load(Path(metadata["config_path"]).read_text(encoding="utf-8"))
        epsilon = str(config["privacy"].get("target_epsilon", "configured")) if config["privacy"]["enabled"] else "inf"
        heterogeneity = str(config["data"].get("heterogeneity"))
        metrics = read_json(run_dir / "metrics.json")
        group = groups[(epsilon, heterogeneity)]
        group["runs"].append(str(run_dir))
        for method, payload in metrics.items():
            bucket = group["methods"][method]
            bucket["test_accuracy"].append(_nested(payload, "test", "accuracy"))
            bucket["test_loss"].append(_nested(payload, "test", "loss"))
            bucket["test_js_to_retrain"].append(_nested(payload, "alignment_to_retrain", "test_js_divergence"))
            bucket["forgotten_js_to_retrain"].append(_nested(payload, "alignment_to_retrain", "forgotten_js_divergence"))
            bucket["forgotten_mia_auc"].append(_nested(payload, "attack_diagnostics", "forgotten_vs_unseen_loss_mia", "auc"))
            bucket["forgotten_mia_advantage"].append(_nested(payload, "attack_diagnostics", "forgotten_vs_unseen_loss_mia", "advantage"))
            bucket["retained_mia_auc"].append(_nested(payload, "attack_diagnostics", "retained_vs_unseen_loss_mia", "auc"))
            bucket["tri_probe_accuracy"].append(_nested(payload, "attack_diagnostics", "tri_population_pre_post_probe", "probe_accuracy"))
            bucket["client_updates"].append(_nested(payload, "cost", "client_updates"))
            bucket["local_examples"].append(_nested(payload, "cost", "local_examples"))
            bucket["persistent_storage_kib"].append(_nested(payload, "cost", "persistent_storage_bytes") / 1024.0)

    summary = {}
    for (epsilon, heterogeneity), group in sorted(groups.items(), key=lambda item: (_epsilon_sort_key(item[0][0]), float(item[0][1]))):
        key = f"epsilon={epsilon}|heterogeneity={heterogeneity}"
        summary[key] = {"runs": group["runs"], "methods": {}}
        for method, values in sorted(group["methods"].items()):
            summary[key]["methods"][method] = {metric: _mean_sd(metric_values) for metric, metric_values in values.items()}
        if key in screen:
            summary[key]["redundancy_screen"] = screen[key]

    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    lines = [
        "# Phase 1 Real CIFAR-10 Binary Results",
        "",
        "Generated from `reports/phase1_v3_reference_corrected_run_index.json`.",
        "",
        "This is a real-data CPU benchmark over CIFAR-10 classes 0 vs 1 using public average-pooled features, 12 simulated clients, one full-client deletion request, five seeds per cell, and the accountant-aligned `poisson_sum_fixed_public_normalizer_v2` DP mechanism.",
        "",
        "Important: this is an exploratory benchmark result, not a state-of-the-art or certified-unlearning claim. The result is strong enough to guide the paper direction, but final claims still require stronger models, published baseline reproductions, calibrated attacks, and confirmatory seeds.",
        "",
        "## Main Result",
        "",
        "| Epsilon | Heterogeneity | Method | Test accuracy | JS to retrain | Forgotten MIA AUC | Retained MIA AUC | Client updates | Storage KiB |",
        "|---:|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key, item in summary.items():
        epsilon = key.split("|")[0].split("=", 1)[1]
        heterogeneity = key.split("|")[1].split("=", 1)[1]
        for method in METHODS:
            if method not in item["methods"]:
                continue
            metrics = item["methods"][method]
            lines.append(
                "| "
                + " | ".join(
                    [
                        epsilon,
                        heterogeneity,
                        METHODS[method],
                        _fmt_mean_sd(metrics["test_accuracy"]),
                        _fmt_mean_sd(metrics["test_js_to_retrain"]),
                        _fmt_mean_sd(metrics["forgotten_mia_auc"]),
                        _fmt_mean_sd(metrics["retained_mia_auc"]),
                        _fmt_mean_sd(metrics["client_updates"]),
                        _fmt_mean_sd(metrics["persistent_storage_kib"]),
                    ]
                )
                + " |"
            )
    lines.extend([
        "",
        "## Redundancy Screen",
        "",
        "| Epsilon | Heterogeneity | Baseline | Mean marginal JS benefit | Retrain-variability q90 | Screen interpretation |",
        "|---:|---:|---|---:|---:|---|",
    ])
    for key, item in summary.items():
        epsilon = key.split("|")[0].split("=", 1)[1]
        heterogeneity = key.split("|")[1].split("=", 1)[1]
        screen_item = item.get("redundancy_screen", {})
        for baseline_key, label in [
            ("retained_finetune", "Retained fine-tune"),
            ("cached_update_reconstruction", "Cached-update reconstruction"),
        ]:
            if baseline_key not in screen_item:
                continue
            payload = screen_item[baseline_key]
            lines.append(
                "| "
                + " | ".join(
                    [
                        epsilon,
                        heterogeneity,
                        label,
                        _fmt(float(payload["marginal_unlearning_benefit"]["mean"])),
                        _fmt(float(payload["practical_tolerance_from_retrain_variability_q90"])),
                        payload["screen_interpretation"],
                    ]
                )
                + " |"
            )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "- Across all six cells, this exploratory screen found no practically material JS-to-retrain improvement from explicit unlearning over DP-only/no-action.",
        "- At epsilon 2 and 8, retrain-to-retrain variability is large because the DP noise dominates the small CPU model. That makes exact functional alignment a weak endpoint unless paired with stronger utility and attack evaluations.",
        "- In the non-private control, all methods are already very close to the retrain target on test JS; this does not prove deletion, but it is a useful sanity check for the evaluator.",
        "- Cached-update reconstruction is not a clean privacy post-processing method because it depends on sensitive per-client update history. It is reported as a historical-server-state baseline only.",
        "",
        "## Reproducibility",
        "",
        f"- Run index: `{args.run_index}`",
        f"- Redundancy JSON: `{args.screen}`",
        f"- Summary JSON: `{args.json}`",
        "- Validation: every run in the index passed `scripts/validate_artifact.py`.",
    ])
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.markdown} and {args.json}")


if __name__ == "__main__":
    main()
