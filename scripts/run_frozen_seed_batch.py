"""Run additional seeds for the frozen multiclass unlearning study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dp_forgetbench.config import load_config
from dp_forgetbench.run import _config_hash
from scripts.analyze_frozen_unlearning import analyze

DEFAULT_CONFIG = ROOT / "configs" / "multiclass_cnn_frozen_unlearning.yaml"
DEFAULT_SEEDS = tuple(range(20260921, 20260928))
INITIAL_SEEDS = {20260918, 20260919, 20260920}


def _validate_frozen_config(config: dict) -> None:
    """Refuse configs that would change the previously frozen study."""
    expected = {
        ("experiment_name",): "multiclass_cnn_frozen_unlearning_eps32_r40_e5",
        ("data", "backend"): "cifar10_multiclass",
        ("data", "n_clients"): 100,
        ("data", "samples_per_client"): 200,
        ("data", "heterogeneity_alpha"): 0.5,
        ("model", "name"): "small_groupnorm_cnn",
        ("model", "num_classes"): 10,
        ("federated", "rounds"): 40,
        ("federated", "local_epochs"): 5,
        ("federated", "client_sample_rate"): 0.5,
        ("privacy", "enabled"): True,
        ("privacy", "noise_multiplier"): 0.85629173,
        ("deletion", "kind"): "client",
        ("deletion", "client_id"): 2,
        ("unlearning", "rounds"): 4,
        ("evaluation", "retrain_ensemble_size"): 5,
    }
    for path, wanted in expected.items():
        actual = config
        for key in path:
            actual = actual.get(key) if isinstance(actual, dict) else None
        if actual != wanted:
            dotted_path = ".".join(path)
            raise ValueError(
                f"Frozen config mismatch at {dotted_path}: expected {wanted!r}, "
                f"got {actual!r}"
            )


def _find_complete_seed_run(
    output_root: Path,
    experiment_name: str,
    seed: int,
    expected_config_hash: str,
) -> Path | None:
    if not output_root.exists():
        return None

    for run_dir in sorted(output_root.iterdir()):
        if not run_dir.is_dir():
            continue
        metadata_path = run_dir / "run_metadata.json"
        if not metadata_path.exists():
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("experiment_name") != experiment_name or metadata.get("seed") != seed:
            continue
        if metadata.get("config_sha256") != expected_config_hash:
            raise ValueError(
                f"{run_dir} uses a different config for seed {seed}; "
                "refusing to mix configurations in one confirmation study."
            )
        required_artifacts = ("metrics.json", "privacy_ledger.json")
        if all((run_dir / name).is_file() for name in required_artifacts):
            return run_dir
    return None


def run_batch(config_path: Path, seeds: list[int], *, rerun: bool, dry_run: bool) -> dict | None:
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    config = load_config(config_path)
    _validate_frozen_config(config)
    if len(seeds) != len(set(seeds)):
        raise ValueError("Seed list contains duplicates.")
    if INITIAL_SEEDS.intersection(seeds):
        raise ValueError(f"Additional seed list must not repeat the initial seeds {sorted(INITIAL_SEEDS)}.")

    output_root = ROOT / config["output"]["root"]
    experiment_name = config["experiment_name"]
    pending: list[tuple[int, str]] = []
    for seed in seeds:
        seed_config = {**config, "seed": seed}
        config_hash = _config_hash(seed_config)
        existing = None if rerun else _find_complete_seed_run(
            output_root, experiment_name, seed, config_hash
        )
        if existing is not None:
            print(f"Skipping completed seed {seed}: {existing}")
        else:
            pending.append((seed, config_hash))

    if dry_run:
        print(f"Frozen study output: {output_root}")
        print(f"Seeds to run: {[seed for seed, _ in pending]}")
        return None

    for index, (seed, _) in enumerate(pending, start=1):
        print(f"\n[{index}/{len(pending)}] Running frozen study seed {seed}")
        subprocess.run(
            [
                sys.executable,
                "-m",
                "dp_forgetbench.run",
                "--config",
                str(config_path),
                "--seed",
                str(seed),
            ],
            cwd=ROOT,
            check=True,
        )

    summary = analyze(output_root)
    selected_seeds = set(summary["selected_seeds"])
    expected_seeds = INITIAL_SEEDS.union(seeds)
    missing = expected_seeds - selected_seeds
    if missing:
        raise RuntimeError(f"Study aggregation is missing expected seeds: {sorted(missing)}")

    summary_path = output_root / "frozen_unlearning_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nAggregated {summary['run_count']} distinct seeds: {summary['selected_seeds']}")
    print(f"Summary written to {summary_path}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="Rerun seeds even when complete matching results already exist.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the frozen config and show which seeds would run.",
    )
    args = parser.parse_args()
    run_batch(args.config, args.seeds, rerun=args.rerun, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
