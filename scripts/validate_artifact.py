"""Validate a benchmark run directory before it is used in a report or paper."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def config_hash(config: dict) -> str:
    encoded = json.dumps(config, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def expected_retained_population(config: dict, metadata: dict) -> int:
    original = int(config["data"]["n_clients"])
    fully_deleted = {
        int(request["client_id"])
        for request in metadata.get("deletion_requests", [])
        if float(request.get("fraction", 1.0)) == 1.0
    }
    return original - len(fully_deleted)


def validate(run_dir: Path) -> list[str]:
    errors = []
    metadata_path = run_dir / "run_metadata.json"
    ledger_path = run_dir / "privacy_ledger.json"
    metrics_path = run_dir / "metrics.json"
    sequence_path = run_dir / "sequence_metrics.json"
    for required in (metadata_path, ledger_path):
        if not required.exists():
            errors.append(f"Missing required file: {required.name}")
    if errors:
        return errors
    metadata = read_json(metadata_path)
    ledger = read_json(ledger_path)
    if ledger.get("enabled"):
        if ledger.get("adjacency") != "add/remove one complete client dataset":
            errors.append("Unexpected DP adjacency for primary client deletion.")
        if ledger.get("mechanism_version") != "poisson_sum_fixed_public_normalizer_v2":
            errors.append("DP mechanism does not match the current accountant-aligned fixed-normalizer implementation.")
        if ledger.get("release_count") != 1:
            errors.append("Ledger does not have exactly one declared final release.")
        if ledger.get("epsilon") is None or ledger.get("delta") is None:
            errors.append("Enabled DP run lacks epsilon/delta.")
    if "deletion_manifest" in metadata:
        manifest = Path(metadata["deletion_manifest"])
        if not manifest.is_absolute():
            manifest = Path.cwd() / manifest
        if not manifest.exists():
            errors.append("Deletion manifest does not exist.")
        elif sha256(manifest) != metadata.get("deletion_manifest_sha256"):
            errors.append("Deletion manifest checksum mismatch.")
    config = None
    config_path = Path(metadata.get("config_path", ""))
    if config_path.exists():
        try:
            import yaml
            config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            if config_hash(config) != metadata.get("config_sha256"):
                errors.append("Frozen configuration hash mismatch; do not aggregate this run.")
        except Exception as error:  # pragma: no cover - defensive artifact validation
            errors.append(f"Could not validate frozen config hash: {error}")
    else:
        errors.append("Frozen configuration file no longer exists.")
    if metrics_path.exists():
        metrics = read_json(metrics_path)
        required_methods = {"dp_only_no_action", "retained_finetune_baseline", "target_retrain", "target_retrain_independent"}
        missing = required_methods - set(metrics)
        if missing:
            errors.append(f"metrics.json missing methods: {sorted(missing)}")
        if ledger.get("enabled") and config is not None:
            effective = metadata.get("effective_privacy_config", {})
            reference = metadata.get("reference_privacy_config")
            if int(effective.get("population_size", -1)) != int(config["data"]["n_clients"]):
                errors.append("Released-model privacy population does not match original client count.")
            if not reference:
                errors.append("Missing reference_privacy_config; retrain-reference normalizer cannot be validated.")
            elif int(reference.get("population_size", -1)) != expected_retained_population(config, metadata):
                errors.append("Retrain-reference privacy population does not match retained client count.")
    elif sequence_path.exists():
        steps = read_json(sequence_path).get("steps", [])
        if not steps:
            errors.append("sequence_metrics.json has no request steps.")
        for step in steps:
            manifest = Path(step["manifest"])
            if not manifest.is_absolute():
                manifest = Path.cwd() / manifest
            if not manifest.exists() or sha256(manifest) != step["manifest_sha256"]:
                errors.append(f"Request {step.get('request_index')} manifest checksum mismatch.")
            if ledger.get("enabled") and "reference_privacy_config" not in step:
                errors.append(f"Request {step.get('request_index')} missing reference_privacy_config.")
    else:
        errors.append("Neither metrics.json nor sequence_metrics.json exists.")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, nargs="+", help="One or more experiment result directories")
    args = parser.parse_args()
    bad = 0
    for run_dir in args.run:
        errors = validate(run_dir)
        if errors:
            bad += 1
            print(f"FAIL {run_dir}")
            for error in errors:
                print(f"  - {error}")
        else:
            print(f"OK   {run_dir}")
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
