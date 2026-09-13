"""Run a simplified offline-LiRA confirmation check against a completed run.

This does not read a saved model checkpoint (run_experiment does not persist
model weights). Instead it deterministically re-derives the exact `full_model`
(the released DP model) from the run's saved config, using the same seed the
original run used -- the whole pipeline is seeded, so this reproduces the
identical model. This keeps the validated Phase 0/1 run-writing code in
run.py completely untouched.

Usage:
    python scripts/run_lira_confirmation.py \
        --run-dir results/phase1_cifar10_binary_client_dp_cpu_..._seed20260830 \
        --n-shadow-models 32 --n-audit-examples 60

A meaningful confirmation run typically wants >=32 shadow models; start with a
small number (e.g. 8) as a smoke test since each shadow model is a full
training run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dp_forgetbench.config import load_config
from dp_forgetbench.data import apply_client_deletion_requests, make_federation, normalized_deletion_requests
from dp_forgetbench.federated import infer_n_features, train_federated
from dp_forgetbench.lira import run_lira_confirmation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True, help="A completed results/<run_name> directory.")
    parser.add_argument("--n-shadow-models", type=int, default=32)
    parser.add_argument("--n-audit-examples", type=int, default=60)
    parser.add_argument("--seed", type=int, default=777, help="Seed for shadow-model training and audit sampling.")
    parser.add_argument("--against", choices=["full_model", "retained_only"], default="full_model",
                         help="Attack the released full-population model (default) or a model trained only on retained clients.")
    args = parser.parse_args()

    metadata_path = args.run_dir / "run_metadata.json"
    if not metadata_path.exists():
        raise SystemExit(f"No run_metadata.json found under {args.run_dir}. Point --run-dir at a completed run directory.")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    config = load_config(metadata["config_path"])
    if _config_hash(config) != metadata["config_sha256"]:
        raise SystemExit("Config file has changed since this run was produced; cannot reproduce the exact target model.")

    seed = int(config["seed"])
    federation = make_federation(config["data"], seed)
    deletion_result = apply_client_deletion_requests(federation, config["deletion"], seed)
    requests = normalized_deletion_requests(config["deletion"])
    n_features = infer_n_features(next(iter(federation.clients.values())).x)
    model_config = config.get("model")

    if args.against == "full_model":
        target_clients = federation.clients
        privacy_config = {**config["privacy"]}
        if privacy_config["enabled"]:
            privacy_config.setdefault("population_size", int(config["data"]["n_clients"]))
        target_seed = seed
        member_client_ids = list(federation.clients)
    else:
        target_clients = deletion_result.retained
        privacy_config = {**metadata["reference_privacy_config"]}
        target_seed = seed + 10_000
        member_client_ids = list(deletion_result.retained)

    target_model, _cost = train_federated(
        clients=target_clients,
        n_features=n_features,
        federated_config=config["federated"],
        privacy_config=privacy_config,
        seed=target_seed,
        model_config=model_config,
    )

    result = run_lira_confirmation(
        target_model=target_model,
        federation=federation,
        member_client_ids=member_client_ids,
        n_shadow_models=args.n_shadow_models,
        n_audit_examples=args.n_audit_examples,
        federated_config=config["federated"],
        model_config=model_config,
        seed=args.seed,
    )
    summary = result.summary()
    summary["run_dir"] = str(args.run_dir)
    summary["attacked_model"] = args.against
    summary["deletion_requests"] = requests
    print(json.dumps(summary, indent=2))

    out_path = args.run_dir / f"lira_confirmation_{args.against}.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSaved to {out_path}")


def _config_hash(config: dict) -> str:
    import hashlib

    encoded = json.dumps(config, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


if __name__ == "__main__":
    main()
