from __future__ import annotations

import json

import pytest

from scripts.run_frozen_seed_batch import (
    DEFAULT_CONFIG,
    DEFAULT_SEEDS,
    _find_complete_seed_run,
    _validate_frozen_config,
)
from dp_forgetbench.config import load_config


def test_default_seed_batch_adds_seven_new_seeds() -> None:
    assert DEFAULT_SEEDS == tuple(range(20260921, 20260928))
    assert not {20260918, 20260919, 20260920}.intersection(DEFAULT_SEEDS)
    _validate_frozen_config(load_config(DEFAULT_CONFIG))


def test_frozen_config_validation_rejects_changed_rounds() -> None:
    config = load_config(DEFAULT_CONFIG)
    config["federated"]["rounds"] = 60

    with pytest.raises(ValueError, match="federated.rounds"):
        _validate_frozen_config(config)


def test_resume_only_accepts_complete_matching_seed_run(tmp_path) -> None:
    run_dir = tmp_path / "frozen_seed20260921"
    run_dir.mkdir()
    (run_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "experiment_name": "frozen-study",
                "seed": 20260921,
                "config_sha256": "expected-hash",
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "metrics.json").write_text("{}", encoding="utf-8")
    (run_dir / "privacy_ledger.json").write_text("{}", encoding="utf-8")

    assert _find_complete_seed_run(
        tmp_path, "frozen-study", 20260921, "expected-hash"
    ) == run_dir

    with pytest.raises(ValueError, match="different config"):
        _find_complete_seed_run(tmp_path, "frozen-study", 20260921, "other-hash")
