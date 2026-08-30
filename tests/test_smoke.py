from pathlib import Path

from dp_forgetbench.config import load_config
from dp_forgetbench.run import run_experiment, run_sequence_experiment


def test_phase0_smoke(tmp_path, monkeypatch) -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs" / "phase0_toy.yaml")
    config["federated"]["rounds"] = 2
    config["unlearning"]["rounds"] = 1
    config["output"]["root"] = str(tmp_path / "results")
    monkeypatch.chdir(tmp_path)
    output = run_experiment(config, root / "configs" / "phase0_toy.yaml")
    assert (output / "metrics.json").exists()
    assert (output / "privacy_ledger.json").exists()
    assert (output / "run_metadata.json").exists()
    metrics = (output / "metrics.json").read_text(encoding="utf-8")
    assert "target_retrain_independent" in metrics
    assert "forgotten_vs_unseen_loss_mia" in metrics


def test_sequential_smoke(tmp_path, monkeypatch) -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs" / "phase0_partial_sequential.yaml")
    config["federated"]["rounds"] = 2
    config["unlearning"]["rounds"] = 1
    config["output"]["root"] = str(tmp_path / "results")
    monkeypatch.chdir(tmp_path)
    output = run_sequence_experiment(config, root / "configs" / "phase0_partial_sequential.yaml")
    assert (output / "sequence_metrics.json").exists()
