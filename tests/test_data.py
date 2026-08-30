from dp_forgetbench.data import apply_client_deletion_requests, make_synthetic_federation, retained_clients, write_client_deletion_manifest


def test_deletion_manifest_and_retention(tmp_path) -> None:
    config = {"n_clients": 4, "samples_per_client": 5, "n_features": 3, "heterogeneity": 0.5, "test_samples": 10}
    federation = make_synthetic_federation(config, seed=9)
    path, checksum = write_client_deletion_manifest(federation, 1, tmp_path)
    retained = retained_clients(federation, 1)
    assert path.exists()
    assert len(checksum) == 64
    assert 1 not in retained
    assert len(retained) == 3


def test_partial_and_cumulative_deletions_are_deterministic() -> None:
    config = {"n_clients": 5, "samples_per_client": 20, "n_features": 3, "heterogeneity": 0.5, "test_samples": 20}
    federation = make_synthetic_federation(config, seed=19)
    request = {"requests": [{"client_id": 1, "fraction": 0.25}, {"client_id": 3, "fraction": 1.0}]}
    first = apply_client_deletion_requests(federation, request, seed=19)
    second = apply_client_deletion_requests(federation, request, seed=19)
    assert len(first.forgotten_y) == 25
    assert len(first.retained[1].y) == 15
    assert 3 not in first.retained
    assert first.rows == second.rows
