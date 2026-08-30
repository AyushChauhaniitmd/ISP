from dp_forgetbench.privacy import make_ledger, poisson_gaussian_epsilon


def test_epsilon_increases_with_rounds() -> None:
    short = poisson_gaussian_epsilon(sample_rate=0.2, noise_multiplier=1.2, rounds=5, delta=1e-5)
    long = poisson_gaussian_epsilon(sample_rate=0.2, noise_multiplier=1.2, rounds=10, delta=1e-5)
    assert 0 < short < long


def test_ledger_is_client_level() -> None:
    ledger = make_ledger(
        {"enabled": True, "sampling": "poisson", "noise_multiplier": 1.0, "clip_norm": 1.0, "delta": 1e-5, "release_count": 1, "population_size": 12},
        {"client_sample_rate": 0.5, "rounds": 3},
    )
    assert ledger.adjacency == "add/remove one complete client dataset"
    assert ledger.epsilon is not None


def test_multiple_releases_are_rejected_until_composed() -> None:
    try:
        make_ledger(
            {"enabled": True, "sampling": "poisson", "noise_multiplier": 1.0, "clip_norm": 1.0, "delta": 1e-5, "release_count": 2, "population_size": 12},
            {"client_sample_rate": 0.5, "rounds": 3},
        )
    except ValueError as error:
        assert "release" in str(error).lower()
    else:
        raise AssertionError("Expected uncomposed multiple releases to be rejected")
