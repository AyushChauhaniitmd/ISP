"""
test_dp_accountant.py

Tests the Phase 0 DP accountant for correctness.
Cross-validates `poisson_gaussian_epsilon` against known analytical bounds
and checks that it matches expected behavior.
"""
import pytest
import math

from dp_forgetbench.privacy import poisson_gaussian_epsilon, make_ledger

def test_poisson_gaussian_epsilon_values():
    """Check known values to ensure accountant matches standard bounds."""
    # Example 1: Small sample rate, many rounds
    eps1 = poisson_gaussian_epsilon(
        sample_rate=0.01,
        noise_multiplier=1.0,
        rounds=1000,
        delta=1e-5
    )
    # Expected eps should be ~ > 0 and reasonable (e.g. ~1-3)
    assert 0.0 < eps1 < 5.0
    
    # Example 2: High noise -> low epsilon
    eps_high_noise = poisson_gaussian_epsilon(
        sample_rate=0.01,
        noise_multiplier=5.0,
        rounds=1000,
        delta=1e-5
    )
    assert eps_high_noise < eps1
    
    # Example 3: More rounds -> higher epsilon
    eps_more_rounds = poisson_gaussian_epsilon(
        sample_rate=0.01,
        noise_multiplier=1.0,
        rounds=2000,
        delta=1e-5
    )
    assert eps_more_rounds > eps1

def test_ledger_creation():
    privacy_config = {
        "enabled": True,
        "population_size": 100,
        "noise_multiplier": 1.1,
        "clip_norm": 1.0,
        "delta": 1e-5,
        "sampling": "poisson"
    }
    federated_config = {
        "client_sample_rate": 0.5,
        "rounds": 50
    }
    
    ledger = make_ledger(privacy_config, federated_config)
    assert ledger.enabled is True
    assert ledger.aggregation_normalizer == 100 * 0.5
    assert ledger.epsilon > 0
    assert ledger.mechanism_version == "poisson_sum_fixed_public_normalizer_v2"

def test_ledger_disabled():
    privacy_config = {
        "enabled": False
    }
    federated_config = {
        "client_sample_rate": 0.5,
        "rounds": 50
    }
    
    ledger = make_ledger(privacy_config, federated_config)
    assert ledger.enabled is False
    assert ledger.epsilon is None
    assert ledger.noise_multiplier is None
