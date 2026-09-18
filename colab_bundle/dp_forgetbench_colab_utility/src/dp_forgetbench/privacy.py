"""Client-level central-DP primitives and an auditable accountant ledger."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import dp_accounting
from dp_accounting import rdp


@dataclass(frozen=True)
class PrivacyLedger:
    enabled: bool
    adjacency: str
    accountant: str
    sampling: str
    client_sample_rate: float
    public_population_size: int | None
    aggregation_normalizer: float | None
    mechanism_version: str
    rounds: int
    release_count: int
    clip_norm: float | None
    noise_multiplier: float | None
    epsilon: float | None
    delta: float | None
    threat_model_note: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def poisson_gaussian_epsilon(
    *, sample_rate: float, noise_multiplier: float, rounds: int, delta: float
) -> float:
    """Compute RDP epsilon for the simulated Poisson-sampled Gaussian mechanism.

    This applies to add/remove-one client adjacency. Callers must not reuse it for
    fixed-size sampling, per-example clipping, or extra public checkpoint releases.
    """
    if not (0.0 < sample_rate <= 1.0):
        raise ValueError("sample_rate must be in (0, 1].")
    if noise_multiplier <= 0.0 or rounds < 1 or not (0.0 < delta < 1.0):
        raise ValueError("Invalid Gaussian-DP accounting parameters.")
    # Integer and selected half-integer orders avoid numerical convergence warnings
    # from the default dense fractional grid at high sampling probabilities.
    orders = [2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 32, 48, 64, 96, 128, 256]
    accountant = rdp.RdpAccountant(orders=orders)
    event = dp_accounting.PoissonSampledDpEvent(
        sample_rate, dp_accounting.GaussianDpEvent(noise_multiplier)
    )
    accountant.compose(dp_accounting.SelfComposedDpEvent(event, rounds))
    epsilon, _ = accountant.get_epsilon_and_optimal_order(delta)
    return float(epsilon)


def make_ledger(privacy_config: dict, federated_config: dict) -> PrivacyLedger:
    enabled = bool(privacy_config["enabled"])
    release_count = int(privacy_config.get("release_count", 1))
    if release_count != 1:
        raise ValueError(
            "Phase 0/1 ledger supports exactly one final model release. "
            "Add explicit release composition before setting release_count > 1."
        )
    if not enabled:
        return PrivacyLedger(
            enabled=False,
            adjacency="none (non-private control)",
            accountant="none",
            sampling="not applicable",
            client_sample_rate=float(federated_config["client_sample_rate"]),
            public_population_size=None,
            aggregation_normalizer=None,
            mechanism_version="non_private_control",
            rounds=int(federated_config["rounds"]),
            release_count=0,
            clip_norm=None,
            noise_multiplier=None,
            epsilon=None,
            delta=None,
            threat_model_note="Non-private control; no DP claim is made.",
        )
    if privacy_config.get("sampling") != "poisson":
        raise ValueError("The Phase 0 accountant supports only Poisson client sampling.")
    public_population_size = int(privacy_config["population_size"])
    if public_population_size < 1:
        raise ValueError("privacy.population_size must be positive and fixed before training.")
    aggregation_normalizer = public_population_size * float(federated_config["client_sample_rate"])
    epsilon = poisson_gaussian_epsilon(
        sample_rate=float(federated_config["client_sample_rate"]),
        noise_multiplier=float(privacy_config["noise_multiplier"]),
        rounds=int(federated_config["rounds"]),
        delta=float(privacy_config["delta"]),
    )
    return PrivacyLedger(
        enabled=True,
        adjacency="add/remove one complete client dataset",
        accountant="dp-accounting RDP accountant; Poisson-sampled Gaussian mechanism",
        sampling="independent Poisson client participation",
        client_sample_rate=float(federated_config["client_sample_rate"]),
        public_population_size=public_population_size,
        aggregation_normalizer=aggregation_normalizer,
        mechanism_version="poisson_sum_fixed_public_normalizer_v2",
        rounds=int(federated_config["rounds"]),
        release_count=release_count,
        clip_norm=float(privacy_config["clip_norm"]),
        noise_multiplier=float(privacy_config["noise_multiplier"]),
        epsilon=epsilon,
        delta=float(privacy_config["delta"]),
        threat_model_note=(
            "Simulator assumes a server-side Gaussian mechanism and protects a final model release. "
            "Each round sums clipped client updates, adds Gaussian noise to that sum, and divides by the fixed "
            "public normalizer q*N; this matches the Poisson-sampled Gaussian accountant under add/remove-one "
            "client adjacency. It is not a production secure-aggregation implementation; releasing additional "
            "checkpoints or accessing raw retained data during unlearning requires a revised privacy ledger."
        ),
    )
