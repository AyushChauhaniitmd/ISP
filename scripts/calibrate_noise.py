"""Find a Gaussian noise multiplier for a target client-DP epsilon.

The calculation is valid only for the Poisson-sampled client mechanism used by
this repository's Phase 0/1 simulator.
"""

from __future__ import annotations

import argparse

from dp_forgetbench.privacy import poisson_gaussian_epsilon


def calibrate(target: float, sample_rate: float, rounds: int, delta: float) -> tuple[float, float]:
    if target <= 0:
        raise ValueError("Target epsilon must be positive.")
    low, high = 1e-3, 1.0
    while poisson_gaussian_epsilon(sample_rate=sample_rate, noise_multiplier=high, rounds=rounds, delta=delta) > target:
        high *= 2.0
        if high > 1e5:
            raise RuntimeError("Could not bracket target epsilon.")
    for _ in range(45):
        middle = (low + high) / 2.0
        epsilon = poisson_gaussian_epsilon(sample_rate=sample_rate, noise_multiplier=middle, rounds=rounds, delta=delta)
        if epsilon > target:
            low = middle
        else:
            high = middle
    achieved = poisson_gaussian_epsilon(sample_rate=sample_rate, noise_multiplier=high, rounds=rounds, delta=delta)
    return high, achieved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epsilon", type=float, nargs="+", required=True, help="Target epsilon values")
    parser.add_argument("--sample-rate", type=float, required=True)
    parser.add_argument("--rounds", type=int, required=True)
    parser.add_argument("--delta", type=float, default=1e-5)
    args = parser.parse_args()
    for target in args.epsilon:
        noise, achieved = calibrate(target, args.sample_rate, args.rounds, args.delta)
        print(f"target_epsilon={target:.4g} noise_multiplier={noise:.8f} achieved_epsilon={achieved:.8f}")


if __name__ == "__main__":
    main()

