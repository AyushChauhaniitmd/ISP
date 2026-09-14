"""
Phase 11: Frontier Discovery & 10+ Seed Statistical Confirmation.

Analyzes Phase 7 screening outputs to detect transitional/frontier cells
(where the model shifts between REDUNDANT and UNLEARNING-BENEFICIAL),
and automatically triggers high-sample confirmation (>=10 seeds)
to bound the Marginal Unlearning Benefit (MUB) with tight 95% Bootstrap CIs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR / "src") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "src"))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from dp_forgetbench.config import load_config
from scripts.run_phase7_factorial import (
    BASE_CONFIG_PATH,
    build_cell_config,
    execute_run,
    aggregate_master_results,
)

FRONTIER_RESULTS_DIR = Path("results/phase11_frontier")
HIGH_POWER_SEEDS = [20261101 + i for i in range(10)]


def find_frontier_cells(screening_records: list[dict]) -> list[dict]:
    """Identify transition cells that show mixed classifications or boundary epsilons."""
    grouped = {}
    for r in screening_records:
        key = (r["epsilon"], r["alpha"], r["deletion_fraction"])
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(r)

    frontier_cells = []
    for (eps, alpha, del_frac), cell_records in grouped.items():
        classes = {r["classification"] for r in cell_records}
        # If there is divergence across seeds or it's a known transition point (e.g. eps=4.0 or eps=8.0)
        is_mixed = len(classes) > 1
        is_boundary_eps = eps in (4.0, 8.0)
        if is_mixed or is_boundary_eps:
            frontier_cells.append({
                "epsilon": eps,
                "alpha": alpha,
                "deletion_fraction": del_frac,
                "screening_classes": list(classes),
            })
    return frontier_cells


def run_frontier_confirmation(
    cells: list[dict],
    seeds: list[int],
    output_root: Path,
    base_cfg: dict,
) -> None:
    print(f"\nTriggering high-power confirmation on {len(cells)} frontier cells with {len(seeds)} seeds each...")
    output_root.mkdir(parents=True, exist_ok=True)

    for cell in cells:
        eps = cell["epsilon"]
        alpha = cell["alpha"]
        del_frac = cell["deletion_fraction"]
        print(f"\n--- Confirming Frontier Cell: eps={eps}, alpha={alpha}, del={del_frac} ---")

        for seed in seeds:
            cfg, run_id = build_cell_config(base_cfg, eps, alpha, del_frac, seed, output_root)
            run_dir = output_root / run_id
            execute_run(cfg, run_dir, expected_deletion_fraction=del_frac)

    summary, records = aggregate_master_results(output_root)
    print(f"\nFrontier confirmation complete. Aggregated {summary['count']} runs.")


def main():
    parser = argparse.ArgumentParser(description="Phase 11 Frontier Confirmation")
    parser.add_argument("--screening_results", type=Path, default=Path("results/phase7_factorial/master_results.json"))
    parser.add_argument("--output_root", type=Path, default=FRONTIER_RESULTS_DIR)
    parser.add_argument("--seeds", type=int, nargs="+", default=HIGH_POWER_SEEDS)
    args = parser.parse_args()

    if not args.screening_results.exists():
        print(f"Screening results not found at {args.screening_results}. Please run Phase 7 screening first.")
        return

    screening_records = json.loads(args.screening_results.read_text(encoding="utf-8"))
    frontier_cells = find_frontier_cells(screening_records)

    print(f"Identified {len(frontier_cells)} frontier transition cells from screening data:")
    for c in frontier_cells:
        print(f"  - Epsilon: {c['epsilon']}, Alpha: {c['alpha']}, Deletion: {c['deletion_fraction']} (Observed: {c['screening_classes']})")

    base_cfg = load_config(BASE_CONFIG_PATH)
    run_frontier_confirmation(frontier_cells, args.seeds, args.output_root, base_cfg)


if __name__ == "__main__":
    main()
