"""Execute an explicitly selected immutable grid and write a run index."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dp_forgetbench.config import load_config
from dp_forgetbench.run import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=Path("configs/generated/phase1/INDEX.txt"))
    parser.add_argument("--limit", type=int, help="Run only the first N configs for a smoke test.")
    parser.add_argument("--execute", action="store_true", help="Required acknowledgement before starting a grid.")
    parser.add_argument("--run-index", type=Path, default=Path("reports/phase1_run_index.json"))
    args = parser.parse_args()
    if not args.execute:
        raise SystemExit("Refusing to start a grid without --execute. Generate/review configs first.")
    paths = [Path(line.strip()) for line in args.index.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.limit is not None:
        paths = paths[: args.limit]
    run_dirs = []
    for path in paths:
        run_dir = run_experiment(load_config(path), path)
        run_dirs.append(str(run_dir))
        print(f"Completed {path.name}: {run_dir}")
        args.run_index.parent.mkdir(parents=True, exist_ok=True)
        args.run_index.write_text(json.dumps({"runs": run_dirs}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

