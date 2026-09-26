"""Print and optionally validate the MiniTrain runtime environment."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Running `python3 scripts/check_env.py` places scripts/, not the repository
# root, on sys.path. Add the root so installation is optional on Kaggle.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from minitrain.environment import (  # noqa: E402
    format_environment_report,
    inspect_environment,
    validate_environment,
)


# Input comes from CLI arguments; output is terminal diagnostics and a process
# status. The status gates future distributed launches on valid GPU resources.
def main() -> int:
    """Inspect the environment and enforce optional GPU requirements."""

    parser = argparse.ArgumentParser(
        description="Inspect the PyTorch/CUDA/NCCL environment used by MiniTrain."
    )
    parser.add_argument(
        "--require-gpus",
        type=int,
        default=0,
        metavar="N",
        help="fail unless CUDA, NCCL, and at least N GPUs are available",
    )
    args = parser.parse_args()
    if args.require_gpus < 0:
        parser.error("--require-gpus must be non-negative")

    report = inspect_environment()
    print(format_environment_report(report))

    if args.require_gpus == 0:
        return 0

    failures = validate_environment(report, args.require_gpus)
    if failures:
        print("\nEnvironment validation: FAILED", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1

    print("\nEnvironment validation: PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

