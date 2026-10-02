"""Initialize one torchrun worker and print its distributed identity."""

from __future__ import annotations

import argparse
from pathlib import Path
import socket
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from minitrain.distributed.runtime import (  # noqa: E402
    cleanup_distributed,
    init_distributed,
)


# Input selects the communication backend. Output is one identity line per
# process, proving torchrun metadata and local device mapping are consistent.
def main() -> int:
    """Initialize, display this worker's context, and cleanly tear it down."""

    parser = argparse.ArgumentParser(description="Inspect torchrun worker identity.")
    parser.add_argument(
        "--backend",
        choices=("auto", "gloo", "nccl"),
        default="auto",
        help="auto selects NCCL when CUDA is visible, otherwise Gloo",
    )
    args = parser.parse_args()

    context = None
    try:
        context = init_distributed(backend=args.backend)
        print(
            f"rank={context.rank} local_rank={context.local_rank} "
            f"world_size={context.world_size} backend={context.backend} "
            f"device={context.device} host={socket.gethostname()}",
            flush=True,
        )
    finally:
        if context is not None:
            cleanup_distributed()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
