"""Benchmark AllReduce latency and bandwidth across message sizes."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from minitrain.distributed.benchmark import (  # noqa: E402
    benchmark_all_reduce,
    generate_message_sizes,
    write_collective_csv,
)
from minitrain.distributed.runtime import (  # noqa: E402
    cleanup_distributed,
    init_distributed,
)


# CLI inputs define the size sweep and repetition policy. Rank 0 outputs one
# human-readable line per size and the shared raw CSV; all ranks perform timing.
def main() -> int:
    """Initialize the group and benchmark AllReduce for every message size."""

    parser = argparse.ArgumentParser(description="Benchmark distributed collectives.")
    parser.add_argument(
        "--backend", choices=("gloo", "nccl"), default="nccl"
    )
    parser.add_argument("--min-bytes", type=int, default=1_024)
    parser.add_argument("--max-bytes", type=int, default=268_435_456)
    parser.add_argument("--factor", type=int, default=4)
    parser.add_argument("--warmup-iterations", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results" / "raw" / "nccl_all_reduce.csv",
    )
    args = parser.parse_args()

    sizes = generate_message_sizes(args.min_bytes, args.max_bytes, args.factor)
    context = None
    try:
        context = init_distributed(backend=args.backend)
        if context.world_size < 2:
            raise RuntimeError("collective benchmark requires WORLD_SIZE >= 2")

        records = []
        for message_size in sizes:
            record = benchmark_all_reduce(
                context=context,
                message_size_bytes=message_size,
                warmup_iterations=args.warmup_iterations,
                iterations=args.iterations,
            )
            records.append(record)
            if context.is_main_process:
                print(
                    f"bytes={record.message_size_bytes} "
                    f"latency_ms={record.latency_ms:.6f} "
                    f"algbw_GBps={record.algorithm_bandwidth_gbps:.3f} "
                    f"busbw_GBps={record.bus_bandwidth_gbps:.3f}",
                    flush=True,
                )

        if context.is_main_process:
            write_collective_csv(args.output, records)
            print(f"metrics_csv={args.output}", flush=True)
    finally:
        if context is not None:
            cleanup_distributed()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
