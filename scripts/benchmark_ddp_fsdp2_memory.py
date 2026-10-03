"""Launch isolated DDP/FSDP2 jobs and build one fair memory table."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from minitrain.config import available_model_configs
from minitrain.distributed.memory import (
    COMPARISON_FIELDS,
    build_memory_comparison,
    write_memory_csv,
)


# Input is a one-row worker CSV. Output parses numeric fields so fairness and
# ratio calculations cannot silently use lexicographic string comparisons.
def read_memory_row(path: Path) -> dict[str, str | int | float]:
    """Read and type one raw memory benchmark row."""

    with path.open(encoding="utf-8", newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))
    if len(rows) != 1:
        raise ValueError(f"expected one row in {path}, got {len(rows)}")
    integer_fields = {
        "world_size",
        "local_batch_size",
        "global_batch_size",
        "seq_len",
        "num_parameters",
        "warmup_steps",
        "measured_steps",
        "parameter_bytes_per_rank",
        "gradient_bytes_per_rank",
        "optimizer_state_bytes_per_rank",
        "theoretical_model_state_bytes_per_rank",
        "observed_peak_allocated_bytes",
        "observed_peak_reserved_bytes",
        "observed_end_allocated_bytes",
        "observed_end_reserved_bytes",
    }
    return {
        key: int(value) if key in integer_fields else value
        for key, value in rows[0].items()
    }


# CLI inputs define one fair paired experiment. Each strategy runs in a fresh
# process group, and output contains raw evidence plus derived relative savings.
def main() -> int:
    """Run isolated DDP and FSDP2 memory jobs and write a comparison CSV."""

    parser = argparse.ArgumentParser(description="Compare DDP and FSDP2 memory.")
    parser.add_argument("--model", choices=available_model_configs(), default="tiny")
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--local-batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-name", default="ddp_vs_fsdp2_memory")
    args = parser.parse_args()
    if args.world_size < 2:
        parser.error("--world-size must be at least 2 for a sharding comparison")

    raw_directory = PROJECT_ROOT / "results" / "raw" / args.run_name
    raw_rows = []
    for strategy in ("ddp", "fsdp2"):
        output = raw_directory / f"{strategy}.csv"
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc-per-node={args.world_size}",
            str(PROJECT_ROOT / "scripts" / "benchmark_memory_worker.py"),
            "--strategy",
            strategy,
            "--backend",
            "nccl",
            "--model",
            args.model,
            "--local-batch-size",
            str(args.local_batch_size),
            "--seq-len",
            str(args.seq_len),
            "--learning-rate",
            str(args.learning_rate),
            "--warmup-steps",
            str(args.warmup_steps),
            "--steps",
            str(args.steps),
            "--seed",
            str(args.seed),
            "--output",
            str(output),
        ]
        print(f"running strategy={strategy}", flush=True)
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)
        raw_rows.append(read_memory_row(output))

    comparison = build_memory_comparison(raw_rows)
    output = PROJECT_ROOT / "results" / "tables" / f"{args.run_name}.csv"
    write_memory_csv(output, comparison, COMPARISON_FIELDS)
    print(f"comparison_csv={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
