"""Run fixed-protocol DDP strong and weak scaling experiments."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from minitrain.benchmark import summarize_metrics_file
from minitrain.config import available_model_configs
from minitrain.distributed.scaling import (
    build_scaling_rows,
    write_scaling_csv,
)


def main() -> int:
    """Launch four fresh DDP jobs and produce one scaling summary table."""

    parser = argparse.ArgumentParser(description="Benchmark DDP scaling.")
    parser.add_argument("--backend", choices=("gloo", "nccl"), default="nccl")
    parser.add_argument("--model", choices=available_model_configs(), default="tiny")
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--strong-global-batch-size", type=int, default=2)
    parser.add_argument("--weak-local-batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-name", default="ddp_scaling")
    args = parser.parse_args()

    raw_directory = PROJECT_ROOT / "results" / "raw" / args.run_name
    experiments: list[tuple[str, int, int, Path]] = []
    for scaling_type in ("strong", "weak"):
        for world_size in (1, 2):
            if scaling_type == "strong":
                if args.strong_global_batch_size % world_size != 0:
                    parser.error(
                        "strong global batch size must be divisible by world size"
                    )
                local_batch_size = args.strong_global_batch_size // world_size
            else:
                local_batch_size = args.weak_local_batch_size
            path = raw_directory / f"{scaling_type}_ws{world_size}.csv"
            experiments.append((scaling_type, world_size, local_batch_size, path))

    summaries: dict[str, list[dict[str, str | int | float]]] = {
        "strong": [],
        "weak": [],
    }
    for scaling_type, world_size, local_batch_size, path in experiments:
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc-per-node={world_size}",
            str(PROJECT_ROOT / "scripts" / "benchmark_ddp_worker.py"),
            "--backend",
            args.backend,
            "--model",
            args.model,
            "--local-batch-size",
            str(local_batch_size),
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
            str(path),
        ]
        print(
            f"running scaling={scaling_type} world_size={world_size} "
            f"local_batch_size={local_batch_size}",
            flush=True,
        )
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)
        summaries[scaling_type].append(summarize_metrics_file(path))

    rows = build_scaling_rows(summaries["strong"], "strong")
    rows.extend(build_scaling_rows(summaries["weak"], "weak"))
    output = PROJECT_ROOT / "results" / "tables" / f"{args.run_name}.csv"
    write_scaling_csv(output, rows)
    print(f"scaling_csv={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
