"""Run the reproducible tiny/small/medium single-device baseline matrix."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from minitrain.benchmark import summarize_metrics_file, write_summary_csv  # noqa: E402
from minitrain.config import available_model_configs  # noqa: E402


# CLI inputs define one fixed workload matrix. Each model runs in a fresh Python
# process so CUDA model/optimizer allocations from an earlier run cannot leak
# into the next model's peak memory measurement.
def main() -> int:
    """Run model presets sequentially and create raw plus summary CSV files."""

    parser = argparse.ArgumentParser(description="Benchmark single-device models.")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=available_model_configs(),
        default=list(available_model_configs()),
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--local-batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()

    run_name = args.run_name or datetime.now(timezone.utc).strftime(
        "single_%Y%m%dT%H%M%SZ"
    )
    raw_directory = PROJECT_ROOT / "results" / "raw" / run_name
    summary_path = PROJECT_ROOT / "results" / "tables" / f"{run_name}.csv"
    raw_paths: list[Path] = []

    for model_name in args.models:
        raw_path = raw_directory / f"{model_name}.csv"
        command = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "train_single.py"),
            "--model",
            model_name,
            "--device",
            args.device,
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
            str(raw_path),
        ]
        print(f"running model={model_name}", flush=True)
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)
        raw_paths.append(raw_path)

    summaries = [summarize_metrics_file(path) for path in raw_paths]
    write_summary_csv(summary_path, summaries)
    print(f"summary_csv={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
