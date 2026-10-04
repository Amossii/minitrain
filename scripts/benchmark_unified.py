"""编排 Step 23 的统一 Single/DDP/FSDP2/ZeRO/TP benchmark。"""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from minitrain.config import available_model_configs, get_model_config
from minitrain.distributed.unified_benchmark import (
    summarize_unified_metrics,
    write_unified_summary_csv,
)
from minitrain.tensor_parallel.transformer import validate_tensor_parallel_config


DATA_PARALLEL_STRATEGIES = ("ddp", "fsdp2", "zero1", "zero2", "zero3")


# 输入是 argv 列表，输出为空；check=True 保证任何策略失败都停止汇总，避免把
# 不完整表误当成正式实验结果。
def run_job(command: list[str]) -> None:
    """打印并同步执行一个全新的 benchmark 子进程。"""

    print(f"running={' '.join(command)}", flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


# 输入是固定实验协议，内部逐策略创建新进程；输出 data-parallel 主表和可选 TP
# 对照表，原始逐 step 数据保存在 results/raw 下。
def main() -> int:
    """运行统一 workload 并生成经过公平性校验的 CSV 表。"""

    parser = argparse.ArgumentParser(description="Run the unified Step 23 benchmark.")
    parser.add_argument("--model", choices=available_model_configs(), default="tiny")
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--global-batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=DATA_PARALLEL_STRATEGIES,
        default=list(DATA_PARALLEL_STRATEGIES),
    )
    parser.add_argument(
        "--include-tp",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--run-name", default="kaggle_unified")
    args = parser.parse_args()

    if args.world_size <= 0 or args.global_batch_size <= 0 or args.steps <= 0:
        parser.error("world size, global batch size, and steps must be positive")
    if args.warmup_steps < 0 or args.learning_rate <= 0:
        parser.error("warmup must be non-negative and learning rate must be positive")
    if args.global_batch_size % args.world_size != 0:
        parser.error("global batch size must be divisible by data-parallel world size")
    model_config = get_model_config(args.model)
    if args.seq_len > model_config.seq_len:
        parser.error(f"--seq-len exceeds the {args.model} maximum")
    if args.include_tp:
        validate_tensor_parallel_config(model_config, args.world_size)
    if any(strategy.startswith("zero") for strategy in args.strategies):
        if importlib.util.find_spec("deepspeed") is None:
            parser.error("DeepSpeed is required; run: pip install -e '.[deepspeed]'")

    raw_directory = PROJECT_ROOT / "results" / "raw" / args.run_name
    table_directory = PROJECT_ROOT / "results" / "tables"
    raw_directory.mkdir(parents=True, exist_ok=True)
    common = [
        "--model",
        args.model,
        "--global-batch-size",
        str(args.global_batch_size),
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
    ]

    single_path = raw_directory / "single.csv"
    run_job(
        [
            sys.executable,
            "scripts/benchmark_native_worker.py",
            "--strategy",
            "single",
            "--device",
            "cuda",
            *common,
            "--output",
            str(single_path),
        ]
    )

    strategy_paths: dict[str, Path] = {"single": single_path}
    for strategy in args.strategies:
        output = raw_directory / f"{strategy}.csv"
        torchrun = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc-per-node={args.world_size}",
        ]
        if strategy.startswith("zero"):
            stage = strategy.removeprefix("zero")
            command = [
                *torchrun,
                "scripts/benchmark_deepspeed_worker.py",
                "--zero-stage",
                stage,
                *common,
                "--output",
                str(output),
            ]
        else:
            command = [
                *torchrun,
                "scripts/benchmark_native_worker.py",
                "--strategy",
                strategy,
                "--backend",
                "nccl",
                *common,
                "--output",
                str(output),
            ]
        run_job(command)
        strategy_paths[strategy] = output

    data_parallel_rows = [
        summarize_unified_metrics(strategy_paths[strategy])
        for strategy in ("single", *args.strategies)
    ]
    main_table = table_directory / f"{args.run_name}.csv"
    write_unified_summary_csv(main_table, data_parallel_rows)
    print(f"data_parallel_table={main_table}", flush=True)

    if args.include_tp:
        tp_path = raw_directory / "tp.csv"
        run_job(
            [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                f"--nproc-per-node={args.world_size}",
                "scripts/benchmark_native_worker.py",
                "--strategy",
                "tp",
                "--backend",
                "nccl",
                *common,
                "--output",
                str(tp_path),
            ]
        )
        tp_table = table_directory / f"{args.run_name}_tp.csv"
        write_unified_summary_csv(
            tp_table,
            [
                summarize_unified_metrics(single_path),
                summarize_unified_metrics(tp_path),
            ],
        )
        print(f"tensor_parallel_table={tp_table}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
