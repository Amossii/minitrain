"""在独立 torchrun 作业中扫描 DDP/FSDP2 最大可训练模型。"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from minitrain.distributed.oom_boundary import (
    build_boundary_config,
    is_cuda_oom,
    maximum_passing_parameters,
    transformer_parameter_count,
    write_boundary_csv,
)


# 输入是完整命令和日志路径，输出是子进程返回码及合并日志；
# 每个点独立运行，确保上一次 OOM 不会污染 CUDA allocator 或进程组。
def run_probe(command: list[str], log_path: Path) -> tuple[int, str]:
    """运行一次隔离探测并保存完整 stdout/stderr。"""

    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(completed.stdout, encoding="utf-8")
    print(completed.stdout, end="", flush=True)
    return completed.returncode, completed.stdout


# 输入是策略列表和目标参数量列表，输出每个真实运行点的 PASS/OOM CSV；
# 非 OOM 异常会立即终止，以免产生误导性的模型边界。
def main() -> int:
    """扫描并记录 DDP/FSDP2 的最大可训练模型规模。"""

    parser = argparse.ArgumentParser(description="Benchmark distributed OOM boundary.")
    parser.add_argument("--strategies", nargs="+", choices=("ddp", "fsdp2"), default=("ddp", "fsdp2"))
    parser.add_argument("--targets-millions", nargs="+", type=float, default=(100, 200, 300, 400))
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--vocab-size", type=int, default=32_000)
    parser.add_argument("--num-layers", type=int, default=12)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--mlp-ratio", type=int, default=3)
    parser.add_argument("--local-batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-name", default="oom_boundary")
    args = parser.parse_args()
    if args.world_size < 2:
        parser.error("--world-size must be at least 2")
    if len(set(args.targets_millions)) != len(args.targets_millions):
        parser.error("--targets-millions cannot contain duplicates")
    if any(target <= 0 for target in args.targets_millions):
        parser.error("--targets-millions must be positive")

    raw_directory = PROJECT_ROOT / "results" / "raw" / args.run_name
    rows: list[dict[str, str | int | float]] = []
    for strategy in args.strategies:
        for target in sorted(args.targets_millions):
            config = build_boundary_config(
                target,
                args.seq_len,
                vocab_size=args.vocab_size,
                num_layers=args.num_layers,
                num_heads=args.num_heads,
                mlp_ratio=args.mlp_ratio,
            )
            actual_parameters = transformer_parameter_count(config)
            label = f"{target:g}m"
            result_path = raw_directory / f"{strategy}_{label}.json"
            log_path = raw_directory / "logs" / f"{strategy}_{label}.log"
            # 删除同名旧成功记录，避免本次 OOM 时误读上一次运行结果。
            result_path.unlink(missing_ok=True)
            command = [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                f"--nproc-per-node={args.world_size}",
                str(PROJECT_ROOT / "scripts" / "oom_boundary_worker.py"),
                "--strategy",
                strategy,
                "--target-parameters-millions",
                str(target),
                "--vocab-size",
                str(args.vocab_size),
                "--num-layers",
                str(args.num_layers),
                "--num-heads",
                str(args.num_heads),
                "--mlp-ratio",
                str(args.mlp_ratio),
                "--local-batch-size",
                str(args.local_batch_size),
                "--seq-len",
                str(args.seq_len),
                "--learning-rate",
                str(args.learning_rate),
                "--steps",
                str(args.steps),
                "--seed",
                str(args.seed),
                "--output",
                str(result_path),
            ]
            print(
                f"running strategy={strategy} target={target:g}M "
                f"actual_parameters={actual_parameters}",
                flush=True,
            )
            returncode, output = run_probe(command, log_path)
            if returncode == 0:
                if not result_path.exists():
                    raise RuntimeError(f"successful probe did not write {result_path}")
                result = json.loads(result_path.read_text(encoding="utf-8"))
                status = "PASS"
                peak_allocated = int(result["peak_memory_allocated"])
                peak_reserved = int(result["peak_memory_reserved"])
            elif is_cuda_oom(output):
                status = "OOM"
                peak_allocated = 0
                peak_reserved = 0
            else:
                raise RuntimeError(
                    f"{strategy} {target:g}M failed for a non-OOM reason; see {log_path}"
                )
            rows.append(
                {
                    "strategy": strategy,
                    "target_parameters_millions": target,
                    "actual_parameters": actual_parameters,
                    "hidden_size": config.hidden_size,
                    "num_layers": config.num_layers,
                    "num_heads": config.num_heads,
                    "intermediate_size": config.intermediate_size,
                    "vocab_size": config.vocab_size,
                    "seq_len": config.seq_len,
                    "world_size": args.world_size,
                    "local_batch_size": args.local_batch_size,
                    "global_batch_size": args.local_batch_size * args.world_size,
                    "precision": "fp32",
                    "optimizer": "adamw",
                    "steps": args.steps,
                    "status": status,
                    "peak_memory_allocated": peak_allocated,
                    "peak_memory_reserved": peak_reserved,
                    "log_path": str(log_path.relative_to(PROJECT_ROOT)),
                }
            )
            write_boundary_csv(
                PROJECT_ROOT / "results" / "tables" / f"{args.run_name}.csv", rows
            )

    for strategy in args.strategies:
        maximum = maximum_passing_parameters(rows, strategy)
        text = "none" if maximum is None else f"{maximum / 1_000_000:.3f}M"
        print(f"strategy={strategy} maximum_verified_parameters={text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
