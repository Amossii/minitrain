"""在固定双卡 workload 下比较 FSDP2 与手写 TP 的容量和吞吐。"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from minitrain.distributed.fsdp_tp_comparison import (
    per_rank_batch_size,
    validate_throughput_target,
)
from minitrain.distributed.oom_boundary import (
    build_boundary_config,
    is_cuda_oom,
    maximum_passing_parameters,
    transformer_parameter_count,
    write_boundary_csv,
)
from minitrain.distributed.unified_benchmark import (
    summarize_unified_metrics,
    write_unified_summary_csv,
)


STRATEGIES = ("fsdp2", "tp")


# 输入是命令和日志路径，输出退出码与完整日志；每个容量点必须使用新进程，
# 因为一次 CUDA OOM 后继续复用 allocator/process group 会污染后续观测。
def run_logged_job(command: list[str], log_path: Path) -> tuple[int, str]:
    """运行隔离任务并保存合并后的 stdout/stderr。"""

    print(f"running={' '.join(command)}", flush=True)
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


# 输入是固定硬件约束，输出两张 GPU 的真实名称；默认严格拒绝非双 T4 环境，
# 防止把其他 GPU 上的结果误标成课程要求的 2 × T4 16GB。
def validate_hardware(require_t4: bool) -> list[str]:
    """验证当前实验确实运行在两张可见 CUDA GPU 上。"""

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")
    if torch.cuda.device_count() != 2:
        raise RuntimeError(
            f"expected exactly 2 visible GPUs, got {torch.cuda.device_count()}"
        )
    names = [torch.cuda.get_device_name(index) for index in range(2)]
    if require_t4 and any("T4" not in name for name in names):
        raise RuntimeError(f"expected 2 x NVIDIA T4, got {names}")
    return names


# 输入是完整且固定的实验协议；输出容量扫描表、共同模型吞吐表和 manifest。
# 容量与吞吐分开运行，避免用不同模型规模制造不公平的吞吐结论。
def main() -> int:
    """执行 FSDP2/TP 最大容量与同模型吞吐比较。"""

    parser = argparse.ArgumentParser(description="Compare FSDP2 and TP fairly.")
    parser.add_argument(
        "--capacity-targets-millions",
        nargs="+",
        type=float,
        default=(100, 200, 400, 600, 800, 1000, 1200, 1400),
    )
    parser.add_argument("--throughput-target-millions", type=float, default=100)
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--global-batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--vocab-size", type=int, default=32_000)
    parser.add_argument("--num-layers", type=int, default=12)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--mlp-ratio", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--capacity-steps", type=int, default=1)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--throughput-steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--require-t4", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run-name", default="kaggle_fsdp2_vs_tp")
    args = parser.parse_args()

    if args.world_size != 2:
        parser.error("this course protocol fixes --world-size=2")
    if args.global_batch_size <= 0 or args.global_batch_size % 2 != 0:
        parser.error("--global-batch-size must be positive and divisible by 2")
    if args.seq_len <= 0 or args.capacity_steps <= 0 or args.throughput_steps <= 0:
        parser.error("sequence length and measured steps must be positive")
    if args.warmup_steps < 0 or args.learning_rate <= 0:
        parser.error("warmup must be non-negative and learning rate must be positive")
    targets = sorted(set(args.capacity_targets_millions))
    if len(targets) != len(args.capacity_targets_millions):
        parser.error("capacity targets cannot contain duplicates")
    if any(target <= 0 for target in targets):
        parser.error("capacity targets must be positive")
    if args.throughput_target_millions not in targets:
        parser.error("throughput target must also appear in capacity targets")

    gpu_names = validate_hardware(args.require_t4)
    raw_directory = PROJECT_ROOT / "results" / "raw" / args.run_name
    table_directory = PROJECT_ROOT / "results" / "tables"
    raw_directory.mkdir(parents=True, exist_ok=True)

    manifest = {
        "gpu_names": gpu_names,
        "world_size": args.world_size,
        "precision": "fp32",
        "optimizer": "adamw",
        "seq_len": args.seq_len,
        "global_batch_size": args.global_batch_size,
        "activation_checkpoint": False,
        "vocab_size": args.vocab_size,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "mlp_ratio": args.mlp_ratio,
        "seed": args.seed,
    }
    (raw_directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    capacity_rows: list[dict[str, str | int | float]] = []
    for strategy in STRATEGIES:
        for target in targets:
            config = build_boundary_config(
                target,
                args.seq_len,
                vocab_size=args.vocab_size,
                num_layers=args.num_layers,
                num_heads=args.num_heads,
                mlp_ratio=args.mlp_ratio,
            )
            label = f"{target:g}m"
            result_path = raw_directory / "capacity" / f"{strategy}_{label}.json"
            log_path = raw_directory / "capacity" / "logs" / f"{strategy}_{label}.log"
            result_path.unlink(missing_ok=True)
            command = [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nproc-per-node=2",
                "scripts/oom_boundary_worker.py",
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
                "--global-batch-size",
                str(args.global_batch_size),
                "--seq-len",
                str(args.seq_len),
                "--learning-rate",
                str(args.learning_rate),
                "--steps",
                str(args.capacity_steps),
                "--seed",
                str(args.seed),
                "--output",
                str(result_path),
            ]
            returncode, output = run_logged_job(command, log_path)
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
            capacity_rows.append(
                {
                    "strategy": strategy,
                    "target_parameters_millions": target,
                    "actual_parameters": transformer_parameter_count(config),
                    "hidden_size": config.hidden_size,
                    "num_layers": config.num_layers,
                    "num_heads": config.num_heads,
                    "intermediate_size": config.intermediate_size,
                    "vocab_size": config.vocab_size,
                    "seq_len": config.seq_len,
                    "world_size": args.world_size,
                    "local_batch_size": per_rank_batch_size(
                        strategy, args.global_batch_size, args.world_size
                    ),
                    "global_batch_size": args.global_batch_size,
                    "precision": "fp32",
                    "optimizer": "adamw",
                    "steps": args.capacity_steps,
                    "status": status,
                    "peak_memory_allocated": peak_allocated,
                    "peak_memory_reserved": peak_reserved,
                    "log_path": str(log_path.relative_to(PROJECT_ROOT)),
                }
            )
            write_boundary_csv(
                table_directory / f"{args.run_name}_capacity.csv", capacity_rows
            )

    validate_throughput_target(capacity_rows, args.throughput_target_millions)
    throughput_paths: dict[str, Path] = {}
    for strategy in STRATEGIES:
        output_path = raw_directory / "throughput" / f"{strategy}.csv"
        log_path = raw_directory / "throughput" / "logs" / f"{strategy}.log"
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=2",
            "scripts/benchmark_native_worker.py",
            "--strategy",
            strategy,
            "--backend",
            "nccl",
            "--target-parameters-millions",
            str(args.throughput_target_millions),
            "--vocab-size",
            str(args.vocab_size),
            "--num-layers",
            str(args.num_layers),
            "--num-heads",
            str(args.num_heads),
            "--mlp-ratio",
            str(args.mlp_ratio),
            "--global-batch-size",
            str(args.global_batch_size),
            "--seq-len",
            str(args.seq_len),
            "--learning-rate",
            str(args.learning_rate),
            "--warmup-steps",
            str(args.warmup_steps),
            "--steps",
            str(args.throughput_steps),
            "--seed",
            str(args.seed),
            "--output",
            str(output_path),
        ]
        returncode, _ = run_logged_job(command, log_path)
        if returncode != 0:
            raise RuntimeError(f"throughput run failed; see {log_path}")
        throughput_paths[strategy] = output_path

    summaries = [
        summarize_unified_metrics(throughput_paths[strategy])
        for strategy in STRATEGIES
    ]
    throughput_table = table_directory / f"{args.run_name}_throughput.csv"
    write_unified_summary_csv(throughput_table, summaries)

    for strategy in STRATEGIES:
        maximum = maximum_passing_parameters(capacity_rows, strategy)
        text = "none" if maximum is None else f"{maximum / 1_000_000:.3f}M"
        print(f"strategy={strategy} maximum_verified_parameters={text}")
    print(f"capacity_table={table_directory / f'{args.run_name}_capacity.csv'}")
    print(f"throughput_table={throughput_table}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
