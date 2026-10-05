"""在一个独立进程任务中测量 Single、DDP、FSDP2 或手写 TP。"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from minitrain.config import available_model_configs, get_model_config
from minitrain.data import SyntheticTokenDataset
from minitrain.distributed.fsdp2 import apply_fsdp2
from minitrain.distributed.oom_boundary import (
    build_boundary_config,
    transformer_parameter_count,
)
from minitrain.distributed.runtime import cleanup_distributed, init_distributed
from minitrain.distributed.unified_benchmark import (
    UnifiedStepMetrics,
    write_unified_metrics_csv,
)
from minitrain.model import MiniTransformer
from minitrain.tensor_parallel.transformer import TensorParallelTransformer
from minitrain.training import StepMeasurement, measure_train_step, train_step


# 输入是一个 rank 的观测，输出同步训练中决定整体速度的全局观测：loss 取均值，
# 时间和显存取最大值，因为最慢、占用最高的 rank 决定系统瓶颈。
def aggregate_measurement(
    measurement: StepMeasurement, device: torch.device, world_size: int
) -> StepMeasurement:
    """将 rank-local 观测聚合为可比较的全局 step 观测。"""

    loss = torch.tensor(measurement.loss, dtype=torch.float64, device=device)
    dist.all_reduce(loss, op=dist.ReduceOp.SUM)
    loss /= world_size
    times = torch.tensor(
        [
            measurement.step_time,
            measurement.forward_time,
            measurement.backward_time,
            measurement.optimizer_time,
        ],
        dtype=torch.float64,
        device=device,
    )
    dist.all_reduce(times, op=dist.ReduceOp.MAX)
    memory = torch.tensor(
        [measurement.peak_memory_allocated, measurement.peak_memory_reserved],
        dtype=torch.int64,
        device=device,
    )
    dist.all_reduce(memory, op=dist.ReduceOp.MAX)
    return StepMeasurement(
        loss=float(loss.item()),
        step_time=float(times[0].item()),
        forward_time=float(times[1].item()),
        backward_time=float(times[2].item()),
        optimizer_time=float(times[3].item()),
        peak_memory_allocated=int(memory[0].item()),
        peak_memory_reserved=int(memory[1].item()),
    )


# 输入是策略、模型配置和设备，输出可训练模块。关键分布式转换直接写在这里，
# 让 benchmark 的模型 ownership 不被多层抽象隐藏。
def build_model(
    strategy: str,
    model_config: object,
    device: torch.device,
    local_rank: int,
) -> nn.Module:
    """创建并按指定策略包装 MiniTransformer。"""

    if strategy == "tp":
        return TensorParallelTransformer(model_config, device=device)  # type: ignore[arg-type]

    model = MiniTransformer(model_config).to(device)  # type: ignore[arg-type]
    if strategy == "single":
        return model
    if strategy == "ddp":
        if device.type == "cuda":
            return DistributedDataParallel(
                model,
                device_ids=[local_rank],
                output_device=local_rank,
                broadcast_buffers=False,
            )
        return DistributedDataParallel(model, broadcast_buffers=False)
    if strategy == "fsdp2":
        return apply_fsdp2(model, device.type)
    raise ValueError(f"unsupported native strategy: {strategy}")


# CLI 输入定义一个固定 global-batch workload；内部状态是模型、optimizer 和
# DataLoader 迭代位置；输出仅由 rank 0 写入逐 step CSV。
def main() -> int:
    """执行 warmup 和正式测量，并写入原始统一指标。"""

    parser = argparse.ArgumentParser(description="Benchmark one native strategy.")
    parser.add_argument("--strategy", choices=("single", "ddp", "fsdp2", "tp"), required=True)
    parser.add_argument("--backend", choices=("gloo", "nccl"), default="nccl")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--model", choices=available_model_configs(), default="tiny")
    parser.add_argument("--target-parameters-millions", type=float)
    parser.add_argument("--vocab-size", type=int, default=32_000)
    parser.add_argument("--num-layers", type=int, default=12)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--mlp-ratio", type=int, default=3)
    parser.add_argument("--global-batch-size", type=int, required=True)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.global_batch_size <= 0 or args.steps <= 0 or args.learning_rate <= 0:
        parser.error("batch size, steps, and learning rate must be positive")
    if args.warmup_steps < 0:
        parser.error("--warmup-steps must be non-negative")

    model_config = (
        build_boundary_config(
            args.target_parameters_millions,
            args.seq_len,
            vocab_size=args.vocab_size,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            mlp_ratio=args.mlp_ratio,
        )
        if args.target_parameters_millions is not None
        else get_model_config(args.model)
    )
    if args.seq_len > model_config.seq_len:
        parser.error(f"--seq-len exceeds the {args.model} maximum")

    context = None
    distributed = args.strategy != "single"
    try:
        if distributed:
            context = init_distributed(args.backend)
            rank = context.rank
            world_size = context.world_size
            local_rank = context.local_rank
            device = context.device
        else:
            rank = 0
            world_size = 1
            local_rank = 0
            if args.device == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("CUDA was requested but is not available")
            device = torch.device("cuda:0" if args.device == "cuda" else "cpu")

        if args.strategy in ("ddp", "fsdp2"):
            if args.global_batch_size % world_size != 0:
                raise ValueError("data-parallel global batch must be divisible by world size")
            batch_size_per_process = args.global_batch_size // world_size
        else:
            # Single 和 TP 都处理完整 global batch；TP ranks 的输入是 replicated。
            batch_size_per_process = args.global_batch_size

        torch.manual_seed(args.seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed)
        model = build_model(args.strategy, model_config, device, local_rank)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

        total_steps = args.warmup_steps + args.steps
        dataset = SyntheticTokenDataset(
            args.global_batch_size * total_steps,
            args.seq_len,
            model_config.vocab_size,
            args.seed,
        )
        sampler = None
        if args.strategy in ("ddp", "fsdp2"):
            sampler = DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=False,
                drop_last=True,
            )
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size_per_process,
            sampler=sampler,
            shuffle=False,
            drop_last=True,
            pin_memory=device.type == "cuda",
        )
        iterator = iter(dataloader)
        for _ in range(args.warmup_steps):
            train_step(model, optimizer, next(iterator), device)

        records: list[UnifiedStepMetrics] = []
        num_parameters = transformer_parameter_count(model_config)
        for step in range(args.steps):
            local = measure_train_step(model, optimizer, next(iterator), device)
            if not math.isfinite(local.loss):
                raise AssertionError(f"non-finite loss at measured step {step}")
            observed = (
                aggregate_measurement(local, device, world_size)
                if distributed
                else local
            )
            if rank == 0:
                record = UnifiedStepMetrics.from_measurement(
                    step=step,
                    measurement=observed,
                    world_size=world_size,
                    batch_size_per_process=batch_size_per_process,
                    global_batch_size=args.global_batch_size,
                    seq_len=args.seq_len,
                    num_parameters=num_parameters,
                    warmup_steps=args.warmup_steps,
                    model_name=(
                        f"target_{args.target_parameters_millions:g}m"
                        if args.target_parameters_millions is not None
                        else args.model
                    ),
                    precision="fp32",
                    optimizer="adamw",
                    strategy=args.strategy,
                )
                records.append(record)
                print(
                    f"strategy={args.strategy} step={step} "
                    f"loss={record.loss:.6f} step_time={record.step_time:.6f}s "
                    f"tokens_per_second={record.tokens_per_second:.2f}",
                    flush=True,
                )

        if rank == 0:
            write_unified_metrics_csv(args.output, records)
            print(f"metrics_csv={args.output}", flush=True)
        if distributed:
            dist.barrier()
    finally:
        if context is not None:
            cleanup_distributed()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
