"""在独立 torchrun 任务中测量 DeepSpeed ZeRO-1/2/3。"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from minitrain.config import available_model_configs, get_model_config
from minitrain.data import SyntheticTokenDataset
from minitrain.distributed.oom_boundary import transformer_parameter_count
from minitrain.distributed.unified_benchmark import (
    UnifiedStepMetrics,
    write_unified_metrics_csv,
)
from minitrain.distributed.zero import build_zero_config
from minitrain.model import MiniTransformer
from minitrain.training import StepMeasurement, causal_lm_loss


# 输入为空，输出可选 DeepSpeed 模块；延迟导入保证未安装 DeepSpeed 时 CPU
# 单元测试仍能运行，并给出明确安装命令。
def import_deepspeed() -> Any:
    """导入 DeepSpeed 或报告可操作的依赖错误。"""

    try:
        import deepspeed
    except ImportError as exc:
        raise RuntimeError(
            "DeepSpeed is not installed; run: pip install -e '.[deepspeed]'"
        ) from exc
    return deepspeed


# 输入是 engine 和一个 batch，输出一次完整参数更新；engine 持有 ZeRO 分片状态，
# 每次调用后 optimizer 状态和参数 shard 都前进一步。
def deepspeed_train_step(
    engine: Any, batch: dict[str, torch.Tensor], device: torch.device
) -> float:
    """执行不计时的 DeepSpeed warmup 更新。"""

    engine.train()
    input_ids = batch["input_ids"].to(device, non_blocking=True)
    labels = batch["labels"].to(device, non_blocking=True)
    loss = causal_lm_loss(engine(input_ids), labels)
    engine.backward(loss)
    engine.step()
    return float(loss.detach())


# 输入是 engine 和 batch，输出 CUDA Event 实测值。阶段时间对应 API 生命周期：
# backward/optimizer 区间可能包含 ZeRO 通信，因此不能解释为纯计算时间。
def measure_deepspeed_step(
    engine: Any, batch: dict[str, torch.Tensor], device: torch.device
) -> StepMeasurement:
    """测量一个 ZeRO forward、backward 和 engine.step 更新。"""

    engine.train()
    input_ids = batch["input_ids"].to(device, non_blocking=True)
    labels = batch["labels"].to(device, non_blocking=True)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    events = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
    events[0].record()
    loss = causal_lm_loss(engine(input_ids), labels)
    events[1].record()
    engine.backward(loss)
    events[2].record()
    engine.step()
    events[3].record()
    events[3].synchronize()
    return StepMeasurement(
        loss=float(loss.detach()),
        step_time=events[0].elapsed_time(events[3]) / 1_000,
        forward_time=events[0].elapsed_time(events[1]) / 1_000,
        backward_time=events[1].elapsed_time(events[2]) / 1_000,
        optimizer_time=events[2].elapsed_time(events[3]) / 1_000,
        peak_memory_allocated=torch.cuda.max_memory_allocated(device),
        peak_memory_reserved=torch.cuda.max_memory_reserved(device),
    )


# 输入是 rank-local 观测，输出 mean loss 与跨 rank 最大时间/显存；聚合发生在
# 被计时的 CUDA Events 之后，不会污染当前 step 的策略时间。
def aggregate_measurement(
    measurement: StepMeasurement, device: torch.device, world_size: int
) -> StepMeasurement:
    """聚合 DeepSpeed 同步任务中决定整体表现的观测值。"""

    values = torch.tensor(
        [
            measurement.loss,
            measurement.step_time,
            measurement.forward_time,
            measurement.backward_time,
            measurement.optimizer_time,
        ],
        dtype=torch.float64,
        device=device,
    )
    loss = values[:1].clone()
    dist.all_reduce(loss, op=dist.ReduceOp.SUM)
    loss /= world_size
    times = values[1:].clone()
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


# CLI 输入定义与 native worker 相同的 global workload；内部由 DeepSpeed engine
# 管理不同 stage 的 shard；输出是 rank 0 的统一逐 step CSV。
def main() -> int:
    """初始化 ZeRO，执行 warmup/测量并保存真实 GPU 观测。"""

    parser = argparse.ArgumentParser(description="Benchmark one DeepSpeed strategy.")
    parser.add_argument("--zero-stage", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--model", choices=available_model_configs(), default="tiny")
    parser.add_argument("--global-batch-size", type=int, required=True)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--local_rank", type=int, default=-1)
    args = parser.parse_args()
    if args.global_batch_size <= 0 or args.steps <= 0 or args.learning_rate <= 0:
        parser.error("batch size, steps, and learning rate must be positive")
    if args.warmup_steps < 0:
        parser.error("--warmup-steps must be non-negative")
    model_config = get_model_config(args.model)
    if args.seq_len > model_config.seq_len:
        parser.error(f"--seq-len exceeds the {args.model} maximum")

    deepspeed = import_deepspeed()
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    deepspeed.init_distributed(dist_backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    try:
        if args.global_batch_size % world_size != 0:
            raise ValueError("data-parallel global batch must be divisible by world size")
        batch_size_per_process = args.global_batch_size // world_size
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        model = MiniTransformer(model_config).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
        engine, _, _, _ = deepspeed.initialize(
            model=model,
            optimizer=optimizer,
            config=build_zero_config(
                args.zero_stage, batch_size_per_process, world_size
            ),
        )

        total_steps = args.warmup_steps + args.steps
        dataset = SyntheticTokenDataset(
            args.global_batch_size * total_steps,
            args.seq_len,
            model_config.vocab_size,
            args.seed,
        )
        sampler = DistributedSampler(
            dataset, world_size, rank, shuffle=False, drop_last=True
        )
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size_per_process,
            sampler=sampler,
            drop_last=True,
            pin_memory=True,
        )
        iterator = iter(dataloader)
        for _ in range(args.warmup_steps):
            deepspeed_train_step(engine, next(iterator), device)

        records: list[UnifiedStepMetrics] = []
        strategy = f"zero{args.zero_stage}"
        num_parameters = transformer_parameter_count(model_config)
        for step in range(args.steps):
            local = measure_deepspeed_step(engine, next(iterator), device)
            if not math.isfinite(local.loss):
                raise AssertionError(f"non-finite loss at measured step {step}")
            observed = aggregate_measurement(local, device, world_size)
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
                    model_name=args.model,
                    precision="fp32",
                    optimizer="adamw",
                    strategy=strategy,
                )
                records.append(record)
                print(
                    f"strategy={strategy} step={step} loss={record.loss:.6f} "
                    f"step_time={record.step_time:.6f}s "
                    f"tokens_per_second={record.tokens_per_second:.2f}",
                    flush=True,
                )
        if rank == 0:
            write_unified_metrics_csv(args.output, records)
            print(f"metrics_csv={args.output}", flush=True)
        dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
