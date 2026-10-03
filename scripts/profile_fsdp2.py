"""采集 FSDP2 的 AllGather、ReduceScatter 与计算时间线。"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FSDPModule
from torch.profiler import profile, record_function
from torch.utils.data import DataLoader, DistributedSampler

from minitrain.config import available_model_configs, get_model_config
from minitrain.data import SyntheticTokenDataset
from minitrain.distributed.fsdp2 import apply_fsdp2, inspect_parameter_shards
from minitrain.distributed.fsdp2_profiling import fsdp2_profile_paths
from minitrain.distributed.profiling import (
    communication_event_names,
    profiler_activities,
)
from minitrain.distributed.runtime import cleanup_distributed, init_distributed
from minitrain.model import MiniTransformer
from minitrain.training import causal_lm_loss, train_step


# 输入是 FSDP2 模型、optimizer、一个 batch 和 rank device；输出是本地 loss。
# 人工区间用于把 collective 放回训练生命周期中理解，而不是替代框架事件。
def profiled_fsdp2_step(
    model: FSDPModule,
    optimizer: torch.optim.Optimizer,
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> float:
    """执行一次带语义标记的 FSDP2 参数更新。"""

    with record_function("minitrain/data_to_device"):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
    optimizer.zero_grad(set_to_none=True)
    with record_function("minitrain/fsdp2_forward"):
        logits = model(input_ids)
    with record_function("minitrain/loss"):
        loss = causal_lm_loss(logits, labels)
    with record_function("minitrain/fsdp2_backward"):
        loss.backward()
    with record_function("minitrain/optimizer"):
        optimizer.step()
    return float(loss.detach())


# CLI 输入固定一个 FSDP2 workload；输出每个 rank 的 trace、算子表、分片信息和
# 通信事件索引。每次 step 都会真实更新本地参数与 optimizer shard。
def main() -> int:
    """预热 FSDP2，采集稳态训练时间线并保存分析产物。"""

    parser = argparse.ArgumentParser(description="Profile FSDP2 collectives.")
    parser.add_argument("--backend", choices=("gloo", "nccl"), default="nccl")
    parser.add_argument("--model", choices=available_model_configs(), default="tiny")
    parser.add_argument("--local-batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--profile-steps", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/raw/profiles/fsdp2")
    )
    args = parser.parse_args()
    if args.local_batch_size <= 0 or args.profile_steps <= 0:
        parser.error("--local-batch-size and --profile-steps must be positive")
    if args.warmup_steps < 0:
        parser.error("--warmup-steps must be non-negative")

    model_config = get_model_config(args.model)
    if args.seq_len > model_config.seq_len:
        parser.error(f"--seq-len exceeds the {args.model} maximum")

    context = None
    try:
        context = init_distributed(backend=args.backend)
        torch.manual_seed(args.seed)
        if context.device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed)
        model = MiniTransformer(model_config).to(context.device)
        fsdp_model = apply_fsdp2(model, context.device.type)
        optimizer = torch.optim.AdamW(fsdp_model.parameters(), lr=args.learning_rate)

        total_steps = args.warmup_steps + args.profile_steps
        global_batch_size = args.local_batch_size * context.world_size
        dataset = SyntheticTokenDataset(
            global_batch_size * total_steps,
            args.seq_len,
            model_config.vocab_size,
            args.seed,
        )
        sampler = DistributedSampler(
            dataset, context.world_size, context.rank, shuffle=False, drop_last=True
        )
        dataloader = DataLoader(
            dataset,
            batch_size=args.local_batch_size,
            sampler=sampler,
            drop_last=True,
            pin_memory=context.device.type == "cuda",
        )
        iterator = iter(dataloader)

        # 预热排除初始化、首次编译路径和 Adam 状态创建，使 trace 更接近稳态训练。
        for _ in range(args.warmup_steps):
            train_step(fsdp_model, optimizer, next(iterator), context.device)
        dist.barrier()

        with profile(
            activities=profiler_activities(context.device),
            record_shapes=True,
            profile_memory=True,
        ) as profiler:
            losses = []
            for _ in range(args.profile_steps):
                losses.append(
                    profiled_fsdp2_step(
                        fsdp_model, optimizer, next(iterator), context.device
                    )
                )
                profiler.step()

        args.output_dir.mkdir(parents=True, exist_ok=True)
        paths = fsdp2_profile_paths(args.output_dir, context.rank)
        profiler.export_chrome_trace(str(paths.trace))
        sort_key = (
            "self_cuda_time_total"
            if context.device.type == "cuda"
            else "self_cpu_time_total"
        )
        averages = profiler.key_averages()
        paths.operators.write_text(
            averages.table(sort_by=sort_key, row_limit=100), encoding="utf-8"
        )

        # shard metadata 不触发参数 AllGather，可用于核对 trace 中通信组的参数规模。
        shard_rows = [asdict(item) for item in inspect_parameter_shards(fsdp_model)]
        paths.shards.write_text(
            json.dumps(shard_rows, indent=2, sort_keys=True), encoding="utf-8"
        )
        communication_names = communication_event_names(
            sorted({event.key for event in averages})
        )
        paths.communications.write_text(
            json.dumps(communication_names, indent=2), encoding="utf-8"
        )
        print(
            f"rank={context.rank} mean_local_loss={sum(losses) / len(losses):.6f} "
            f"communication_events={len(communication_names)} trace={paths.trace}",
            flush=True,
        )
        dist.barrier()
    finally:
        if context is not None:
            cleanup_distributed()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
