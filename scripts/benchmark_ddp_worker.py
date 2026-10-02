"""Measure one DDP world-size/local-batch configuration and write rank-0 CSV."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from minitrain.config import (
    TrainConfig,
    available_model_configs,
    get_model_config,
)
from minitrain.data import SyntheticTokenDataset
from minitrain.distributed.runtime import (
    cleanup_distributed,
    init_distributed,
)
from minitrain.metrics import StepMetrics, write_metrics_csv
from minitrain.model import MiniTransformer, count_parameters
from minitrain.training import (
    StepMeasurement,
    measure_train_step,
    train_step,
)


# Input is one rank's observed step. Output uses mean loss and maximum timing/
# memory across ranks, matching the slowest worker that controls synchronous DDP.
def aggregate_measurement(
    measurement: StepMeasurement, device: torch.device, world_size: int
) -> StepMeasurement:
    """Aggregate one measurement into globally meaningful DDP observations."""

    loss = torch.tensor([measurement.loss], dtype=torch.float64, device=device)
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


def main() -> int:
    """Run one warmup/measurement DDP job selected by the scaling orchestrator."""

    parser = argparse.ArgumentParser(description="Measure one DDP configuration.")
    parser.add_argument("--backend", choices=("gloo", "nccl"), default="nccl")
    parser.add_argument("--model", choices=available_model_configs(), default="tiny")
    parser.add_argument("--local-batch-size", type=int, required=True)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.warmup_steps < 0:
        parser.error("--warmup-steps must be non-negative")

    model_config = get_model_config(args.model)
    if args.seq_len > model_config.seq_len:
        parser.error(f"--seq-len exceeds the {args.model} maximum")
    train_config = TrainConfig(
        local_batch_size=args.local_batch_size,
        learning_rate=args.learning_rate,
        max_steps=args.steps,
        precision="fp32",
        seed=args.seed,
    )

    context = None
    try:
        context = init_distributed(backend=args.backend)
        torch.manual_seed(train_config.seed)
        model = MiniTransformer(model_config).to(context.device)
        if context.device.type == "cuda":
            ddp_model = DistributedDataParallel(
                model,
                device_ids=[context.local_rank],
                output_device=context.local_rank,
                broadcast_buffers=False,
            )
        else:
            ddp_model = DistributedDataParallel(model, broadcast_buffers=False)
        optimizer = torch.optim.AdamW(
            ddp_model.parameters(), lr=train_config.learning_rate
        )

        total_steps = args.warmup_steps + train_config.max_steps
        global_batch_size = train_config.global_batch_size(context.world_size)
        dataset = SyntheticTokenDataset(
            global_batch_size * total_steps,
            args.seq_len,
            model_config.vocab_size,
            train_config.seed,
        )
        sampler = DistributedSampler(
            dataset, context.world_size, context.rank, shuffle=False, drop_last=True
        )
        dataloader = DataLoader(
            dataset,
            batch_size=train_config.local_batch_size,
            sampler=sampler,
            drop_last=True,
            pin_memory=context.device.type == "cuda",
        )
        iterator = iter(dataloader)
        for _ in range(args.warmup_steps):
            train_step(ddp_model, optimizer, next(iterator), context.device)

        parameter_count = count_parameters(model)
        records: list[StepMetrics] = []
        for step in range(train_config.max_steps):
            local = measure_train_step(
                ddp_model, optimizer, next(iterator), context.device
            )
            observed = aggregate_measurement(local, context.device, context.world_size)
            if context.is_main_process:
                record = StepMetrics.from_measurement(
                    step=step,
                    loss=observed.loss,
                    step_time=observed.step_time,
                    forward_time=observed.forward_time,
                    backward_time=observed.backward_time,
                    optimizer_time=observed.optimizer_time,
                    peak_memory_allocated=observed.peak_memory_allocated,
                    peak_memory_reserved=observed.peak_memory_reserved,
                    world_size=context.world_size,
                    local_batch_size=train_config.local_batch_size,
                    seq_len=args.seq_len,
                    num_parameters=parameter_count,
                    model_name=args.model,
                    precision="fp32",
                    strategy="ddp",
                )
                records.append(record)
                print(
                    f"step={step} world_size={context.world_size} "
                    f"step_time={record.step_time:.6f}s "
                    f"tokens_per_second={record.tokens_per_second:.2f}",
                    flush=True,
                )
        if context.is_main_process:
            write_metrics_csv(args.output, records)
            print(f"metrics_csv={args.output}", flush=True)
        dist.barrier()
    finally:
        if context is not None:
            cleanup_distributed()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
