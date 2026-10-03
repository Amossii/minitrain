"""Measure one DDP or FSDP2 workload's per-rank CUDA memory."""

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

from minitrain.config import TrainConfig, available_model_configs, get_model_config
from minitrain.data import SyntheticTokenDataset
from minitrain.distributed.fsdp2 import apply_fsdp2
from minitrain.distributed.memory import (
    model_state_bytes,
    optimizer_state_bytes,
    theoretical_adam_bytes_per_rank,
    write_memory_csv,
)
from minitrain.distributed.runtime import cleanup_distributed, init_distributed
from minitrain.model import MiniTransformer, count_parameters
from minitrain.training import train_step


# Inputs are rank-local integer observations. Output takes the maximum because
# the most memory-consuming rank determines whether synchronous training fits.
def reduce_max_int(value: int, device: torch.device) -> int:
    """Return the maximum integer observation across the process group."""

    tensor = torch.tensor(value, dtype=torch.int64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return int(tensor.item())


# CLI inputs define one isolated strategy run. Rank 0 outputs one raw CSV row;
# model, gradients, optimizer state, and allocator peaks are all per-rank maxima.
def main() -> int:
    """Warm up one strategy, measure CUDA memory, and write its raw record."""

    parser = argparse.ArgumentParser(description="Measure DDP or FSDP2 memory.")
    parser.add_argument("--strategy", choices=("ddp", "fsdp2"), required=True)
    parser.add_argument("--backend", choices=("gloo", "nccl"), default="nccl")
    parser.add_argument("--model", choices=available_model_configs(), default="tiny")
    parser.add_argument("--local-batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.backend != "nccl":
        parser.error("GPU memory benchmark requires --backend nccl")
    if args.warmup_steps <= 0:
        parser.error("--warmup-steps must be positive to initialize Adam state")

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
        torch.cuda.manual_seed_all(train_config.seed)
        model = MiniTransformer(model_config).to(context.device)
        num_parameters = count_parameters(model)
        if args.strategy == "ddp":
            train_model = DistributedDataParallel(
                model,
                device_ids=[context.local_rank],
                output_device=context.local_rank,
                broadcast_buffers=False,
            )
        else:
            train_model = apply_fsdp2(model, context.device.type)
        optimizer = torch.optim.AdamW(
            train_model.parameters(), lr=train_config.learning_rate
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
            pin_memory=True,
        )
        iterator = iter(dataloader)

        # Adam allocates moment tensors lazily on the first update. Warmup makes
        # both strategies enter measurement with equivalent persistent state.
        for _ in range(args.warmup_steps):
            train_step(train_model, optimizer, next(iterator), context.device)
        torch.cuda.synchronize(context.device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(context.device)

        for _ in range(train_config.max_steps):
            train_step(train_model, optimizer, next(iterator), context.device)
        torch.cuda.synchronize(context.device)

        parameter_bytes, gradient_bytes = model_state_bytes(train_model)
        optimizer_bytes = optimizer_state_bytes(optimizer)
        observations = {
            "parameter_bytes_per_rank": parameter_bytes,
            "gradient_bytes_per_rank": gradient_bytes,
            "optimizer_state_bytes_per_rank": optimizer_bytes,
            "observed_peak_allocated_bytes": torch.cuda.max_memory_allocated(
                context.device
            ),
            "observed_peak_reserved_bytes": torch.cuda.max_memory_reserved(
                context.device
            ),
            "observed_end_allocated_bytes": torch.cuda.memory_allocated(context.device),
            "observed_end_reserved_bytes": torch.cuda.memory_reserved(context.device),
        }
        maxima = {
            name: reduce_max_int(value, context.device)
            for name, value in observations.items()
        }
        if context.is_main_process:
            row: dict[str, str | int | float] = {
                "strategy": args.strategy,
                "model_name": args.model,
                "precision": "fp32",
                "world_size": context.world_size,
                "local_batch_size": train_config.local_batch_size,
                "global_batch_size": global_batch_size,
                "seq_len": args.seq_len,
                "num_parameters": num_parameters,
                "warmup_steps": args.warmup_steps,
                "measured_steps": train_config.max_steps,
                **maxima,
                "theoretical_model_state_bytes_per_rank": theoretical_adam_bytes_per_rank(
                    num_parameters, context.world_size, args.strategy
                ),
            }
            write_memory_csv(args.output, [row])
            print(
                f"strategy={args.strategy} "
                f"peak_allocated={row['observed_peak_allocated_bytes']} "
                f"peak_reserved={row['observed_peak_reserved_bytes']} "
                f"output={args.output}",
                flush=True,
            )
        dist.barrier()
    finally:
        if context is not None:
            cleanup_distributed()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
