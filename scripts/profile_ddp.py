"""Capture per-rank PyTorch Profiler traces for DDP training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.profiler import profile, record_function
from torch.utils.data import DataLoader, DistributedSampler

from minitrain.config import available_model_configs, get_model_config
from minitrain.data import SyntheticTokenDataset
from minitrain.distributed.profiling import (
    profile_output_paths,
    profiler_activities,
)
from minitrain.distributed.runtime import cleanup_distributed, init_distributed
from minitrain.model import MiniTransformer
from minitrain.training import causal_lm_loss, train_step


# Input is private DDP diagnostic data containing scalar and occasionally byte
# values. Output is JSON-safe metadata used to inspect actual reducer buckets.
def make_json_safe(values: dict[str, Any]) -> dict[str, Any]:
    """Convert DDP logging values into JSON-serializable primitives."""

    result: dict[str, Any] = {}
    for key, value in values.items():
        if isinstance(value, bytes):
            result[key] = value.decode("utf-8", errors="replace")
        elif isinstance(value, (str, int, float, bool)) or value is None:
            result[key] = value
        else:
            result[key] = str(value)
    return result


# Inputs are a DDP model, optimizer, batch, and rank device. Output is one loss;
# the named regions expose when backward launches bucket AllReduce operations.
def profiled_train_step(
    model: DistributedDataParallel,
    optimizer: torch.optim.Optimizer,
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> float:
    """Run one optimizer update with semantic profiler annotations."""

    with record_function("minitrain/data_to_device"):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
    optimizer.zero_grad(set_to_none=True)
    with record_function("minitrain/forward"):
        logits = model(input_ids)
    with record_function("minitrain/loss"):
        loss = causal_lm_loss(logits, labels)
    with record_function("minitrain/backward"):
        loss.backward()
    with record_function("minitrain/optimizer"):
        optimizer.step()
    return float(loss.detach())


# CLI inputs define one fixed DDP workload. Outputs are per-rank traces,
# operator tables, and reducer metadata; model state advances once per step.
def main() -> int:
    """Warm up DDP, profile training steps, and save rank-local artifacts."""

    parser = argparse.ArgumentParser(description="Profile DDP communication overlap.")
    parser.add_argument("--backend", choices=("gloo", "nccl"), default="nccl")
    parser.add_argument("--model", choices=available_model_configs(), default="tiny")
    parser.add_argument("--local-batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--profile-steps", type=int, default=3)
    parser.add_argument("--bucket-cap-mb", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/raw/profiles/ddp")
    )
    args = parser.parse_args()
    if args.local_batch_size <= 0 or args.profile_steps <= 0:
        parser.error("--local-batch-size and --profile-steps must be positive")
    if args.warmup_steps < 0 or args.bucket_cap_mb <= 0:
        parser.error("--warmup-steps must be non-negative and --bucket-cap-mb positive")

    model_config = get_model_config(args.model)
    if args.seq_len > model_config.seq_len:
        parser.error(f"--seq-len exceeds the {args.model} maximum")

    context = None
    try:
        context = init_distributed(backend=args.backend)
        torch.manual_seed(args.seed)
        model = MiniTransformer(model_config).to(context.device)
        ddp_kwargs: dict[str, Any] = {
            "broadcast_buffers": False,
            "bucket_cap_mb": args.bucket_cap_mb,
        }
        if context.device.type == "cuda":
            ddp_kwargs.update(
                device_ids=[context.local_rank], output_device=context.local_rank
            )
        ddp_model = DistributedDataParallel(model, **ddp_kwargs)
        optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=args.learning_rate)

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

        # Warmup excludes initialization and lets DDP observe gradient readiness
        # before the trace used to reason about steady-state bucket overlap.
        for _ in range(args.warmup_steps):
            train_step(ddp_model, optimizer, next(iterator), context.device)
        dist.barrier()

        with profile(
            activities=profiler_activities(context.device),
            record_shapes=True,
            profile_memory=True,
        ) as profiler:
            losses = []
            for _ in range(args.profile_steps):
                losses.append(
                    profiled_train_step(
                        ddp_model, optimizer, next(iterator), context.device
                    )
                )
                profiler.step()

        args.output_dir.mkdir(parents=True, exist_ok=True)
        paths = profile_output_paths(args.output_dir, context.rank)
        profiler.export_chrome_trace(str(paths.trace))
        sort_key = (
            "self_cuda_time_total"
            if context.device.type == "cuda"
            else "self_cpu_time_total"
        )
        paths.operators.write_text(
            profiler.key_averages().table(sort_by=sort_key, row_limit=100),
            encoding="utf-8",
        )
        logging_data = make_json_safe(ddp_model._get_ddp_logging_data())
        paths.ddp_logging.write_text(
            json.dumps(logging_data, indent=2, sort_keys=True), encoding="utf-8"
        )
        print(
            f"rank={context.rank} mean_local_loss={sum(losses) / len(losses):.6f} "
            f"trace={paths.trace} operators={paths.operators} "
            f"ddp_logging={paths.ddp_logging}",
            flush=True,
        )
        dist.barrier()
    finally:
        if context is not None:
            cleanup_distributed()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
