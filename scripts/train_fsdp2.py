"""Train MiniTransformer with native composable PyTorch FSDP2."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from minitrain.config import TrainConfig, available_model_configs, get_model_config
from minitrain.data import SyntheticTokenDataset
from minitrain.distributed.fsdp2 import apply_fsdp2, inspect_parameter_shards
from minitrain.distributed.runtime import cleanup_distributed, init_distributed
from minitrain.model import MiniTransformer, count_parameters
from minitrain.training import train_step


# CLI inputs define one fully-sharded data-parallel workload. Each call advances
# local parameter, gradient, and Adam shards while reporting global mean loss.
def main() -> int:
    """Initialize FSDP2 and execute explicit forward/backward/update steps."""

    parser = argparse.ArgumentParser(description="Train MiniTransformer with FSDP2.")
    parser.add_argument("--backend", choices=("gloo", "nccl"), default="nccl")
    parser.add_argument("--model", choices=available_model_configs(), default="tiny")
    parser.add_argument("--local-batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    model_config = get_model_config(args.model)
    if args.seq_len > model_config.seq_len:
        parser.error(
            f"--seq-len cannot exceed the {args.model} maximum of {model_config.seq_len}"
        )
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
        # FSDP2 does not use DDP's constructor-time parameter broadcast. Every
        # rank therefore starts from the same seed before parameters are sharded.
        torch.manual_seed(train_config.seed)
        if context.device.type == "cuda":
            torch.cuda.manual_seed_all(train_config.seed)
        model = MiniTransformer(model_config).to(context.device)
        num_parameters = count_parameters(model)
        fsdp_model = apply_fsdp2(model, context.device.type)
        shard_info = inspect_parameter_shards(fsdp_model)
        optimizer = torch.optim.AdamW(
            fsdp_model.parameters(), lr=train_config.learning_rate
        )

        global_batch_size = train_config.global_batch_size(context.world_size)
        dataset = SyntheticTokenDataset(
            num_samples=global_batch_size * train_config.max_steps,
            seq_len=args.seq_len,
            vocab_size=model_config.vocab_size,
            seed=train_config.seed,
        )
        sampler = DistributedSampler(
            dataset,
            num_replicas=context.world_size,
            rank=context.rank,
            shuffle=False,
            drop_last=True,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=train_config.local_batch_size,
            sampler=sampler,
            drop_last=True,
            pin_memory=context.device.type == "cuda",
        )

        local_parameter_numel = sum(item.local_numel for item in shard_info)
        print(
            f"rank={context.rank} device={context.device} "
            f"local_parameter_numel={local_parameter_numel} "
            f"first_shard={shard_info[0]}",
            flush=True,
        )
        if context.is_main_process:
            print(
                f"strategy=fsdp2 world_size={context.world_size} "
                f"local_batch_size={train_config.local_batch_size} "
                f"global_batch_size={global_batch_size} parameters={num_parameters:,}",
                flush=True,
            )

        for step, batch in enumerate(dataloader):
            local_loss = train_step(fsdp_model, optimizer, batch, context.device)
            # This collective only aggregates logging. FSDP2 already performed
            # ReduceScatter on gradients during backward.
            mean_loss = torch.tensor(local_loss, device=context.device)
            dist.all_reduce(mean_loss, op=dist.ReduceOp.SUM)
            mean_loss /= context.world_size
            if context.is_main_process:
                print(f"step={step} mean_loss={mean_loss.item():.6f}", flush=True)
    finally:
        if context is not None:
            cleanup_distributed()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
