"""Verify FSDP2 shards and one update against a global-batch reference."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.distributed as dist
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
)
from torch.distributed.tensor import DTensor, Shard
from torch.utils.data import DistributedSampler

from minitrain.config import ModelConfig
from minitrain.data import SyntheticTokenDataset
from minitrain.distributed.fsdp2 import apply_fsdp2, inspect_parameter_shards
from minitrain.distributed.runtime import cleanup_distributed, init_distributed
from minitrain.model import MiniTransformer
from minitrain.training import causal_lm_loss
from scripts.verify_ddp_correctness import stack_dataset_indices


# Input is sequence length. Output keeps the real Transformer structure small
# enough for rank 0 to hold both sharded and reference models during verification.
def correctness_model_config(seq_len: int) -> ModelConfig:
    """Return the compact Transformer used for FSDP2 correctness checks."""

    return ModelConfig(
        vocab_size=64,
        seq_len=seq_len,
        hidden_size=32,
        num_layers=2,
        num_heads=4,
        intermediate_size=96,
    )


# Inputs are model shards and world size. The function collectively proves that
# local ownership covers every global parameter exactly once across FSDP ranks.
def assert_parameter_sharding(model, world_size: int, device: torch.device) -> None:
    """Assert parameters use Shard(0) and local sizes cover global tensors."""

    for info, (_, parameter) in zip(
        inspect_parameter_shards(model), model.named_parameters(), strict=True
    ):
        if not isinstance(parameter, DTensor):
            raise AssertionError(f"{info.name} is not a DTensor")
        if tuple(parameter.placements) != (Shard(0),):
            raise AssertionError(
                f"{info.name} expected Shard(0), got {parameter.placements}"
            )
        local_numel = torch.tensor(info.local_numel, dtype=torch.int64, device=device)
        dist.all_reduce(local_numel, op=dist.ReduceOp.SUM)
        if local_numel.item() != info.global_numel:
            raise AssertionError(
                f"{info.name} local shards sum to {local_numel.item()}, "
                f"expected {info.global_numel}"
            )
    if world_size <= 1:
        raise AssertionError("FSDP2 correctness requires at least two ranks")


# Inputs select one deterministic two-or-more-rank workload. Outputs are PASS
# markers for parameter shards, gradient shards, loss, and global update parity.
def main() -> int:
    """Run one FSDP2 update and compare it with a global-batch reference."""

    parser = argparse.ArgumentParser(description="Verify FSDP2 correctness.")
    parser.add_argument("--backend", choices=("gloo", "nccl"), default="nccl")
    parser.add_argument("--local-batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.local_batch_size <= 0 or args.seq_len <= 0 or args.learning_rate <= 0:
        parser.error("batch size, sequence length, and learning rate must be positive")

    context = None
    try:
        context = init_distributed(backend=args.backend)
        config = correctness_model_config(args.seq_len)
        torch.manual_seed(args.seed)
        model = MiniTransformer(config)
        initial_state = None
        if context.is_main_process:
            initial_state = {
                name: tensor.detach().clone()
                for name, tensor in model.state_dict().items()
            }
        model.to(context.device)
        fsdp_model = apply_fsdp2(model, context.device.type)
        assert_parameter_sharding(fsdp_model, context.world_size, context.device)
        if context.is_main_process:
            print("parameter_sharding=PASS", flush=True)

        global_batch_size = args.local_batch_size * context.world_size
        dataset = SyntheticTokenDataset(
            global_batch_size, args.seq_len, config.vocab_size, args.seed
        )
        sampler = DistributedSampler(
            dataset,
            context.world_size,
            context.rank,
            shuffle=False,
            drop_last=True,
        )
        local_batch = stack_dataset_indices(dataset, list(iter(sampler)))
        input_ids = local_batch["input_ids"].to(context.device)
        labels = local_batch["labels"].to(context.device)

        optimizer = torch.optim.SGD(fsdp_model.parameters(), lr=args.learning_rate)
        optimizer.zero_grad(set_to_none=True)
        local_loss = causal_lm_loss(fsdp_model(input_ids), labels)
        mean_local_loss = local_loss.detach().clone()
        dist.all_reduce(mean_local_loss, op=dist.ReduceOp.SUM)
        mean_local_loss /= context.world_size
        local_loss.backward()

        for name, parameter in fsdp_model.named_parameters():
            if parameter.grad is None:
                raise AssertionError(f"missing gradient for {name}")
            if not isinstance(parameter.grad, DTensor):
                raise AssertionError(f"gradient {name} is not a sharded DTensor")
            if tuple(parameter.grad.placements) != (Shard(0),):
                raise AssertionError(
                    f"gradient {name} expected Shard(0), got {parameter.grad.placements}"
                )
        if context.is_main_process:
            print("gradient_sharding=PASS", flush=True)
        optimizer.step()

        # Full-state gathering is collective. CPU offload intentionally returns
        # tensors only on rank 0, which alone owns the unsharded reference model.
        full_state = get_model_state_dict(
            fsdp_model,
            options=StateDictOptions(full_state_dict=True, cpu_offload=True),
        )
        if context.is_main_process:
            assert initial_state is not None
            reference_model = MiniTransformer(config).to(context.device)
            reference_model.load_state_dict(initial_state)
            reference_optimizer = torch.optim.SGD(
                reference_model.parameters(), lr=args.learning_rate
            )
            global_batch = stack_dataset_indices(
                dataset, list(range(global_batch_size))
            )
            global_input_ids = global_batch["input_ids"].to(context.device)
            global_labels = global_batch["labels"].to(context.device)
            reference_optimizer.zero_grad(set_to_none=True)
            reference_loss = causal_lm_loss(
                reference_model(global_input_ids), global_labels
            )
            torch.testing.assert_close(
                mean_local_loss, reference_loss.detach(), atol=1e-6, rtol=1e-5
            )
            reference_loss.backward()
            reference_optimizer.step()

            reference_state = reference_model.state_dict()
            if full_state.keys() != reference_state.keys():
                raise AssertionError("FSDP2 and reference state keys differ")
            for name, tensor in full_state.items():
                torch.testing.assert_close(
                    tensor,
                    reference_state[name].detach().cpu(),
                    atol=1e-5,
                    rtol=1e-4,
                    msg=lambda _: name,
                )
            print(
                f"global_batch_reference=PASS mean_local_loss={mean_local_loss.item():.6f} "
                f"reference_loss={reference_loss.item():.6f}",
                flush=True,
            )
        dist.barrier()
        if context.is_main_process:
            print("FSDP2 CORRECTNESS CHECKS PASSED", flush=True)
    finally:
        if context is not None:
            cleanup_distributed()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
