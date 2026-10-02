"""Verify DDP gradient/parameter synchronization against a global-batch reference."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.distributed as dist
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DistributedSampler

from minitrain.config import ModelConfig
from minitrain.data import SyntheticTokenDataset
from minitrain.distributed.runtime import (
    cleanup_distributed,
    init_distributed,
)
from minitrain.model import MiniTransformer
from minitrain.training import causal_lm_loss


# This model uses the production Transformer implementation with smaller
# dimensions so rank 0 can also hold a reference model during correctness tests.
def correctness_model_config(seq_len: int) -> ModelConfig:
    """Return the compact Transformer configuration used for DDP verification."""

    return ModelConfig(
        vocab_size=64,
        seq_len=seq_len,
        hidden_size=32,
        num_layers=2,
        num_heads=4,
        intermediate_size=96,
    )


# Inputs are deterministic dataset indices. Output is the same `[B,T]` batch
# dictionary consumed by training, making local and global sample sets explicit.
def stack_dataset_indices(
    dataset: SyntheticTokenDataset,
    indices: list[int],
) -> dict[str, Tensor]:
    """Stack selected dataset samples into one batch without a DataLoader."""

    if not indices:
        raise ValueError("indices cannot be empty")
    samples = [dataset[index] for index in indices]
    return {
        "input_ids": torch.stack([sample["input_ids"] for sample in samples]),
        "labels": torch.stack([sample["labels"] for sample in samples]),
    }


# Every named tensor is compared against a temporary copy broadcast from rank 0.
# Processing one tensor at a time avoids allocating a second flattened model.
def assert_tensors_match_rank_zero(
    named_tensors: list[tuple[str, Tensor]],
    rank: int,
    *,
    atol: float = 1e-6,
    rtol: float = 1e-5,
) -> None:
    """Assert each rank's tensors are close to rank 0 tensors."""

    for name, tensor in named_tensors:
        expected = torch.empty_like(tensor)
        if rank == 0:
            expected.copy_(tensor.detach())
        dist.broadcast(expected, src=0)
        torch.testing.assert_close(
            tensor.detach(), expected, atol=atol, rtol=rtol, msg=lambda _: name
        )


# Inputs select backend and one equal-sized local batch. Output is PASS markers
# for initial parameters, reduced gradients, updated replicas, and reference.
def main() -> int:
    """Run one DDP update and prove its replicated/global-batch equivalence."""

    parser = argparse.ArgumentParser(description="Verify DDP correctness.")
    parser.add_argument("--backend", choices=("gloo", "nccl"), default="nccl")
    parser.add_argument("--local-batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.local_batch_size <= 0:
        parser.error("--local-batch-size must be positive")
    if args.seq_len <= 0:
        parser.error("--seq-len must be positive")
    if args.learning_rate <= 0:
        parser.error("--learning-rate must be positive")

    context = None
    try:
        context = init_distributed(backend=args.backend)
        config = correctness_model_config(args.seq_len)

        # Different seeds make rank 1 start differently on purpose. DDP's
        # constructor must overwrite it with rank 0's parameter state.
        torch.manual_seed(args.seed + context.rank)
        model = MiniTransformer(config).to(context.device)
        if context.device.type == "cuda":
            ddp_model = DistributedDataParallel(
                model,
                device_ids=[context.local_rank],
                output_device=context.local_rank,
                broadcast_buffers=False,
            )
        else:
            ddp_model = DistributedDataParallel(model, broadcast_buffers=False)

        assert_tensors_match_rank_zero(
            list(ddp_model.module.named_parameters()), context.rank
        )
        if context.is_main_process:
            print("initial_parameter_sync=PASS", flush=True)

        initial_state = None
        if context.is_main_process:
            initial_state = {
                name: tensor.detach().clone()
                for name, tensor in ddp_model.module.state_dict().items()
            }

        global_batch_size = args.local_batch_size * context.world_size
        dataset = SyntheticTokenDataset(
            num_samples=global_batch_size,
            seq_len=args.seq_len,
            vocab_size=config.vocab_size,
            seed=args.seed,
        )
        sampler = DistributedSampler(
            dataset,
            num_replicas=context.world_size,
            rank=context.rank,
            shuffle=False,
            drop_last=True,
        )
        local_indices = list(iter(sampler))
        local_batch = stack_dataset_indices(dataset, local_indices)
        input_ids = local_batch["input_ids"].to(context.device)
        labels = local_batch["labels"].to(context.device)

        optimizer = torch.optim.SGD(ddp_model.parameters(), lr=args.learning_rate)
        optimizer.zero_grad(set_to_none=True)
        local_loss = causal_lm_loss(ddp_model(input_ids), labels)
        mean_local_loss = local_loss.detach().clone()
        dist.all_reduce(mean_local_loss, op=dist.ReduceOp.SUM)
        mean_local_loss /= context.world_size
        local_loss.backward()

        named_gradients = []
        for name, parameter in ddp_model.module.named_parameters():
            if parameter.grad is None:
                raise AssertionError(f"missing synchronized gradient for {name}")
            named_gradients.append((name, parameter.grad))
        assert_tensors_match_rank_zero(named_gradients, context.rank)
        if context.is_main_process:
            print("gradient_sync=PASS", flush=True)

        optimizer.step()
        assert_tensors_match_rank_zero(
            list(ddp_model.module.named_parameters()), context.rank
        )
        if context.is_main_process:
            print("updated_parameter_sync=PASS", flush=True)

        # Rank 0 repeats the update without DDP using the union of all local
        # samples. Equal local batch sizes make averaged DDP gradients identical
        # to mean loss over this global batch, up to floating-point order.
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

            for (ddp_name, ddp_parameter), (ref_name, ref_parameter) in zip(
                ddp_model.module.named_parameters(),
                reference_model.named_parameters(),
                strict=True,
            ):
                if ddp_name != ref_name:
                    raise AssertionError(
                        f"parameter order mismatch: {ddp_name} != {ref_name}"
                    )
                torch.testing.assert_close(
                    ddp_parameter,
                    ref_parameter,
                    atol=1e-6,
                    rtol=1e-5,
                    msg=lambda _: ddp_name,
                )
            print(
                f"global_batch_reference=PASS mean_local_loss={mean_local_loss.item():.6f} "
                f"reference_loss={reference_loss.item():.6f}",
                flush=True,
            )

        dist.barrier()
        if context.is_main_process:
            print("DDP CORRECTNESS CHECKS PASSED", flush=True)
    finally:
        if context is not None:
            cleanup_distributed()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
