"""Verify fundamental torch.distributed collectives with inspectable tensors."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

from minitrain.distributed.runtime import (  # noqa: E402
    DistributedContext,
    cleanup_distributed,
    init_distributed,
)


# Every rank contributes rank + 1, so this closed form is the expected SUM for
# Reduce and AllReduce under any positive world size.
def expected_rank_sum(world_size: int) -> float:
    """Return 1 + 2 + ... + world_size as a floating-point value."""

    return float(world_size * (world_size + 1) // 2)


# Input describes this rank's output chunk. The returned tensor is what remains
# after summing all full inputs and scattering consecutive chunks by rank.
def expected_reduce_scatter_chunk(
    rank: int,
    world_size: int,
    chunk_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Compute the expected ReduceScatter SUM output for one rank."""

    start = rank * chunk_size
    positions = torch.arange(start, start + chunk_size, device=device, dtype=torch.float32)
    rank_offsets = 10.0 * (world_size * (world_size - 1) / 2)
    return world_size * positions + rank_offsets


# Broadcast input is meaningful only on rank 0. Output is the source tensor on
# every rank, demonstrating one-to-all replication without reduction.
def check_broadcast(context: DistributedContext) -> None:
    """Verify rank 0 broadcasts `[10, 20]` to every worker."""

    tensor = torch.tensor(
        [10.0, 20.0] if context.rank == 0 else [-1.0, -1.0],
        device=context.device,
    )
    dist.broadcast(tensor, src=0)
    torch.testing.assert_close(
        tensor, torch.tensor([10.0, 20.0], device=context.device)
    )
    print(f"rank={context.rank} broadcast={tensor.tolist()} PASS", flush=True)


# All ranks contribute one scalar. Only destination rank 0 may rely on the
# reduced output; non-destination buffers are deliberately not asserted.
def check_reduce(context: DistributedContext) -> None:
    """Verify Reduce SUM places the rank-value sum only on rank 0."""

    tensor = torch.tensor([float(context.rank + 1)], device=context.device)
    dist.reduce(tensor, dst=0, op=dist.ReduceOp.SUM)
    if context.rank == 0:
        expected = torch.tensor(
            [expected_rank_sum(context.world_size)], device=context.device
        )
        torch.testing.assert_close(tensor, expected)
        print(f"rank=0 reduce={tensor.tolist()} PASS", flush=True)
    else:
        print(f"rank={context.rank} reduce=destination-rank-only PASS", flush=True)


# AllReduce combines reduction and replication: every rank receives the same
# sum, which later explains how DDP synchronizes replicated gradients.
def check_all_reduce(context: DistributedContext) -> None:
    """Verify AllReduce SUM returns the rank-value sum on every worker."""

    tensor = torch.tensor([float(context.rank + 1)], device=context.device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    expected = torch.tensor(
        [expected_rank_sum(context.world_size)], device=context.device
    )
    torch.testing.assert_close(tensor, expected)
    print(f"rank={context.rank} all_reduce={tensor.tolist()} PASS", flush=True)


# Each rank owns [rank, rank + 10]. AllGather returns world_size chunks ordered
# by source rank, increasing total data held by every rank.
def check_all_gather(context: DistributedContext) -> None:
    """Verify every worker gathers one two-element tensor from every rank."""

    local_tensor = torch.tensor(
        [float(context.rank), float(context.rank + 10)], device=context.device
    )
    gathered = [torch.empty_like(local_tensor) for _ in range(context.world_size)]
    dist.all_gather(gathered, local_tensor)
    expected = [
        torch.tensor([float(rank), float(rank + 10)], device=context.device)
        for rank in range(context.world_size)
    ]
    for actual_chunk, expected_chunk in zip(gathered, expected, strict=True):
        torch.testing.assert_close(actual_chunk, expected_chunk)
    values = [chunk.tolist() for chunk in gathered]
    print(f"rank={context.rank} all_gather={values} PASS", flush=True)


# Every rank supplies a full `[world_size * chunk_size]` input. SUM happens
# elementwise, then rank r receives consecutive chunk r of the reduced tensor.
def check_reduce_scatter(context: DistributedContext) -> None:
    """Verify ReduceScatter SUM returns the correct chunk on each worker."""

    chunk_size = 2
    input_tensor = torch.arange(
        context.world_size * chunk_size,
        device=context.device,
        dtype=torch.float32,
    )
    input_tensor = input_tensor + 10.0 * context.rank
    output = torch.empty(chunk_size, device=context.device)
    dist.reduce_scatter_tensor(output, input_tensor, op=dist.ReduceOp.SUM)
    expected = expected_reduce_scatter_chunk(
        context.rank, context.world_size, chunk_size, context.device
    )
    torch.testing.assert_close(output, expected)
    print(f"rank={context.rank} reduce_scatter={output.tolist()} PASS", flush=True)


# Barrier has no tensor result. Returning proves every worker reached this point
# before any worker printed its post-barrier success marker.
def check_barrier(context: DistributedContext) -> None:
    """Verify all workers can enter and leave a process-group barrier."""

    print(f"rank={context.rank} barrier=entered", flush=True)
    dist.barrier()
    print(f"rank={context.rank} barrier=exited PASS", flush=True)


# Input selects Gloo or NCCL. Output consists of per-rank, human-inspectable
# tensors plus assertions that fail the distributed job on any wrong result.
def main() -> int:
    """Initialize the group and execute all collective correctness checks."""

    parser = argparse.ArgumentParser(description="Verify distributed collectives.")
    parser.add_argument(
        "--backend", choices=("auto", "gloo", "nccl"), default="auto"
    )
    args = parser.parse_args()

    context = None
    try:
        context = init_distributed(backend=args.backend)
        if context.world_size < 2:
            raise RuntimeError("collective correctness requires WORLD_SIZE >= 2")

        check_broadcast(context)
        check_reduce(context)
        check_all_reduce(context)
        check_all_gather(context)
        check_reduce_scatter(context)
        check_barrier(context)
        if context.is_main_process:
            print("ALL COLLECTIVE CORRECTNESS CHECKS PASSED", flush=True)
    finally:
        if context is not None:
            cleanup_distributed()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
