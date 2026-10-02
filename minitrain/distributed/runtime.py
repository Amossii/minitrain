"""Minimal torchrun process-group initialization and device mapping."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import timedelta
import os

import torch
import torch.distributed as dist


@dataclass(frozen=True, slots=True)
class DistributedContext:
    """Describe one worker's identity inside a distributed process group.

    Inputs originate from torchrun and backend resolution. The immutable state
    identifies this process; consumers use the output device for local tensor
    allocation and rank/world size for future collective correctness checks.
    """

    rank: int
    local_rank: int
    world_size: int
    backend: str
    device: torch.device

    @property
    def is_main_process(self) -> bool:
        """Return whether this worker is global rank zero."""

        return self.rank == 0


# Input is normally os.environ; output is validated integer process identity.
# Keeping parsing separate allows CPU unit tests without opening network ports.
def read_torchrun_environment(
    environment: Mapping[str, str] | None = None,
) -> tuple[int, int, int]:
    """Read RANK, LOCAL_RANK, and WORLD_SIZE injected by torchrun."""

    source = os.environ if environment is None else environment
    required = ("RANK", "LOCAL_RANK", "WORLD_SIZE")
    missing = [name for name in required if name not in source]
    if missing:
        raise RuntimeError(
            "missing torchrun environment variables: "
            f"{', '.join(missing)}; launch this program with torchrun"
        )

    try:
        rank = int(source["RANK"])
        local_rank = int(source["LOCAL_RANK"])
        world_size = int(source["WORLD_SIZE"])
    except ValueError as exc:
        raise RuntimeError("torchrun rank variables must be integers") from exc

    if rank < 0 or local_rank < 0:
        raise RuntimeError("RANK and LOCAL_RANK must be non-negative")
    if world_size <= 0:
        raise RuntimeError("WORLD_SIZE must be positive")
    if rank >= world_size:
        raise RuntimeError(f"RANK {rank} must be smaller than WORLD_SIZE {world_size}")
    return rank, local_rank, world_size


# Backend selection maps NCCL workers to local CUDA devices and Gloo workers to
# CPU. Global rank is deliberately not used for device selection: on multiple
# nodes only LOCAL_RANK identifies a GPU within the current host.
def resolve_distributed_device(
    backend: str,
    local_rank: int,
) -> tuple[str, torch.device]:
    """Resolve auto/gloo/nccl and validate the worker's local device mapping."""

    resolved_backend = backend
    if backend == "auto":
        resolved_backend = "nccl" if torch.cuda.is_available() else "gloo"
    if resolved_backend not in ("gloo", "nccl"):
        raise ValueError("backend must be one of: auto, gloo, nccl")

    if resolved_backend == "gloo":
        if not dist.is_gloo_available():
            raise RuntimeError("the Gloo backend is not available in this PyTorch build")
        return resolved_backend, torch.device("cpu")

    if not dist.is_nccl_available():
        raise RuntimeError("the NCCL backend is not available in this PyTorch build")
    if not torch.cuda.is_available():
        raise RuntimeError("NCCL requires CUDA, but CUDA is not available")
    if local_rank >= torch.cuda.device_count():
        raise RuntimeError(
            f"LOCAL_RANK {local_rank} cannot map to {torch.cuda.device_count()} "
            "visible CUDA device(s)"
        )
    return resolved_backend, torch.device("cuda", local_rank)


# Input comes from torchrun plus a backend choice. Output is the worker context;
# the side effect is creation of PyTorch's default process group used by every
# later collective, DDP, and FSDP operation.
def init_distributed(
    backend: str = "auto",
    timeout_seconds: int = 120,
) -> DistributedContext:
    """Map this worker to a device and initialize the default process group."""

    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    if not dist.is_available():
        raise RuntimeError("torch.distributed is not available in this PyTorch build")
    if dist.is_initialized():
        raise RuntimeError("the default process group is already initialized")

    rank, local_rank, world_size = read_torchrun_environment()
    resolved_backend, device = resolve_distributed_device(backend, local_rank)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    # env:// consumes MASTER_ADDR/MASTER_PORT provided by torchrun. Every worker
    # blocks here until rendezvous succeeds or the explicit timeout is reached.
    dist.init_process_group(
        backend=resolved_backend,
        init_method="env://",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=timeout_seconds),
    )
    return DistributedContext(
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        backend=resolved_backend,
        device=device,
    )


# There is no input/output data. The function releases the global distributed
# runtime so tests and orderly process shutdown do not retain group resources.
def cleanup_distributed() -> None:
    """Destroy the default process group if this process initialized one."""

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
