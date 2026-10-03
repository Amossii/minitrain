"""Small, testable helpers shared by distributed profiler entry points."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity


@dataclass(frozen=True, slots=True)
class ProfileOutputPaths:
    """Hold the three rank-local artifacts produced by one profiler process."""

    trace: Path
    operators: Path
    ddp_logging: Path


# Input is the device used by this rank. Output selects CPU launch activity and,
# for a CUDA rank, GPU kernels as well; this keeps the same script CPU-testable.
def profiler_activities(device: torch.device) -> list[ProfilerActivity]:
    """Return profiler activities appropriate for the rank's training device."""

    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)
    return activities


# Inputs identify one output directory and distributed rank. Outputs are unique
# per rank so concurrent workers never overwrite one another's trace or report.
def profile_output_paths(output_dir: Path, rank: int) -> ProfileOutputPaths:
    """Build deterministic, rank-local profiler artifact paths."""

    if rank < 0:
        raise ValueError("rank must be non-negative")
    return ProfileOutputPaths(
        trace=output_dir / f"ddp_rank{rank}.json",
        operators=output_dir / f"ddp_rank{rank}_operators.txt",
        ddp_logging=output_dir / f"ddp_rank{rank}_ddp_logging.json",
    )


# Input is profiler event names. Output keeps likely distributed communication
# entries so tests and later report tooling can locate NCCL/c10d work explicitly.
def communication_event_names(event_names: list[str]) -> list[str]:
    """Filter profiler event names that likely represent DDP communication."""

    markers = ("nccl", "allreduce", "all_reduce", "c10d")
    return [name for name in event_names if any(key in name.lower() for key in markers)]
