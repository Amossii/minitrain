"""Memory accounting and fair DDP/FSDP2 comparison helpers."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.distributed.tensor import DTensor


MEMORY_FIELDS = (
    "strategy",
    "model_name",
    "precision",
    "world_size",
    "local_batch_size",
    "global_batch_size",
    "seq_len",
    "num_parameters",
    "warmup_steps",
    "measured_steps",
    "parameter_bytes_per_rank",
    "gradient_bytes_per_rank",
    "optimizer_state_bytes_per_rank",
    "theoretical_model_state_bytes_per_rank",
    "observed_peak_allocated_bytes",
    "observed_peak_reserved_bytes",
    "observed_end_allocated_bytes",
    "observed_end_reserved_bytes",
)

COMPARISON_FIELDS = MEMORY_FIELDS + (
    "peak_allocated_ratio_vs_ddp",
    "peak_allocated_saving_vs_ddp",
)


# Input may be a local Tensor, sharded DTensor, or non-tensor optimizer value.
# Output counts only rank-local tensor payload and never triggers an AllGather.
def local_tensor_bytes(value: Any) -> int:
    """Return bytes physically represented by one rank's tensor value."""

    if isinstance(value, DTensor):
        local = value.to_local()
        return local.numel() * local.element_size()
    if isinstance(value, Tensor):
        return value.numel() * value.element_size()
    return 0


# Input is the live model after backward/warmup. Outputs separate persistent
# local parameter and gradient storage for replicated versus sharded strategies.
def model_state_bytes(model: nn.Module) -> tuple[int, int]:
    """Count rank-local parameter and gradient tensor payload bytes."""

    parameter_bytes = 0
    gradient_bytes = 0
    for parameter in model.parameters():
        parameter_bytes += local_tensor_bytes(parameter)
        if parameter.grad is not None:
            gradient_bytes += local_tensor_bytes(parameter.grad)
    return parameter_bytes, gradient_bytes


# Input is an initialized optimizer. Output recursively counts its rank-local
# tensor state, including Adam moments but excluding Python object overhead.
def optimizer_state_bytes(optimizer: torch.optim.Optimizer) -> int:
    """Count tensor payload bytes currently held in optimizer state."""

    def count(value: Any) -> int:
        if isinstance(value, dict):
            return sum(count(item) for item in value.values())
        if isinstance(value, (list, tuple)):
            return sum(count(item) for item in value)
        return local_tensor_bytes(value)

    return count(optimizer.state)


# Inputs describe P parameters trained with FP32 Adam. Output is the idealized
# persistent parameter+gradient+two-moment bytes owned by one data-parallel rank.
def theoretical_adam_bytes_per_rank(
    num_parameters: int,
    world_size: int,
    strategy: str,
    element_size: int = 4,
) -> int:
    """Estimate persistent model-state bytes per rank for DDP or FSDP2."""

    if num_parameters <= 0 or world_size <= 0 or element_size <= 0:
        raise ValueError("parameter count, world size, and element size must be positive")
    if strategy not in ("ddp", "fsdp2"):
        raise ValueError("strategy must be ddp or fsdp2")
    total = num_parameters * element_size * 4
    return total if strategy == "ddp" else (total + world_size - 1) // world_size


# Input rows must describe the same workload and contain one DDP and one FSDP2
# observation. Output adds ratios against DDP without inventing benchmark data.
def build_memory_comparison(
    rows: list[dict[str, str | int | float]],
) -> list[dict[str, str | int | float]]:
    """Validate fairness and derive FSDP2 peak-memory savings versus DDP."""

    if {str(row["strategy"]) for row in rows} != {"ddp", "fsdp2"}:
        raise ValueError("comparison requires exactly one ddp and one fsdp2 row")
    if len(rows) != 2:
        raise ValueError("comparison requires exactly two rows")
    fixed = (
        "model_name",
        "precision",
        "world_size",
        "local_batch_size",
        "global_batch_size",
        "seq_len",
        "num_parameters",
        "warmup_steps",
        "measured_steps",
    )
    for field in fixed:
        if len({row[field] for row in rows}) != 1:
            raise ValueError(f"{field} must remain fixed for a fair comparison")

    ddp = next(row for row in rows if row["strategy"] == "ddp")
    baseline = float(ddp["observed_peak_allocated_bytes"])
    if baseline <= 0:
        raise ValueError("DDP observed peak allocated memory must be positive")
    result = []
    for row in sorted(rows, key=lambda item: str(item["strategy"])):
        peak = float(row["observed_peak_allocated_bytes"])
        result.append(
            {
                **row,
                "peak_allocated_ratio_vs_ddp": peak / baseline,
                "peak_allocated_saving_vs_ddp": 1.0 - peak / baseline,
            }
        )
    return result


# Inputs are a destination, rows, and an explicit schema. Output is a stable CSV
# used as raw evidence or as the derived strategy-comparison table.
def write_memory_csv(
    path: Path,
    rows: list[dict[str, str | int | float]],
    field_names: tuple[str, ...] = MEMORY_FIELDS,
) -> None:
    """Write memory benchmark rows using a deterministic schema."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=field_names)
        writer.writeheader()
        writer.writerows(rows)
