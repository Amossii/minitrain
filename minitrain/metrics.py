"""Stable metrics schema and CSV persistence for MiniTrain experiments."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, fields
from pathlib import Path


@dataclass(frozen=True, slots=True)
class StepMetrics:
    """Describe one measured optimizer step and its fixed workload metadata.

    Inputs combine observed timing/memory with experiment configuration. The
    immutable record is the internal state and one CSV row is its output. A new
    record is created for every measured step; previous rows never mutate.
    """

    step: int
    loss: float
    step_time: float
    forward_time: float
    backward_time: float
    optimizer_time: float
    tokens_per_second: float
    samples_per_second: float
    peak_memory_allocated: int
    peak_memory_reserved: int
    world_size: int
    local_batch_size: int
    global_batch_size: int
    seq_len: int
    num_parameters: int
    precision: str
    strategy: str

    @classmethod
    def from_measurement(
        cls,
        *,
        step: int,
        loss: float,
        step_time: float,
        forward_time: float,
        backward_time: float,
        optimizer_time: float,
        peak_memory_allocated: int,
        peak_memory_reserved: int,
        world_size: int,
        local_batch_size: int,
        seq_len: int,
        num_parameters: int,
        precision: str,
        strategy: str,
    ) -> "StepMetrics":
        """Build a row and derive global batch and throughput consistently."""

        if step < 0:
            raise ValueError("step must be non-negative")
        if step_time <= 0:
            raise ValueError("step_time must be positive")
        if world_size <= 0 or local_batch_size <= 0 or seq_len <= 0:
            raise ValueError("world_size, local_batch_size, and seq_len must be positive")

        global_batch_size = local_batch_size * world_size
        samples_per_second = global_batch_size / step_time
        tokens_per_second = global_batch_size * seq_len / step_time
        return cls(
            step=step,
            loss=loss,
            step_time=step_time,
            forward_time=forward_time,
            backward_time=backward_time,
            optimizer_time=optimizer_time,
            tokens_per_second=tokens_per_second,
            samples_per_second=samples_per_second,
            peak_memory_allocated=peak_memory_allocated,
            peak_memory_reserved=peak_memory_reserved,
            world_size=world_size,
            local_batch_size=local_batch_size,
            global_batch_size=global_batch_size,
            seq_len=seq_len,
            num_parameters=num_parameters,
            precision=precision,
            strategy=strategy,
        )

    def to_dict(self) -> dict[str, int | float | str]:
        """Return one dictionary whose key order matches the CSV schema."""

        return asdict(self)


# Input is a destination and homogeneous records; output is a complete CSV.
# Writing the header even for no rows makes downstream schema failures visible.
def write_metrics_csv(path: Path, records: list[StepMetrics]) -> None:
    """Create parent directories and write metrics using the stable schema."""

    path.parent.mkdir(parents=True, exist_ok=True)
    field_names = [field.name for field in fields(StepMetrics)]
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=field_names)
        writer.writeheader()
        writer.writerows(record.to_dict() for record in records)
