"""NCCL/Gloo collective timing and bandwidth calculations."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, fields
from pathlib import Path
import time

import torch
import torch.distributed as dist

from minitrain.distributed.runtime import DistributedContext


@dataclass(frozen=True, slots=True)
class CollectiveBenchmarkRecord:
    """Store one message-size measurement from a collective benchmark.

    Inputs describe the collective, workload, and observed maximum-rank time.
    The immutable output row is written to CSV. Repeated calls do not retain
    timing state, so every message size has an independent measurement.
    """

    collective: str
    backend: str
    world_size: int
    dtype: str
    message_size_bytes: int
    num_elements: int
    warmup_iterations: int
    iterations: int
    latency_ms: float
    algorithm_bandwidth_gbps: float
    bus_bandwidth_gbps: float

    def to_dict(self) -> dict[str, str | int | float]:
        """Return one dictionary whose order matches the CSV schema."""

        return asdict(self)


# Inputs define a geometric message-size sweep. Output includes both endpoints
# and rejects a factor that could otherwise create an infinite loop.
def generate_message_sizes(
    minimum_bytes: int,
    maximum_bytes: int,
    factor: int,
) -> list[int]:
    """Generate geometrically increasing message sizes in bytes."""

    if minimum_bytes <= 0 or maximum_bytes <= 0:
        raise ValueError("message sizes must be positive")
    if minimum_bytes > maximum_bytes:
        raise ValueError("minimum_bytes cannot exceed maximum_bytes")
    if factor <= 1:
        raise ValueError("factor must be greater than 1")

    sizes: list[int] = []
    current = minimum_bytes
    while current <= maximum_bytes:
        sizes.append(current)
        current *= factor
    if sizes[-1] != maximum_bytes:
        sizes.append(maximum_bytes)
    return sizes


# Algorithm bandwidth describes useful tensor bytes processed per second. The
# ring-equivalent bus correction makes AllReduce comparable to link bandwidth.
def all_reduce_bandwidths(
    message_size_bytes: int,
    latency_seconds: float,
    world_size: int,
) -> tuple[float, float]:
    """Return decimal algorithm and bus bandwidth in GB/s for AllReduce."""

    if message_size_bytes <= 0 or latency_seconds <= 0 or world_size <= 1:
        raise ValueError("message size/time must be positive and world_size > 1")
    algorithm_bandwidth = message_size_bytes / latency_seconds / 1_000_000_000
    bus_bandwidth = algorithm_bandwidth * 2 * (world_size - 1) / world_size
    return algorithm_bandwidth, bus_bandwidth


# Input is one per-rank tensor size and iteration policy. Output uses the
# slowest rank's elapsed time, because synchronous training waits for that rank.
def benchmark_all_reduce(
    context: DistributedContext,
    message_size_bytes: int,
    warmup_iterations: int,
    iterations: int,
) -> CollectiveBenchmarkRecord:
    """Benchmark repeated float32 AllReduce SUM operations for one message size."""

    element_size = torch.empty((), dtype=torch.float32).element_size()
    if message_size_bytes % element_size != 0:
        raise ValueError(
            f"message_size_bytes must be divisible by float32 size {element_size}"
        )
    if warmup_iterations < 0 or iterations <= 0:
        raise ValueError("warmup_iterations must be non-negative and iterations positive")

    num_elements = message_size_bytes // element_size
    tensor = torch.zeros(num_elements, dtype=torch.float32, device=context.device)

    for _ in range(warmup_iterations):
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    # Align workers before timing. The barrier itself is outside the timed region.
    dist.barrier()
    if context.device.type == "cuda":
        torch.cuda.synchronize(context.device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        end.record()
        end.synchronize()
        local_elapsed_seconds = start.elapsed_time(end) / 1_000
    else:
        start_time = time.perf_counter()
        for _ in range(iterations):
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        local_elapsed_seconds = time.perf_counter() - start_time

    # A collective is complete only as fast as its slowest rank. MAX reduction
    # happens after timing and therefore does not contaminate the measurement.
    elapsed = torch.tensor([local_elapsed_seconds], device=context.device)
    dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
    latency_seconds = float(elapsed.item()) / iterations
    algorithm_bandwidth, bus_bandwidth = all_reduce_bandwidths(
        message_size_bytes, latency_seconds, context.world_size
    )
    return CollectiveBenchmarkRecord(
        collective="all_reduce",
        backend=context.backend,
        world_size=context.world_size,
        dtype="float32",
        message_size_bytes=message_size_bytes,
        num_elements=num_elements,
        warmup_iterations=warmup_iterations,
        iterations=iterations,
        latency_ms=latency_seconds * 1_000,
        algorithm_bandwidth_gbps=algorithm_bandwidth,
        bus_bandwidth_gbps=bus_bandwidth,
    )


# Only rank 0 calls this function. It persists raw size-sweep observations using
# a stable schema so later plots never need to parse terminal text.
def write_collective_csv(
    path: Path,
    records: list[CollectiveBenchmarkRecord],
) -> None:
    """Write collective benchmark records to a CSV file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    field_names = [field.name for field in fields(CollectiveBenchmarkRecord)]
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=field_names)
        writer.writeheader()
        writer.writerows(record.to_dict() for record in records)
