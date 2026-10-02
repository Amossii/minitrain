"""Aggregation helpers for reproducible MiniTrain benchmark runs."""

from __future__ import annotations

import csv
from pathlib import Path
import statistics


SUMMARY_FIELDS = (
    "model_name",
    "strategy",
    "precision",
    "world_size",
    "local_batch_size",
    "global_batch_size",
    "seq_len",
    "num_parameters",
    "measured_steps",
    "median_step_time",
    "median_forward_time",
    "median_backward_time",
    "median_optimizer_time",
    "median_tokens_per_second",
    "median_samples_per_second",
    "max_peak_memory_allocated",
    "max_peak_memory_reserved",
)


# Input is one raw per-step CSV. Output is a single summary row. Configuration
# columns must remain constant so the aggregation cannot hide mixed workloads.
def summarize_metrics_file(path: Path) -> dict[str, str | int | float]:
    """Validate one raw run and summarize its measured steps with medians."""

    with path.open(encoding="utf-8", newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))
    if not rows:
        raise ValueError(f"metrics file contains no measured steps: {path}")

    fixed_fields = (
        "model_name",
        "strategy",
        "precision",
        "world_size",
        "local_batch_size",
        "global_batch_size",
        "seq_len",
        "num_parameters",
    )
    for field in fixed_fields:
        values = {row[field] for row in rows}
        if len(values) != 1:
            raise ValueError(f"{field} changes within one benchmark run: {path}")

    def median(field: str) -> float:
        return statistics.median(float(row[field]) for row in rows)

    def maximum(field: str) -> int:
        return max(int(row[field]) for row in rows)

    first = rows[0]
    return {
        "model_name": first["model_name"],
        "strategy": first["strategy"],
        "precision": first["precision"],
        "world_size": int(first["world_size"]),
        "local_batch_size": int(first["local_batch_size"]),
        "global_batch_size": int(first["global_batch_size"]),
        "seq_len": int(first["seq_len"]),
        "num_parameters": int(first["num_parameters"]),
        "measured_steps": len(rows),
        "median_step_time": median("step_time"),
        "median_forward_time": median("forward_time"),
        "median_backward_time": median("backward_time"),
        "median_optimizer_time": median("optimizer_time"),
        "median_tokens_per_second": median("tokens_per_second"),
        "median_samples_per_second": median("samples_per_second"),
        "max_peak_memory_allocated": maximum("peak_memory_allocated"),
        "max_peak_memory_reserved": maximum("peak_memory_reserved"),
    }


# Inputs are validated summary rows and a destination. Output is the compact
# comparison table kept separately from immutable raw per-step measurements.
def write_summary_csv(
    path: Path, rows: list[dict[str, str | int | float]]
) -> None:
    """Write benchmark summary rows using a deterministic column order."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
