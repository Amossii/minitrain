"""Validation and summary calculations for DDP scaling experiments."""

from __future__ import annotations

import csv
from pathlib import Path


SCALING_FIELDS = (
    "scaling_type",
    "model_name",
    "world_size",
    "local_batch_size",
    "global_batch_size",
    "seq_len",
    "median_step_time",
    "median_tokens_per_second",
    "scaling_factor",
    "scaling_efficiency",
    "max_peak_memory_allocated",
    "max_peak_memory_reserved",
)


# Input rows are per-world-size summaries from the shared benchmark schema.
# Output adds scaling factors while rejecting unfair mixed workloads.
def build_scaling_rows(
    summaries: list[dict[str, str | int | float]],
    scaling_type: str,
) -> list[dict[str, str | int | float]]:
    """Validate a strong/weak experiment and compute speedup plus efficiency."""

    if scaling_type not in ("strong", "weak"):
        raise ValueError("scaling_type must be strong or weak")
    ordered = sorted(summaries, key=lambda row: int(row["world_size"]))
    if not ordered or int(ordered[0]["world_size"]) != 1:
        raise ValueError("scaling summaries require a world_size=1 baseline")

    fixed_fields = ("model_name", "precision", "seq_len", "num_parameters")
    for field in fixed_fields:
        if len({row[field] for row in ordered}) != 1:
            raise ValueError(f"{field} must remain fixed during scaling")
    batch_field = "global_batch_size" if scaling_type == "strong" else "local_batch_size"
    if len({row[batch_field] for row in ordered}) != 1:
        raise ValueError(f"{batch_field} must remain fixed for {scaling_type} scaling")

    baseline = ordered[0]
    rows: list[dict[str, str | int | float]] = []
    for summary in ordered:
        world_size = int(summary["world_size"])
        if scaling_type == "strong":
            scaling_factor = float(baseline["median_step_time"]) / float(
                summary["median_step_time"]
            )
        else:
            scaling_factor = float(summary["median_tokens_per_second"]) / float(
                baseline["median_tokens_per_second"]
            )
        rows.append(
            {
                "scaling_type": scaling_type,
                "model_name": summary["model_name"],
                "world_size": world_size,
                "local_batch_size": summary["local_batch_size"],
                "global_batch_size": summary["global_batch_size"],
                "seq_len": summary["seq_len"],
                "median_step_time": summary["median_step_time"],
                "median_tokens_per_second": summary["median_tokens_per_second"],
                "scaling_factor": scaling_factor,
                "scaling_efficiency": scaling_factor / world_size,
                "max_peak_memory_allocated": summary["max_peak_memory_allocated"],
                "max_peak_memory_reserved": summary["max_peak_memory_reserved"],
            }
        )
    return rows


def write_scaling_csv(
    path: Path, rows: list[dict[str, str | int | float]]
) -> None:
    """Write validated strong and weak scaling rows to one comparison table."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=SCALING_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
