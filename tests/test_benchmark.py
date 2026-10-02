"""Correctness tests for raw benchmark aggregation."""

from __future__ import annotations

import csv
from pathlib import Path
import tempfile
import unittest

from minitrain.benchmark import summarize_metrics_file, write_summary_csv
from minitrain.metrics import StepMetrics, write_metrics_csv


def _record(step: int, step_time: float) -> StepMetrics:
    """Create a consistent raw record with predictable throughput."""

    return StepMetrics.from_measurement(
        step=step,
        loss=10.0,
        step_time=step_time,
        forward_time=step_time * 0.25,
        backward_time=step_time * 0.5,
        optimizer_time=step_time * 0.25,
        peak_memory_allocated=100 + step,
        peak_memory_reserved=200 + step,
        world_size=1,
        local_batch_size=2,
        seq_len=8,
        num_parameters=123,
        model_name="tiny",
        precision="fp32",
        strategy="single",
    )


class BenchmarkSummaryTest(unittest.TestCase):
    def test_summary_uses_medians_and_maximum_memory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            raw_path = Path(temporary_directory) / "raw.csv"
            write_metrics_csv(raw_path, [_record(0, 1.0), _record(1, 3.0)])

            summary = summarize_metrics_file(raw_path)

        self.assertEqual(summary["model_name"], "tiny")
        self.assertEqual(summary["measured_steps"], 2)
        self.assertEqual(summary["median_step_time"], 2.0)
        self.assertEqual(summary["median_tokens_per_second"], 32 / 3)
        self.assertEqual(summary["max_peak_memory_allocated"], 101)

    def test_mixed_workloads_are_rejected(self) -> None:
        first = _record(0, 1.0)
        second_values = first.to_dict()
        second_values["seq_len"] = 16
        second = StepMetrics(**second_values)

        with tempfile.TemporaryDirectory() as temporary_directory:
            raw_path = Path(temporary_directory) / "mixed.csv"
            write_metrics_csv(raw_path, [first, second])
            with self.assertRaisesRegex(ValueError, "seq_len changes"):
                summarize_metrics_file(raw_path)

    def test_summary_csv_has_one_row_per_model(self) -> None:
        row = {
            "model_name": "tiny",
            "strategy": "single",
            "precision": "fp32",
            "world_size": 1,
            "local_batch_size": 2,
            "global_batch_size": 2,
            "seq_len": 8,
            "num_parameters": 123,
            "measured_steps": 2,
            "median_step_time": 1.0,
            "median_forward_time": 0.25,
            "median_backward_time": 0.5,
            "median_optimizer_time": 0.25,
            "median_tokens_per_second": 16.0,
            "median_samples_per_second": 2.0,
            "max_peak_memory_allocated": 100,
            "max_peak_memory_reserved": 200,
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "table.csv"
            write_summary_csv(path, [row])
            with path.open(encoding="utf-8", newline="") as csv_file:
                rows = list(csv.DictReader(csv_file))

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["model_name"], "tiny")


if __name__ == "__main__":
    unittest.main()
