"""CPU correctness tests for timing records and stable CSV output."""

from __future__ import annotations

import csv
from pathlib import Path
import tempfile
import unittest

import torch

from minitrain.config import ModelConfig
from minitrain.data import create_synthetic_batch
from minitrain.metrics import StepMetrics, write_metrics_csv
from minitrain.model import MiniTransformer
from minitrain.training import measure_train_step


class StepMetricsTest(unittest.TestCase):
    def test_throughput_and_global_batch_formulas(self) -> None:
        record = StepMetrics.from_measurement(
            step=0,
            loss=1.5,
            step_time=2.0,
            forward_time=0.5,
            backward_time=1.0,
            optimizer_time=0.5,
            peak_memory_allocated=100,
            peak_memory_reserved=200,
            world_size=2,
            local_batch_size=4,
            seq_len=16,
            num_parameters=1_000,
            precision="fp32",
            strategy="test",
        )

        self.assertEqual(record.global_batch_size, 8)
        self.assertEqual(record.samples_per_second, 4.0)
        self.assertEqual(record.tokens_per_second, 64.0)

    def test_csv_header_and_values_are_stable(self) -> None:
        record = StepMetrics.from_measurement(
            step=3,
            loss=2.0,
            step_time=1.0,
            forward_time=0.2,
            backward_time=0.6,
            optimizer_time=0.2,
            peak_memory_allocated=0,
            peak_memory_reserved=0,
            world_size=1,
            local_batch_size=2,
            seq_len=8,
            num_parameters=123,
            precision="fp32",
            strategy="single",
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "nested" / "metrics.csv"
            write_metrics_csv(path, [record])
            with path.open(encoding="utf-8", newline="") as csv_file:
                rows = list(csv.DictReader(csv_file))

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["strategy"], "single")
        self.assertEqual(rows[0]["global_batch_size"], "2")
        self.assertIn("peak_memory_reserved", rows[0])


class MeasuredTrainStepTest(unittest.TestCase):
    def test_cpu_measurement_is_positive_and_updates_model(self) -> None:
        config = ModelConfig(32, 8, 16, 1, 4, 48)
        model = MiniTransformer(config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        batch = create_synthetic_batch(2, 8, 32, seed=5)

        measurement = measure_train_step(
            model, optimizer, batch, torch.device("cpu")
        )

        self.assertGreater(measurement.step_time, 0)
        self.assertGreater(measurement.forward_time, 0)
        self.assertGreater(measurement.backward_time, 0)
        self.assertGreater(measurement.optimizer_time, 0)
        self.assertAlmostEqual(
            measurement.step_time,
            measurement.forward_time
            + measurement.backward_time
            + measurement.optimizer_time,
            places=9,
        )
        self.assertEqual(measurement.peak_memory_allocated, 0)
        self.assertEqual(measurement.peak_memory_reserved, 0)


if __name__ == "__main__":
    unittest.main()
