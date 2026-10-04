"""统一 benchmark schema 和公平性规则的 CPU 单元测试。"""

from __future__ import annotations

import csv
from pathlib import Path
import tempfile
import unittest

from minitrain.distributed.unified_benchmark import (
    UnifiedStepMetrics,
    strategy_semantics,
    summarize_unified_metrics,
    validate_comparable_summaries,
    write_unified_metrics_csv,
)
from minitrain.training import StepMeasurement


def measurement(step_time: float = 2.0) -> StepMeasurement:
    """构造确定性观测值，让测试只验证指标语义。"""

    return StepMeasurement(1.5, step_time, 0.5, 1.0, 0.5, 100, 200)


class UnifiedBenchmarkTest(unittest.TestCase):
    def test_data_parallel_uses_distinct_samples_from_all_ranks(self) -> None:
        record = UnifiedStepMetrics.from_measurement(
            step=0,
            measurement=measurement(),
            world_size=2,
            batch_size_per_process=3,
            global_batch_size=6,
            seq_len=10,
            num_parameters=1000,
            warmup_steps=1,
            model_name="tiny",
            precision="fp32",
            optimizer="adamw",
            strategy="ddp",
        )
        self.assertEqual(record.samples_per_second, 3.0)
        self.assertEqual(record.tokens_per_second, 30.0)

    def test_tensor_parallel_does_not_multiply_replicated_batch(self) -> None:
        record = UnifiedStepMetrics.from_measurement(
            step=0,
            measurement=measurement(),
            world_size=2,
            batch_size_per_process=3,
            global_batch_size=3,
            seq_len=10,
            num_parameters=1000,
            warmup_steps=1,
            model_name="tiny",
            precision="fp32",
            optimizer="adamw",
            strategy="tp",
        )
        self.assertEqual(record.samples_per_second, 1.5)
        self.assertEqual(record.tokens_per_second, 15.0)

    def test_invalid_parallel_batch_semantics_are_rejected(self) -> None:
        common = dict(
            step=0,
            measurement=measurement(),
            world_size=2,
            batch_size_per_process=2,
            seq_len=8,
            num_parameters=100,
            warmup_steps=0,
            model_name="tiny",
            precision="fp32",
            optimizer="adamw",
        )
        with self.assertRaises(ValueError):
            UnifiedStepMetrics.from_measurement(
                **common, global_batch_size=2, strategy="ddp"
            )
        with self.assertRaises(ValueError):
            UnifiedStepMetrics.from_measurement(
                **common, global_batch_size=4, strategy="tp"
            )

    def test_raw_csv_round_trip_and_summary(self) -> None:
        records = [
            UnifiedStepMetrics.from_measurement(
                step=step,
                measurement=measurement(step_time),
                world_size=1,
                batch_size_per_process=2,
                global_batch_size=2,
                seq_len=8,
                num_parameters=100,
                warmup_steps=1,
                model_name="tiny",
                precision="fp32",
                optimizer="adamw",
                strategy="single",
            )
            for step, step_time in enumerate((1.0, 3.0))
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "raw.csv"
            write_unified_metrics_csv(path, records)
            with path.open(encoding="utf-8", newline="") as csv_file:
                self.assertEqual(len(list(csv.DictReader(csv_file))), 2)
            summary = summarize_unified_metrics(path)
        self.assertEqual(summary["median_step_time"], 2.0)
        self.assertEqual(summary["measured_steps"], 2)

    def test_comparison_rejects_different_global_batch(self) -> None:
        base: dict[str, str | int | float] = {
            "model_name": "tiny",
            "precision": "fp32",
            "optimizer": "adamw",
            "global_batch_size": 2,
            "seq_len": 8,
            "num_parameters": 100,
            "warmup_steps": 1,
            "measured_steps": 2,
        }
        other = dict(base)
        other["global_batch_size"] = 4
        with self.assertRaisesRegex(ValueError, "global_batch_size"):
            validate_comparable_summaries([base, other])

    def test_strategy_communication_is_explicit(self) -> None:
        self.assertEqual(strategy_semantics("ddp").parallelism_family, "data_parallel")
        self.assertIn("all_reduce", strategy_semantics("tp").communication)


if __name__ == "__main__":
    unittest.main()
