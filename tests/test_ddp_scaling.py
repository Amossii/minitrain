"""CPU unit tests for strong/weak scaling validation and formulas."""

from __future__ import annotations

import unittest

from minitrain.distributed.scaling import build_scaling_rows


def _summary(world_size: int, local_batch: int, step_time: float) -> dict:
    return {
        "model_name": "tiny",
        "strategy": "ddp",
        "precision": "fp32",
        "world_size": world_size,
        "local_batch_size": local_batch,
        "global_batch_size": local_batch * world_size,
        "seq_len": 128,
        "num_parameters": 100,
        "median_step_time": step_time,
        "median_tokens_per_second": local_batch * world_size * 128 / step_time,
        "max_peak_memory_allocated": 10,
        "max_peak_memory_reserved": 20,
    }


class DDPScalingTest(unittest.TestCase):
    def test_strong_scaling_uses_fixed_global_batch_and_time_speedup(self) -> None:
        rows = build_scaling_rows(
            [_summary(1, 2, 1.0), _summary(2, 1, 0.6)], "strong"
        )
        self.assertAlmostEqual(rows[1]["scaling_factor"], 1.0 / 0.6)
        self.assertAlmostEqual(rows[1]["scaling_efficiency"], 1.0 / 1.2)

    def test_weak_scaling_uses_fixed_local_batch_and_throughput(self) -> None:
        rows = build_scaling_rows(
            [_summary(1, 1, 1.0), _summary(2, 1, 1.0)], "weak"
        )
        self.assertEqual(rows[1]["scaling_factor"], 2.0)
        self.assertEqual(rows[1]["scaling_efficiency"], 1.0)

    def test_rejects_unfair_strong_scaling(self) -> None:
        with self.assertRaisesRegex(ValueError, "global_batch_size"):
            build_scaling_rows([_summary(1, 1, 1.0), _summary(2, 1, 0.7)], "strong")

    def test_rejects_unfair_weak_scaling(self) -> None:
        with self.assertRaisesRegex(ValueError, "local_batch_size"):
            build_scaling_rows([_summary(1, 1, 1.0), _summary(2, 2, 1.0)], "weak")


if __name__ == "__main__":
    unittest.main()
