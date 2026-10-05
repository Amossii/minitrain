"""FSDP2/TP 固定 workload 协议的 CPU 单元测试。"""

from __future__ import annotations

import unittest

from minitrain.distributed.fsdp_tp_comparison import (
    per_rank_batch_size,
    validate_throughput_target,
)


class FSDPTPComparisonTest(unittest.TestCase):
    def test_effective_global_batch_has_different_rank_local_shapes(self) -> None:
        self.assertEqual(per_rank_batch_size("fsdp2", 2, 2), 1)
        self.assertEqual(per_rank_batch_size("tp", 2, 2), 2)

    def test_fsdp_batch_must_be_divisible(self) -> None:
        with self.assertRaisesRegex(ValueError, "divisible"):
            per_rank_batch_size("fsdp2", 3, 2)

    def test_throughput_target_requires_both_capacity_passes(self) -> None:
        rows = [
            {
                "strategy": "fsdp2",
                "target_parameters_millions": 100.0,
                "status": "PASS",
            },
            {
                "strategy": "tp",
                "target_parameters_millions": 100.0,
                "status": "OOM",
            },
        ]
        with self.assertRaisesRegex(ValueError, "tp"):
            validate_throughput_target(rows, 100.0)


if __name__ == "__main__":
    unittest.main()
