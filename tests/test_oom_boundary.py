"""OOM boundary 架构生成、错误分类和边界汇总的 CPU 测试。"""

from __future__ import annotations

import unittest

from minitrain.distributed.oom_boundary import (
    build_boundary_config,
    is_cuda_oom,
    maximum_passing_parameters,
    transformer_parameter_count,
)
from minitrain.model import MiniTransformer, count_parameters


class OOMBoundaryTest(unittest.TestCase):
    def test_parameter_formula_matches_real_model(self) -> None:
        config = build_boundary_config(5, seq_len=32, vocab_size=128, num_layers=2)
        self.assertEqual(
            transformer_parameter_count(config),
            count_parameters(MiniTransformer(config)),
        )

    def test_generated_width_is_valid_and_near_target(self) -> None:
        config = build_boundary_config(100, seq_len=128)
        actual = transformer_parameter_count(config)
        self.assertEqual(config.hidden_size % config.num_heads, 0)
        self.assertLess(abs(actual - 100_000_000) / 100_000_000, 0.03)

    def test_only_cuda_oom_is_classified_as_boundary(self) -> None:
        self.assertTrue(is_cuda_oom("torch.OutOfMemoryError: CUDA out of memory"))
        self.assertFalse(is_cuda_oom("torch.OutOfMemoryError: CPU allocation failed"))
        self.assertFalse(is_cuda_oom("RuntimeError: NCCL connection refused"))
        self.assertFalse(is_cuda_oom("RuntimeError: shape mismatch"))

    def test_maximum_uses_only_real_pass_rows(self) -> None:
        rows = [
            {"strategy": "ddp", "status": "PASS", "actual_parameters": 100},
            {"strategy": "ddp", "status": "OOM", "actual_parameters": 200},
            {"strategy": "fsdp2", "status": "PASS", "actual_parameters": 200},
        ]
        self.assertEqual(maximum_passing_parameters(rows, "ddp"), 100)
        self.assertEqual(maximum_passing_parameters(rows, "fsdp2"), 200)
        self.assertIsNone(maximum_passing_parameters(rows, "missing"))


if __name__ == "__main__":
    unittest.main()
