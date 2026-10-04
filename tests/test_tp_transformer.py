"""Transformer TP 配置和 correctness workload 的 CPU 单元测试。"""

from __future__ import annotations

import unittest

from minitrain.config import ModelConfig
from minitrain.tensor_parallel.transformer import validate_tensor_parallel_config
from scripts.verify_tp_transformer import correctness_config


class TensorParallelConfigTest(unittest.TestCase):
    def test_correctness_config_is_divisible_for_two_ranks(self) -> None:
        config = correctness_config(seq_len=8)
        validate_tensor_parallel_config(config, world_size=2)
        self.assertEqual(config.num_heads // 2, 2)
        self.assertEqual(config.intermediate_size // 2, 48)

    def test_rejects_non_divisible_attention_heads(self) -> None:
        config = ModelConfig(64, 8, 24, 2, 3, 72)
        with self.assertRaisesRegex(ValueError, "num_heads"):
            validate_tensor_parallel_config(config, world_size=2)

    def test_rejects_non_divisible_intermediate_width(self) -> None:
        config = ModelConfig(64, 8, 24, 2, 4, 73)
        with self.assertRaisesRegex(ValueError, "intermediate_size"):
            validate_tensor_parallel_config(config, world_size=2)


if __name__ == "__main__":
    unittest.main()
