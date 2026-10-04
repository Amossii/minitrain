"""ZeRO stage 状态 ownership、理论显存和配置的 CPU 单元测试。"""

from __future__ import annotations

import unittest

from minitrain.distributed.zero import (
    build_zero_config,
    theoretical_zero_bytes_per_rank,
    zero_state_ownership,
)


class ZeroConfigurationTest(unittest.TestCase):
    def test_stage_ownership_is_cumulative(self) -> None:
        stage1 = zero_state_ownership(1)
        stage2 = zero_state_ownership(2)
        stage3 = zero_state_ownership(3)
        self.assertTrue(stage1.optimizer_states_sharded)
        self.assertFalse(stage1.gradients_sharded)
        self.assertTrue(stage2.gradients_sharded)
        self.assertFalse(stage2.parameters_sharded)
        self.assertTrue(stage3.parameters_sharded)

    def test_fp32_adam_theory_for_two_ranks(self) -> None:
        # P=100 时 DDP 为 1600 bytes；stage 越高依次分片更多持久状态。
        self.assertEqual(theoretical_zero_bytes_per_rank(100, 2, 1), 1200)
        self.assertEqual(theoretical_zero_bytes_per_rank(100, 2, 2), 1000)
        self.assertEqual(theoretical_zero_bytes_per_rank(100, 2, 3), 800)

    def test_config_keeps_batch_semantics_explicit(self) -> None:
        config = build_zero_config(2, local_batch_size=3, world_size=2)
        self.assertEqual(config["train_micro_batch_size_per_gpu"], 3)
        self.assertEqual(config["train_batch_size"], 6)
        self.assertEqual(config["zero_optimization"]["stage"], 2)
        self.assertTrue(config["zero_optimization"]["reduce_scatter"])

    def test_invalid_stage_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "stage"):
            build_zero_config(0, local_batch_size=1, world_size=2)


if __name__ == "__main__":
    unittest.main()
