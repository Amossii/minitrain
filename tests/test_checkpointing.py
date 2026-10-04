"""Distributed Checkpoint 的单进程 CPU correctness 测试。"""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import torch

from minitrain.config import ModelConfig
from minitrain.data import create_synthetic_batch
from minitrain.distributed.checkpointing import (
    load_training_checkpoint,
    save_training_checkpoint,
)
from minitrain.model import MiniTransformer
from minitrain.training import train_step


def compact_config() -> ModelConfig:
    """返回可快速训练、但仍包含完整 Transformer 结构的配置。"""

    return ModelConfig(32, 8, 16, 1, 4, 32)


class DistributedCheckpointTest(unittest.TestCase):
    def test_missing_checkpoint_is_rejected(self) -> None:
        model = MiniTransformer(compact_config())
        optimizer = torch.optim.AdamW(model.parameters())
        with self.assertRaises(FileNotFoundError):
            load_training_checkpoint(Path("/tmp/does-not-exist-minitrain"), model, optimizer)

    def test_model_optimizer_and_step_resume_exactly(self) -> None:
        config = compact_config()
        device = torch.device("cpu")
        torch.manual_seed(7)
        uninterrupted = MiniTransformer(config)
        uninterrupted_optimizer = torch.optim.AdamW(
            uninterrupted.parameters(), lr=1e-3
        )
        first_batch = create_synthetic_batch(2, 8, 32, seed=11)
        resume_batch = create_synthetic_batch(2, 8, 32, seed=12)
        train_step(uninterrupted, uninterrupted_optimizer, first_batch, device)

        with tempfile.TemporaryDirectory() as directory:
            checkpoint_dir = Path(directory) / "step_000001"
            save_training_checkpoint(
                checkpoint_dir,
                uninterrupted,
                uninterrupted_optimizer,
                step=1,
            )
            torch.manual_seed(999)
            restored = MiniTransformer(config)
            restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3)
            restored_step = load_training_checkpoint(
                checkpoint_dir, restored, restored_optimizer
            )

        self.assertEqual(restored_step, 1)
        for expected, actual in zip(
            uninterrupted.parameters(), restored.parameters(), strict=True
        ):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)

        # 两条分支处理相同的下一批数据。若 Adam moments 未恢复，更新后参数会分叉。
        expected_loss = train_step(
            uninterrupted, uninterrupted_optimizer, resume_batch, device
        )
        actual_loss = train_step(restored, restored_optimizer, resume_batch, device)
        self.assertEqual(actual_loss, expected_loss)
        for expected, actual in zip(
            uninterrupted.parameters(), restored.parameters(), strict=True
        ):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
