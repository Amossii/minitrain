"""CPU tests for DDP correctness workload construction and loss equivalence."""

from __future__ import annotations

import unittest

import torch

from minitrain.data import SyntheticTokenDataset
from minitrain.training import causal_lm_loss
from scripts.verify_ddp_correctness import (
    correctness_model_config,
    stack_dataset_indices,
)


class DDPCorrectnessWorkloadTest(unittest.TestCase):
    def test_compact_model_config_is_structurally_valid(self) -> None:
        config = correctness_model_config(seq_len=16)
        self.assertEqual(config.hidden_size, config.num_heads * config.head_dim)
        self.assertEqual(config.seq_len, 16)

    def test_stacked_local_batches_form_global_batch(self) -> None:
        dataset = SyntheticTokenDataset(4, seq_len=8, vocab_size=32, seed=7)
        rank_zero = stack_dataset_indices(dataset, [0, 2])
        rank_one = stack_dataset_indices(dataset, [1, 3])
        combined = {
            "input_ids": torch.cat([rank_zero["input_ids"], rank_one["input_ids"]]),
            "labels": torch.cat([rank_zero["labels"], rank_one["labels"]]),
        }

        self.assertEqual(tuple(combined["input_ids"].shape), (4, 8))
        observed_rows = {tuple(row.tolist()) for row in combined["input_ids"]}
        expected = stack_dataset_indices(dataset, [0, 1, 2, 3])
        expected_rows = {tuple(row.tolist()) for row in expected["input_ids"]}
        self.assertEqual(observed_rows, expected_rows)

    # For equal local batch sizes, averaging local mean losses equals the mean
    # loss computed over their concatenated global batch.
    def test_mean_local_losses_equal_global_batch_loss(self) -> None:
        logits = torch.randn(4, 3, 7)
        labels = torch.randint(0, 7, (4, 3))

        rank_zero_loss = causal_lm_loss(logits[:2], labels[:2])
        rank_one_loss = causal_lm_loss(logits[2:], labels[2:])
        averaged_local_loss = (rank_zero_loss + rank_one_loss) / 2
        global_loss = causal_lm_loss(logits, labels)

        torch.testing.assert_close(averaged_local_loss, global_loss)

    def test_empty_index_list_is_rejected(self) -> None:
        dataset = SyntheticTokenDataset(2, seq_len=4, vocab_size=8)
        with self.assertRaisesRegex(ValueError, "cannot be empty"):
            stack_dataset_indices(dataset, [])


if __name__ == "__main__":
    unittest.main()
