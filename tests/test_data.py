"""CPU correctness tests for deterministic synthetic token data."""

from __future__ import annotations

import unittest

try:
    import torch
    from torch.utils.data import DataLoader
except ImportError:
    torch = None
    DataLoader = None


@unittest.skipIf(torch is None, "PyTorch is not installed")
class SyntheticTokenDatasetTest(unittest.TestCase):
    def setUp(self) -> None:
        from minitrain.data import SyntheticTokenDataset

        self.dataset = SyntheticTokenDataset(
            num_samples=6,
            seq_len=8,
            vocab_size=32,
            seed=123,
        )

    # A sample must have the exact dtype/range accepted by nn.Embedding.
    def test_sample_shape_dtype_and_range(self) -> None:
        sample = self.dataset[0]

        self.assertEqual(tuple(sample["input_ids"].shape), (8,))
        self.assertEqual(tuple(sample["labels"].shape), (8,))
        self.assertEqual(sample["input_ids"].dtype, torch.long)
        self.assertGreaterEqual(int(sample["input_ids"].min()), 0)
        self.assertLess(int(sample["input_ids"].max()), 32)

    # Labels are the same token stream shifted left by one position.
    def test_labels_are_next_tokens(self) -> None:
        sample = self.dataset[2]
        torch.testing.assert_close(sample["input_ids"][1:], sample["labels"][:-1])

    # Index-local RNG makes values stable regardless of access order.
    def test_seed_and_index_are_deterministic(self) -> None:
        first = self.dataset[3]
        _ = self.dataset[1]
        repeated = self.dataset[3]

        torch.testing.assert_close(first["input_ids"], repeated["input_ids"])
        torch.testing.assert_close(first["labels"], repeated["labels"])

        from minitrain.data import SyntheticTokenDataset

        other_seed = SyntheticTokenDataset(6, 8, 32, seed=124)[3]
        self.assertFalse(torch.equal(first["input_ids"], other_seed["input_ids"]))

    # Default collation turns per-sample [T] tensors into local batches [B, T].
    def test_dataloader_batch_shape(self) -> None:
        loader = DataLoader(self.dataset, batch_size=3, shuffle=False)
        batch = next(iter(loader))

        self.assertEqual(tuple(batch["input_ids"].shape), (3, 8))
        self.assertEqual(tuple(batch["labels"].shape), (3, 8))

    def test_dataset_does_not_advance_global_rng(self) -> None:
        torch.manual_seed(999)
        expected = torch.randint(0, 100, (4,))

        torch.manual_seed(999)
        _ = self.dataset[0]
        actual = torch.randint(0, 100, (4,))

        torch.testing.assert_close(actual, expected)

    def test_invalid_arguments_and_index_fail_early(self) -> None:
        from minitrain.data import SyntheticTokenDataset, create_synthetic_batch

        with self.assertRaisesRegex(ValueError, "num_samples"):
            SyntheticTokenDataset(0, 8, 32)
        with self.assertRaisesRegex(ValueError, "vocab_size"):
            SyntheticTokenDataset(2, 8, 1)
        with self.assertRaises(IndexError):
            _ = self.dataset[6]
        with self.assertRaisesRegex(ValueError, "batch_size"):
            create_synthetic_batch(0, 8, 32)


if __name__ == "__main__":
    unittest.main()
