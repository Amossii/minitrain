"""CPU tests for the data partitioning contract used by DDP training."""

from __future__ import annotations

import unittest

from torch.utils.data import DistributedSampler

from minitrain.data import SyntheticTokenDataset


class DDPDataPartitionTest(unittest.TestCase):
    # With shuffle disabled, the two ranks take alternating global indices.
    def test_two_rank_samplers_are_disjoint_and_complete(self) -> None:
        dataset = SyntheticTokenDataset(8, seq_len=4, vocab_size=16, seed=42)
        rank_zero = DistributedSampler(
            dataset, num_replicas=2, rank=0, shuffle=False, drop_last=True
        )
        rank_one = DistributedSampler(
            dataset, num_replicas=2, rank=1, shuffle=False, drop_last=True
        )

        zero_indices = list(iter(rank_zero))
        one_indices = list(iter(rank_one))

        self.assertEqual(zero_indices, [0, 2, 4, 6])
        self.assertEqual(one_indices, [1, 3, 5, 7])
        self.assertTrue(set(zero_indices).isdisjoint(one_indices))
        self.assertEqual(sorted(zero_indices + one_indices), list(range(8)))

    def test_each_rank_receives_local_batch_times_steps_samples(self) -> None:
        local_batch_size = 2
        world_size = 2
        steps = 3
        dataset = SyntheticTokenDataset(
            local_batch_size * world_size * steps,
            seq_len=4,
            vocab_size=16,
        )
        sampler = DistributedSampler(
            dataset, num_replicas=world_size, rank=0, shuffle=False, drop_last=True
        )

        self.assertEqual(len(sampler), local_batch_size * steps)


if __name__ == "__main__":
    unittest.main()
