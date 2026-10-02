"""CPU unit tests for the hand-computable collective expectations."""

from __future__ import annotations

import unittest

import torch

from scripts.collective_correctness import (
    expected_rank_sum,
    expected_reduce_scatter_chunk,
)


class CollectiveExpectationTest(unittest.TestCase):
    def test_rank_sum_for_two_workers(self) -> None:
        self.assertEqual(expected_rank_sum(world_size=2), 3.0)

    def test_rank_sum_generalizes_beyond_kaggle(self) -> None:
        self.assertEqual(expected_rank_sum(world_size=4), 10.0)

    def test_reduce_scatter_expected_chunks_for_two_workers(self) -> None:
        rank_zero = expected_reduce_scatter_chunk(
            rank=0, world_size=2, chunk_size=2, device=torch.device("cpu")
        )
        rank_one = expected_reduce_scatter_chunk(
            rank=1, world_size=2, chunk_size=2, device=torch.device("cpu")
        )

        torch.testing.assert_close(rank_zero, torch.tensor([10.0, 12.0]))
        torch.testing.assert_close(rank_one, torch.tensor([14.0, 16.0]))


if __name__ == "__main__":
    unittest.main()
