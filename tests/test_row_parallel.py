"""Row Parallel 输入维分片范围的 CPU 单元测试。"""

from __future__ import annotations

import unittest

from minitrain.tensor_parallel.row_linear import row_partition_range


class RowPartitionTest(unittest.TestCase):
    def test_two_ranks_own_disjoint_complete_input_columns(self) -> None:
        self.assertEqual(row_partition_range(12, 2, 0), (0, 6))
        self.assertEqual(row_partition_range(12, 2, 1), (6, 12))

    def test_rejects_non_divisible_input(self) -> None:
        with self.assertRaisesRegex(ValueError, "divisible"):
            row_partition_range(11, 2, 0)

    def test_rejects_invalid_rank(self) -> None:
        with self.assertRaisesRegex(ValueError, "rank"):
            row_partition_range(12, 2, -1)


if __name__ == "__main__":
    unittest.main()
