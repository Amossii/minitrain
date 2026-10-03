"""Column Parallel 分片范围的 CPU 单元测试。"""

from __future__ import annotations

import unittest

from minitrain.tensor_parallel.column_linear import column_partition_range


class ColumnPartitionTest(unittest.TestCase):
    def test_two_ranks_own_disjoint_complete_output_rows(self) -> None:
        self.assertEqual(column_partition_range(12, 2, 0), (0, 6))
        self.assertEqual(column_partition_range(12, 2, 1), (6, 12))

    def test_rejects_non_divisible_output(self) -> None:
        with self.assertRaisesRegex(ValueError, "divisible"):
            column_partition_range(11, 2, 0)

    def test_rejects_invalid_rank(self) -> None:
        with self.assertRaisesRegex(ValueError, "rank"):
            column_partition_range(12, 2, 2)


if __name__ == "__main__":
    unittest.main()
