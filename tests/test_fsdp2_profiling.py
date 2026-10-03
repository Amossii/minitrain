"""FSDP2 profiler 路径和通信事件筛选的 CPU 单元测试。"""

from __future__ import annotations

from pathlib import Path
import unittest

from minitrain.distributed.fsdp2_profiling import fsdp2_profile_paths
from minitrain.distributed.profiling import communication_event_names


class FSDP2ProfilingTest(unittest.TestCase):
    def test_paths_are_unique_for_each_rank(self) -> None:
        rank_zero = fsdp2_profile_paths(Path("profiles"), 0)
        rank_one = fsdp2_profile_paths(Path("profiles"), 1)
        self.assertEqual(rank_zero.trace, Path("profiles/fsdp2_rank0.json"))
        self.assertNotEqual(rank_zero.trace, rank_one.trace)
        self.assertIn("rank1", rank_one.communications.name)

    def test_negative_rank_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "rank"):
            fsdp2_profile_paths(Path("profiles"), -1)

    def test_filter_finds_all_gather_and_reduce_scatter(self) -> None:
        names = [
            "aten::mm",
            "c10d::allgather_",
            "ncclDevKernel_ReduceScatter",
            "fsdp::all_gather",
            "fsdp::reduce_scatter",
        ]
        self.assertEqual(communication_event_names(names), names[1:])


if __name__ == "__main__":
    unittest.main()
