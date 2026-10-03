"""CPU unit tests for DDP profiler configuration and artifact naming."""

from __future__ import annotations

from pathlib import Path
import unittest

import torch
from torch.profiler import ProfilerActivity

from minitrain.distributed.profiling import (
    communication_event_names,
    profile_output_paths,
    profiler_activities,
)


class DDPProfilingTest(unittest.TestCase):
    def test_cpu_activity_does_not_request_cuda(self) -> None:
        self.assertEqual(profiler_activities(torch.device("cpu")), [ProfilerActivity.CPU])

    def test_cuda_activity_captures_cpu_launches_and_gpu_kernels(self) -> None:
        self.assertEqual(
            profiler_activities(torch.device("cuda:1")),
            [ProfilerActivity.CPU, ProfilerActivity.CUDA],
        )

    def test_output_paths_are_rank_local(self) -> None:
        rank_zero = profile_output_paths(Path("profiles"), 0)
        rank_one = profile_output_paths(Path("profiles"), 1)
        self.assertEqual(rank_zero.trace, Path("profiles/ddp_rank0.json"))
        self.assertNotEqual(rank_zero.trace, rank_one.trace)
        self.assertIn("rank1", rank_one.ddp_logging.name)

    def test_communication_event_filter_keeps_nccl_and_c10d(self) -> None:
        names = ["aten::mm", "ncclDevKernel_AllReduce", "c10d::allreduce_"]
        self.assertEqual(communication_event_names(names), names[1:])

    def test_negative_rank_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "rank"):
            profile_output_paths(Path("profiles"), -1)


if __name__ == "__main__":
    unittest.main()
