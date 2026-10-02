"""CPU unit tests for torchrun metadata and backend/device resolution."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import torch

from minitrain.distributed.runtime import (
    DistributedContext,
    read_torchrun_environment,
    resolve_distributed_device,
)


class TorchrunEnvironmentTest(unittest.TestCase):
    def test_reads_worker_identity(self) -> None:
        rank, local_rank, world_size = read_torchrun_environment(
            {"RANK": "3", "LOCAL_RANK": "1", "WORLD_SIZE": "4"}
        )
        self.assertEqual((rank, local_rank, world_size), (3, 1, 4))

    def test_missing_environment_explains_torchrun_requirement(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "launch this program with torchrun"):
            read_torchrun_environment({})

    def test_rejects_rank_outside_world(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "smaller than WORLD_SIZE"):
            read_torchrun_environment(
                {"RANK": "2", "LOCAL_RANK": "0", "WORLD_SIZE": "2"}
            )


class DistributedDeviceTest(unittest.TestCase):
    def test_gloo_maps_to_cpu(self) -> None:
        backend, device = resolve_distributed_device("gloo", local_rank=7)
        self.assertEqual(backend, "gloo")
        self.assertEqual(device, torch.device("cpu"))

    @patch("minitrain.distributed.runtime.torch.cuda.device_count", return_value=2)
    @patch("minitrain.distributed.runtime.torch.cuda.is_available", return_value=True)
    def test_nccl_maps_local_rank_to_cuda(
        self, _cuda_available: object, _device_count: object
    ) -> None:
        backend, device = resolve_distributed_device("nccl", local_rank=1)
        self.assertEqual(backend, "nccl")
        self.assertEqual(device, torch.device("cuda", 1))

    @patch("minitrain.distributed.runtime.torch.cuda.is_available", return_value=False)
    def test_nccl_rejects_missing_cuda(self, _cuda_available: object) -> None:
        with self.assertRaisesRegex(RuntimeError, "NCCL requires CUDA"):
            resolve_distributed_device("nccl", local_rank=0)

    def test_context_identifies_main_process(self) -> None:
        main = DistributedContext(0, 0, 2, "gloo", torch.device("cpu"))
        worker = DistributedContext(1, 1, 2, "gloo", torch.device("cpu"))
        self.assertTrue(main.is_main_process)
        self.assertFalse(worker.is_main_process)


if __name__ == "__main__":
    unittest.main()
