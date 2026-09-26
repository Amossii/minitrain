"""CPU-only correctness tests for environment inspection and validation."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from minitrain.environment import (
    _format_nccl_version,
    format_environment_report,
    inspect_environment,
    validate_environment,
)


class _FakeNccl:
    @staticmethod
    def version() -> tuple[int, int, int]:
        return (2, 21, 5)


class _FakeCuda:
    nccl = _FakeNccl()

    @staticmethod
    def is_available() -> bool:
        return True

    @staticmethod
    def device_count() -> int:
        return 2

    @staticmethod
    def get_device_name(index: int) -> str:
        return f"Fake GPU {index}"


class _FakeDistributed:
    @staticmethod
    def is_nccl_available() -> bool:
        return True


class _FakeVersion:
    cuda = "12.1"


class _FakeTorch:
    __version__ = "2.4.0"
    cuda = _FakeCuda()
    distributed = _FakeDistributed()
    version = _FakeVersion()


class EnvironmentTest(unittest.TestCase):
    # This verifies the successful two-GPU path without requiring test hardware.
    @patch("minitrain.environment._read_gpu_topology", return_value="GPU0 GPU1")
    def test_inspect_environment(self, _topology: object) -> None:
        report = inspect_environment(_FakeTorch())

        self.assertEqual(report.pytorch_version, "2.4.0")
        self.assertEqual(report.cuda_build_version, "12.1")
        self.assertEqual(report.nccl_version, "2.21.5")
        self.assertEqual(report.gpu_names, ("Fake GPU 0", "Fake GPU 1"))
        self.assertEqual(validate_environment(report, required_gpus=2), [])

        output = format_environment_report(report)
        self.assertIn("GPU count: 2", output)
        self.assertIn("GPU 1: Fake GPU 1", output)

    # Integer formatting covers the alternate representation used by PyTorch.
    def test_format_integer_nccl_version(self) -> None:
        self.assertEqual(_format_nccl_version(22105), "2.21.5")

    # Validation must report every missing prerequisite, not merely the first.
    def test_validation_explains_missing_gpu_requirements(self) -> None:
        report = inspect_environment()
        failures = validate_environment(report, required_gpus=2)

        if report.gpu_count < 2:
            self.assertTrue(any("expected at least 2" in item for item in failures))


if __name__ == "__main__":
    unittest.main()

