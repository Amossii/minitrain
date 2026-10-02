"""CPU unit tests for collective size sweeps, bandwidth, and CSV schema."""

from __future__ import annotations

import csv
from pathlib import Path
import tempfile
import unittest

from minitrain.distributed.benchmark import (
    CollectiveBenchmarkRecord,
    all_reduce_bandwidths,
    generate_message_sizes,
    write_collective_csv,
)


class CollectiveBenchmarkMathTest(unittest.TestCase):
    def test_message_size_sweep_includes_maximum(self) -> None:
        self.assertEqual(generate_message_sizes(1_024, 16_384, 4), [1_024, 4_096, 16_384])
        self.assertEqual(generate_message_sizes(1_024, 10_000, 4), [1_024, 4_096, 10_000])

    def test_rejects_non_growing_factor(self) -> None:
        with self.assertRaisesRegex(ValueError, "greater than 1"):
            generate_message_sizes(1_024, 4_096, 1)

    def test_two_rank_all_reduce_bandwidth_formula(self) -> None:
        algorithm, bus = all_reduce_bandwidths(
            message_size_bytes=1_000_000_000,
            latency_seconds=1.0,
            world_size=2,
        )
        self.assertEqual(algorithm, 1.0)
        self.assertEqual(bus, 1.0)

    def test_four_rank_bus_correction(self) -> None:
        algorithm, bus = all_reduce_bandwidths(1_000_000_000, 1.0, 4)
        self.assertEqual(algorithm, 1.0)
        self.assertEqual(bus, 1.5)

    def test_csv_preserves_units_and_workload(self) -> None:
        record = CollectiveBenchmarkRecord(
            collective="all_reduce",
            backend="nccl",
            world_size=2,
            dtype="float32",
            message_size_bytes=1_024,
            num_elements=256,
            warmup_iterations=5,
            iterations=20,
            latency_ms=0.1,
            algorithm_bandwidth_gbps=0.01024,
            bus_bandwidth_gbps=0.01024,
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "collective.csv"
            write_collective_csv(path, [record])
            with path.open(encoding="utf-8", newline="") as csv_file:
                rows = list(csv.DictReader(csv_file))

        self.assertEqual(rows[0]["message_size_bytes"], "1024")
        self.assertEqual(rows[0]["latency_ms"], "0.1")
        self.assertEqual(rows[0]["bus_bandwidth_gbps"], "0.01024")


if __name__ == "__main__":
    unittest.main()
