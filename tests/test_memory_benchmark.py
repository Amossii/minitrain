"""CPU unit tests for DDP/FSDP2 memory accounting and fairness."""

from __future__ import annotations

import unittest

import torch

from minitrain.distributed.memory import (
    build_memory_comparison,
    local_tensor_bytes,
    theoretical_adam_bytes_per_rank,
)


def _row(strategy: str, peak: int) -> dict[str, str | int | float]:
    return {
        "strategy": strategy,
        "model_name": "tiny",
        "precision": "fp32",
        "world_size": 2,
        "local_batch_size": 1,
        "global_batch_size": 2,
        "seq_len": 128,
        "num_parameters": 100,
        "warmup_steps": 3,
        "measured_steps": 5,
        "observed_peak_allocated_bytes": peak,
    }


class MemoryAccountingTest(unittest.TestCase):
    def test_local_tensor_bytes_uses_dtype_size(self) -> None:
        self.assertEqual(local_tensor_bytes(torch.zeros(7, dtype=torch.float32)), 28)
        self.assertEqual(local_tensor_bytes("not a tensor"), 0)

    def test_fp32_adam_theory_shards_all_four_model_states(self) -> None:
        self.assertEqual(theoretical_adam_bytes_per_rank(100, 2, "ddp"), 1600)
        self.assertEqual(theoretical_adam_bytes_per_rank(100, 2, "fsdp2"), 800)

    def test_comparison_calculates_observed_saving(self) -> None:
        rows = build_memory_comparison([_row("ddp", 1000), _row("fsdp2", 700)])
        fsdp2 = next(row for row in rows if row["strategy"] == "fsdp2")
        self.assertAlmostEqual(fsdp2["peak_allocated_ratio_vs_ddp"], 0.7)
        self.assertAlmostEqual(fsdp2["peak_allocated_saving_vs_ddp"], 0.3)

    def test_comparison_rejects_unfair_workloads(self) -> None:
        ddp = _row("ddp", 1000)
        fsdp2 = _row("fsdp2", 700)
        fsdp2["local_batch_size"] = 2
        with self.assertRaisesRegex(ValueError, "local_batch_size"):
            build_memory_comparison([ddp, fsdp2])


if __name__ == "__main__":
    unittest.main()
