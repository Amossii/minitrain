"""CPU unit tests for FSDP2 workload metadata and model grouping inputs."""

from __future__ import annotations

import unittest

from minitrain.model import MiniTransformer
from scripts.verify_fsdp2_correctness import correctness_model_config


class FSDP2WorkloadTest(unittest.TestCase):
    def test_correctness_model_uses_real_transformer_blocks(self) -> None:
        config = correctness_model_config(seq_len=16)
        model = MiniTransformer(config)
        self.assertEqual(len(model.blocks), 2)
        self.assertEqual(config.hidden_size, config.num_heads * config.head_dim)
        self.assertEqual(config.seq_len, 16)

    def test_correctness_model_rejects_invalid_sequence_length(self) -> None:
        with self.assertRaisesRegex(ValueError, "seq_len"):
            correctness_model_config(seq_len=0)


if __name__ == "__main__":
    unittest.main()
