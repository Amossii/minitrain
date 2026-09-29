"""CPU-only correctness tests for MiniTrain experiment configuration."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import unittest

from minitrain.config import (
    ModelConfig,
    TrainConfig,
    available_model_configs,
    get_model_config,
)


class ModelConfigTest(unittest.TestCase):
    # All public presets must satisfy the same structural invariants.
    def test_model_presets_are_valid(self) -> None:
        self.assertEqual(available_model_configs(), ("tiny", "small", "medium"))

        for name in available_model_configs():
            config = get_model_config(name)
            self.assertEqual(config.hidden_size, config.num_heads * config.head_dim)
            self.assertGreater(config.intermediate_size, config.hidden_size)

    # Uneven head widths would make attention reshaping mathematically invalid.
    def test_hidden_size_must_be_divisible_by_heads(self) -> None:
        with self.assertRaisesRegex(ValueError, "divisible"):
            ModelConfig(
                vocab_size=100,
                seq_len=16,
                hidden_size=10,
                num_layers=2,
                num_heads=3,
                intermediate_size=32,
            )

    def test_unknown_preset_lists_choices(self) -> None:
        with self.assertRaisesRegex(ValueError, "tiny, small, medium"):
            get_model_config("large")

    # Immutable shared presets cannot leak mutations between benchmark runs.
    def test_model_config_is_immutable(self) -> None:
        config = get_model_config("tiny")
        with self.assertRaises(FrozenInstanceError):
            config.hidden_size = 256  # type: ignore[misc]


class TrainConfigTest(unittest.TestCase):
    def test_global_batch_size_is_explicit(self) -> None:
        config = TrainConfig(local_batch_size=8)
        self.assertEqual(config.global_batch_size(world_size=1), 8)
        self.assertEqual(config.global_batch_size(world_size=2), 16)

    def test_invalid_training_values_fail_early(self) -> None:
        with self.assertRaisesRegex(ValueError, "local_batch_size"):
            TrainConfig(local_batch_size=0)
        with self.assertRaisesRegex(ValueError, "precision"):
            TrainConfig(precision="tf32")  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, "world_size"):
            TrainConfig().global_batch_size(world_size=0)


if __name__ == "__main__":
    unittest.main()
