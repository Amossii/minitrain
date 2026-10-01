"""CPU correctness tests for the explicit single-device training lifecycle."""

from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F

from minitrain.config import ModelConfig
from minitrain.data import create_synthetic_batch
from minitrain.model import MiniTransformer
from minitrain.training import causal_lm_loss, resolve_device, train_step


def _test_config() -> ModelConfig:
    """Return a small model that keeps optimizer tests fast on CPU."""

    return ModelConfig(
        vocab_size=32,
        seq_len=8,
        hidden_size=16,
        num_layers=1,
        num_heads=4,
        intermediate_size=48,
    )


class CausalLMLossTest(unittest.TestCase):
    # Flattening must preserve the correspondence of every [b,t] prediction.
    def test_matches_pytorch_reference(self) -> None:
        logits = torch.randn(2, 3, 5)
        labels = torch.randint(0, 5, (2, 3))

        actual = causal_lm_loss(logits, labels)
        expected = F.cross_entropy(logits.reshape(-1, 5), labels.reshape(-1))

        torch.testing.assert_close(actual, expected)

    def test_rejects_mismatched_shapes(self) -> None:
        with self.assertRaisesRegex(ValueError, "share"):
            causal_lm_loss(torch.randn(2, 3, 5), torch.ones(2, 4, dtype=torch.long))


class TrainStepTest(unittest.TestCase):
    def test_one_step_updates_parameters_and_optimizer_state(self) -> None:
        torch.manual_seed(7)
        model = MiniTransformer(_test_config())
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        batch = create_synthetic_batch(2, 8, 32, seed=11)
        before = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
        }

        loss = train_step(model, optimizer, batch, torch.device("cpu"))

        self.assertTrue(torch.isfinite(torch.tensor(loss)))
        self.assertGreater(len(optimizer.state), 0)
        changed = [
            not torch.equal(before[name], parameter.detach())
            for name, parameter in model.named_parameters()
        ]
        self.assertTrue(any(changed))
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, msg=f"missing gradient for {name}")
            self.assertTrue(torch.isfinite(parameter.grad).all())

    # Equal seeds, model state, data, and update order must reproduce losses.
    def test_training_step_is_reproducible(self) -> None:
        losses: list[float] = []
        batch = create_synthetic_batch(2, 8, 32, seed=19)
        for _ in range(2):
            torch.manual_seed(13)
            model = MiniTransformer(_test_config())
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            losses.append(train_step(model, optimizer, batch, torch.device("cpu")))

        self.assertEqual(losses[0], losses[1])

    def test_cpu_device_resolution(self) -> None:
        self.assertEqual(resolve_device("cpu"), torch.device("cpu"))
        with self.assertRaisesRegex(ValueError, "auto, cpu, cuda"):
            resolve_device("tpu")


if __name__ == "__main__":
    unittest.main()
