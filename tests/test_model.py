"""CPU correctness tests for the handwritten decoder-only Transformer."""

from __future__ import annotations

import unittest

import torch

from minitrain.config import ModelConfig
from minitrain.model import MiniTransformer, RMSNorm, TransformerBlock


# This deliberately tiny shape keeps mathematical tests fast on CPU while
# exercising exactly the same implementation used by named model presets.
def _test_config() -> ModelConfig:
    """Return a compact valid configuration for unit tests."""

    return ModelConfig(
        vocab_size=32,
        seq_len=8,
        hidden_size=16,
        num_layers=2,
        num_heads=4,
        intermediate_size=48,
    )


class RMSNormTest(unittest.TestCase):
    def test_matches_reference_formula(self) -> None:
        norm = RMSNorm(hidden_size=3, eps=1e-5)
        hidden_states = torch.tensor([[[1.0, 2.0, 3.0]]])

        actual = norm(hidden_states)
        expected = hidden_states * torch.rsqrt(
            hidden_states.pow(2).mean(dim=-1, keepdim=True) + 1e-5
        )

        torch.testing.assert_close(actual, expected)


class TransformerBlockTest(unittest.TestCase):
    def test_block_preserves_hidden_shape(self) -> None:
        block = TransformerBlock(_test_config())
        hidden_states = torch.randn(2, 5, 16)

        output = block(hidden_states)

        self.assertEqual(tuple(output.shape), (2, 5, 16))


class MiniTransformerTest(unittest.TestCase):
    def test_model_maps_token_ids_to_vocab_logits(self) -> None:
        model = MiniTransformer(_test_config())
        input_ids = torch.randint(0, 32, (2, 5))

        logits = model(input_ids)

        self.assertEqual(tuple(logits.shape), (2, 5, 32))
        self.assertTrue(torch.isfinite(logits).all())

    # Changing future tokens must not alter logits at earlier positions.
    def test_attention_is_causal(self) -> None:
        torch.manual_seed(7)
        model = MiniTransformer(_test_config()).eval()
        first = torch.tensor([[1, 2, 3, 4, 5, 6]])
        changed_future = torch.tensor([[1, 2, 3, 20, 21, 22]])

        with torch.no_grad():
            first_logits = model(first)
            changed_logits = model(changed_future)

        torch.testing.assert_close(first_logits[:, :3], changed_logits[:, :3])
        self.assertFalse(torch.equal(first_logits[:, 3:], changed_logits[:, 3:]))

    def test_backward_reaches_every_parameter(self) -> None:
        model = MiniTransformer(_test_config())
        input_ids = torch.randint(0, 32, (2, 5))

        model(input_ids).square().mean().backward()

        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, msg=f"missing gradient for {name}")
            self.assertTrue(
                torch.isfinite(parameter.grad).all(),
                msg=f"non-finite gradient for {name}",
            )

    def test_rejects_invalid_input_contract(self) -> None:
        model = MiniTransformer(_test_config())

        with self.assertRaisesRegex(ValueError, "shape"):
            model(torch.ones(8, dtype=torch.long))
        with self.assertRaisesRegex(TypeError, "int64"):
            model(torch.ones(1, 8, dtype=torch.float32))
        with self.assertRaisesRegex(ValueError, "exceeds"):
            model(torch.ones(1, 9, dtype=torch.long))


if __name__ == "__main__":
    unittest.main()
