"""Run a small MiniTransformer forward/backward correctness inspection."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch  # noqa: E402

from minitrain.config import available_model_configs, get_model_config  # noqa: E402
from minitrain.data import create_synthetic_batch  # noqa: E402
from minitrain.model import MiniTransformer, count_parameters  # noqa: E402


# CLI input selects a preset and compact shape; output confirms the complete
# model's tensor contract and gradient connectivity before training is added.
def main() -> int:
    """Construct a model, run forward/backward, and print observable facts."""

    parser = argparse.ArgumentParser(description="Inspect the MiniTransformer.")
    parser.add_argument(
        "--model", choices=available_model_configs(), default="tiny"
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = get_model_config(args.model)
    if args.seq_len > config.seq_len:
        parser.error(
            f"--seq-len cannot exceed the {args.model} maximum of {config.seq_len}"
        )

    torch.manual_seed(args.seed)
    model = MiniTransformer(config)
    batch = create_synthetic_batch(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        vocab_size=config.vocab_size,
        seed=args.seed,
    )
    logits = model(batch["input_ids"])

    # A simple scalar is sufficient to verify autograd connectivity here. The
    # real causal-language-model loss belongs to Step 5.
    logits.float().mean().backward()
    parameters_with_grad = sum(
        parameter.grad is not None for parameter in model.parameters()
    )

    print(f"input shape: {tuple(batch['input_ids'].shape)}")
    print(f"logits shape: {tuple(logits.shape)}")
    print(f"parameters: {count_parameters(model):,}")
    print(
        f"parameters with gradients: {parameters_with_grad}/"
        f"{sum(1 for _ in model.parameters())}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
