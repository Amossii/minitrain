"""Inspect a small deterministic synthetic language-model batch."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from minitrain.config import get_model_config  # noqa: E402
from minitrain.data import create_synthetic_batch  # noqa: E402


# CLI inputs select a workload and compact display size; output exposes tensor
# shape, dtype, and shift alignment before data reaches a future model.
def main() -> int:
    """Generate and print one synthetic batch for manual verification."""

    parser = argparse.ArgumentParser(description="Inspect MiniTrain synthetic data.")
    parser.add_argument("--model", choices=("tiny", "small", "medium"), default="tiny")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument(
        "--seq-len",
        type=int,
        default=8,
        help="short display length; defaults to 8 instead of the model maximum",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    model_config = get_model_config(args.model)
    if args.seq_len > model_config.seq_len:
        parser.error(
            f"--seq-len cannot exceed the {args.model} maximum "
            f"of {model_config.seq_len}"
        )

    batch = create_synthetic_batch(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        vocab_size=model_config.vocab_size,
        seed=args.seed,
    )
    print(f"input_ids shape: {tuple(batch['input_ids'].shape)}")
    print(f"labels shape: {tuple(batch['labels'].shape)}")
    print(f"dtype: {batch['input_ids'].dtype}")
    print(f"input_ids[0]: {batch['input_ids'][0].tolist()}")
    print(f"labels[0]:    {batch['labels'][0].tolist()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
