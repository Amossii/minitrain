"""Display a validated MiniTrain model and training configuration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from minitrain.config import (  # noqa: E402
    TrainConfig,
    available_model_configs,
    get_model_config,
)


# Input comes from explicit CLI experiment choices; output is JSON that humans
# and later automation can inspect before a training process is launched.
def main() -> int:
    """Build, validate, and print the selected experiment configuration."""

    parser = argparse.ArgumentParser(description="Show a MiniTrain configuration.")
    parser.add_argument(
        "--model",
        choices=available_model_configs(),
        default="tiny",
        help="model workload preset",
    )
    parser.add_argument("--local-batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument(
        "--precision", choices=("fp32", "fp16", "bf16"), default="fp32"
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    model_config = get_model_config(args.model)
    train_config = TrainConfig(
        local_batch_size=args.local_batch_size,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        precision=args.precision,
        seed=args.seed,
    )
    output = {
        "model_name": args.model,
        "model": model_config.to_dict(),
        "train": train_config.to_dict(),
    }
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
