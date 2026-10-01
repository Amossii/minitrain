"""Train MiniTransformer with an explicit single-device PyTorch loop."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from minitrain.config import (  # noqa: E402
    TrainConfig,
    available_model_configs,
    get_model_config,
)
from minitrain.data import SyntheticTokenDataset  # noqa: E402
from minitrain.model import MiniTransformer, count_parameters  # noqa: E402
from minitrain.training import resolve_device, train_step  # noqa: E402


# CLI inputs define one reproducible single-device workload. Output is one loss
# per optimizer update; standardized timing and CSV metrics arrive in Step 6.
def main() -> int:
    """Build the workload and run the explicit FP32 training lifecycle."""

    parser = argparse.ArgumentParser(description="Train MiniTrain on one device.")
    parser.add_argument(
        "--model", choices=available_model_configs(), default="tiny"
    )
    parser.add_argument("--local-batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()

    model_config = get_model_config(args.model)
    if args.seq_len > model_config.seq_len:
        parser.error(
            f"--seq-len cannot exceed the {args.model} maximum "
            f"of {model_config.seq_len}"
        )
    train_config = TrainConfig(
        local_batch_size=args.local_batch_size,
        learning_rate=args.learning_rate,
        max_steps=args.steps,
        precision="fp32",
        seed=args.seed,
    )
    device = resolve_device(args.device)

    torch.manual_seed(train_config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(train_config.seed)

    model = MiniTransformer(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=train_config.learning_rate
    )
    dataset = SyntheticTokenDataset(
        num_samples=train_config.local_batch_size * train_config.max_steps,
        seq_len=args.seq_len,
        vocab_size=model_config.vocab_size,
        seed=train_config.seed,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=train_config.local_batch_size,
        shuffle=False,
        drop_last=True,
        pin_memory=device.type == "cuda",
    )

    print(f"device={device}")
    print(f"parameters={count_parameters(model):,}")
    for step, batch in enumerate(dataloader):
        loss = train_step(model, optimizer, batch, device)
        print(f"step={step} loss={loss:.6f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
