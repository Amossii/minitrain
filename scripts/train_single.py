"""Train MiniTransformer with an explicit single-device PyTorch loop."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
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
from minitrain.metrics import StepMetrics, write_metrics_csv  # noqa: E402
from minitrain.training import (  # noqa: E402
    measure_train_step,
    resolve_device,
    train_step,
)


# CLI inputs define one reproducible single-device workload. Output is a stable
# row per measured optimizer update plus a CSV consumable by later benchmarks.
def main() -> int:
    """Warm up, measure FP32 training steps, and persist unified metrics."""

    parser = argparse.ArgumentParser(description="Train MiniTrain on one device.")
    parser.add_argument(
        "--model", choices=available_model_configs(), default="tiny"
    )
    parser.add_argument("--local-batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="CSV path; defaults to a timestamped file under results/raw",
    )
    args = parser.parse_args()
    if args.warmup_steps < 0:
        parser.error("--warmup-steps must be non-negative")

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
        num_samples=train_config.local_batch_size
        * (args.warmup_steps + train_config.max_steps),
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
    dataloader_iterator = iter(dataloader)
    for _ in range(args.warmup_steps):
        train_step(model, optimizer, next(dataloader_iterator), device)

    num_parameters = count_parameters(model)
    records: list[StepMetrics] = []
    for step in range(train_config.max_steps):
        measurement = measure_train_step(
            model, optimizer, next(dataloader_iterator), device
        )
        record = StepMetrics.from_measurement(
            step=step,
            loss=measurement.loss,
            step_time=measurement.step_time,
            forward_time=measurement.forward_time,
            backward_time=measurement.backward_time,
            optimizer_time=measurement.optimizer_time,
            peak_memory_allocated=measurement.peak_memory_allocated,
            peak_memory_reserved=measurement.peak_memory_reserved,
            world_size=1,
            local_batch_size=train_config.local_batch_size,
            seq_len=args.seq_len,
            num_parameters=num_parameters,
            model_name=args.model,
            precision=train_config.precision,
            strategy="single",
        )
        records.append(record)
        print(
            f"step={step} loss={record.loss:.6f} "
            f"step_time={record.step_time:.6f}s "
            f"tokens_per_second={record.tokens_per_second:.2f}"
        )

    output_path = args.output
    if output_path is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_path = PROJECT_ROOT / "results" / "raw" / f"single_{timestamp}.csv"
    write_metrics_csv(output_path, records)
    print(f"metrics_csv={output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
