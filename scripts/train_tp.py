"""使用两张 GPU 训练手写 Tensor Parallel MiniTransformer。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader

from minitrain.config import TrainConfig, available_model_configs, get_model_config
from minitrain.data import SyntheticTokenDataset
from minitrain.distributed.runtime import cleanup_distributed, init_distributed
from minitrain.tensor_parallel.transformer import TensorParallelTransformer
from minitrain.training import train_step


# CLI 输入定义一个 TP workload；所有 rank 消费同一个 batch，各自更新本地 Linear
# shard 和 replicated 参数。输出由 rank 0 打印 loss 与本地参数 ownership。
def main() -> int:
    """初始化并训练 Tensor Parallel MiniTransformer。"""

    parser = argparse.ArgumentParser(description="Train Transformer with TP.")
    parser.add_argument("--backend", choices=("gloo", "nccl"), default="nccl")
    parser.add_argument("--model", choices=available_model_configs(), default="tiny")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    model_config = get_model_config(args.model)
    if args.seq_len > model_config.seq_len:
        parser.error(f"--seq-len exceeds the {args.model} maximum")
    train_config = TrainConfig(
        local_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_steps=args.steps,
        precision="fp32",
        seed=args.seed,
    )

    context = None
    try:
        context = init_distributed(backend=args.backend)
        torch.manual_seed(args.seed)
        if context.device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed)
        model = TensorParallelTransformer(model_config, device=context.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
        dataset = SyntheticTokenDataset(
            args.batch_size * args.steps,
            args.seq_len,
            model_config.vocab_size,
            args.seed,
        )
        # TP ranks 合作处理同一批样本，因此这里不能使用 DistributedSampler。
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=True,
            pin_memory=context.device.type == "cuda",
        )
        local_parameters = sum(parameter.numel() for parameter in model.parameters())
        print(
            f"rank={context.rank} tp_world_size={context.world_size} "
            f"local_parameters={local_parameters}",
            flush=True,
        )
        for step, batch in enumerate(dataloader):
            loss = train_step(model, optimizer, batch, context.device)
            if context.is_main_process:
                print(f"step={step} loss={loss:.6f}", flush=True)
    finally:
        if context is not None:
            cleanup_distributed()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
