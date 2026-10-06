"""训练包含 vocabulary sharding 的完整 Tensor Parallel MiniTransformer。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader

from minitrain.config import available_model_configs, get_model_config
from minitrain.data import SyntheticTokenDataset
from minitrain.distributed.runtime import cleanup_distributed, init_distributed
from minitrain.tensor_parallel.full_transformer import FullTensorParallelTransformer
from minitrain.training import train_step


# CLI 输入定义一个 Full TP workload；内部状态包括 vocabulary/block shards 和本地
# AdamW moments；每次调用后所有 rank 共同完成一个相同 global batch 的参数更新。
def main() -> int:
    """初始化并训练 vocabulary + block Tensor Parallel Transformer。"""

    parser = argparse.ArgumentParser(description="Train Transformer with full TP.")
    parser.add_argument("--backend", choices=("gloo", "nccl"), default="nccl")
    parser.add_argument("--model", choices=available_model_configs(), default="tiny")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if min(args.batch_size, args.seq_len, args.steps) <= 0:
        parser.error("batch size, sequence length, and steps must be positive")
    if args.learning_rate <= 0:
        parser.error("learning rate must be positive")
    model_config = get_model_config(args.model)
    if args.seq_len > model_config.seq_len:
        parser.error(f"--seq-len exceeds the {args.model} maximum")

    context = None
    try:
        context = init_distributed(backend=args.backend)
        torch.manual_seed(args.seed)
        if context.device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed)
        model = FullTensorParallelTransformer(model_config, device=context.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
        dataset = SyntheticTokenDataset(
            args.batch_size * args.steps,
            args.seq_len,
            model_config.vocab_size,
            args.seed,
        )
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
            f"local_parameters={local_parameters} "
            f"local_vocab_size={model.token_embedding.local_vocab_size}",
            flush=True,
        )
        for step, batch in enumerate(dataloader):
            loss = train_step(
                model, optimizer, batch, context.device, model.loss
            )
            if context.is_main_process:
                print(f"step={step} loss={loss:.6f}", flush=True)
    finally:
        if context is not None:
            cleanup_distributed()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
