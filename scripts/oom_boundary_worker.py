"""运行一个隔离的 DDP、FSDP2 或 TP 模型规模探测任务。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from minitrain.data import SyntheticTokenDataset
from minitrain.distributed.fsdp2 import apply_fsdp2
from minitrain.distributed.oom_boundary import (
    build_boundary_config,
    transformer_parameter_count,
)
from minitrain.distributed.runtime import cleanup_distributed, init_distributed
from minitrain.distributed.fsdp_tp_comparison import per_rank_batch_size
from minitrain.model import MiniTransformer
from minitrain.tensor_parallel.transformer import TensorParallelTransformer
from minitrain.training import train_step


# 输入是一个策略和一个模型目标规模，内部状态经历完整训练更新；
# 仅当所有 rank 成功时，rank 0 才输出带真实峰值的 JSON 成功记录。
def main() -> int:
    """执行一次模型规模探测，并在成功时写入结果。"""

    parser = argparse.ArgumentParser(description="Probe one distributed OOM point.")
    parser.add_argument("--strategy", choices=("ddp", "fsdp2", "tp"), required=True)
    parser.add_argument("--backend", choices=("nccl",), default="nccl")
    parser.add_argument("--target-parameters-millions", type=float, required=True)
    parser.add_argument("--vocab-size", type=int, default=32_000)
    parser.add_argument("--num-layers", type=int, default=12)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--mlp-ratio", type=int, default=3)
    parser.add_argument("--local-batch-size", type=int, default=1)
    parser.add_argument("--global-batch-size", type=int)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.local_batch_size, args.steps) <= 0:
        parser.error("--local-batch-size and --steps must be positive")
    if args.global_batch_size is not None and args.global_batch_size <= 0:
        parser.error("--global-batch-size must be positive")

    config = build_boundary_config(
        args.target_parameters_millions,
        args.seq_len,
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
    )
    num_parameters = transformer_parameter_count(config)

    context = None
    try:
        context = init_distributed(backend=args.backend)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(context.device)

        # 先在 CPU 构造模型。FSDP2 会依据 CUDA DeviceMesh 直接生成本地 shard，
        # 避免“先复制完整 GPU 模型再分片”人为降低它的最大可训练边界。
        model = MiniTransformer(config) if args.strategy != "tp" else None
        if args.strategy == "ddp":
            assert model is not None
            model.to(context.device)
            train_model = DistributedDataParallel(
                model,
                device_ids=[context.local_rank],
                output_device=context.local_rank,
                broadcast_buffers=False,
            )
        elif args.strategy == "fsdp2":
            assert model is not None
            train_model = apply_fsdp2(model, context.device.type)
        else:
            train_model = TensorParallelTransformer(config, device=context.device)
        optimizer = torch.optim.AdamW(train_model.parameters(), lr=args.learning_rate)

        if args.global_batch_size is None:
            # 保留原有 DDP/FSDP2 CLI 语义，旧实验仍以 local batch 推导 global batch。
            global_batch_size = args.local_batch_size * context.world_size
            batch_size_per_rank = args.local_batch_size
        else:
            global_batch_size = args.global_batch_size
            if args.strategy == "ddp":
                if global_batch_size % context.world_size != 0:
                    raise ValueError("DDP global batch must be divisible by world size")
                batch_size_per_rank = global_batch_size // context.world_size
            else:
                batch_size_per_rank = per_rank_batch_size(
                    args.strategy, global_batch_size, context.world_size
                )
        dataset = SyntheticTokenDataset(
            global_batch_size * args.steps,
            args.seq_len,
            config.vocab_size,
            args.seed,
        )
        sampler = None
        if args.strategy in ("ddp", "fsdp2"):
            sampler = DistributedSampler(
                dataset, context.world_size, context.rank, shuffle=False, drop_last=True
            )
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size_per_rank,
            sampler=sampler,
            drop_last=True,
            pin_memory=True,
        )
        for batch in dataloader:
            train_step(train_model, optimizer, batch, context.device)
        torch.cuda.synchronize(context.device)

        peak_allocated = torch.tensor(
            torch.cuda.max_memory_allocated(context.device),
            dtype=torch.int64,
            device=context.device,
        )
        peak_reserved = torch.tensor(
            torch.cuda.max_memory_reserved(context.device),
            dtype=torch.int64,
            device=context.device,
        )
        dist.all_reduce(peak_allocated, op=dist.ReduceOp.MAX)
        dist.all_reduce(peak_reserved, op=dist.ReduceOp.MAX)
        if context.is_main_process:
            result = {
                "actual_parameters": num_parameters,
                "hidden_size": config.hidden_size,
                "intermediate_size": config.intermediate_size,
                "peak_memory_allocated": int(peak_allocated.item()),
                "peak_memory_reserved": int(peak_reserved.item()),
            }
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
            print(
                f"strategy={args.strategy} parameters={num_parameters} "
                f"peak_allocated={result['peak_memory_allocated']}",
                flush=True,
            )
        dist.barrier()
    finally:
        if context is not None:
            cleanup_distributed()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
