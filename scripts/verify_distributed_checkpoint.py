"""验证 DDP/FSDP2 checkpoint 恢复后可无缝继续训练。"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict
from torch.nn.parallel import DistributedDataParallel

from minitrain.config import available_model_configs, get_model_config
from minitrain.data import create_synthetic_batch
from minitrain.distributed.checkpointing import (
    load_training_checkpoint,
    save_training_checkpoint,
)
from minitrain.distributed.fsdp2 import apply_fsdp2
from minitrain.distributed.runtime import cleanup_distributed, init_distributed
from minitrain.model import MiniTransformer
from minitrain.training import train_step


# 输入是策略、结构和当前 rank 设备，输出包装后的可训练模型。DDP 保留完整副本，
# FSDP2 在构造后立即把参数转为 DTensor shard。
def build_model(
    strategy: str,
    model_config: object,
    device: torch.device,
    local_rank: int,
) -> nn.Module:
    """为 checkpoint correctness 构造 DDP 或 FSDP2 模型。"""

    model = MiniTransformer(model_config).to(device)  # type: ignore[arg-type]
    if strategy == "ddp":
        if device.type == "cuda":
            return DistributedDataParallel(
                model,
                device_ids=[local_rank],
                output_device=local_rank,
                broadcast_buffers=False,
            )
        return DistributedDataParallel(model, broadcast_buffers=False)
    return apply_fsdp2(model, device.type)


# 输入是 global batch 描述和 rank，输出当前 DP rank 独占的连续样本区间。
# source/restored 两条分支重复调用相同 step 时会得到完全相同的本地 batch。
def local_batch_for_step(
    *,
    step: int,
    local_batch_size: int,
    seq_len: int,
    vocab_size: int,
    seed: int,
    rank: int,
    world_size: int,
) -> dict[str, torch.Tensor]:
    """生成确定性 global batch，并切出当前 rank 的 data-parallel shard。"""

    global_batch = create_synthetic_batch(
        local_batch_size * world_size,
        seq_len,
        vocab_size,
        seed=seed + step * local_batch_size * world_size,
    )
    start = rank * local_batch_size
    end = start + local_batch_size
    return {name: tensor[start:end] for name, tensor in global_batch.items()}


# 输入是两份相同逻辑模型，输出为空；所有 rank 参与必要的 FSDP AllGather，仅 rank 0
# 持有 CPU full state 并逐 tensor 比较。
def assert_models_equal(expected: nn.Module, actual: nn.Module, rank: int) -> None:
    """比较 DDP/FSDP2 的逻辑完整参数，而不是误比本地 shard。"""

    options = StateDictOptions(full_state_dict=True, cpu_offload=True)
    expected_state = get_model_state_dict(expected, options=options)
    actual_state = get_model_state_dict(actual, options=options)
    if rank == 0:
        if expected_state.keys() != actual_state.keys():
            raise AssertionError("restored model state keys do not match")
        for name in expected_state:
            torch.testing.assert_close(
                actual_state[name], expected_state[name], atol=0, rtol=0, msg=lambda _: name
            )


# CLI 输入指定策略和共享 checkpoint 路径；内部状态经过“训练→保存→新对象恢复→
# 两分支继续一步”；输出 PASS 证明参数、optimizer moments 和 step 都正确恢复。
def main() -> int:
    """执行分布式 checkpoint 的端到端 resume correctness 验证。"""

    parser = argparse.ArgumentParser(description="Verify distributed checkpoint resume.")
    parser.add_argument("--strategy", choices=("ddp", "fsdp2"), required=True)
    parser.add_argument("--backend", choices=("gloo", "nccl"), default="nccl")
    parser.add_argument("--model", choices=available_model_configs(), default="tiny")
    parser.add_argument("--local-batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--pre-steps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    args = parser.parse_args()
    if min(args.local_batch_size, args.seq_len, args.pre_steps) <= 0:
        parser.error("batch size, sequence length, and pre-steps must be positive")
    if args.learning_rate <= 0:
        parser.error("learning rate must be positive")
    model_config = get_model_config(args.model)
    if args.seq_len > model_config.seq_len:
        parser.error(f"--seq-len exceeds the {args.model} maximum")

    context = None
    try:
        context = init_distributed(args.backend)
        torch.manual_seed(args.seed)
        if context.device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed)
        source = build_model(
            args.strategy, model_config, context.device, context.local_rank
        )
        source_optimizer = torch.optim.AdamW(
            source.parameters(), lr=args.learning_rate
        )

        for step in range(args.pre_steps):
            batch = local_batch_for_step(
                step=step,
                local_batch_size=args.local_batch_size,
                seq_len=args.seq_len,
                vocab_size=model_config.vocab_size,
                seed=args.seed,
                rank=context.rank,
                world_size=context.world_size,
            )
            loss = train_step(source, source_optimizer, batch, context.device)
            if not math.isfinite(loss):
                raise AssertionError(f"non-finite pre-checkpoint loss at step {step}")

        save_training_checkpoint(
            args.checkpoint_dir, source, source_optimizer, step=args.pre_steps
        )
        if context.is_main_process:
            print(
                f"checkpoint_saved={args.checkpoint_dir} step={args.pre_steps}",
                flush=True,
            )

        # 使用不同 seed 构造新对象，确保相等来自 checkpoint，而不是初始化巧合。
        torch.manual_seed(args.seed + 10_000)
        if context.device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed + 10_000)
        restored = build_model(
            args.strategy, model_config, context.device, context.local_rank
        )
        restored_optimizer = torch.optim.AdamW(
            restored.parameters(), lr=args.learning_rate
        )
        restored_step = load_training_checkpoint(
            args.checkpoint_dir, restored, restored_optimizer
        )
        if restored_step != args.pre_steps:
            raise AssertionError(
                f"expected restored step {args.pre_steps}, got {restored_step}"
            )
        assert_models_equal(source, restored, context.rank)

        resume_batch = local_batch_for_step(
            step=restored_step,
            local_batch_size=args.local_batch_size,
            seq_len=args.seq_len,
            vocab_size=model_config.vocab_size,
            seed=args.seed,
            rank=context.rank,
            world_size=context.world_size,
        )
        source_loss = train_step(
            source, source_optimizer, resume_batch, context.device
        )
        restored_loss = train_step(
            restored, restored_optimizer, resume_batch, context.device
        )
        torch.testing.assert_close(
            torch.tensor(restored_loss),
            torch.tensor(source_loss),
            atol=0,
            rtol=0,
        )
        assert_models_equal(source, restored, context.rank)
        if context.is_main_process:
            print("model_state_after_load=PASS", flush=True)
            print("optimizer_state_after_resumed_step=PASS", flush=True)
            print("training_step_resume=PASS", flush=True)
            print(f"{args.strategy.upper()} DISTRIBUTED CHECKPOINT PASSED", flush=True)
        dist.barrier()
    finally:
        if context is not None:
            cleanup_distributed()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
