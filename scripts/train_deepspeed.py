"""使用同一 MiniTransformer workload 运行 DeepSpeed ZeRO-1/2/3。"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import math
import os
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from minitrain.config import TrainConfig, available_model_configs, get_model_config
from minitrain.data import SyntheticTokenDataset
from minitrain.distributed.zero import (
    build_zero_config,
    theoretical_zero_bytes_per_rank,
    zero_state_ownership,
)
from minitrain.model import MiniTransformer, count_parameters
from minitrain.training import causal_lm_loss


# 输入为空，输出外部 DeepSpeed 模块；延迟导入让没有可选依赖的 CPU 单测仍可运行，
# 同时为 Kaggle 用户提供明确安装提示，而不是难懂的顶层 ImportError。
def import_deepspeed() -> Any:
    """导入可选 DeepSpeed 依赖或抛出可操作的错误。"""

    try:
        import deepspeed
    except ImportError as exc:
        raise RuntimeError(
            "DeepSpeed is not installed; run: pip install deepspeed"
        ) from exc
    return deepspeed


# 输入是 engine 和 stage，输出完整参数数值的稳定标量签名；ZeRO-3 只在 context
# 内临时 AllGather，退出后参数立刻恢复 shard，不改变训练常驻 ownership。
def parameter_signature(engine: Any, stage: int, deepspeed_module: Any) -> float:
    """计算完整参数的平方和，用于证明 optimizer 确实更新模型。"""

    parameters = list(engine.module.parameters())
    context = (
        deepspeed_module.zero.GatheredParameters(parameters, modifier_rank=None)
        if stage == 3
        else nullcontext()
    )
    with context:
        signature = sum(
            parameter.detach().double().square().sum().item()
            for parameter in parameters
        )
    return signature


# 输入是 ZeRO engine 和 stage，输出为空；对每个完整参数广播 rank 0 副本并比较。
# Stage 3 必须逐参数 Gather，不能把本地 shard 错当成完整参数进行 correctness。
def assert_parameters_match_rank_zero(
    engine: Any, stage: int, deepspeed_module: Any
) -> None:
    """验证一次更新后所有 data-parallel rank 表示同一个完整模型。"""

    rank = dist.get_rank()
    for name, parameter in engine.module.named_parameters():
        context = (
            deepspeed_module.zero.GatheredParameters([parameter], modifier_rank=None)
            if stage == 3
            else nullcontext()
        )
        with context:
            expected = torch.empty_like(parameter)
            if rank == 0:
                expected.copy_(parameter.detach())
            dist.broadcast(expected, src=0)
            torch.testing.assert_close(
                parameter.detach(), expected, atol=1e-6, rtol=1e-5, msg=lambda _: name
            )


# CLI 输入定义一个与其他策略相同的数据并行 workload；内部状态由 DeepSpeed engine
# 根据 stage 分片，输出 loss、理论 ownership 和可选的分布式 correctness 结果。
def main() -> int:
    """初始化 DeepSpeed ZeRO 并执行显式 forward/backward/step。"""

    parser = argparse.ArgumentParser(description="Train MiniTransformer with ZeRO.")
    parser.add_argument("--zero-stage", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--model", choices=available_model_configs(), default="tiny")
    parser.add_argument("--local-batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verify-parameters", action="store_true")
    parser.add_argument("--local_rank", type=int, default=-1)
    args = parser.parse_args()

    model_config = get_model_config(args.model)
    if args.seq_len > model_config.seq_len:
        parser.error(f"--seq-len exceeds the {args.model} maximum")
    train_config = TrainConfig(
        local_batch_size=args.local_batch_size,
        learning_rate=args.learning_rate,
        max_steps=args.steps,
        precision="fp32",
        seed=args.seed,
    )
    deepspeed = import_deepspeed()
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    deepspeed.init_distributed(dist_backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    try:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        model = MiniTransformer(model_config).to(device)
        num_parameters = count_parameters(model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
        ds_config = build_zero_config(
            args.zero_stage, args.local_batch_size, world_size
        )
        engine, optimizer, _, _ = deepspeed.initialize(
            model=model,
            optimizer=optimizer,
            config=ds_config,
        )
        initial_signature = parameter_signature(engine, args.zero_stage, deepspeed)

        global_batch_size = args.local_batch_size * world_size
        dataset = SyntheticTokenDataset(
            global_batch_size * args.steps,
            args.seq_len,
            model_config.vocab_size,
            args.seed,
        )
        sampler = DistributedSampler(
            dataset, world_size, rank, shuffle=False, drop_last=True
        )
        dataloader = DataLoader(
            dataset,
            batch_size=args.local_batch_size,
            sampler=sampler,
            drop_last=True,
            pin_memory=True,
        )
        ownership = zero_state_ownership(args.zero_stage)
        if rank == 0:
            print(
                f"strategy=zero{args.zero_stage} world_size={world_size} "
                f"local_batch_size={args.local_batch_size} "
                f"global_batch_size={global_batch_size} parameters={num_parameters} "
                f"ownership={ownership} "
                f"theoretical_state_bytes_per_rank="
                f"{theoretical_zero_bytes_per_rank(num_parameters, world_size, args.zero_stage)}",
                flush=True,
            )

        for step, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            logits = engine(input_ids)
            loss = causal_lm_loss(logits, labels)
            if not math.isfinite(float(loss.detach())):
                raise AssertionError(f"non-finite loss at step {step}")
            engine.backward(loss)
            engine.step()

            mean_loss = loss.detach().clone()
            dist.all_reduce(mean_loss, op=dist.ReduceOp.SUM)
            mean_loss /= world_size
            if rank == 0:
                print(f"step={step} mean_loss={mean_loss.item():.6f}", flush=True)

        final_signature = parameter_signature(engine, args.zero_stage, deepspeed)
        if math.isclose(initial_signature, final_signature, rel_tol=0.0, abs_tol=1e-12):
            raise AssertionError("optimizer did not change the model parameters")
        if rank == 0:
            print("parameter_update=PASS", flush=True)
        if args.verify_parameters:
            assert_parameters_match_rank_zero(engine, args.zero_stage, deepspeed)
            if rank == 0:
                print("full_parameter_sync=PASS", flush=True)
                print(f"ZERO-{args.zero_stage} CORRECTNESS CHECKS PASSED", flush=True)
        dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
