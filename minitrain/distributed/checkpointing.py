"""使用 PyTorch Distributed Checkpoint 保存和恢复训练状态。"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch import nn
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_state_dict,
    set_state_dict,
)


# 分布式 checkpoint 保存的是逻辑完整状态，但每个 rank 只写自己持有的 tensor
# shard。DDP 参数是 replicated，FSDP2 参数是 DTensor shard；调用方不需要先手工
# AllGather 成一个巨大的 rank-0 checkpoint。
def save_training_checkpoint(
    checkpoint_dir: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    step: int,
) -> None:
    """保存模型、optimizer 和下一训练步编号。

    输入是当前模型/optimizer 状态与非负 step；输出是 checkpoint 目录。保存不会
    修改训练状态。下一次调用 load 后，应从返回的 step 继续读取数据并训练。
    """

    if step < 0:
        raise ValueError("step must be non-negative")
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.parent.mkdir(parents=True, exist_ok=True)
    options = StateDictOptions(full_state_dict=False, cpu_offload=False)
    model_state, optimizer_state = get_state_dict(
        model, optimizer, options=options
    )
    state = {
        "model": model_state,
        "optimizer": optimizer_state,
        # 每个 rank 保存相同的逻辑 step；tensor 形式可由 DCP planner 处理。
        "step": torch.tensor(step, dtype=torch.int64),
    }
    dcp.save(
        state,
        checkpoint_id=checkpoint_dir,
        no_dist=not dist.is_initialized(),
    )


# 恢复时先从新模型/optimizer 取得相同结构的 state-dict 模板，DCP 将存储 shard
# 直接加载到当前 world-size/layout，再由 set_state_dict 写回真实对象。这样同一份
# checkpoint 可以由 DDP 或 FSDP2 的当前分片布局消费。
def load_training_checkpoint(
    checkpoint_dir: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """恢复模型和 optimizer，返回下一次应执行的训练步编号。

    输入必须是结构相同的新模型和 optimizer。函数会原地替换二者状态；输出 step
    让 DataLoader/Sampler 恢复到一致位置，避免参数恢复但数据进度回退。
    """

    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"checkpoint directory does not exist: {checkpoint_dir}")
    options = StateDictOptions(full_state_dict=False, cpu_offload=False)
    model_state, optimizer_state = get_state_dict(
        model, optimizer, options=options
    )
    state = {
        "model": model_state,
        "optimizer": optimizer_state,
        "step": torch.zeros((), dtype=torch.int64),
    }
    dcp.load(
        state,
        checkpoint_id=checkpoint_dir,
        no_dist=not dist.is_initialized(),
    )
    set_state_dict(
        model,
        optimizer,
        model_state_dict=state["model"],
        optim_state_dict=state["optimizer"],
        options=options,
    )
    step = int(state["step"].item())
    if step < 0:
        raise ValueError("checkpoint contains a negative training step")
    return step
