"""DeepSpeed ZeRO 配置、状态分片语义和理论显存模型。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ZeroStateOwnership:
    """描述一个 ZeRO stage 是否对三类持久模型状态进行分片。"""

    optimizer_states_sharded: bool
    gradients_sharded: bool
    parameters_sharded: bool


# 输入是 ZeRO stage，输出明确三类状态的 ownership；该映射用于配置说明、
# 理论显存计算和测试，避免只记住模糊的“stage 越高越省显存”。
def zero_state_ownership(stage: int) -> ZeroStateOwnership:
    """返回 ZeRO-1/2/3 分别分片哪些训练状态。"""

    if stage not in (1, 2, 3):
        raise ValueError("ZeRO stage must be 1, 2, or 3")
    return ZeroStateOwnership(
        optimizer_states_sharded=True,
        gradients_sharded=stage >= 2,
        parameters_sharded=stage >= 3,
    )


# 输入是参数量、world size 和 stage，输出 FP32 AdamW 理想持久状态字节数。
# 公式不包含 activation、通信 buffer、CUDA context 和 allocator 碎片。
def theoretical_zero_bytes_per_rank(
    num_parameters: int,
    world_size: int,
    stage: int,
    element_size: int = 4,
) -> int:
    """估算 ZeRO 每个 rank 的 parameter/gradient/Adam moments 字节数。"""

    if num_parameters <= 0 or world_size <= 0 or element_size <= 0:
        raise ValueError("parameter count, world size, and element size must be positive")
    ownership = zero_state_ownership(stage)
    parameter_factor = 1 / world_size if ownership.parameters_sharded else 1
    gradient_factor = 1 / world_size if ownership.gradients_sharded else 1
    # AdamW 有 exp_avg 和 exp_avg_sq 两份与参数同 shape 的状态。
    optimizer_factor = 2 / world_size
    total_elements = num_parameters * (
        parameter_factor + gradient_factor + optimizer_factor
    )
    return int(total_elements * element_size)


# 输入定义固定 workload，输出可以直接交给 deepspeed.initialize 的配置。
# optimizer 由训练脚本显式创建，确保 DDP/FSDP2/ZeRO 后续使用同一种 AdamW。
def build_zero_config(
    stage: int,
    local_batch_size: int,
    world_size: int,
    *,
    gradient_accumulation_steps: int = 1,
) -> dict[str, Any]:
    """构造最小、显式且可比较的 FP32 DeepSpeed ZeRO 配置。"""

    zero_state_ownership(stage)
    if min(local_batch_size, world_size, gradient_accumulation_steps) <= 0:
        raise ValueError("batch sizes, world size, and accumulation steps must be positive")
    global_batch_size = (
        local_batch_size * world_size * gradient_accumulation_steps
    )
    zero_config: dict[str, Any] = {
        "stage": stage,
        "overlap_comm": False,
        "contiguous_gradients": True,
    }
    if stage >= 2:
        zero_config["reduce_scatter"] = True
    return {
        "train_micro_batch_size_per_gpu": local_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "train_batch_size": global_batch_size,
        "zero_optimization": zero_config,
        "fp16": {"enabled": False},
        "bf16": {"enabled": False},
        "steps_per_print": 1_000_000,
        "wall_clock_breakdown": False,
    }
