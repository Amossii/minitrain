"""沿输出维切分权重的 ColumnParallelLinear。"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.distributed as dist
from torch import Tensor, nn
import torch.nn.functional as F


# 输入是全局输出宽度、TP world size 和当前 rank，输出当前 rank 连续负责的
# [start, end) 行范围；连续切片便于与普通 nn.Linear 权重直接对齐。
def column_partition_range(
    out_features: int, world_size: int, rank: int
) -> tuple[int, int]:
    """返回 Column Parallel 当前 rank 的输出维切片范围。"""

    if out_features <= 0 or world_size <= 0:
        raise ValueError("out_features and world_size must be positive")
    if out_features % world_size != 0:
        raise ValueError("out_features must be divisible by world_size")
    if rank < 0 or rank >= world_size:
        raise ValueError("rank must be in [0, world_size)")
    local_size = out_features // world_size
    start = rank * local_size
    return start, start + local_size


class _CopyToColumnParallelRegion(torch.autograd.Function):
    """前向保持 replicated 输入，反向汇总所有列分片产生的输入梯度。"""

    @staticmethod
    def forward(ctx: Any, input_tensor: Tensor, group: dist.ProcessGroup | None) -> Tensor:
        ctx.group = group
        return input_tensor

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> tuple[Tensor, None]:
        # 每个 rank 只计算自己 weight 行对 dX 的贡献；求和后才是完整 Linear 的 dX。
        grad_input = grad_output.contiguous().clone()
        dist.all_reduce(grad_input, op=dist.ReduceOp.SUM, group=ctx.group)
        return grad_input, None


class _GatherFromColumnParallelRegion(torch.autograd.Function):
    """前向拼接输出分片，反向取回当前 rank 对应的梯度切片。"""

    @staticmethod
    def forward(ctx: Any, local_output: Tensor, group: dist.ProcessGroup | None) -> Tensor:
        world_size = dist.get_world_size(group)
        rank = dist.get_rank(group)
        ctx.group = group
        ctx.rank = rank
        ctx.local_size = local_output.size(-1)
        gathered = [torch.empty_like(local_output) for _ in range(world_size)]
        dist.all_gather(gathered, local_output.contiguous(), group=group)
        return torch.cat(gathered, dim=-1)

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> tuple[Tensor, None]:
        start = ctx.rank * ctx.local_size
        local_gradient = grad_output.narrow(-1, start, ctx.local_size).contiguous()
        return local_gradient, None


class ColumnParallelLinear(nn.Module):
    """将 `nn.Linear` 权重沿输出维切到多个 TP rank。

    输入在所有 rank 上都是 replicated `[..., in_features]`。每个 rank 持久保存
    `weight=[out_features/world_size, in_features]` 和对应 bias，先输出本地 shard；
    `gather_output=True` 时再沿最后一维 AllGather 为 `[..., out_features]`。
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        gather_output: bool = False,
        process_group: dist.ProcessGroup | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if not dist.is_initialized():
            raise RuntimeError("ColumnParallelLinear requires an initialized process group")
        if in_features <= 0:
            raise ValueError("in_features must be positive")

        self.in_features = in_features
        self.out_features = out_features
        self.process_group = process_group
        self.world_size = dist.get_world_size(process_group)
        self.rank = dist.get_rank(process_group)
        self.output_start, self.output_end = column_partition_range(
            out_features, self.world_size, self.rank
        )
        self.local_out_features = self.output_end - self.output_start
        self.gather_output = gather_output

        factory_kwargs = {"device": device, "dtype": dtype}
        self.weight = nn.Parameter(
            torch.empty(self.local_out_features, in_features, **factory_kwargs)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(self.local_out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    # 输入为空；函数临时生成完整初始化张量，再只保留本 rank 的连续切片。
    # 这样相同 seed 的各 rank 能共同组成一个标准 Linear 初始化，而不会得到重复 shard。
    def reset_parameters(self) -> None:
        """使用与 nn.Linear 相同的分布初始化全局权重的本地切片。"""

        with torch.no_grad():
            full_weight = torch.empty(
                self.out_features,
                self.in_features,
                device=self.weight.device,
                dtype=self.weight.dtype,
            )
            nn.init.kaiming_uniform_(full_weight, a=math.sqrt(5))
            self.weight.copy_(full_weight[self.output_start : self.output_end])
            if self.bias is not None:
                bound = 1 / math.sqrt(self.in_features)
                full_bias = torch.empty(
                    self.out_features,
                    device=self.bias.device,
                    dtype=self.bias.dtype,
                )
                nn.init.uniform_(full_bias, -bound, bound)
                self.bias.copy_(full_bias[self.output_start : self.output_end])

    # 输入是完整 nn.Linear，输出为空；参数复制后，本模块的所有本地 shard 与 reference
    # 对应区间完全一致，主要用于 checkpoint 转换和数值 correctness 测试。
    def load_from_linear(self, linear: nn.Linear) -> None:
        """从普通 Linear 复制当前 rank 应持有的 weight/bias 切片。"""

        if linear.in_features != self.in_features or linear.out_features != self.out_features:
            raise ValueError("linear shape does not match ColumnParallelLinear")
        if (linear.bias is None) != (self.bias is None):
            raise ValueError("linear bias setting does not match ColumnParallelLinear")
        with torch.no_grad():
            weight_shard = linear.weight[self.output_start : self.output_end]
            self.weight.copy_(weight_shard.to(self.weight.device, self.weight.dtype))
            if self.bias is not None and linear.bias is not None:
                bias_shard = linear.bias[self.output_start : self.output_end]
                self.bias.copy_(bias_shard.to(self.bias.device, self.bias.dtype))

    # 输入为 replicated `[..., in_features]`；内部状态是本地 weight/bias shard；
    # 输出默认是本地 shard，也可按调用参数 AllGather 为完整输出。
    def forward(self, input_tensor: Tensor, gather_output: bool | None = None) -> Tensor:
        """执行本地矩阵乘，并按需拼接完整输出。"""

        if input_tensor.size(-1) != self.in_features:
            raise ValueError(
                f"expected input last dimension {self.in_features}, "
                f"got {input_tensor.size(-1)}"
            )
        replicated_input = _CopyToColumnParallelRegion.apply(
            input_tensor, self.process_group
        )
        local_output = F.linear(replicated_input, self.weight, self.bias)
        should_gather = self.gather_output if gather_output is None else gather_output
        if should_gather:
            return _GatherFromColumnParallelRegion.apply(
                local_output, self.process_group
            )
        return local_output

    def extra_repr(self) -> str:
        """显示全局/本地 shape 和当前 rank，便于检查 TP ownership。"""

        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"local_out_features={self.local_out_features}, rank={self.rank}, "
            f"world_size={self.world_size}, bias={self.bias is not None}, "
            f"gather_output={self.gather_output}"
        )
