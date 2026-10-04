"""沿输入维切分权重的 RowParallelLinear。"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn


# 输入是全局输入宽度、TP world size 和当前 rank，输出当前 rank 连续负责的
# [start, end) 列范围；每个 rank 保存完整输出行、部分输入列。
def row_partition_range(
    in_features: int, world_size: int, rank: int
) -> tuple[int, int]:
    """返回 Row Parallel 当前 rank 的输入维切片范围。"""

    if in_features <= 0 or world_size <= 0:
        raise ValueError("in_features and world_size must be positive")
    if in_features % world_size != 0:
        raise ValueError("in_features must be divisible by world_size")
    if rank < 0 or rank >= world_size:
        raise ValueError("rank must be in [0, world_size)")
    local_size = in_features // world_size
    start = rank * local_size
    return start, start + local_size


class _ScatterToRowParallelRegion(torch.autograd.Function):
    """前向本地切分 replicated 输入，反向 AllGather 完整输入梯度。"""

    @staticmethod
    def forward(
        ctx: Any, input_tensor: Tensor, group: dist.ProcessGroup | None
    ) -> Tensor:
        world_size = dist.get_world_size(group)
        rank = dist.get_rank(group)
        if input_tensor.size(-1) % world_size != 0:
            raise ValueError("input last dimension must be divisible by world size")
        local_size = input_tensor.size(-1) // world_size
        ctx.group = group
        ctx.world_size = world_size
        return input_tensor.narrow(-1, rank * local_size, local_size).contiguous()

    @staticmethod
    def backward(ctx: Any, local_gradient: Tensor) -> tuple[Tensor, None]:
        # forward 的输入在每个 rank 都是完整副本，因此 backward 也返回完整 dX；
        # 每个 rank 只拥有一段 dX，按 rank 顺序拼接即可恢复最后一维。
        gathered = [torch.empty_like(local_gradient) for _ in range(ctx.world_size)]
        dist.all_gather(gathered, local_gradient.contiguous(), group=ctx.group)
        return torch.cat(gathered, dim=-1), None


class _ReduceFromRowParallelRegion(torch.autograd.Function):
    """前向求和局部输出贡献，反向把相同完整 dY 传给每个 rank。"""

    @staticmethod
    def forward(
        ctx: Any, local_output: Tensor, group: dist.ProcessGroup | None
    ) -> Tensor:
        ctx.group = group
        output = local_output.contiguous().clone()
        dist.all_reduce(output, op=dist.ReduceOp.SUM, group=group)
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> tuple[Tensor, None]:
        # y=sum(y_rank)，所以每个局部分支都有 dy_rank/dy=1，无需额外通信。
        return grad_output, None


class RowParallelLinear(nn.Module):
    """将 `nn.Linear` 权重沿输入维切到多个 TP rank。

    每个 rank 持久保存 `weight=[out_features, in_features/world_size]`。
    输入可以已经是本地 shard，也可以由模块从 replicated 输入中切分。各 rank 的
    `[...，out_features]` 局部贡献经 AllReduce 求和后才得到完整输出。
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        input_is_parallel: bool = True,
        process_group: dist.ProcessGroup | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if not dist.is_initialized():
            raise RuntimeError(
                "RowParallelLinear requires an initialized process group"
            )
        if out_features <= 0:
            raise ValueError("out_features must be positive")

        self.in_features = in_features
        self.out_features = out_features
        self.process_group = process_group
        self.world_size = dist.get_world_size(process_group)
        self.rank = dist.get_rank(process_group)
        self.input_start, self.input_end = row_partition_range(
            in_features, self.world_size, self.rank
        )
        self.local_in_features = self.input_end - self.input_start
        self.input_is_parallel = input_is_parallel

        factory_kwargs = {"device": device, "dtype": dtype}
        self.weight = nn.Parameter(
            torch.empty(out_features, self.local_in_features, **factory_kwargs)
        )
        # Row Parallel 的 bias 对应完整输出维，在所有 rank 上保持 replicated；
        # 它必须在局部贡献 AllReduce 完成后添加，确保只出现一次。
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    # 输入为空；初始化时临时生成标准 Linear 完整权重，各 rank 只持久保留对应列。
    # bias 不切分，各 rank 使用相同 seed 时会得到相同副本。
    def reset_parameters(self) -> None:
        """使用与 nn.Linear 相同的分布初始化本地列分片和 replicated bias。"""

        with torch.no_grad():
            full_weight = torch.empty(
                self.out_features,
                self.in_features,
                device=self.weight.device,
                dtype=self.weight.dtype,
            )
            nn.init.kaiming_uniform_(full_weight, a=math.sqrt(5))
            self.weight.copy_(full_weight[:, self.input_start : self.input_end])
            if self.bias is not None:
                bound = 1 / math.sqrt(self.in_features)
                nn.init.uniform_(self.bias, -bound, bound)

    # 输入是普通完整 Linear，输出为空；weight 复制本 rank 的输入列，bias 完整复制。
    # 该接口使 checkpoint 转换和与 reference 的梯度比较保持直接可见。
    def load_from_linear(self, linear: nn.Linear) -> None:
        """从普通 Linear 复制当前 rank 的 weight 列分片和完整 bias。"""

        if (
            linear.in_features != self.in_features
            or linear.out_features != self.out_features
        ):
            raise ValueError("linear shape does not match RowParallelLinear")
        if (linear.bias is None) != (self.bias is None):
            raise ValueError("linear bias setting does not match RowParallelLinear")
        with torch.no_grad():
            weight_shard = linear.weight[:, self.input_start : self.input_end]
            self.weight.copy_(weight_shard.to(self.weight.device, self.weight.dtype))
            if self.bias is not None and linear.bias is not None:
                self.bias.copy_(linear.bias.to(self.bias.device, self.bias.dtype))

    # 输入可以是 `[..., I/TP]` shard 或 `[..., I]` replicated tensor；内部先计算
    # `[..., O]` 局部贡献，再 AllReduce 并添加一次 bias；输出始终为完整 `[..., O]`。
    def forward(
        self, input_tensor: Tensor, input_is_parallel: bool | None = None
    ) -> Tensor:
        """计算本地输出贡献并通过 AllReduce 恢复完整输出。"""

        is_parallel = (
            self.input_is_parallel if input_is_parallel is None else input_is_parallel
        )
        if is_parallel:
            if input_tensor.size(-1) != self.local_in_features:
                raise ValueError(
                    f"expected sharded input last dimension {self.local_in_features}, "
                    f"got {input_tensor.size(-1)}"
                )
            local_input = input_tensor
        else:
            if input_tensor.size(-1) != self.in_features:
                raise ValueError(
                    f"expected replicated input last dimension {self.in_features}, "
                    f"got {input_tensor.size(-1)}"
                )
            local_input = _ScatterToRowParallelRegion.apply(
                input_tensor, self.process_group
            )

        local_contribution = F.linear(local_input, self.weight, bias=None)
        output = _ReduceFromRowParallelRegion.apply(
            local_contribution, self.process_group
        )
        return output if self.bias is None else output + self.bias

    def extra_repr(self) -> str:
        """显示全局/本地 shape、rank 和输入布局，便于检查 TP ownership。"""

        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"local_in_features={self.local_in_features}, rank={self.rank}, "
            f"world_size={self.world_size}, bias={self.bias is not None}, "
            f"input_is_parallel={self.input_is_parallel}"
        )
