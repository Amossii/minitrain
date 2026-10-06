"""按词表维度切分 Embedding、LM Head 和 cross entropy。"""

from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist
from torch import Tensor, nn
import torch.nn.functional as F

from minitrain.tensor_parallel.column_linear import ColumnParallelLinear


# 输入是全局词表大小、TP world size 和 rank，输出该 rank 连续持有的 token 区间；
# 连续分片让 embedding、LM Head 和 target ownership 使用完全相同的映射。
def vocabulary_partition_range(
    vocab_size: int, world_size: int, rank: int
) -> tuple[int, int]:
    """返回当前 rank 的 `[vocab_start, vocab_end)`。"""

    if vocab_size <= 0 or world_size <= 0:
        raise ValueError("vocab_size and world_size must be positive")
    if vocab_size % world_size != 0:
        raise ValueError("vocab_size must be divisible by world_size")
    if rank < 0 or rank >= world_size:
        raise ValueError("rank must be in [0, world_size)")
    local_size = vocab_size // world_size
    start = rank * local_size
    return start, start + local_size


class _ReduceVocabParallelRegion(torch.autograd.Function):
    """前向汇总 vocab shard 的贡献，反向把完整梯度传回每个本地分支。"""

    @staticmethod
    def forward(ctx: Any, tensor: Tensor, group: dist.ProcessGroup | None) -> Tensor:
        ctx.group = group
        output = tensor.contiguous().clone()
        dist.all_reduce(output, op=dist.ReduceOp.SUM, group=group)
        return output

    @staticmethod
    def backward(ctx: Any, gradient: Tensor) -> tuple[Tensor, None]:
        # 前向是各 rank 局部贡献之和，所以每个分支对总和的导数都是 1。
        return gradient, None


class VocabParallelEmbedding(nn.Module):
    """沿 vocabulary 维切分 token embedding。

    每个 rank 保存 `[V/TP,C]`。输入 token IDs 在 TP group 内 replicated；本 rank
    只 lookup 属于自己区间的 token，其他位置输出零，最后 AllReduce 得到完整
    `[B,T,C]` hidden states。
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        process_group: dist.ProcessGroup | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if not dist.is_initialized():
            raise RuntimeError("VocabParallelEmbedding requires a process group")
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.process_group = process_group
        self.world_size = dist.get_world_size(process_group)
        self.rank = dist.get_rank(process_group)
        self.vocab_start, self.vocab_end = vocabulary_partition_range(
            vocab_size, self.world_size, self.rank
        )
        self.local_vocab_size = self.vocab_end - self.vocab_start
        self.weight = nn.Parameter(
            torch.empty(self.local_vocab_size, hidden_size, device=device, dtype=dtype)
        )
        self.reset_parameters()

    # 输入为空；各 rank 用相同 seed 生成完整 reference 初始化后只保留自己的行，
    # 从而使分片组合与普通 nn.Embedding 的初始化完全一致。
    def reset_parameters(self) -> None:
        """初始化当前 rank 的 embedding rows。"""

        with torch.no_grad():
            full_weight = torch.empty(
                self.vocab_size,
                self.hidden_size,
                device=self.weight.device,
                dtype=self.weight.dtype,
            )
            nn.init.normal_(full_weight)
            self.weight.copy_(full_weight[self.vocab_start : self.vocab_end])

    # 输入是普通完整 embedding，输出为空；用于 correctness 和 checkpoint 转换。
    def load_from_embedding(self, embedding: nn.Embedding) -> None:
        """复制 reference embedding 中属于当前 rank 的连续行。"""

        if embedding.weight.shape != (self.vocab_size, self.hidden_size):
            raise ValueError("embedding shape does not match vocab-parallel embedding")
        with torch.no_grad():
            shard = embedding.weight[self.vocab_start : self.vocab_end]
            self.weight.copy_(shard.to(self.weight.device, self.weight.dtype))

    # 输入是 replicated `[B,T]` 全局 token IDs；输出是 replicated `[B,T,C]`，
    # 内部 mask 保证每个 token 只有 owner rank 提供非零 lookup 结果。
    def forward(self, input_ids: Tensor) -> Tensor:
        """执行本地 lookup 并 AllReduce 为完整 embedding。"""

        if input_ids.dtype != torch.long:
            raise TypeError("input_ids must use torch.int64")
        outside = (input_ids < self.vocab_start) | (input_ids >= self.vocab_end)
        local_ids = (input_ids - self.vocab_start).masked_fill(outside, 0)
        local_output = F.embedding(local_ids, self.weight)
        local_output = local_output.masked_fill(outside.unsqueeze(-1), 0)
        return _ReduceVocabParallelRegion.apply(local_output, self.process_group)


class VocabParallelLMHead(ColumnParallelLinear):
    """沿 vocabulary 输出维切分且不 Gather logits 的 LM Head。"""

    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        process_group: dist.ProcessGroup | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(
            hidden_size,
            vocab_size,
            bias=False,
            gather_output=False,
            process_group=process_group,
            device=device,
            dtype=dtype,
        )


# 输入是 local logits `[B,T,V/TP]` 和全局 labels `[B,T]`，输出 replicated scalar；
# 全局 max/sum 和 target logit 通过 collective 得到，全程不构造 `[B,T,V]`。
def vocab_parallel_cross_entropy(
    local_logits: Tensor,
    labels: Tensor,
    *,
    vocab_start: int,
    vocab_end: int,
    process_group: dist.ProcessGroup | None = None,
) -> Tensor:
    """数值稳定地计算 vocabulary-sharded mean cross entropy。"""

    if local_logits.ndim != 3 or labels.ndim != 2:
        raise ValueError("local_logits must be [B,T,V_local] and labels must be [B,T]")
    if local_logits.shape[:2] != labels.shape:
        raise ValueError("local_logits and labels must share batch/sequence dimensions")
    if labels.dtype != torch.long:
        raise TypeError("labels must use torch.int64")
    if local_logits.size(-1) != vocab_end - vocab_start:
        raise ValueError("local logits width does not match vocabulary partition")
    world_size = dist.get_world_size(process_group)
    global_vocab_size = local_logits.size(-1) * world_size
    if global_vocab_size <= 0:
        raise ValueError("global vocabulary must be positive")

    # max 只用于 softmax 数值平移，detach 后不参与梯度；跨 rank 取最大值。
    global_max = local_logits.detach().amax(dim=-1)
    dist.all_reduce(global_max, op=dist.ReduceOp.MAX, group=process_group)
    shifted = local_logits - global_max.unsqueeze(-1)
    local_exp_sum = shifted.exp().sum(dim=-1)
    global_exp_sum = _ReduceVocabParallelRegion.apply(
        local_exp_sum, process_group
    )

    outside = (labels < vocab_start) | (labels >= vocab_end)
    local_labels = (labels - vocab_start).masked_fill(outside, 0)
    local_target = shifted.gather(-1, local_labels.unsqueeze(-1)).squeeze(-1)
    local_target = local_target.masked_fill(outside, 0)
    global_target = _ReduceVocabParallelRegion.apply(local_target, process_group)
    return (global_exp_sum.log() - global_target).mean()
