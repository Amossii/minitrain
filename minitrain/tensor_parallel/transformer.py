"""使用手写 Column/Row Linear 构建 Tensor Parallel Transformer。"""

from __future__ import annotations

import math

import torch
import torch.distributed as dist
from torch import Tensor, nn
import torch.nn.functional as F

from minitrain.config import ModelConfig
from minitrain.model import MiniTransformer, RMSNorm
from minitrain.tensor_parallel.column_linear import ColumnParallelLinear
from minitrain.tensor_parallel.row_linear import RowParallelLinear


# 输入是完整模型配置和 TP world size，输出为空；该检查保证 attention head、
# hidden width 和 MLP width 都能均匀映射到每个 rank。
def validate_tensor_parallel_config(config: ModelConfig, world_size: int) -> None:
    """检查 Transformer 是否能按给定 TP world size 无余数切分。"""

    if world_size <= 0:
        raise ValueError("world_size must be positive")
    for name, value in (
        ("hidden_size", config.hidden_size),
        ("num_heads", config.num_heads),
        ("intermediate_size", config.intermediate_size),
    ):
        if value % world_size != 0:
            raise ValueError(f"{name} must be divisible by tensor parallel world size")


class TensorParallelSelfAttention(nn.Module):
    """让每个 rank 只计算一部分 attention heads。

    输入和最终输出都是 replicated `[B,T,C]`。Q/K/V 使用 Column Parallel 产生
    `[B,T,C/TP]`，对应 `H/TP` 个本地 head；O projection 使用 Row Parallel 将
    各 head 的输出贡献 AllReduce 回完整 `[B,T,C]`。
    """

    def __init__(
        self,
        config: ModelConfig,
        process_group: dist.ProcessGroup | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        world_size = dist.get_world_size(process_group)
        validate_tensor_parallel_config(config, world_size)
        self.hidden_size = config.hidden_size
        self.head_dim = config.head_dim
        self.local_num_heads = config.num_heads // world_size
        self.local_hidden_size = config.hidden_size // world_size

        self.q_proj = ColumnParallelLinear(
            config.hidden_size,
            config.hidden_size,
            bias=False,
            gather_output=False,
            process_group=process_group,
            device=device,
        )
        self.k_proj = ColumnParallelLinear(
            config.hidden_size,
            config.hidden_size,
            bias=False,
            gather_output=False,
            process_group=process_group,
            device=device,
        )
        self.v_proj = ColumnParallelLinear(
            config.hidden_size,
            config.hidden_size,
            bias=False,
            gather_output=False,
            process_group=process_group,
            device=device,
        )
        self.o_proj = RowParallelLinear(
            config.hidden_size,
            config.hidden_size,
            bias=False,
            input_is_parallel=True,
            process_group=process_group,
            device=device,
        )
        causal_mask = torch.tril(
            torch.ones(config.seq_len, config.seq_len, dtype=torch.bool, device=device)
        )
        self.register_buffer("causal_mask", causal_mask, persistent=False)

    # 输入 `[B,T,C/TP]`，输出 `[B,H/TP,T,D]`；这里只重排本地 shard，
    # 不需要 AllGather，因为每个 attention head 可以独立计算。
    def _split_local_heads(self, tensor: Tensor) -> Tensor:
        """把本地 hidden shard 重排为本地 attention heads。"""

        batch_size, seq_len, local_hidden = tensor.shape
        if local_hidden != self.local_hidden_size:
            raise ValueError(
                f"expected local hidden size {self.local_hidden_size}, got {local_hidden}"
            )
        return tensor.view(
            batch_size, seq_len, self.local_num_heads, self.head_dim
        ).transpose(1, 2)

    # 输入和输出均为 replicated `[B,T,C]`；内部只保存、计算当前 rank 的 heads，
    # O projection 的 AllReduce 是本模块 forward 中唯一必需的 collective。
    def forward(self, hidden_states: Tensor) -> Tensor:
        """计算本地 attention heads 并恢复完整 hidden 输出。"""

        batch_size, seq_len, hidden_size = hidden_states.shape
        if hidden_size != self.hidden_size:
            raise ValueError(f"expected hidden size {self.hidden_size}, got {hidden_size}")
        if seq_len > self.causal_mask.size(0):
            raise ValueError("sequence length exceeds configured maximum")

        query = self._split_local_heads(self.q_proj(hidden_states))
        key = self._split_local_heads(self.k_proj(hidden_states))
        value = self._split_local_heads(self.v_proj(hidden_states))
        scores = query @ key.transpose(-2, -1)
        scores = scores / math.sqrt(self.head_dim)
        visible = self.causal_mask[:seq_len, :seq_len]
        scores = scores.masked_fill(~visible, float("-inf"))
        probabilities = F.softmax(scores, dim=-1, dtype=torch.float32).to(value.dtype)
        local_context = probabilities @ value
        local_context = local_context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.local_hidden_size
        )
        return self.o_proj(local_context)


class TensorParallelSwiGLU(nn.Module):
    """用 Column→Row 配对实现不需要中间 AllGather 的 SwiGLU。"""

    def __init__(
        self,
        config: ModelConfig,
        process_group: dist.ProcessGroup | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.gate_proj = ColumnParallelLinear(
            config.hidden_size,
            config.intermediate_size,
            bias=False,
            gather_output=False,
            process_group=process_group,
            device=device,
        )
        self.up_proj = ColumnParallelLinear(
            config.hidden_size,
            config.intermediate_size,
            bias=False,
            gather_output=False,
            process_group=process_group,
            device=device,
        )
        self.down_proj = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            bias=False,
            input_is_parallel=True,
            process_group=process_group,
            device=device,
        )

    # 输入是 replicated `[B,T,C]`；gate/up 输出均为相同的 `[B,T,I/TP]`
    # shard，可直接逐元素相乘；down projection AllReduce 后恢复 `[B,T,C]`。
    def forward(self, hidden_states: Tensor) -> Tensor:
        """在本地 intermediate shard 上计算 SwiGLU。"""

        local_gated = F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
        return self.down_proj(local_gated)


class TensorParallelBlock(nn.Module):
    """保持 residual/norm replicated，只分片 Attention 和 MLP 的 Linear。"""

    def __init__(
        self,
        config: ModelConfig,
        process_group: dist.ProcessGroup | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.attention_norm = RMSNorm(config.hidden_size).to(device)
        self.attention = TensorParallelSelfAttention(config, process_group, device)
        self.mlp_norm = RMSNorm(config.hidden_size).to(device)
        self.mlp = TensorParallelSwiGLU(config, process_group, device)

    def forward(self, hidden_states: Tensor) -> Tensor:
        """执行两个 replicated residual 路径并保持 `[B,T,C]`。"""

        hidden_states = hidden_states + self.attention(
            self.attention_norm(hidden_states)
        )
        hidden_states = hidden_states + self.mlp(self.mlp_norm(hidden_states))
        return hidden_states


class TensorParallelTransformer(nn.Module):
    """仅将 Transformer Blocks 的 Linear 做 TP 的教学模型。

    token/position embedding、RMSNorm 和 LM Head 在所有 rank 上 replicated；Block
    内 Q/K/V/Gate/Up 为 Column Parallel，O/Down 为 Row Parallel。输入 token 和
    输出 logits 在整个 TP group 上相同。
    """

    def __init__(
        self,
        config: ModelConfig,
        process_group: dist.ProcessGroup | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        if not dist.is_initialized():
            raise RuntimeError("TensorParallelTransformer requires a process group")
        validate_tensor_parallel_config(config, dist.get_world_size(process_group))
        self.config = config
        self.process_group = process_group
        self.token_embedding = nn.Embedding(
            config.vocab_size, config.hidden_size, device=device
        )
        self.position_embedding = nn.Embedding(
            config.seq_len, config.hidden_size, device=device
        )
        self.blocks = nn.ModuleList(
            TensorParallelBlock(config, process_group, device)
            for _ in range(config.num_layers)
        )
        self.final_norm = RMSNorm(config.hidden_size).to(device)
        self.lm_head = nn.Linear(
            config.hidden_size, config.vocab_size, bias=False, device=device
        )

    # 输入是完整 reference Transformer；输出为空。Column 复制输出行，Row 复制输入列，
    # replicated 模块完整复制，使两种模型可做逐参数 correctness。
    def load_from_transformer(self, reference: MiniTransformer) -> None:
        """从普通 MiniTransformer 加载 replicated 参数和 TP shards。"""

        if reference.config != self.config:
            raise ValueError("reference model config does not match TP model")
        with torch.no_grad():
            self.token_embedding.weight.copy_(reference.token_embedding.weight)
            self.position_embedding.weight.copy_(reference.position_embedding.weight)
            self.final_norm.weight.copy_(reference.final_norm.weight)
            self.lm_head.weight.copy_(reference.lm_head.weight)
            for tp_block, ref_block in zip(
                self.blocks, reference.blocks, strict=True
            ):
                tp_block.attention_norm.weight.copy_(ref_block.attention_norm.weight)
                tp_block.mlp_norm.weight.copy_(ref_block.mlp_norm.weight)
                tp_block.attention.q_proj.load_from_linear(ref_block.attention.q_proj)
                tp_block.attention.k_proj.load_from_linear(ref_block.attention.k_proj)
                tp_block.attention.v_proj.load_from_linear(ref_block.attention.v_proj)
                tp_block.attention.o_proj.load_from_linear(ref_block.attention.o_proj)
                tp_block.mlp.gate_proj.load_from_linear(ref_block.mlp.gate_proj)
                tp_block.mlp.up_proj.load_from_linear(ref_block.mlp.up_proj)
                tp_block.mlp.down_proj.load_from_linear(ref_block.mlp.down_proj)

    # 输入 `[B,T]` token IDs，内部每个 Block 在 shard/replicated 布局间转换；
    # 输出是每个 rank 都相同的完整 `[B,T,V]` logits。
    def forward(self, input_ids: Tensor) -> Tensor:
        """运行 Tensor Parallel decoder 并返回 replicated logits。"""

        if input_ids.ndim != 2 or input_ids.dtype != torch.long:
            raise ValueError("input_ids must have shape [B,T] and dtype torch.int64")
        _, seq_len = input_ids.shape
        if seq_len > self.config.seq_len:
            raise ValueError("sequence length exceeds configured maximum")
        positions = torch.arange(seq_len, device=input_ids.device)
        hidden_states = self.token_embedding(input_ids)
        hidden_states = hidden_states + self.position_embedding(positions)
        for block in self.blocks:
            hidden_states = block(hidden_states)
        return self.lm_head(self.final_norm(hidden_states))


# 输入是 TP 模型以及是否读取 gradient；输出将所有 Column/Row shard 恢复成与普通
# Transformer 同名、同 shape 的 tensor，供 correctness 使用，不参与训练图。
def gather_tensor_parallel_tensors(
    model: TensorParallelTransformer, *, gradients: bool = False
) -> dict[str, Tensor]:
    """Gather TP 参数或梯度为完整 named tensor 字典。"""

    full_tensors: dict[str, Tensor] = {}
    for name, parameter in model.named_parameters():
        value = parameter.grad if gradients else parameter
        if value is None:
            raise AssertionError(f"missing gradient for {name}")
        module_name, _, parameter_name = name.rpartition(".")
        owner = model.get_submodule(module_name) if module_name else model
        detached = value.detach().contiguous()
        if isinstance(owner, ColumnParallelLinear):
            pieces = [torch.empty_like(detached) for _ in range(owner.world_size)]
            dist.all_gather(pieces, detached, group=owner.process_group)
            full_tensors[name] = torch.cat(pieces, dim=0)
        elif isinstance(owner, RowParallelLinear) and parameter_name == "weight":
            pieces = [torch.empty_like(detached) for _ in range(owner.world_size)]
            dist.all_gather(pieces, detached, group=owner.process_group)
            full_tensors[name] = torch.cat(pieces, dim=1)
        else:
            full_tensors[name] = detached.clone()
    return full_tensors
