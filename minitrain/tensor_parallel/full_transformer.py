"""带 vocabulary sharding 的完整 Tensor Parallel Transformer。"""

from __future__ import annotations

import torch
import torch.distributed as dist
from torch import Tensor, nn

from minitrain.config import ModelConfig
from minitrain.model import MiniTransformer, RMSNorm
from minitrain.tensor_parallel.transformer import (
    TensorParallelBlock,
    validate_tensor_parallel_config,
)
from minitrain.tensor_parallel.vocab import (
    VocabParallelEmbedding,
    VocabParallelLMHead,
    vocab_parallel_cross_entropy,
)


class FullTensorParallelTransformer(nn.Module):
    """同时分片 vocabulary 模块和 Transformer Block Linear。

    输入 token IDs `[B,T]` 在 TP ranks 上复制；embedding AllReduce 后 residual
    stream 为 `[B,T,C]`，Block 内使用 Column/Row TP；LM Head 只输出当前 rank 的
    `[B,T,V/TP]` logits，loss 由 vocabulary-parallel cross entropy 直接计算。
    """

    def __init__(
        self,
        config: ModelConfig,
        process_group: dist.ProcessGroup | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        if not dist.is_initialized():
            raise RuntimeError("FullTensorParallelTransformer requires a process group")
        world_size = dist.get_world_size(process_group)
        validate_tensor_parallel_config(config, world_size)
        if config.vocab_size % world_size != 0:
            raise ValueError("vocab_size must be divisible by tensor parallel world size")
        self.config = config
        self.process_group = process_group
        self.token_embedding = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size, process_group, device
        )
        self.position_embedding = nn.Embedding(
            config.seq_len, config.hidden_size, device=device
        )
        self.blocks = nn.ModuleList(
            TensorParallelBlock(config, process_group, device)
            for _ in range(config.num_layers)
        )
        self.final_norm = RMSNorm(config.hidden_size).to(device)
        self.lm_head = VocabParallelLMHead(
            config.hidden_size, config.vocab_size, process_group, device
        )

    # 输入是完整 reference，输出为空；vocabulary 和 block weights 分别复制对应 shard，
    # replicated position/norm 完整复制，使 full TP 可与原模型逐项验证。
    def load_from_transformer(self, reference: MiniTransformer) -> None:
        """从普通 MiniTransformer 加载所有 replicated 参数和本地 shards。"""

        if reference.config != self.config:
            raise ValueError("reference model config does not match full TP model")
        with torch.no_grad():
            self.token_embedding.load_from_embedding(reference.token_embedding)
            self.position_embedding.weight.copy_(reference.position_embedding.weight)
            self.final_norm.weight.copy_(reference.final_norm.weight)
            self.lm_head.load_from_linear(reference.lm_head)
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

    # 输入 `[B,T]`，输出当前 rank 的 `[B,T,V/TP]` logits；不进行最终 AllGather，
    # 从而避免完整 vocabulary activation 成为每卡显存瓶颈。
    def forward(self, input_ids: Tensor) -> Tensor:
        """运行完整 TP decoder 并返回本地 vocabulary logits。"""

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

    # 输入 local logits 和 replicated labels，输出每个 rank 相同的 scalar loss；
    # 下一次 optimizer step 只更新本 rank 持有的 vocabulary/Linear shards。
    def loss(self, local_logits: Tensor, labels: Tensor) -> Tensor:
        """计算不 Gather 完整 logits 的 vocabulary-parallel loss。"""

        return vocab_parallel_cross_entropy(
            local_logits,
            labels,
            vocab_start=self.lm_head.output_start,
            vocab_end=self.lm_head.output_end,
            process_group=self.process_group,
        )


# 输入 local logits，输出仅供 correctness 使用的完整 `[B,T,V]`；训练和 benchmark
# 禁止调用它，否则会失去 vocabulary-parallel activation memory 收益。
def gather_full_logits(
    local_logits: Tensor, process_group: dist.ProcessGroup | None = None
) -> Tensor:
    """按 rank 顺序 Gather vocabulary logits 供数值验证。"""

    pieces = [
        torch.empty_like(local_logits)
        for _ in range(dist.get_world_size(process_group))
    ]
    dist.all_gather(pieces, local_logits.contiguous(), group=process_group)
    return torch.cat(pieces, dim=-1)
