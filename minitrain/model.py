"""A small, explicit decoder-only Transformer for systems experiments."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from minitrain.config import ModelConfig


class RMSNorm(nn.Module):
    """Normalize hidden states by root mean square without mean subtraction.

    Input and output both have shape `[..., hidden_size]`. The module stores one
    learned scale per hidden channel. Normalization uses float32 arithmetic for
    stability, then returns to the input dtype for mixed-precision training.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-5) -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if eps <= 0:
            raise ValueError("eps must be positive")

        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, hidden_states: Tensor) -> Tensor:
        """Scale each token vector to unit root-mean-square magnitude."""

        input_dtype = hidden_states.dtype
        hidden_states_float = hidden_states.float()
        variance = hidden_states_float.pow(2).mean(dim=-1, keepdim=True)
        normalized = hidden_states_float * torch.rsqrt(variance + self.eps)
        return self.weight * normalized.to(input_dtype)


class CausalSelfAttention(nn.Module):
    """Apply multi-head self-attention while hiding future token positions.

    Input and output are `[B, T, C]`. Internally, Q/K/V become `[B, H, T, D]`,
    attention scores become `[B, H, T, T]`, and heads are merged back to C.
    Separate projections keep future tensor-parallel boundaries visible.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size

        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

        # True entries are visible. Registering this as a non-persistent buffer
        # moves it with the model but avoids storing a derivable tensor in checkpoints.
        causal_mask = torch.tril(
            torch.ones(config.seq_len, config.seq_len, dtype=torch.bool)
        )
        self.register_buffer("causal_mask", causal_mask, persistent=False)

    def _split_heads(self, tensor: Tensor) -> Tensor:
        """Transform `[B, T, C]` into `[B, H, T, D]` for head-wise attention."""

        batch_size, seq_len, _ = tensor.shape
        return tensor.view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)

    def forward(self, hidden_states: Tensor) -> Tensor:
        """Return contextualized `[B, T, C]` states using causal attention."""

        batch_size, seq_len, hidden_size = hidden_states.shape
        if hidden_size != self.hidden_size:
            raise ValueError(
                f"expected hidden size {self.hidden_size}, got {hidden_size}"
            )
        if seq_len > self.causal_mask.size(0):
            raise ValueError(
                f"sequence length {seq_len} exceeds configured maximum "
                f"{self.causal_mask.size(0)}"
            )

        query = self._split_heads(self.q_proj(hidden_states))
        key = self._split_heads(self.k_proj(hidden_states))
        value = self._split_heads(self.v_proj(hidden_states))

        # [B,H,T,D] @ [B,H,D,T] -> [B,H,T,T]. Scaling prevents logits from
        # growing with head width and driving softmax into saturation.
        attention_scores = query @ key.transpose(-2, -1)
        attention_scores = attention_scores / math.sqrt(self.head_dim)
        visible_positions = self.causal_mask[:seq_len, :seq_len]
        attention_scores = attention_scores.masked_fill(
            ~visible_positions, float("-inf")
        )

        # Compute probabilities in float32 for numerical stability, then return
        # to the value dtype before the attention-value matrix multiplication.
        attention_probs = F.softmax(
            attention_scores, dim=-1, dtype=torch.float32
        ).to(value.dtype)
        context = attention_probs @ value

        # [B,H,T,D] -> [B,T,H,D] -> [B,T,C]. contiguous() is required because
        # transpose changes strides and view needs contiguous logical storage.
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.hidden_size
        )
        return self.o_proj(context)


class SwiGLU(nn.Module):
    """Apply the gated feed-forward network used in modern decoder LLMs.

    Input/output shape is `[B, T, C]`; gate and up projections expand to
    `[B, T, I]`, are multiplied elementwise, then down-projected to C.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        """Return `down(silu(gate(x)) * up(x))` with the original hidden width."""

        gated = F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
        return self.down_proj(gated)


class TransformerBlock(nn.Module):
    """Combine pre-norm attention and SwiGLU with two residual paths.

    Input, internal persistent state between calls, and output all use hidden
    width C. Each call reads learned parameters but does not mutate them.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attention_norm = RMSNorm(config.hidden_size)
        self.attention = CausalSelfAttention(config)
        self.mlp_norm = RMSNorm(config.hidden_size)
        self.mlp = SwiGLU(config)

    def forward(self, hidden_states: Tensor) -> Tensor:
        """Transform `[B, T, C]` while preserving its shape for block stacking."""

        hidden_states = hidden_states + self.attention(
            self.attention_norm(hidden_states)
        )
        hidden_states = hidden_states + self.mlp(self.mlp_norm(hidden_states))
        return hidden_states


class MiniTransformer(nn.Module):
    """Map token IDs `[B, T]` to vocabulary logits `[B, T, V]`.

    Persistent state consists of token/position embeddings, Transformer blocks,
    final normalization, and the LM head. Forward produces logits without
    changing parameter state; Step 5 will introduce optimizer-driven updates.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embedding = nn.Embedding(config.seq_len, config.hidden_size)
        self.blocks = nn.ModuleList(
            TransformerBlock(config) for _ in range(config.num_layers)
        )
        self.final_norm = RMSNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids: Tensor) -> Tensor:
        """Run the complete decoder and return unnormalized next-token logits."""

        if input_ids.ndim != 2:
            raise ValueError(
                f"input_ids must have shape [B, T], got {tuple(input_ids.shape)}"
            )
        if input_ids.dtype != torch.long:
            raise TypeError(f"input_ids must use torch.int64, got {input_ids.dtype}")

        _, seq_len = input_ids.shape
        if seq_len > self.config.seq_len:
            raise ValueError(
                f"sequence length {seq_len} exceeds configured maximum "
                f"{self.config.seq_len}"
            )

        positions = torch.arange(seq_len, device=input_ids.device)
        hidden_states = self.token_embedding(input_ids)
        hidden_states = hidden_states + self.position_embedding(positions)

        for block in self.blocks:
            hidden_states = block(hidden_states)

        hidden_states = self.final_norm(hidden_states)
        return self.lm_head(hidden_states)


# Input is any nn.Module; output is a scalar used in benchmark metadata.
# Buffers such as causal masks are deliberately excluded from parameter count.
def count_parameters(model: nn.Module) -> int:
    """Return the total number of learned scalar parameters in a model."""

    return sum(parameter.numel() for parameter in model.parameters())
