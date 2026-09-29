"""Typed experiment configuration for reproducible MiniTrain workloads."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal


Precision = Literal["fp32", "fp16", "bf16"]


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Describe the model shape shared by every future training strategy.

    Inputs are architectural dimensions and the maximum sequence length. The
    immutable fields are the complete internal state. Model constructors will
    consume this object and produce parameters; calling them does not mutate
    the configuration, which keeps benchmark workloads comparable.
    """

    vocab_size: int
    seq_len: int
    hidden_size: int
    num_layers: int
    num_heads: int
    intermediate_size: int

    def __post_init__(self) -> None:
        """Reject shapes that cannot form a valid decoder-only Transformer."""

        fields = asdict(self)
        for name, value in fields.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.hidden_size % self.num_heads != 0:
            raise ValueError(
                "hidden_size must be divisible by num_heads so each attention "
                "head has the same width"
            )

    @property
    def head_dim(self) -> int:
        """Return the hidden width owned by one attention head."""

        return self.hidden_size // self.num_heads

    def to_dict(self) -> dict[str, Any]:
        """Return plain data suitable for logs, CSV metadata, or serialization."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class TrainConfig:
    """Describe one training workload independently of execution strategy.

    Inputs control optimization and per-process workload size. The immutable
    fields form the full state, and training loops will consume them without
    mutation. `local_batch_size` is explicit because global batch size changes
    with world size once distributed training is introduced.
    """

    local_batch_size: int = 8
    learning_rate: float = 3e-4
    max_steps: int = 100
    precision: Precision = "fp32"
    seed: int = 42

    def __post_init__(self) -> None:
        """Validate values before an experiment allocates model or GPU state."""

        if self.local_batch_size <= 0:
            raise ValueError("local_batch_size must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if self.precision not in ("fp32", "fp16", "bf16"):
            raise ValueError(
                "precision must be one of: fp32, fp16, bf16; "
                f"got {self.precision!r}"
            )
        if self.seed < 0:
            raise ValueError("seed must be non-negative")

    def global_batch_size(self, world_size: int) -> int:
        """Compute samples processed globally per step for a data-parallel run."""

        if world_size <= 0:
            raise ValueError("world_size must be positive")
        return self.local_batch_size * world_size

    def to_dict(self) -> dict[str, Any]:
        """Return plain data suitable for logs, CSV metadata, or serialization."""

        return asdict(self)


# Presets intentionally vary only model workload. Training settings stay
# separate so changing an optimizer experiment cannot silently change shape.
_MODEL_PRESETS: dict[str, ModelConfig] = {
    "tiny": ModelConfig(
        vocab_size=32_000,
        seq_len=256,
        hidden_size=128,
        num_layers=4,
        num_heads=4,
        intermediate_size=384,
    ),
    "small": ModelConfig(
        vocab_size=32_000,
        seq_len=512,
        hidden_size=256,
        num_layers=8,
        num_heads=8,
        intermediate_size=768,
    ),
    "medium": ModelConfig(
        vocab_size=32_000,
        seq_len=512,
        hidden_size=512,
        num_layers=12,
        num_heads=8,
        intermediate_size=1_536,
    ),
}


# Input is a stable preset name; output is a validated immutable model shape.
# Every future strategy uses this same entry point to avoid benchmark drift.
def get_model_config(name: str) -> ModelConfig:
    """Return the named model preset or explain the available choices."""

    normalized_name = name.strip().lower()
    try:
        return _MODEL_PRESETS[normalized_name]
    except KeyError as exc:
        choices = ", ".join(available_model_configs())
        raise ValueError(
            f"unknown model config {name!r}; available configs: {choices}"
        ) from exc


def available_model_configs() -> tuple[str, ...]:
    """Return model preset names in deterministic display order."""

    return tuple(_MODEL_PRESETS)
