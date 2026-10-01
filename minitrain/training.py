"""Explicit single-device training primitives for MiniTrain."""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


# Inputs are vocabulary logits [B,T,V] and target IDs [B,T]. The scalar output
# is the mean next-token negative log likelihood used to train the decoder.
def causal_lm_loss(logits: Tensor, labels: Tensor) -> Tensor:
    """Compute mean causal language-model cross-entropy loss."""

    if logits.ndim != 3:
        raise ValueError(f"logits must have shape [B, T, V], got {tuple(logits.shape)}")
    if labels.ndim != 2:
        raise ValueError(f"labels must have shape [B, T], got {tuple(labels.shape)}")
    if logits.shape[:2] != labels.shape:
        raise ValueError(
            "logits and labels must share [B, T], got "
            f"{tuple(logits.shape[:2])} and {tuple(labels.shape)}"
        )
    if labels.dtype != torch.long:
        raise TypeError(f"labels must use torch.int64, got {labels.dtype}")

    vocab_size = logits.size(-1)
    return F.cross_entropy(
        logits.reshape(-1, vocab_size),
        labels.reshape(-1),
    )


# Inputs are a model, its optimizer, one CPU/device batch, and target device.
# Output is the detached scalar loss. This function owns exactly one parameter
# update, making the training state transition visible and testable.
def train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: dict[str, Tensor],
    device: torch.device,
) -> float:
    """Run zero-grad, forward, loss, backward, and optimizer update once."""

    if "input_ids" not in batch or "labels" not in batch:
        raise KeyError("batch must contain input_ids and labels")

    model.train()
    input_ids = batch["input_ids"].to(device, non_blocking=True)
    labels = batch["labels"].to(device, non_blocking=True)

    # set_to_none avoids writing zeros into every gradient buffer. Backward
    # allocates only the gradients needed for this step, with no accumulation.
    optimizer.zero_grad(set_to_none=True)
    logits = model(input_ids)
    loss = causal_lm_loss(logits, labels)
    loss.backward()
    optimizer.step()

    return float(loss.detach())


# The requested string is resolved once before model allocation. Distributed
# rank-to-device mapping is intentionally deferred until Step 8.
def resolve_device(requested: str) -> torch.device:
    """Resolve auto/cpu/cuda into one valid single-process torch device."""

    if requested == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        return torch.device("cuda:0")
    raise ValueError("device must be one of: auto, cpu, cuda")
