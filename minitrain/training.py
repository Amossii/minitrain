"""Explicit single-device training primitives for MiniTrain."""

from __future__ import annotations

from dataclasses import dataclass
import time

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass(frozen=True, slots=True)
class StepMeasurement:
    """Hold raw observations from exactly one measured optimizer update."""

    loss: float
    step_time: float
    forward_time: float
    backward_time: float
    optimizer_time: float
    peak_memory_allocated: int
    peak_memory_reserved: int


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


# Inputs match train_step; output additionally contains raw timing and CUDA
# memory observations. CUDA Events measure asynchronous work on the active
# stream without placing a CPU synchronization barrier between every phase.
def measure_train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: dict[str, Tensor],
    device: torch.device,
) -> StepMeasurement:
    """Run one update and measure forward, backward, optimizer, and peak memory."""

    if "input_ids" not in batch or "labels" not in batch:
        raise KeyError("batch must contain input_ids and labels")

    model.train()
    input_ids = batch["input_ids"].to(device, non_blocking=True)
    labels = batch["labels"].to(device, non_blocking=True)
    optimizer.zero_grad(set_to_none=True)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        events = [torch.cuda.Event(enable_timing=True) for _ in range(4)]

        events[0].record()
        logits = model(input_ids)
        loss = causal_lm_loss(logits, labels)
        events[1].record()
        loss.backward()
        events[2].record()
        optimizer.step()
        events[3].record()
        events[3].synchronize()

        forward_time = events[0].elapsed_time(events[1]) / 1_000
        backward_time = events[1].elapsed_time(events[2]) / 1_000
        optimizer_time = events[2].elapsed_time(events[3]) / 1_000
        step_time = events[0].elapsed_time(events[3]) / 1_000
        peak_memory_allocated = torch.cuda.max_memory_allocated(device)
        peak_memory_reserved = torch.cuda.max_memory_reserved(device)
    else:
        step_start = time.perf_counter()
        logits = model(input_ids)
        loss = causal_lm_loss(logits, labels)
        forward_end = time.perf_counter()
        loss.backward()
        backward_end = time.perf_counter()
        optimizer.step()
        optimizer_end = time.perf_counter()

        forward_time = forward_end - step_start
        backward_time = backward_end - forward_end
        optimizer_time = optimizer_end - backward_end
        step_time = optimizer_end - step_start
        # These fields describe the CUDA caching allocator. Zero means not
        # applicable on CPU; it is not a measurement of process RSS.
        peak_memory_allocated = 0
        peak_memory_reserved = 0

    return StepMeasurement(
        loss=float(loss.detach()),
        step_time=step_time,
        forward_time=forward_time,
        backward_time=backward_time,
        optimizer_time=optimizer_time,
        peak_memory_allocated=peak_memory_allocated,
        peak_memory_reserved=peak_memory_reserved,
    )


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
