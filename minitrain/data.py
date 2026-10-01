"""Deterministic synthetic token data for training-system experiments."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.utils.data import Dataset


class SyntheticTokenDataset(Dataset[dict[str, Tensor]]):
    """Generate deterministic causal-language-model samples by index.

    Inputs define the number of samples, token sequence shape, vocabulary, and
    random seed. The dataset stores only this metadata rather than all token
    tensors. Each call returns `input_ids [T]` and shifted `labels [T]`; no
    mutable RNG state changes between calls, so distributed samplers and
    DataLoader workers observe stable data for every index.
    """

    def __init__(
        self,
        num_samples: int,
        seq_len: int,
        vocab_size: int,
        seed: int = 42,
    ) -> None:
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
        if seq_len <= 0:
            raise ValueError("seq_len must be positive")
        if vocab_size <= 1:
            raise ValueError("vocab_size must be greater than 1")
        if seed < 0:
            raise ValueError("seed must be non-negative")

        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.seed = seed

    def __len__(self) -> int:
        """Return the fixed number of samples available to a sampler."""

        return self.num_samples

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """Return one reproducible next-token prediction sample.

        The input is a dataset index. The output contains two int64 tensors of
        shape `[seq_len]`. In distributed training, a DistributedSampler can
        divide indices among ranks without changing the sample values.
        """

        if not isinstance(index, int):
            raise TypeError(f"index must be an integer, got {type(index).__name__}")
        if index < 0:
            index += self.num_samples
        if index < 0 or index >= self.num_samples:
            raise IndexError(
                f"sample index {index} is out of range for {self.num_samples} samples"
            )

        # A local CPU generator makes each sample a pure function of seed/index.
        # It avoids changing the global RNG used for model initialization.
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed + index)
        tokens = torch.randint(
            low=0,
            high=self.vocab_size,
            size=(self.seq_len + 1,),
            generator=generator,
            dtype=torch.long,
        )

        # Adjacent views encode causal next-token targets without extra copies.
        return {
            "input_ids": tokens[:-1],
            "labels": tokens[1:],
        }


# Input describes a single local batch; output matches the dictionary and
# `[B, T]` shape that future single-GPU and distributed training loops consume.
def create_synthetic_batch(
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    seed: int = 42,
) -> dict[str, Tensor]:
    """Create one deterministic batch without constructing a DataLoader."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    dataset = SyntheticTokenDataset(
        num_samples=batch_size,
        seq_len=seq_len,
        vocab_size=vocab_size,
        seed=seed,
    )
    samples = [dataset[index] for index in range(batch_size)]
    return {
        "input_ids": torch.stack([sample["input_ids"] for sample in samples]),
        "labels": torch.stack([sample["labels"] for sample in samples]),
    }
