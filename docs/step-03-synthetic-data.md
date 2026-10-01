# Step 3: Deterministic synthetic token data

Synthetic data isolates the training system from tokenizer, network, disk, and
variable-length preprocessing costs. This makes later DDP/FSDP/ZeRO comparisons
about model execution and communication rather than an unrelated input
pipeline bottleneck.

`SyntheticTokenDataset[index]` deterministically generates `seq_len + 1`
random tokens and returns two overlapping views:

```text
tokens:    [t0, t1, t2, ..., tT]
input_ids: [t0, t1, t2, ..., t(T-1)]
labels:    [t1, t2, t3, ..., tT]
```

One sample has shape `[T]`. PyTorch's default DataLoader collation stacks `B`
samples into `[B, T]`. Tokens use `torch.int64`, as required by
`torch.nn.Embedding`.

Each index uses a local generator seeded with `seed + index`. The dataset is
therefore independent of access order and does not alter the global RNG used
for model initialization. It stores metadata rather than preallocating the
whole token corpus.

## Run

```bash
python3 scripts/inspect_data.py --model tiny --batch-size 2 --seq-len 8
python3 -m unittest discover -s tests -v
```

This step is a correctness component, not a data-loading benchmark. No
throughput conclusion should be reported from the inspection command.
