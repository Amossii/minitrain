# Step 9: Collective correctness

Every experiment uses small tensors whose results can be calculated manually.
With two ranks, Reduce and AllReduce start from rank values 1 and 2, so their
SUM is 3. Reduce only guarantees this result on destination rank 0; AllReduce
returns it on both ranks.

AllGather starts with `[0,10]` on rank 0 and `[1,11]` on rank 1. Both ranks end
with `[[0,10],[1,11]]`, ordered by source rank.

ReduceScatter uses chunk size 2. Inputs are:

```text
rank 0: [0, 1, 2, 3]
rank 1: [10, 11, 12, 13]
SUM:    [10, 12, 14, 16]
```

The reduced tensor is scattered into consecutive chunks:

```text
rank 0 output: [10, 12]
rank 1 output: [14, 16]
```

## Local two-process Gloo run

```bash
conda run -n sglang python -m torch.distributed.run --standalone --nproc-per-node=2 scripts/collective_correctness.py --backend gloo
```

## Kaggle two-GPU NCCL run

```bash
torchrun --standalone --nproc-per-node=2 scripts/collective_correctness.py --backend nccl
```

Output ordering is nondeterministic, but every operation must print `PASS` on
the ranks where its result is defined, and rank 0 must finally print
`ALL COLLECTIVE CORRECTNESS CHECKS PASSED`.
