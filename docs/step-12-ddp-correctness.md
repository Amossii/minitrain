# Step 12: DDP correctness verification

The verification deliberately initializes ranks with different random seeds.
After constructing DDP, every parameter must match rank 0, proving constructor
synchronization rather than merely relying on identical initialization.

Each rank then processes a disjoint local batch. Backward must leave every
corresponding gradient equal across ranks. After the same SGD update, every
parameter must remain equal.

Finally, rank 0 restores the synchronized initial state into a non-DDP model and
updates it once using the union of all local samples. With equal local batch
sizes and mean cross-entropy reduction:

```text
average(local mean gradients) = gradient(global batch mean loss)
```

The resulting DDP and reference parameters are compared individually with
`torch.testing.assert_close`. A checksum alone would be insufficient because
different parameter errors can cancel in a sum.

## Local two-process Gloo command

```bash
conda run -n sglang python -m torch.distributed.run --standalone --nproc-per-node=2 scripts/verify_ddp_correctness.py --backend gloo --local-batch-size 2 --seq-len 16 --learning-rate 0.01 --seed 42
```

## Kaggle two-GPU NCCL command

```bash
torchrun --standalone --nproc-per-node=2 scripts/verify_ddp_correctness.py --backend nccl --local-batch-size 2 --seq-len 16 --learning-rate 0.01 --seed 42
```

The test uses FP32 and SGD to isolate gradient synchronization semantics. Mixed
precision, adaptive optimizers, and performance measurements are separate
concerns and are deliberately excluded.
