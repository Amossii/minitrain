# Step 11: DistributedDataParallel training

DDP keeps a complete model replica on every rank. `DistributedSampler` assigns
different samples to each process, so one optimizer step consumes:

```text
global_batch_size = local_batch_size * world_size
```

During construction, DDP synchronizes initial model state from rank 0. During
backward, autograd hooks mark gradients ready and DDP reduces gradient buckets
across ranks. By the time backward returns, replicas have equivalent averaged
gradients. Each rank then performs the same local optimizer update.

The model's causal masks are deterministic buffers created from the same
configuration, so `broadcast_buffers=False` avoids rebroadcasting them on every
forward. It does not disable parameter or gradient synchronization.

The script performs another scalar AllReduce after the optimizer step only to
print mean loss. This logging collective is separate from DDP gradient sync and
must not be mistaken for DDP's internal communication.

## Local two-process Gloo smoke test

```bash
conda run -n sglang python -m torch.distributed.run --standalone --nproc-per-node=2 scripts/train_ddp.py --backend gloo --model tiny --local-batch-size 1 --seq-len 8 --steps 2
```

## Kaggle two-GPU NCCL training

```bash
torchrun --standalone --nproc-per-node=2 scripts/train_ddp.py --backend nccl --model tiny --local-batch-size 2 --seq-len 128 --steps 5 --seed 42
```

Step 11 proves the training path executes. Step 12 will explicitly gather and
compare parameters across ranks and compare DDP against an equivalent global-
batch reference update.
