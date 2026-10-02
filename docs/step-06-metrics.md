# Step 6: Metrics and CSV infrastructure

Warmup steps update the model and optimizer but are excluded from recorded
rows. This prevents initialization, allocator growth, and first-use kernel
costs from contaminating steady-state measurements.

CPU phases use `time.perf_counter`. CUDA phases use four Events recorded on the
active stream. Only the final Event is synchronized, after which elapsed GPU
times are read for forward, backward, optimizer, and the entire step. Python
wall time alone is not used to claim CUDA kernel performance.

Throughput uses the explicit distributed definitions:

```text
global_batch_size = local_batch_size * world_size
samples_per_second = global_batch_size / step_time
tokens_per_second = global_batch_size * seq_len / step_time
```

CUDA peak memory reports both caching-allocator values:

- allocated: bytes occupied by live tensors;
- reserved: bytes held by PyTorch's CUDA allocator, including reusable blocks.

CPU rows use zero for these CUDA-only fields; this does not represent process
RSS. CSV rows are written under `results/raw/` and use one stable schema for
future single-GPU, DDP, FSDP2, tensor-parallel, and DeepSpeed experiments.

## Run

```bash
conda run -n sglang python scripts/train_single.py \
  --model tiny --device cpu --local-batch-size 1 --seq-len 8 \
  --warmup-steps 1 --steps 2 --output /tmp/minitrain_metrics.csv
```

This command validates the infrastructure. Step 7 will define and run the
formal single-GPU benchmark matrix.
