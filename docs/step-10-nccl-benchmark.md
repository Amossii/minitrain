# Step 10: NCCL AllReduce benchmark

This benchmark sweeps per-rank float32 tensor sizes and measures repeated
AllReduce SUM operations. Warmup collectives are excluded. CUDA Events measure
GPU-stream elapsed time, and an untimed MAX reduction selects the slowest rank.

Definitions use decimal GB/s:

```text
latency = maximum rank elapsed time / iterations
algorithm bandwidth = message bytes / latency / 1e9
bus bandwidth = algorithm bandwidth * 2 * (world_size - 1) / world_size
```

Small messages are generally latency-sensitive, while sufficiently large
messages can approach a bandwidth-dominated regime. Actual transition points
and bandwidth must come from the target GPU/topology experiment.

## Kaggle two-GPU NCCL command

```bash
torchrun --standalone --nproc-per-node=2 scripts/benchmark_collectives.py --backend nccl --min-bytes 1024 --max-bytes 268435456 --factor 4 --warmup-iterations 10 --iterations 50 --output results/raw/kaggle_nccl_all_reduce.csv
```

## Optional ordinary-host Gloo diagnostic

```bash
conda run -n sglang python -m torch.distributed.run --standalone --nproc-per-node=2 scripts/benchmark_collectives.py --backend gloo --min-bytes 1024 --max-bytes 1048576 --factor 4 --warmup-iterations 2 --iterations 5 --output /tmp/gloo_all_reduce.csv
```

Gloo/CPU results validate the execution path but are not NCCL/GPU performance
results. Preserve `nvidia-smi topo -m` and environment-check output with the
Kaggle CSV before drawing communication conclusions.
