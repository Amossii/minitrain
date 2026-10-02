# Step 13: DDP strong and weak scaling

Strong scaling fixes global batch size. For the default two-GPU matrix, one
rank uses local batch 2 and two ranks use local batch 1. Speedup is baseline
step time divided by distributed step time.

Weak scaling fixes local batch size at 1. Global batch grows from 1 to 2 as
world size grows. Scaling is distributed tokens/s divided by baseline tokens/s.

Both use `efficiency = scaling_factor / world_size`. Every configuration runs
in a fresh torchrun process, with fixed model, sequence length, precision,
optimizer, learning rate, warmup, measured steps, and seed.

The worker records the slowest rank's timing and largest rank memory. Its CUDA
Event backward interval includes DDP communication. Aggregation collectives run
after timing and do not enter measured step time.

## Kaggle command

```bash
python3 scripts/benchmark_ddp_scaling.py --backend nccl --model tiny --seq-len 128 --strong-global-batch-size 2 --weak-local-batch-size 1 --warmup-steps 5 --steps 20 --seed 42 --run-name kaggle_ddp_scaling
```

The command creates four raw files under `results/raw/kaggle_ddp_scaling/` and
one comparison table at `results/tables/kaggle_ddp_scaling.csv`. No scaling
claim is valid until these GPU measurements complete successfully.
