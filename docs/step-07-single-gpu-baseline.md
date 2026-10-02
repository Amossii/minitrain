# Step 7: Single-GPU baseline protocol

The formal baseline compares tiny, small, and medium while holding constant:

- one GPU and world size 1;
- local/global batch size;
- sequence length;
- FP32 precision;
- AdamW learning rate;
- warmup and measured step counts;
- synthetic-data and model seed.

Only model size changes. This is a workload-size comparison, not a scaling or
speedup experiment. Every model runs in a fresh process so CUDA allocations and
allocator state from the previous model cannot contaminate peak memory.

## Kaggle run

```bash
python3 scripts/check_env.py --require-gpus 1
python3 scripts/benchmark_single.py \
  --device cuda \
  --models tiny small medium \
  --local-batch-size 1 \
  --seq-len 256 \
  --warmup-steps 5 \
  --steps 20 \
  --seed 42 \
  --run-name kaggle_single_fp32
```

Outputs:

```text
results/raw/kaggle_single_fp32/tiny.csv
results/raw/kaggle_single_fp32/small.csv
results/raw/kaggle_single_fp32/medium.csv
results/tables/kaggle_single_fp32.csv
```

The summary reports medians for time and throughput, because isolated slow
steps can skew a mean, and maxima for peak CUDA memory. Actual hardware output
must be preserved alongside results. This repository does not claim baseline
numbers until the command has run successfully on the target Kaggle GPU.
