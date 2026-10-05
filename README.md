# MiniTrain

MiniTrain is a small, inspectable lab for learning how modern LLM training
systems work. The project starts from a single-GPU PyTorch baseline and will
progress toward DDP, FSDP2, tensor parallelism, and DeepSpeed on a single node
with two GPUs.

The 24-step implementation is complete. It includes explicit distributed
correctness checks, reproducible CSV benchmarks, profiler entry points, manual
2-way tensor parallelism, and sharded training checkpoints. See the
[final report](report/final_report.md) for measured results and limitations.

## Key measured result

On the recorded 2×T4 FP32 OOM sweep, DDP trained up to approximately 601M
parameters before the next 650M candidate failed, while FSDP2 trained 1.448B
before the next 1.502B candidate failed: a 2.41× increase in largest verified
trainable model. The tiny-model throughput experiments intentionally show that
distributed execution can be slower when communication dominates computation.

## Step 1: verify the environment

The environment checker reports the versions and hardware that later steps
depend on. It does not run a training workload.

```bash
python3 scripts/check_env.py
```

On the target Kaggle two-GPU runtime, use strict validation:

```bash
python3 scripts/check_env.py --require-gpus 2
```

Run the CPU-compatible tests with:

```bash
python3 -m unittest discover -s tests -v
```

The checker exits with a non-zero status in strict mode if PyTorch, CUDA,
NCCL, or the requested number of GPUs is unavailable. GPU topology is read
from `nvidia-smi topo -m` when that command exists.

## Repository layout

```text
minitrain/       Reusable Python implementation
scripts/         Executable experiment entry points
configs/         Experiment configuration files (introduced in Step 2)
tests/           CPU, GPU, and distributed correctness tests
results/raw/     Raw benchmark records
results/tables/  Derived result tables
results/figures/ Benchmark and profiler figures
docs/            Design notes and experiment reports
```

Core implementations live in Python files rather than notebook cells, so the
same commands can run on Kaggle and ordinary Linux machines.

## Step 2: inspect experiment configuration

Model shapes and training parameters are validated before allocating GPU state.
The built-in model presets are `tiny`, `small`, and `medium`.

```bash
python3 scripts/show_config.py --model tiny
python3 scripts/show_config.py --model small --local-batch-size 4 --precision fp16
```

`local_batch_size` always means samples processed by one process. Future
data-parallel code will derive `global_batch_size` from it and `world_size`.

## Step 3: inspect synthetic token data

MiniTrain uses deterministic random tokens to isolate systems experiments from
tokenization and storage overhead. Each label is the next token corresponding
to an input position.

```bash
python3 scripts/inspect_data.py --model tiny --batch-size 2 --seq-len 8
python3 -m unittest discover -s tests -v
```

## Step 4: inspect the handwritten Transformer

The decoder uses pre-norm causal self-attention and a SwiGLU feed-forward
network. Its explicit projection layers will become tensor-parallel boundaries
in later steps.

```bash
conda run -n sglang python scripts/inspect_model.py \
  --model tiny --batch-size 2 --seq-len 8
conda run -n sglang python -m unittest discover -s tests -v
```

## Step 5: run the single-device training loop

The training entry point exposes the complete PyTorch update lifecycle without
Trainer-style framework abstraction.

```bash
conda run -n sglang python scripts/train_single.py \
  --model tiny --device cpu --local-batch-size 2 --seq-len 16 --steps 3
```

## Step 6: record unified metrics

Warmup steps are excluded from measurement. CPU uses a monotonic clock; CUDA
uses Events for asynchronous GPU timing. Each measured step is written using a
stable CSV schema.

```bash
conda run -n sglang python scripts/train_single.py \
  --model tiny --device cpu --local-batch-size 1 --seq-len 8 \
  --warmup-steps 1 --steps 2 --output /tmp/minitrain_metrics.csv
```

## Step 7: run the formal single-GPU baseline

The runner executes every model in a fresh process and creates both raw
per-step CSV files and one median summary table.

```bash
python3 scripts/benchmark_single.py \
  --device cuda --models tiny small medium \
  --local-batch-size 1 --seq-len 256 \
  --warmup-steps 5 --steps 20 \
  --run-name kaggle_single_fp32
```

## Step 8: initialize a distributed process group

Use Gloo for a local two-process CPU correctness check and NCCL for the target
Kaggle two-GPU runtime.

```bash
conda run -n sglang python -m torch.distributed.run --standalone --nproc-per-node=2 scripts/distributed_hello.py --backend gloo
torchrun --standalone --nproc-per-node=2 scripts/distributed_hello.py --backend nccl
```

## Step 9: verify collective semantics

The correctness script uses hand-computable tensors and assertions for
Broadcast, Reduce, AllReduce, AllGather, ReduceScatter, and Barrier.

```bash
torchrun --standalone --nproc-per-node=2 scripts/collective_correctness.py --backend nccl
```

## Step 10: benchmark NCCL AllReduce

The benchmark sweeps per-rank message sizes, excludes warmup, uses CUDA Events,
and writes latency plus algorithm/bus bandwidth to CSV.

```bash
torchrun --standalone --nproc-per-node=2 scripts/benchmark_collectives.py --backend nccl --min-bytes 1024 --max-bytes 268435456 --factor 4 --warmup-iterations 10 --iterations 50 --output results/raw/kaggle_nccl_all_reduce.csv
```

## Step 11: train with DistributedDataParallel

Each rank owns a full model replica and a different local data shard. DDP
synchronizes gradients during backward before every rank updates its replica.

```bash
torchrun --standalone --nproc-per-node=2 scripts/train_ddp.py --backend nccl --model tiny --local-batch-size 2 --seq-len 128 --steps 5 --seed 42
```

## Step 12: verify DDP correctness

The check compares every synchronized gradient and parameter across ranks, then
compares one DDP update with an equivalent single-process global-batch update.

```bash
torchrun --standalone --nproc-per-node=2 scripts/verify_ddp_correctness.py --backend nccl --local-batch-size 2 --seq-len 16 --learning-rate 0.01 --seed 42
```

## Step 13: benchmark DDP scaling

The runner performs world-size 1/2 strong and weak scaling jobs in fresh
processes and writes raw plus summarized CSV results.

```bash
python3 scripts/benchmark_ddp_scaling.py --backend nccl --model tiny --seq-len 128 --strong-global-batch-size 2 --weak-local-batch-size 1 --warmup-steps 5 --steps 20 --seed 42 --run-name kaggle_ddp_scaling
```

## Step 14: profile DDP communication overlap

Warm up the reducer, then capture rank-local CPU/CUDA/NCCL timelines and DDP
bucket metadata without adding measurement collectives to the profiled region.

```bash
torchrun --standalone --nproc-per-node=2 scripts/profile_ddp.py --backend nccl --model tiny --local-batch-size 2 --seq-len 128 --warmup-steps 3 --profile-steps 3 --bucket-cap-mb 1 --output-dir results/raw/profiles/ddp_tiny
```

## Step 15: train and verify FSDP2

FSDP2 shards each Transformer block bottom-up, then shards the parameters left
at the root. The correctness job checks DTensor ownership and compares one
sharded update against an unsharded global-batch reference.

```bash
torchrun --standalone --nproc-per-node=2 scripts/verify_fsdp2_correctness.py --backend nccl --local-batch-size 2 --seq-len 16 --learning-rate 0.01 --seed 42
```

```bash
torchrun --standalone --nproc-per-node=2 scripts/train_fsdp2.py --backend nccl --model tiny --local-batch-size 2 --seq-len 128 --steps 5 --seed 42
```

## Step 16: compare DDP and FSDP2 GPU memory

Run each strategy in a fresh process group with an identical workload, then
compare measured allocated/reserved peaks with persistent-state estimates.

```bash
python scripts/benchmark_ddp_fsdp2_memory.py --model tiny --world-size 2 --local-batch-size 1 --seq-len 128 --warmup-steps 3 --steps 5 --seed 42 --run-name kaggle_tiny_memory
```

## Step 17: find the DDP and FSDP2 OOM boundary

Sweep one fixed Transformer family in fresh jobs. Only explicit CUDA
out-of-memory failures count as OOM; other failures stop the experiment.

```bash
python scripts/benchmark_oom_boundary.py --strategies ddp fsdp2 --targets-millions 100 200 300 400 --world-size 2 --local-batch-size 1 --seq-len 128 --steps 1 --seed 42 --run-name kaggle_oom_boundary
```

## Step 18: profile FSDP2 communication

Capture per-rank traces and inspect parameter AllGather, gradient
ReduceScatter, layer computation, and their possible overlap.

```bash
torchrun --standalone --nproc-per-node=2 scripts/profile_fsdp2.py --backend nccl --model tiny --local-batch-size 2 --seq-len 128 --warmup-steps 3 --profile-steps 3 --seed 42 --output-dir results/raw/profiles/fsdp2_tiny
```

## Step 19: implement ColumnParallelLinear

Shard `nn.Linear.weight=[out_features, in_features]` along its output dimension,
optionally AllGather the output, and AllReduce input-gradient contributions.

```bash
torchrun --standalone --nproc-per-node=2 scripts/verify_column_parallel.py --backend nccl --batch-size 2 --seq-len 3 --in-features 8 --out-features 12 --seed 42
```

## Step 20: implement RowParallelLinear

Shard `nn.Linear.weight=[out_features, in_features]` along its input dimension,
sum local output contributions with AllReduce, then add replicated bias once.

```bash
torchrun --standalone --nproc-per-node=2 scripts/verify_row_parallel.py --backend nccl --batch-size 2 --seq-len 3 --in-features 12 --out-features 8 --seed 42
```

## Step 21: train a tensor-parallel Transformer

Use Column Parallel for Q/K/V/Gate/Up and Row Parallel for Attention Output/Down,
keeping intermediate shards local and restoring replicated residual streams.

```bash
torchrun --standalone --nproc-per-node=2 scripts/verify_tp_transformer.py --backend nccl --batch-size 2 --seq-len 8 --learning-rate 0.01 --seed 42
```

```bash
torchrun --standalone --nproc-per-node=2 scripts/train_tp.py --backend nccl --model tiny --batch-size 2 --seq-len 128 --steps 5 --seed 42
```

## Step 22: train with DeepSpeed ZeRO-1/2/3

Use the same MiniTransformer workload while progressively sharding optimizer
states, gradients, and parameters. DeepSpeed remains an optional dependency.

```bash
torchrun --standalone --nproc-per-node=2 scripts/train_deepspeed.py --zero-stage 3 --model tiny --local-batch-size 2 --seq-len 128 --steps 3 --seed 42 --verify-parameters
```

## Step 23: run the unified benchmark

Run Single, DDP, FSDP2, DeepSpeed ZeRO-1/2/3, and handwritten TP with one
fixed workload. Data-parallel strategies and tensor parallelism are written to
separate comparison tables because their per-rank batch semantics differ.

```bash
python scripts/benchmark_unified.py --model tiny --world-size 2 --global-batch-size 2 --seq-len 128 --learning-rate 0.0003 --warmup-steps 5 --steps 20 --seed 42 --run-name kaggle_unified_tiny
```

See `docs/step-23-unified-benchmark.md` for metric definitions, fairness rules,
expected files, and result-analysis questions.

## Step 24: verify distributed checkpoint resume

Save and restore logical model state, AdamW state, and the next training step
without gathering FSDP2 shards into a rank-0 `torch.save` file. Correctness is
proven by continuing both uninterrupted and restored branches for one update.

```bash
torchrun --standalone --nproc-per-node=2 scripts/verify_distributed_checkpoint.py --strategy ddp --backend nccl --model tiny --local-batch-size 1 --seq-len 16 --pre-steps 2 --checkpoint-dir /kaggle/working/minitrain_checkpoints/ddp_step2
```

```bash
torchrun --standalone --nproc-per-node=2 scripts/verify_distributed_checkpoint.py --strategy fsdp2 --backend nccl --model tiny --local-batch-size 1 --seq-len 16 --pre-steps 2 --checkpoint-dir /kaggle/working/minitrain_checkpoints/fsdp2_step2
```

The full design, expected output, and validation status are documented in
`docs/step-24-checkpoint-and-final-report.md`.

## Course experiment: compare FSDP2 and TP capacity

Run an isolated FP32 OOM sweep on 2 × T4, then compare throughput using one
model size that both strategies have passed. The effective global batch and
model structure stay fixed; FSDP2 ranks receive distinct local samples while
TP ranks cooperate on the same replicated batch.

```bash
python scripts/benchmark_fsdp2_vs_tp.py --capacity-targets-millions 100 200 400 600 800 1000 1200 1400 --throughput-target-millions 100 --global-batch-size 2 --seq-len 128 --run-name kaggle_fsdp2_vs_tp
```

See `docs/course-fsdp2-vs-tp.md` for the fixed workload, output schema,
correctness prerequisites, and rules for interpreting the results.
