# MiniTrain

MiniTrain is a small, inspectable lab for learning how modern LLM training
systems work. The project starts from a single-GPU PyTorch baseline and will
progress toward DDP, FSDP2, tensor parallelism, and DeepSpeed on a single node
with two GPUs.

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
conda run -n sglang python -m torch.distributed.run \
  --standalone --nproc-per-node=2 \
  scripts/distributed_hello.py --backend gloo

torchrun --standalone --nproc-per-node=2 \
  scripts/distributed_hello.py --backend nccl
```
