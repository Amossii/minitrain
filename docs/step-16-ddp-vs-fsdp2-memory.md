# Step 16 — DDP vs FSDP2 Memory Benchmark

## 1. 本 Step 要解决的问题

Step 15 证明 FSDP2 数值正确。本 Step 回答它是否真的减少每张 GPU 的显存，以及节省
来自参数、梯度还是 optimizer state。比较固定同一模型、global/local batch、seq_len、
FP32、AdamW、训练步数和 seed，只改变 `ddp`/`fsdp2` 策略。

## 2. 理论模型

对 `P` 个 FP32 参数，忽略 Adam step scalar 和框架 bookkeeping：

```text
parameters       = 4P bytes
gradients        = 4P bytes
Adam exp_avg     = 4P bytes
Adam exp_avg_sq  = 4P bytes
DDP per rank     = 16P bytes
FSDP2 ideal/rank = 16P / world_size bytes
```

这是持久 model state，不包含 activation、临时 full parameter、collective buffer、CUDA
context 和 allocator 碎片。因此 FSDP2 的真实 peak 不会简单等于 DDP peak 除以卡数。

## 3. 测量设计

DDP 和 FSDP2 分别在全新的 `torchrun` 作业中运行，避免 caching allocator 互相污染。
至少一个 warmup step 先创建 Adam moments，随后调用 `empty_cache()` 和
`reset_peak_memory_stats()`。正式 steps 完成后同步 CUDA，再读取：

- `max_memory_allocated`：活跃 tensor 曾经占用的峰值。
- `max_memory_reserved`：PyTorch caching allocator 向 CUDA 保留的峰值。
- end allocated/reserved：测量结束时的常驻状态。
- parameter/gradient/optimizer bytes：实际 rank-local tensor payload。

比较时以两个 rank 的最大值为准，因为最满的 rank 决定是否 OOM。

## 4. 修改文件

- `minitrain/distributed/memory.py`：本地 tensor 计数、理论值、公平性验证和 CSV。
- `scripts/benchmark_memory_worker.py`：单策略 GPU memory worker。
- `scripts/benchmark_ddp_fsdp2_memory.py`：隔离运行两种策略并生成对比表。
- `tests/test_memory_benchmark.py`：公式和公平性 CPU 单测。

## 5. Kaggle/Linux 执行命令

正式双 GPU 对比（单行）：

```bash
python scripts/benchmark_ddp_fsdp2_memory.py --model tiny --world-size 2 --local-batch-size 1 --seq-len 128 --warmup-steps 3 --steps 5 --seed 42 --run-name kaggle_tiny_memory
```

查看对比表（单行）：

```bash
python -c "import pandas as pd; print(pd.read_csv('results/tables/kaggle_tiny_memory.csv').to_string(index=False))"
```

运行 CPU 单元测试（单行）：

```bash
conda run -n sglang python -m unittest discover -s tests -v
```

## 6. 预期输出

原始证据保存在 `results/raw/kaggle_tiny_memory/ddp.csv` 和 `fsdp2.csv`，最终表保存在
`results/tables/kaggle_tiny_memory.csv`。表中 saving 来自真实
`observed_peak_allocated_bytes`，不是理论估算。

## 7. 如何解释结果

若 FSDP2 的 parameter/gradient/optimizer 本地字节接近 DDP 的一半，说明持久状态成功
分片。若 peak allocated 没有减少一半，通常来自 activation 没有分片，以及 forward/
backward 的临时 AllGather full parameters。reserved 明显高于 allocated 反映 allocator
缓存或碎片，不能把 reserved 全部解释成活跃模型状态。

如果 tiny 模型收益很小甚至更高，也不能判定实现失败：固定的 CUDA/NCCL buffer 和
FSDP runtime 开销可能压过小模型的分片收益。应如实记录，并在 Step 17 增大模型寻找
OOM boundary。

## 8. Checkpoint

- IMPLEMENTED：隔离进程、公平 workload、理论与实测 memory CSV。
- CPU-VERIFIED：字节公式、ratio 和公平性校验。
- NOT YET GPU-VERIFIED：真实 DDP/FSDP2 峰值必须在 Kaggle 双 GPU 测量。

## 9. 推荐 Git commit message

`perf: compare DDP and FSDP2 GPU memory`

## 10. 对实习/面试的价值

你能区分 allocated/reserved、持久状态/临时 buffer，并解释为什么 FSDP2 理论 model
state 按 world size 分片，但真实 peak 不会严格线性下降。
