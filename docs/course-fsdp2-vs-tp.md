# 课程实验：FSDP2 vs Tensor Parallel 容量与吞吐

## 问题与系统位置

FSDP2 和 TP 都能让单卡放不下的模型在多卡上训练，但切分对象不同：FSDP2 在参数
使用前 AllGather、反向后 ReduceScatter；本项目的 TP 切分每层 Linear，并在 attention
输出和 MLP 输出执行 AllReduce。本实验比较两件事：2 × T4 16GB 上能完成一个 AdamW
更新的最大模型，以及同一个模型上的训练吞吐。

## 固定实验协议

- 2 × NVIDIA T4 16GB，NCCL，单机双进程。
- FP32 参数、前反向和 AdamW；不启用 AMP。
- activation checkpoint 关闭。
- 默认 `seq_len=128`、有效 `global_batch_size=2`。
- 固定 12 层、8 heads、SwiGLU ratio=3、vocab=32,000，只扫描 hidden size。
- 容量点分别在新进程中运行，非 CUDA OOM 错误不会被记成 OOM。
- 吞吐使用两种策略都通过容量验证的同一个模型，5 步 warmup、20 步测量。

FSDP2 是数据并行，因此 global batch=2 时每卡读取 1 个不同样本。TP 两卡共同计算
同一批样本，因此每卡都读取完整 batch=2。两者的有效 global batch 相同，但每卡
物理输入 batch 不同；同时固定这两个数在数学上不可能。

## 输入、状态与输出

输入是候选参数量、共同吞吐模型规模和上述固定 workload。每个 worker 的内部状态是
模型 shard、AdamW state、activation 和 CUDA allocator。输出包括：

- `results/raw/<run>/manifest.json`：硬件与固定协议。
- `results/tables/<run>_capacity.csv`：每个真实 PASS/OOM 点及峰值显存。
- `results/tables/<run>_throughput.csv`：同模型的 median step time、tokens/s 和显存。
- `results/raw/<run>/**/logs/`：每个子进程的完整日志。

## Kaggle 执行命令

```bash
python scripts/benchmark_fsdp2_vs_tp.py --capacity-targets-millions 100 200 400 600 800 1000 1200 1400 --throughput-target-millions 100 --global-batch-size 2 --seq-len 128 --run-name kaggle_fsdp2_vs_tp
```

先用较稀疏点定位区间，再增加区间内候选点，才能把“最大已验证 PASS”收紧为有意义的
OOM boundary。吞吐目标必须同时出现在容量候选中，且 FSDP2/TP 都为 PASS，否则脚本
拒绝生成吞吐对比。

## Correctness 与结果解释

该脚本复用 Step 15/21 已有的 FSDP2/TP correctness 实现，但 benchmark 前仍建议运行：

```bash
torchrun --standalone --nproc-per-node=2 scripts/verify_fsdp2_correctness.py --backend nccl
torchrun --standalone --nproc-per-node=2 scripts/verify_tp_transformer.py --backend nccl
```

容量结果只表述为“候选集合中的最大已验证值”，不能声称是精确极限。吞吐差异要结合
通信解释：FSDP2 通信围绕参数生命周期，TP 每层产生 collective；短序列、小 batch
通常更难摊薄 TP 的逐层通信。任何具体容量或吞吐结论必须来自生成的 CSV，不能预写。

## Checkpoint

- [x] 固定模型结构、FP32、AdamW、seq_len、有效 global batch。
- [x] activation checkpoint 关闭。
- [x] 容量点隔离运行并严格区分 OOM 与其他错误。
- [x] 同模型吞吐比较和 warmup。
- [x] 保存原始日志、manifest 与 CSV。
- [ ] 在 Kaggle 2 × T4 上执行并分析真实结果。

状态：`IMPLEMENTED / NOT YET GPU-VERIFIED`。

推荐 commit message：

```text
perf: compare FSDP2 and tensor parallel capacity
```
