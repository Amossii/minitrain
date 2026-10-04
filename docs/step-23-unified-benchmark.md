# Step 23：Unified Benchmark

## 1. 本 Step 要解决的问题

前面的实验分别验证了 Single GPU、DDP、FSDP2、Tensor Parallel 和
DeepSpeed ZeRO，但分散脚本产生的结果不能直接横向比较。Step 23 建立同一套
workload、计时边界、CSV schema 和公平性检查，回答：在相同模型和一次更新的
有效样本数下，不同并行策略的吞吐、阶段耗时和峰值显存有什么差异？

## 2. 为什么需要它

如果 DDP 使用 `global_batch=4`，Single 使用 `batch=1`，或某个策略少做 warmup，
得到的速度排名没有意义。统一 benchmark 固定：

- 模型结构、参数量和序列长度
- global batch size、AdamW、FP32 和学习率
- 随机种子、warmup steps 和 measured steps
- CUDA Event 计时方式
- synthetic dataset 和数据顺序

程序会在写汇总表前检查这些字段；不一致会直接报错。

## 3. 核心原理

数据并行和 Tensor Parallel 的 batch 语义不同：

```text
DDP/FSDP2/ZeRO: 每个 rank 持有不同样本，global_batch = per_process_batch × world_size
TP:             每个 rank 持有同一批样本，global_batch = per_process_batch
```

因此 TP 不进入数据并行主排名，而是与 Single 单独生成 TP 对照表。否则把 TP 的
batch 再乘 world size，会把 tokens/s 虚增两倍。

每个策略在新进程中运行。每个 measured step 使用 rank 间 mean loss、最大耗时和
最大显存：同步训练的 step 结束时间由最慢 rank 决定，容量边界由显存最高 rank
决定。原始数据保存每个 step，汇总表取中位数，降低单次抖动的影响。

阶段时间代表 API 生命周期而非纯计算：例如 DDP 的 AllReduce 位于 backward，
ZeRO 的通信可能分布在 `engine.backward()` 和 `engine.step()` 中。

## 4. 本 Step 修改的文件

- `minitrain/distributed/unified_benchmark.py`：统一 schema、策略语义、汇总与公平性检查。
- `scripts/benchmark_native_worker.py`：测量 Single、DDP、FSDP2、TP。
- `scripts/benchmark_deepspeed_worker.py`：测量 ZeRO-1/2/3。
- `scripts/benchmark_unified.py`：用独立进程编排所有策略并生成两张表。
- `tests/test_unified_benchmark.py`：验证 batch/吞吐语义、CSV 和公平性规则。

## 5. 实现中的状态变化

worker 的输入是固定 workload 和策略；内部状态包括模型参数、梯度、optimizer
状态、分布式 shard 和 DataLoader 位置。一次调用的输出是一条 `StepMeasurement`，
同时模型完成一次参数更新。下一次调用读取更新后的参数和下一个数据 batch。

原始 CSV 中同时保留 `batch_size_per_process` 与 `global_batch_size`、并行类别和主要
通信。`num_parameters` 始终是完整逻辑模型参数量，而不是某 rank 的本地 shard
大小，因此 FSDP2、ZeRO-3 和 TP 仍能与 Single 对齐。

## 6. 测试代码

CPU 单元测试不需要 DeepSpeed 或 GPU，覆盖：

- DDP 的 global batch 必须等于 per-process batch 乘 world size。
- TP 的 global batch 必须等于每个 rank 看到的 replicated batch。
- TP tokens/s 不得乘 world size。
- 不同 global batch 的汇总行不得进入同一比较表。
- 原始 CSV 可以稳定汇总为中位数结果。

## 7. Kaggle/Linux 执行命令

安装项目和 DeepSpeed（单行）：

```bash
pip install -e '.[deepspeed]'
```

运行完整双卡实验（单行）：

```bash
python scripts/benchmark_unified.py --model tiny --world-size 2 --global-batch-size 2 --seq-len 128 --learning-rate 0.0003 --warmup-steps 5 --steps 20 --seed 42 --run-name kaggle_unified_tiny
```

只先验证原生策略、跳过 DeepSpeed 和 TP（单行）：

```bash
python scripts/benchmark_unified.py --model tiny --world-size 2 --global-batch-size 2 --seq-len 128 --warmup-steps 5 --steps 20 --strategies ddp fsdp2 --no-include-tp --run-name kaggle_native_tiny
```

运行全部 CPU 单元测试（单行）：

```bash
python -m unittest discover -s tests -v
```

查看数据并行汇总表（单行）：

```bash
python -c "import pandas as pd; print(pd.read_csv('results/tables/kaggle_unified_tiny.csv').to_string(index=False))"
```

查看 TP 对照表（单行）：

```bash
python -c "import pandas as pd; print(pd.read_csv('results/tables/kaggle_unified_tiny_tp.csv').to_string(index=False))"
```

## 8. 预期输出

运行期间每个策略会打印 `loss`、`step_time`、`tokens_per_second` 和原始 CSV 路径。
完成后生成：

```text
results/raw/kaggle_unified_tiny/single.csv
results/raw/kaggle_unified_tiny/ddp.csv
results/raw/kaggle_unified_tiny/fsdp2.csv
results/raw/kaggle_unified_tiny/zero1.csv
results/raw/kaggle_unified_tiny/zero2.csv
results/raw/kaggle_unified_tiny/zero3.csv
results/raw/kaggle_unified_tiny/tp.csv
results/tables/kaggle_unified_tiny.csv
results/tables/kaggle_unified_tiny_tp.csv
```

本文不预写任何性能数字；实际排序必须以目标 Kaggle 双卡运行结果为准。

## 9. Correctness 验证

Step 23 的职责是公平测量而不是替代前面各策略的数值对齐测试。worker 仍检查每个
measured loss 为有限值，schema 检查 DP/TP batch 语义，汇总器检查固定 workload。
DDP、FSDP2、TP 和 ZeRO 的参数/梯度 correctness 由 Step 12、15、19–22 的专用
测试负责；benchmark 不能把“程序跑完”当成这些实现正确的唯一证据。

## 10. Benchmark 验证与结果解释

分析结果时至少回答：

1. 哪个策略的 median step time 最低，通信发生在哪个阶段？
2. allocated 与 reserved 的差值多大，是否可能存在 allocator 缓存或碎片？
3. FSDP2/ZeRO 是否用额外通信换到了显存下降？
4. tiny 模型是否太小，使通信和框架开销压过计算收益？
5. TP 每个 rank 是否因为 replicated embedding/LM head 和频繁 AllReduce 而没有加速？

`max_peak_memory_*` 是 measured step 的 CUDA allocator 峰值，不等价于 `nvidia-smi`
进程总显存；`reserved` 也不等价于活跃 tensor 占用。

## 11. 本 Step Checkpoint

- [x] 统一逐 step schema 和中位数汇总表。
- [x] 显式区分 DP 与 TP 的 batch/吞吐语义。
- [x] 每个策略使用独立进程和相同 workload。
- [x] CUDA Event、warmup、跨 rank 最大耗时/显存。
- [x] 自动拒绝不公平的汇总数据。
- [x] CPU 单元测试通过。
- [ ] Kaggle 双 GPU + NCCL + DeepSpeed 实测。

当前状态：`IMPLEMENTED / CPU-VERIFIED / NOT YET GPU-VERIFIED`。

## 12. 推荐 Git commit message

```text
perf: add unified distributed training benchmark
```

## 13. 对实习和面试的价值

这个 Step 展示的不只是“会跑多个框架”，而是能够定义公平实验、识别 DP/TP 的
吞吐口径差异、正确处理 CUDA 异步计时、保存可复查原始数据，并把显存、通信与
计算 trade-off 联系起来。面试时应结合真实 CSV 和 profiler trace 解释结果，
包括没有加速或出现回归的实验。
