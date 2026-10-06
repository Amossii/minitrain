# FSDP2 vs Tensor Parallel：双 T4 训练容量与吞吐实验报告

## 摘要

本实验在 2 × NVIDIA Tesla T4 上，对 PyTorch FSDP2 和 MiniTrain 手写的两路
Tensor Parallel（TP）进行训练容量与吞吐比较。两种策略使用相同的 Decoder-only
Transformer 结构、FP32 AdamW、sequence length 128、有效 global batch size 2，
并关闭 activation checkpoint。

实验得到两个互补结论：

1. FSDP2 的最大已验证模型为 1.448B 参数，TP 为 1.300B。FSDP2 在本次候选点中
   多容纳约 148M 参数，最大已验证容量高约 11.4%。
2. 在两种策略都能训练的同一个 1.200B 参数模型上，TP 达到 499.34 tokens/s，
   FSDP2 为 167.31 tokens/s；TP 吞吐为 FSDP2 的 2.98 倍，但 peak allocated
   memory 高 1.52 GiB。

因此，在这个特定双卡 workload 下，FSDP2 更适合最大化模型容量，TP 更适合提高
已经能够放入显存的模型的训练吞吐。该结论不能脱离本实验的模型规模、序列长度、
batch size、精度和互联条件外推。

## 1. 问题是什么

FSDP2 和 TP 都能将单卡无法独立承担的训练工作分布到多张 GPU，但它们解决显存问题
的方式不同：

- FSDP2 按数据并行语义运行，在参数使用前 AllGather，并在反向后通过
  ReduceScatter 保存分片梯度；参数、梯度和 optimizer state 均可分片。
- TP 将单层 Linear 的权重和计算切到多张 GPU。当前实现把 Q/K/V、Gate、Up 按
  输出维切分，把 Attention Output、Down 按输入维切分，并在必要位置执行
  AllReduce。

实验回答两个不同问题：

1. 固定训练协议后，每种策略能够完成完整 AdamW update 的最大已验证模型有多大？
2. 对同一个、两种策略都能运行的模型，哪种策略的训练吞吐更高？

容量和吞吐必须分开比较。若分别使用两种策略各自的最大模型测吞吐，模型规模不同，
结果无法归因于并行策略。

## 2. 固定实验协议

实验 manifest：`results/raw/kaggle_fsdp2_vs_tp/manifest.json`。

| 配置项 | 固定值 |
|---|---|
| Hardware | 2 × Tesla T4 |
| World size | 2 |
| Precision | FP32 |
| Optimizer | AdamW |
| Sequence length | 128 |
| Effective global batch size | 2 |
| Activation checkpoint | Disabled |
| Vocabulary size | 32,000 |
| Transformer layers | 12 |
| Attention heads | 8 |
| SwiGLU intermediate ratio | 3 |
| Seed | 42 |

模型规模仅通过 hidden size 改变，层数、head 数、词表和 MLP ratio 保持不变。
每个容量点在全新的 `torchrun` 作业中运行，避免一次 OOM 污染后续 CUDA allocator
或 process group。只有日志中明确出现 CUDA OOM 才记为 OOM；NCCL、shape 或其他
程序错误不能作为容量边界。

### Batch 语义

两种策略固定的是一次 optimizer update 包含的不同样本数，即有效 global batch=2：

- FSDP2 是数据并行，每个 rank 读取 1 个不同样本。
- TP 的两个 rank 共同计算同一批样本，因此每个 rank 都持有完整 batch=2。

所以两者的有效 workload 相同，但每卡物理输入 batch 不同。这是数据并行与模型并行
的固有语义，不应通过把 TP batch 乘以 world size 来虚增吞吐。

## 3. 模型容量实验

原始汇总：`results/tables/kaggle_fsdp2_vs_tp_capacity.csv`。

| Strategy | 最大已验证 PASS | 第一个 OOM | 当前边界区间 |
|---|---:|---:|---:|
| FSDP2 | 1,448,039,968 | 1,501,882,008 | [1.448B, 1.502B) |
| TP | 1,299,596,928 | 1,350,643,448 | [1.300B, 1.351B) |

按最大已验证 PASS 计算：

- FSDP2 比 TP 多容纳 148,443,040 个参数，约 148M。
- FSDP2 最大已验证容量是 TP 的 1.114 倍，即高约 11.4%。

这不是精确数学极限。模型宽度只能取满足 head/TP 切分要求的离散值，而且边界会受到
PyTorch、CUDA allocator 和其他进程显存占用影响。严谨表述应是“最大已验证 PASS”
和“当前边界区间”。

边界日志证明失败原因确实是显存不足：

- FSDP2 1.502B：两个 rank 均发生 CUDA OOM，申请额外 50 MiB 时失败。
- TP 1.351B：两个 rank 均发生 CUDA OOM，申请额外 16 MiB 时失败。

### 为什么 FSDP2 容量更高

FSDP2 对参数、梯度和 AdamW state 进行分片。当前 TP 只分片 Transformer Block
中的主要 Linear，以下状态仍在两个 rank 上复制：

- Token embedding。
- Position embedding。
- RMSNorm 参数。
- LM Head。
- 完整输入 batch 和最终 logits。
- 上述 replicated 参数对应的 gradient 与 AdamW state。

词表为 32,000 时，embedding 和 LM Head 及其 optimizer state 并不小。因此 TP
虽然切分了 Block 中的大部分矩阵，整体状态分片仍不如 FSDP2 完整。

## 4. 同模型吞吐实验

原始逐 step 数据：

- `results/raw/kaggle_fsdp2_vs_tp/throughput/fsdp2.csv`
- `results/raw/kaggle_fsdp2_vs_tp/throughput/tp.csv`

汇总表：`results/tables/kaggle_fsdp2_vs_tp_throughput.csv`。

吞吐比较固定使用 1,200,439,184 参数模型，执行 5 个 warmup steps 和 20 个
measured steps。表中时间与吞吐均为逐 step 中位数，显存为测量阶段最大值。

| 指标 | FSDP2 | TP | 对比 |
|---|---:|---:|---:|
| Median step time | 1.5301 s | 0.5127 s | TP step time 低 66.5% |
| Median tokens/s | 167.31 | 499.34 | TP 为 2.98× |
| Median samples/s | 1.307 | 3.901 | TP 为 2.98× |
| Peak allocated | 11.25 GiB | 12.77 GiB | TP 多 1.52 GiB |
| Peak reserved | 13.28 GiB | 13.49 GiB | TP 多 0.21 GiB |

TP 在这个共同模型上的吞吐约为 FSDP2 的 2.98 倍，同时 peak allocated memory
高约 13.5%。这展示了典型的 memory-throughput trade-off：FSDP2 用更完整的状态
分片换取容量，而 TP 保留更多 replicated 状态，但减少每卡 Linear 计算量。

### 分阶段时间

| 阶段 | FSDP2 | TP | FSDP2 / TP |
|---|---:|---:|---:|
| Forward | 489.84 ms | 92.95 ms | 5.27× |
| Backward | 835.39 ms | 189.71 ms | 4.40× |
| Optimizer | 204.91 ms | 230.42 ms | 0.89× |

TP 的主要优势来自 forward 和 backward。FSDP2 必须围绕每层参数生命周期执行
AllGather 和 ReduceScatter；TP 让每张卡只执行部分 Linear 和 attention heads，
并在 Row Parallel 输出、Column Parallel 输入梯度位置同步。

TP 的 optimizer 阶段反而略慢。一个合理解释是 TP 仍完整保存 embedding、LM Head
等 replicated 参数及其 AdamW state，而 FSDP2 对 optimizer state 进行了完整分片。
这是依据代码 ownership 得出的解释，不是 profiler trace 的直接证据。

## 5. Correctness 依据

性能结果只有建立在数值正确性上才有意义。当前 TP 实现已有以下验证：

- ColumnParallelLinear 与普通 `nn.Linear` 的输出和梯度对齐。
- RowParallelLinear 与普通 `nn.Linear` 的输出和梯度对齐。
- TP Transformer 与未切分 Transformer 的 forward、gradient 和一次 update 对齐。
- FSDP2 与未分片 reference 的 loss、gradient behavior 和 parameter update 对齐。
- 所有浮点比较使用 `torch.testing.assert_close` 或合理容差。

容量探测要求完成 forward、loss、backward 和 AdamW step，而不是只成功构造模型。
因此 PASS 表示该配置能够执行真实训练状态转移。

## 6. 结论与工程判断

本实验不能简单总结为某个策略“绝对更好”。它们优化的是不同瓶颈：

- 如果目标是让尽可能大的模型装入两张 T4，本实验应选择 FSDP2。
- 如果 1.2B 模型已经能够放入显存，并且目标是提高当前 workload 的训练吞吐，
  本实验中的 TP 明显更快。
- FSDP2 的优势来自更完整的 model-state sharding，代价是训练期间频繁重建参数。
- TP 的优势来自切分层内矩阵计算，代价是逐层 collective 和未切分模块的复制。

一句话总结：

> 在 2 × T4、FP32 AdamW、seq_len 128、global batch 2 的固定实验中，FSDP2
> 将最大已验证训练容量从 TP 的 1.300B 提高到 1.448B；而在共同的 1.200B
> 模型上，手写 TP 实现达到 499 tokens/s，是 FSDP2 的 2.98 倍，但多使用约
> 1.52 GiB peak allocated memory。

## 7. 实验限制

1. 吞吐只测量了一个共同模型规模 1.2B，不能证明 2.98× 对所有规模成立。
2. sequence length 128、global batch 2 较小，改变计算通信比后结论可能变化。
3. 实验只覆盖 FP32；T4 没有现代 GPU 的 BF16/FP8 计算能力。
4. 当前 TP 是教学实现，没有 sequence parallel、vocab-parallel LM Head、通信融合
   或成熟框架中的其他优化。
5. 没有保存本次实验的 profiler trace，因此不能量化 NCCL 时间或证明 overlap。
6. 单次独立实验没有置信区间；严格性能报告应进行多次独立进程重复实验。
7. 结果只适用于单机双 T4，不能外推到 NVLink、多节点或更大 TP degree。

这些限制意味着简历中应该明确写出硬件和 workload，并使用 “measured” 或
“achieved under a fixed workload”，不要写成普遍性能承诺。

## 8. 简历表述建议

### 中文版本

> 从 PyTorch collective 原语实现两路 Tensor Parallel Transformer，并设计
> FSDP2/TP 公平 benchmark；在 2 × T4、FP32 AdamW、1.2B 参数的固定 workload
> 上测得 TP 499 tokens/s（FSDP2 的 2.98 倍），同时验证 FSDP2 最大训练容量
> 1.448B，相比 TP 的 1.300B 高 11.4%，分析了 model-state sharding、逐层通信与
> replicated embedding/LM Head 之间的显存—吞吐权衡。

更精简的一行版本：

> Implemented 2-way tensor parallelism from PyTorch collectives and benchmarked it
> against FSDP2 on 2×T4 GPUs, measuring 2.98× the throughput with TP on a fixed
> 1.2B FP32 workload and 11.4% higher verified model capacity for FSDP2.

### English version

> Implemented a 2-way tensor-parallel Transformer from PyTorch collectives and
> built a controlled FSDP2-vs-TP benchmark on 2×T4 GPUs; measured 2.98× the
> throughput with TP on an identical 1.2B-parameter FP32 AdamW workload, while FSDP2
> increased the largest verified trainable model from 1.300B to 1.448B parameters,
> exposing the trade-off between state sharding, layer-wise communication, and
> replicated model components.

推荐使用完整英文版本，因为它同时说明了：

- 关键机制是自己实现，而不只是调用框架。
- 比较使用 controlled workload。
- 数字来自真实测量。
- 能解释性能与容量的 trade-off。

## 9. 面试讲解提纲

如果面试官追问，可以按以下顺序回答：

1. **为什么做实验：** FSDP2 和 TP 都能突破单卡限制，但一个切训练状态，一个切层内
   计算，因此需要分别比较 capacity 和 throughput。
2. **如何保证公平：** 固定逻辑模型、参数量、FP32、AdamW、seq_len、有效 global
   batch、warmup 和 measurement steps；吞吐只比较同一个 1.2B 模型。
3. **如何验证正确：** 将 Column/Row Linear、完整 TP Transformer 和 FSDP2 update
   分别与未分片 reference 数值对齐。
4. **为什么 FSDP2 容量更高：** 参数、梯度、optimizer state 全部分片；当前 TP 的
   embedding、norm、LM Head 和对应 optimizer state 仍复制。
5. **为什么 TP 更快：** 每卡只执行部分 Linear/attention heads；当前大模型 workload
   足以摊薄逐层 collective，而 FSDP2 的参数 AllGather/梯度 ReduceScatter 位于训练
   critical path。
6. **不能声称什么：** 没有 profiler trace，不能声称已经证明通信 overlap；只测一个
   吞吐点，不能把 2.98× 推广到其他模型、batch、序列长度或硬件。

## 10. 可复现实验

```bash
python scripts/benchmark_fsdp2_vs_tp.py --capacity-targets-millions 1200 1300 1350 1400 1450 1500 1600 --throughput-target-millions 1200 --global-batch-size 2 --seq-len 128 --warmup-steps 5 --throughput-steps 20 --run-name kaggle_fsdp2_vs_tp
```

状态：`IMPLEMENTED / GPU-MEASURED ON 2 × T4 / RESULTS RECORDED`。
