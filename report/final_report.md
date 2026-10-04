# MiniTrain 最终实验报告

## 项目结论

MiniTrain 从手写 Decoder-only Transformer 出发，在单机双 GPU 上实现并验证了
collectives、DDP、FSDP2、手写 Tensor Parallel、DeepSpeed ZeRO-1/2/3、统一
benchmark 和 Distributed Checkpoint。核心目标是让通信和状态 ownership 可见，
而不是用上层 Trainer 隐藏训练生命周期。

本文中的性能数字均来自仓库已提交 CSV。统一实验配置为 tiny 模型（9,077,888
参数）、FP32 AdamW、sequence length 128、global batch size 2、5 个 warmup steps
和 20 个 measured steps。OOM 实验使用 2×T4、FP32 AdamW、local batch size 1。

## 统一吞吐实验

| Strategy | Median step time (ms) | Tokens/s | Peak allocated (MiB) |
|---|---:|---:|---:|
| Single | 14.39 | 17,786.79 | 264.09 |
| DDP | 21.09 | 12,141.45 | 245.40 |
| FSDP2 | 31.31 | 8,175.62 | 172.08 |
| ZeRO-1 | 32.70 | 7,828.52 | 2,047.33 |
| ZeRO-2 | 34.46 | 7,428.68 | 2,046.52 |
| ZeRO-3 | 66.26 | 3,863.72 | 2,106.58 |

这个 tiny/global-batch-2 workload 中，双卡策略都没有超过单卡。DDP step time 是
Single 的 1.47 倍；FSDP2 是 2.18 倍；ZeRO-3 是 4.60 倍。这是合理的失败实验：
单卡计算量太小，collective、框架调度和分片管理开销无法被矩阵计算摊薄。

ZeRO 的 allocator 峰值约 2 GiB，明显高于原生 PyTorch tiny workload。该结果只能
说明当前 DeepSpeed 配置/运行时的实测进程内 CUDA allocator 峰值，不能推出 ZeRO
对大模型不省显存。CUDA context、通信 bucket、临时 gather buffer 和框架预分配
在小模型上占主导；应该结合更大模型 sweep 后再评价状态分片收益。

## Tensor Parallel

TP=2 的 median step time 为 21.47 ms、吞吐为 11,924.77 tokens/s，相比 Single
慢 1.49 倍。当前实现每个 Block 的 attention output 和 MLP down projection 都要
AllReduce，同时 embedding、norm 和 LM head 仍 replicated。tiny 模型的本地 GEMM
变得更小，通信延迟却没有消失，因此没有加速。

TP 的 global batch 没有乘 world size：两个 rank 共同处理同一批样本。这一口径
与数据并行不同，所以项目把 TP 与 DP 主表分开，避免虚报吞吐。

## DDP scaling

固定 global batch=2 的 strong scaling 中，world-size 2 相对单卡 speedup 为 0.757，
效率为 37.9%。固定 local batch=1 的 weak scaling中，双卡吞吐相对单卡增长 1.470
倍，效率为 73.5%。这再次说明小 workload 的通信/启动开销显著，但增加总工作量后
两张卡仍提供了部分吞吐增益。

## FSDP2 memory 与容量边界

tiny workload 的独立 memory benchmark 中：

- DDP peak allocated：255,882,752 bytes。
- FSDP2 peak allocated：180,443,648 bytes。
- FSDP2 实测下降 29.48%。

峰值没有按 world size 理想减半，因为 activation、临时 AllGather buffer、allocator
碎片和未分片状态仍存在。OOM sweep 给出的真实边界是：

- DDP 最大 PASS 约 601M，650M 明确 OOM。
- FSDP2 最大 PASS 约 1.448B，1.502B 明确 OOM。
- 按最大 PASS 计算，FSDP2 可训练模型容量约为 DDP 的 2.41 倍。

这里报告区间而不是声称精确极限；模型宽度是离散候选，边界还会随 allocator 状态、
PyTorch 版本和 GPU 上其他进程变化。

## Correctness 证据

- Collectives 使用可人工计算小 tensor 验证 Broadcast、Reduce、AllReduce、
  AllGather、ReduceScatter 和 Barrier。
- DDP 比较跨 rank 梯度/参数，并与等价单进程 global-batch 更新对齐。
- FSDP2 比较 shard ownership，并与未分片 reference 更新对齐。
- Column/Row Parallel Linear 与 `nn.Linear` 输出、梯度数值对齐。
- TP Transformer 与普通 Transformer 的 forward、gradient 和 update 对齐。
- ZeRO 检查参数实际更新，并可验证完整参数的跨 rank 一致性。
- DCP 检查模型、Adam moments、next step，并验证恢复后继续训练不分叉。

## 通信与状态 ownership

| Strategy | 常驻模型状态 | 关键通信 |
|---|---|---|
| DDP | 每 rank 完整参数/梯度/optimizer | backward gradient AllReduce |
| FSDP2 | 参数、梯度、optimizer 分片 | parameter AllGather + gradient ReduceScatter |
| TP | Linear 权重按输入/输出维切分 | Row output / Column input-gradient AllReduce |
| ZeRO-1 | optimizer 分片 | gradient sync + updated parameter exchange |
| ZeRO-2 | optimizer、gradient 分片 | ReduceScatter + parameter exchange |
| ZeRO-3 | optimizer、gradient、parameter 分片 | parameter AllGather + ReduceScatter |

## 限制与下一轮实验

1. 统一表只有 tiny、FP32、global batch 2，不能外推到计算密集的大模型。
2. 已有 CSV 没有置信区间，后续应重复独立进程运行并报告分布。
3. 仓库未提交 profiler trace，当前不能用事件级证据声称通信计算 overlap。
4. DCP 已通过 CPU/Gloo，仍需在 Kaggle NCCL 上验证实际 GPU shard I/O。
5. ZeRO 小模型显存受固定框架开销主导，应增加模型规模 sweep。

这些限制不是隐藏的缺陷，而是下一轮工程问题：扩大计算/通信比、保存 profiler
证据、重复测量，并验证 checkpoint 跨 world-size reshard。

## 简历描述建议

> Built MiniTrain, a PyTorch distributed LLM training lab implementing DDP,
> FSDP2, 2-way tensor parallelism, DeepSpeed ZeRO-1/2/3, sharded checkpoints,
> correctness tests, and reproducible CUDA/NCCL benchmarks; measured a 2.41×
> trainable-model capacity increase from FSDP2 over DDP on 2×T4 GPUs.

面试时应同时说明 tiny 模型的多卡吞吐回归，并解释它来自低计算通信比。能够解释
失败实验，通常比只展示一个脱离 workload 的“加速百分比”更可信。
