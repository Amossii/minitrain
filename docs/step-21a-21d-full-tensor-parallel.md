# Step 21A–21D：Vocabulary Parallel 与 Full TP Benchmark

## 1. 本 Step 要解决的问题

原 Step 21 只切分 Transformer Block 内的 Linear。Token embedding、LM Head 和完整
`[B,T,V]` logits 仍在每张 GPU 上复制，使大词表的参数、梯度、AdamW state 和
activation 成为 TP 容量瓶颈。

本扩展依次完成：

- Step 21A：VocabParallelEmbedding。
- Step 21B：VocabParallelLMHead。
- Step 21C：VocabParallelCrossEntropy。
- Step 21D：FSDP2 vs Full TP 容量与同模型吞吐实验。

## 2. 为什么需要它

只切 Transformer Block 可以学习 Column/Row Parallel 的核心通信，但不能代表完整
LLM TP 的显存 ownership。若 LM Head 之后 AllGather 完整 logits，再调用普通
cross entropy，`[B,T,V]` activation 会重新出现在每张卡上，也会增加一次大通信。
因此 Full TP 必须让 loss 直接消费 local `[B,T,V/TP]` logits。

## 3. 核心原理与 shape

### Step 21A：VocabParallelEmbedding

```text
weight [V,C]
  rank 0: [0,V/2)   → [V/2,C]
  rank 1: [V/2,V)   → [V/2,C]

replicated input_ids [B,T]
  → owner rank local lookup，非 owner 输出 0
  → AllReduce(SUM)
  → replicated hidden [B,T,C]
```

输入是全局 token IDs；内部状态是本 rank 的 vocabulary rows；输出是完整 hidden。
下一次调用仍读取同一 shard，optimizer 只为本地 rows 保存 moments。

### Step 21B：VocabParallelLMHead

LM Head 等价于沿输出维切分的 ColumnParallelLinear：

```text
replicated hidden [B,T,C]
  × local weight [V/TP,C]
  → local logits [B,T,V/TP]
```

训练路径不执行 logits AllGather。只有 correctness helper 会临时 Gather 为 `[B,T,V]`
并与普通 Transformer 比较。

### Step 21C：VocabParallelCrossEntropy

每个 rank 持有 local logits。数值稳定的 global softmax 需要：

1. local max 后 AllReduce(MAX)，得到每个 token 的 global max。
2. 计算 `exp(local_logits - global_max)`。
3. local sum 后 AllReduce(SUM)，得到 global denominator。
4. 只有 target token owner 读取 local target logit，其他 rank 贡献 0。
5. AllReduce(SUM) 得到 global target logit。
6. `loss = log(global_exp_sum) - shifted_target_logit`，最后取 mean。

整个过程只持有 `[B,T,V/TP]` logits，不构造完整 vocabulary activation。

### Full TP 数据流

```text
input_ids [B,T]
  → VocabParallelEmbedding + AllReduce
hidden [B,T,C]
  → Block Column/Row Parallel
hidden [B,T,C]
  → VocabParallelLMHead
local_logits [B,T,V/TP]
  → VocabParallelCrossEntropy
replicated scalar loss
```

## 4. 修改文件

- `minitrain/tensor_parallel/vocab.py`：vocabulary partition、embedding、LM Head、loss。
- `minitrain/tensor_parallel/full_transformer.py`：Full TP 模型与 correctness logits Gather。
- `minitrain/training.py`：训练/计时函数支持显式 loss function。
- `scripts/train_full_tp.py`：Full TP 训练入口。
- `scripts/verify_full_tp_transformer.py`：端到端 distributed correctness。
- `scripts/benchmark_native_worker.py`：`full_tp` 吞吐 worker。
- `scripts/oom_boundary_worker.py`：`full_tp` 容量 worker。
- `scripts/benchmark_fsdp2_vs_tp.py`：Step 21D 编排入口。
- `tests/test_tp_transformer.py`：vocabulary partition CPU 测试。

旧 `TensorParallelTransformer` 和 `train_tp.py` 保留，继续作为 Block-only TP 学习与
历史实验入口。新 benchmark 使用独立策略名 `full_tp`，不能与旧 `tp` CSV 混合。

## 5. Correctness 验证

CPU/Gloo 双进程命令：

```bash
torchrun --standalone --nproc-per-node=2 scripts/verify_full_tp_transformer.py --backend gloo --batch-size 2 --seq-len 8
```

当前已验证输出：

```text
local_logits_shape=(2, 8, 32)
vocab_parallel_embedding=PASS
vocab_parallel_lm_head=PASS
vocab_parallel_cross_entropy=PASS
full_model_gradients=PASS
updated_parameters=PASS
FULL TP CORRECTNESS CHECKS PASSED
```

Kaggle NCCL 验证：

```bash
torchrun --standalone --nproc-per-node=2 scripts/verify_full_tp_transformer.py --backend nccl --batch-size 2 --seq-len 8
```

## 6. 训练命令

```bash
torchrun --standalone --nproc-per-node=2 scripts/train_full_tp.py --backend nccl --model tiny --batch-size 2 --seq-len 128 --steps 5 --seed 42
```

## 7. Step 21D：重新执行 FSDP2 vs Full TP

旧报告中的 TP 数据属于 Block-only TP，不能当作 Full TP 结果。必须使用新 run name：

```bash
python scripts/benchmark_fsdp2_vs_tp.py --capacity-targets-millions 1200 1300 1350 1400 1450 1500 1600 --throughput-target-millions 1200 --global-batch-size 2 --seq-len 128 --warmup-steps 5 --throughput-steps 20 --run-name kaggle_fsdp2_vs_full_tp
```

预期输出：

```text
results/raw/kaggle_fsdp2_vs_full_tp/manifest.json
results/tables/kaggle_fsdp2_vs_full_tp_capacity.csv
results/tables/kaggle_fsdp2_vs_full_tp_throughput.csv
```

脚本固定 2 × T4、FP32、AdamW、global batch 2、seq_len 128，并关闭 activation
checkpoint。FSDP2 每 rank batch=1；Full TP 两个 rank 共同处理 replicated batch=2。

## 8. 结果解释

尚未运行新的双 T4 实验，因此不能声称 Full TP 相比旧 TP 提升了多少容量或吞吐。
理论上 vocabulary 参数、gradient、AdamW moments 和 logits 被分片，应降低显存；
但 embedding forward 和 distributed cross entropy 增加 collectives，真实吞吐可能上升
也可能下降，必须以新 CSV 为准。

## 9. Checkpoint

- [x] Step 21A VocabParallelEmbedding。
- [x] Step 21B VocabParallelLMHead。
- [x] Step 21C VocabParallelCrossEntropy。
- [x] Full TP forward/gradient/update 两进程 Gloo correctness。
- [x] Step 21D benchmark 编排与独立结果路径。
- [ ] Kaggle 双 T4 NCCL correctness。
- [ ] Kaggle 双 T4 Full TP 容量与吞吐实测。

状态：`IMPLEMENTED / CPU-GLOO VERIFIED / NOT YET GPU-BENCHMARKED`。

推荐 commit message：

```text
feat: add vocabulary parallelism and full TP benchmark
```

## 10. 对实习和面试的价值

这组实现证明不仅理解 Linear weight sharding，还能处理 vocabulary ownership、
distributed softmax 的数值稳定性和 loss backward。面试时应能直接画出 local logits
shape，并解释为什么 AllReduce(MAX/SUM) 可以替代完整 logits AllGather。
