# Step 21 — Transformer Tensor Parallel

## 1. 本 Step 要解决的问题

Step 19/20 分别验证了 Column 和 Row Linear。本 Step 将它们接入完整 decoder-only
Transformer，并证明 forward、全部梯度和一次 optimizer update 与普通模型数值一致。

## 2. Attention 数据流

```text
replicated [B,T,C]
  → Q/K/V Column Parallel
  → 每卡 [B,T,C/TP] = [B,T,H/TP,D]
  → 每卡独立计算本地 attention heads
  → local context [B,T,C/TP]
  → O Row Parallel
  → AllReduce
  → replicated [B,T,C]
```

attention head 之间彼此独立，所以 Q/K/V 后不需要 AllGather。

## 3. SwiGLU 数据流

```text
replicated [B,T,C]
  → Gate/Up Column Parallel
  → 两个 [B,T,I/TP] shard
  → 本地 silu(gate) * up
  → Down Row Parallel
  → AllReduce
  → replicated [B,T,C]
```

Column 输出直接交给 Row，因此中间没有 AllGather。每个 Block forward 主要产生两次
AllReduce：一次来自 attention output，一次来自 MLP down projection。

## 4. 当前模型边界

Embedding、RMSNorm、residual 和 LM Head 保持 replicated。本 Step 的目标是清楚验证
Block 内 Tensor Parallel，而不是一次性实现 vocabulary parallel、sequence parallel 或
TP+DP 多维并行。Kaggle 双卡只能真实运行 TP=2，不能假装验证 TP=2 + DP=2。

## 5. Correctness

验证脚本先从普通 `MiniTransformer` 加载完整参数：Column 取 weight 输出行，Row 取
weight 输入列，replicated 参数完整复制。随后验证：

1. 完整 logits 和 loss 对齐。
2. Gather 所有 TP shard 后，每个同名梯度与 reference 对齐。
3. SGD 更新后，Gather 的完整参数与 reference 对齐。

## 6. Kaggle/Linux 命令

双 GPU correctness（单行）：

```bash
torchrun --standalone --nproc-per-node=2 scripts/verify_tp_transformer.py --backend nccl --batch-size 2 --seq-len 8 --learning-rate 0.01 --seed 42
```

双 GPU 训练（单行）：

```bash
torchrun --standalone --nproc-per-node=2 scripts/train_tp.py --backend nccl --model tiny --batch-size 2 --seq-len 128 --steps 5 --seed 42
```

本机 CPU/Gloo correctness（单行）：

```bash
conda run -n sglang python -m torch.distributed.run --standalone --nproc-per-node=2 scripts/verify_tp_transformer.py --backend gloo --batch-size 2 --seq-len 8 --learning-rate 0.01 --seed 42
```

## 7. 预期输出

```text
forward=PASS
loss=PASS
full_model_gradients=PASS
updated_parameters=PASS
TRANSFORMER TP CORRECTNESS CHECKS PASSED
```

## 8. Checkpoint

- IMPLEMENTED：Attention TP、SwiGLU TP、完整模型训练和 shard Gather correctness。
- CPU/GLOO DISTRIBUTED VERIFIED：配置单测、完整 correctness 和训练 smoke test。
- NOT YET NCCL GPU-VERIFIED：需要在 Kaggle 双 GPU 复现 correctness 和训练。

## 9. 推荐 Git commit message

`feat: implement Transformer tensor parallelism`

## 10. 对实习/面试的价值

你能够逐 shape 解释 Q/K/V head 分片和 MLP intermediate 分片，指出 Column→Row 为什么
避免 AllGather，并用完整参数/梯度重建证明手写 TP 与普通 Transformer 等价。
