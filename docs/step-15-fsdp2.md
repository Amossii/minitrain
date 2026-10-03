# Step 15 — FSDP2 Training and Correctness

## 1. 本 Step 要解决的问题

DDP 的每个 rank 都长期保存完整参数、完整梯度和完整 optimizer state。FSDP2 将三者按
data-parallel rank 分片，以通信换显存。本 Step 接入原生 `fully_shard`，并证明分片训练
的一次更新与普通 global-batch 训练数值一致。

## 2. 核心状态变化

静止状态下，每个参数是 `DTensor(..., placements=(Shard(dim=0),))`，每张 GPU 只持有
dim 0 的一部分。一次调用的状态变化是：

```text
local parameter shard
  -> pre-forward AllGather -> full parameter
  -> forward
  -> reshard local parameter
  -> pre-backward AllGather -> full parameter
  -> backward
  -> ReduceScatter gradient -> local gradient shard
  -> optimizer updates local parameter/optimizer shards
```

输入是每个 rank 不同的 `[local_batch_size, seq_len]` token；持久内部状态是参数、梯度和
optimizer shard；forward 输出仍是本 rank 的 `[local_batch_size, seq_len, vocab_size]`
logits。下一次调用从已更新的 local shard 开始。

## 3. 为什么 bottom-up `fully_shard`

代码先对每个 Transformer Block 调用 `fully_shard`，最后对根模型调用。每个 Block
因此形成自己的通信组，根组只接管 Embedding、FinalNorm 和 LM Head。逐层 group 才能
及时释放 full parameter，并让后续实验观察通信计算重叠。只 shard 根模型虽然能运行，
但会形成过大的 group，降低峰值显存收益和 overlap 空间。

## 4. 修改文件

- `minitrain/distributed/fsdp2.py`：bottom-up wrapping 和 shard metadata。
- `scripts/train_fsdp2.py`：FSDP2 训练入口。
- `scripts/verify_fsdp2_correctness.py`：分片与 global-batch 数值验证。
- `tests/test_fsdp2.py`：CPU 可运行的 workload 测试。

## 5. Kaggle/Linux 执行命令

运行 correctness（单行）：

```bash
torchrun --standalone --nproc-per-node=2 scripts/verify_fsdp2_correctness.py --backend nccl --local-batch-size 2 --seq-len 16 --learning-rate 0.01 --seed 42
```

运行 FSDP2 训练（单行）：

```bash
torchrun --standalone --nproc-per-node=2 scripts/train_fsdp2.py --backend nccl --model tiny --local-batch-size 2 --seq-len 128 --steps 5 --seed 42
```

运行 CPU 单元测试（单行）：

```bash
conda run -n sglang python -m unittest discover -s tests -v
```

## 6. 预期输出与 correctness

correctness 必须依次打印：

```text
parameter_sharding=PASS
gradient_sharding=PASS
global_batch_reference=PASS ...
FSDP2 CORRECTNESS CHECKS PASSED
```

验证不是“没有报错”，而是检查：每个参数/梯度都是 `Shard(0)` DTensor；各 rank local
numel 求和等于 global numel；完整更新后 state dict 与相同 global batch 的普通模型在
浮点容差内一致。

## 7. 尚未在本 Step 声称的结论

本 Step 不比较 DDP/FSDP2 显存，也不声称 FSDP2 更快。显存公平 benchmark 属于 Step
16；最大可训练模型属于 Step 17；AllGather/ReduceScatter timeline 属于 Step 18。

## 8. Checkpoint

- IMPLEMENTED：FSDP2 training、DTensor shard inspection、global reference check。
- CPU-VERIFIED：配置和模型 workload 单测。
- NOT YET GPU-VERIFIED：双卡 NCCL 的真实分片更新需要在 Kaggle 执行。

## 9. 推荐 Git commit message

`feat: add FSDP2 training and correctness verification`

## 10. 对实习/面试的价值

你能明确解释 FSDP2 分片了参数、梯度和 optimizer state，指出 forward/backward 前的
AllGather 与 backward 后的 ReduceScatter，并通过 DTensor ownership 和完整参数更新
对齐来证明实现正确，而不只是会调用一个框架 API。
