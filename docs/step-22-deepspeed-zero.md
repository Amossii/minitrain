# Step 22 — DeepSpeed ZeRO-1 / ZeRO-2 / ZeRO-3

## 1. 本 Step 要解决的问题

用同一个 MiniTransformer、Synthetic Dataset、FP32、AdamW、local/global batch 和训练
步数接入 DeepSpeed ZeRO，并明确三个 stage 分片的状态与通信代价。本项目将 DeepSpeed
作为对照框架，不用它隐藏前面已经手写过的分布式原理。

## 2. 三个 Stage 分片什么

| 策略 | Parameters | Gradients | Adam states |
|---|---|---|---|
| DDP | replicated | replicated | replicated |
| ZeRO-1 | replicated | replicated | sharded |
| ZeRO-2 | replicated | sharded | sharded |
| ZeRO-3 | sharded | sharded | sharded |

对 `P` 个 FP32 参数，AdamW 使用两份 moment，双卡理想持久状态为：

```text
DDP    = 16P bytes/rank
ZeRO-1 = 12P bytes/rank
ZeRO-2 = 10P bytes/rank
ZeRO-3 =  8P bytes/rank
```

这些值不包括 activation、通信临时 buffer、CUDA context 和 allocator 碎片。

## 3. 通信变化

- ZeRO-1：梯度仍像 DDP 一样同步，optimizer step 只更新本 rank ownership，随后同步参数。
- ZeRO-2：梯度通过 ReduceScatter 直接留下 shard，减少 gradient 常驻显存。
- ZeRO-3：forward/backward 前按需 AllGather 参数，backward 后 ReduceScatter 梯度；模式
  与 FSDP2 相近，但 runtime、分组和调度实现不同。

## 4. Correctness

训练脚本验证 loss 始终有限、完整参数签名在 optimizer step 后发生变化，并可用
`--verify-parameters` 检查各 rank 表示同一个更新后模型。ZeRO-3 不能直接比较本地
parameter shard，必须进入 `GatheredParameters` context 临时重建完整参数。

## 5. 安装与运行命令

安装 DeepSpeed（单行）：

```bash
pip install deepspeed
```

ZeRO-1 correctness（单行）：

```bash
torchrun --standalone --nproc-per-node=2 scripts/train_deepspeed.py --zero-stage 1 --model tiny --local-batch-size 2 --seq-len 128 --steps 3 --seed 42 --verify-parameters
```

ZeRO-2 correctness（单行）：

```bash
torchrun --standalone --nproc-per-node=2 scripts/train_deepspeed.py --zero-stage 2 --model tiny --local-batch-size 2 --seq-len 128 --steps 3 --seed 42 --verify-parameters
```

ZeRO-3 correctness（单行）：

```bash
torchrun --standalone --nproc-per-node=2 scripts/train_deepspeed.py --zero-stage 3 --model tiny --local-batch-size 2 --seq-len 128 --steps 3 --seed 42 --verify-parameters
```

CPU 单元测试（单行）：

```bash
conda run -n sglang python -m unittest discover -s tests -v
```

## 6. 预期输出

每个 stage 必须看到有限 `mean_loss`，最后看到：

```text
parameter_update=PASS
full_parameter_sync=PASS
ZERO-N CORRECTNESS CHECKS PASSED
```

## 7. 当前验证状态

- IMPLEMENTED：动态 ZeRO 配置、统一训练、状态理论模型、参数更新与同步检查。
- CPU-VERIFIED：stage ownership、显存公式和 batch/config 单测。
- NOT YET GPU/DEEPSPEED-VERIFIED：当前 Conda 环境未安装 DeepSpeed，需在 Kaggle 运行。

## 8. 推荐 Git commit message

`feat: add DeepSpeed ZeRO training`

## 9. 对实习/面试的价值

你不仅能说 ZeRO-1/2/3 “越来越省显存”，还能准确说明每一级新增分片的状态、理论显存
公式、通信模式，以及为什么 ZeRO-3 correctness 必须先 Gather 完整参数。
