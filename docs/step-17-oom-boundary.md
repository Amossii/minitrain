# Step 17 — Maximum Trainable Model / OOM Boundary

## 1. 本 Step 要解决的问题

Step 16 比较固定模型的显存。本 Step 沿同一个 Transformer 架构族逐渐增加 hidden size，
实测 DDP 和 FSDP2 能完成完整 AdamW 更新的最大模型规模，并记录首次 CUDA OOM。

## 2. 为什么不能只计算理论值

`16P/world_size` 只描述理想 FP32 model state。真正 OOM 还受 activation、临时
AllGather 参数、CUDA/NCCL buffer、allocator 碎片和模型初始化方式影响。因此边界必须
通过真实 forward、backward、optimizer step 得到。

## 3. 实验设计

目标参数量通过解析参数公式反解 hidden size。vocab、层数、head 数、MLP ratio、
seq_len、local/global batch、FP32 和 AdamW 均保持不变。每个“策略 × 参数规模”在新的
`torchrun` 作业中运行，避免上一个 OOM 污染后续实验。

FSDP2 从 CPU 模型直接按照 CUDA DeviceMesh 分片，避免先在每张 GPU 建立完整模型；
DDP 必须在每张 GPU 保存完整模型副本。峰值覆盖模型放置和完整训练更新。

## 4. 错误分类原则

只有日志明确包含 CUDA OOM 特征才记录 `OOM`。NCCL、代码、shape 或数据错误会立即
终止 orchestrator，并保留日志，绝不把普通失败伪装成显存边界。每完成一个点就更新
CSV，因此后续运行意外中断时，已经完成的真实结果仍然保留。

## 5. 修改文件

- `minitrain/distributed/oom_boundary.py`：参数公式、架构生成、OOM 分类、边界汇总。
- `scripts/oom_boundary_worker.py`：单个策略/规模的隔离训练探测。
- `scripts/benchmark_oom_boundary.py`：完整 sweep 与日志、CSV 管理。
- `tests/test_oom_boundary.py`：公式、配置、错误分类和最大 PASS 测试。

## 6. Kaggle/Linux 执行命令

双 GPU 正式 sweep（单行）：

```bash
python scripts/benchmark_oom_boundary.py --strategies ddp fsdp2 --targets-millions 100 200 300 400 --world-size 2 --local-batch-size 1 --seq-len 128 --steps 1 --seed 42 --run-name kaggle_oom_boundary
```

查看结果（单行）：

```bash
python -c "import pandas as pd; print(pd.read_csv('results/tables/kaggle_oom_boundary.csv').to_string(index=False))"
```

运行 CPU 测试（单行）：

```bash
conda run -n sglang python -m unittest discover -s tests -v
```

## 7. 如何判定边界

CSV 中某策略最大 `PASS` 是“当前已验证最大规模”，最小 `OOM` 是“当前已验证失败上界”。
如果 400M 仍 PASS，只能说边界大于等于 400M，不能声称 400M 就是最大值。需要继续加入
更大目标，再在 PASS/OOM 区间内缩小步长。

目标规模和实际规模不会完全相等，应使用 `actual_parameters` 写报告。`peak_memory_*`
在 OOM 行为 0，表示没有可信的完整训练峰值，而不是 OOM 时显存为零；诊断证据在对应
log 文件中。

## 8. Checkpoint

- IMPLEMENTED：架构规模生成、隔离 sweep、严格 OOM 分类、增量 CSV 和日志。
- CPU-VERIFIED：参数公式、目标逼近、错误分类和最大 PASS 计算。
- NOT YET GPU-VERIFIED：DDP/FSDP2 实际边界必须在 Kaggle 双 GPU 测量。

## 9. 推荐 Git commit message

`perf: add DDP and FSDP2 OOM boundary benchmark`

## 10. 对实习/面试的价值

你可以说明“理论分片收益”和“真实最大可训练模型”为什么不同，并展示可复现的边界
搜索、错误分类和失败日志，而不是只报告一次偶然 OOM。
