# Step 24：Distributed Checkpoint、测试整理与最终报告

## 1. 本 Step 要解决的问题

训练系统不能只会启动，还必须能在中断后恢复。最后一步使用 PyTorch Distributed
Checkpoint（DCP）保存模型、AdamW 和训练 step，并把 24 个 Step 的代码、测试、
真实 benchmark 与限制整理为可复查的项目交付物。

## 2. 为什么需要它

普通 `torch.save(model.state_dict())` 可以处理小型 DDP 模型，但让 rank 0 聚合大型
FSDP 参数会形成 CPU 内存和 I/O 瓶颈。DCP 看到的是一个逻辑 state dict，而每个
rank 直接读写自己持有的 shard；恢复时 planner 按当前布局放置 tensor。

只验证“文件存在”没有意义。真正的 resume correctness 要求：

1. 新模型恢复后与保存时的完整逻辑参数一致。
2. Adam 的一阶/二阶 moments 恢复。
3. 训练 step 恢复，使数据进度不会回退。
4. 恢复分支继续训练一步后，与未中断分支仍然逐参数一致。

## 3. 核心原理

```text
model + optimizer
       │ get_state_dict（DDP replica / FSDP2 DTensor shard）
       ▼
逻辑 state dict + step
       │ dcp.save：各 rank 写本地持有部分
       ▼
checkpoint directory
       │ dcp.load：按当前 layout 读取/重分片
       ▼
set_state_dict
       │
       ▼
恢复后的 model + optimizer + next_step
```

`step` 表示下一次要执行的训练步。模型和 optimizer 恢复但数据游标不恢复，会重复
消费旧 batch；本实验使用确定性 step→batch 映射验证这个状态转移。

## 4. 本 Step 修改的文件

- `minitrain/distributed/checkpointing.py`：DCP 保存/加载核心实现。
- `scripts/verify_distributed_checkpoint.py`：DDP/FSDP2 resume correctness。
- `tests/test_checkpointing.py`：单进程 CPU 精确恢复测试。
- `report/final_report.md`：基于仓库真实 CSV 的最终实验报告。
- `README.md`：项目入口、能力地图、结果与复现入口。

## 5. 实现代码与状态

`save_training_checkpoint` 的输入是模型、optimizer、目录和 next step；内部状态不
变化，输出是分布式 checkpoint。`load_training_checkpoint` 输入相同结构的新对象，
原地恢复参数与 optimizer state，输出 next step。下一次调用训练函数后，模型和
Adam moments 从保存点继续演进。

这里没有封装掉 `get_state_dict → dcp.save/load → set_state_dict`，目的是让 shard
checkpoint 的关键数据流保持可见。

## 6. 测试代码

CPU 单测先训练一步、保存、用不同随机种子创建新模型并恢复，然后让未中断与恢复
分支处理相同下一批数据。两条分支的 loss 和所有参数要求 `atol=0, rtol=0`。

分布式脚本对 DDP 和 FSDP2 执行相同协议。FSDP2 比较使用 full logical state，
不会把 rank-local DTensor shard 错当成完整模型。

## 7. Kaggle/Linux 执行命令

CPU 全量单元测试（单行）：

```bash
python -m unittest discover -s tests -v
```

双卡 DDP checkpoint correctness（单行）：

```bash
torchrun --standalone --nproc-per-node=2 scripts/verify_distributed_checkpoint.py --strategy ddp --backend nccl --model tiny --local-batch-size 1 --seq-len 16 --pre-steps 2 --checkpoint-dir /kaggle/working/minitrain_checkpoints/ddp_step2
```

双卡 FSDP2 shard checkpoint correctness（单行）：

```bash
torchrun --standalone --nproc-per-node=2 scripts/verify_distributed_checkpoint.py --strategy fsdp2 --backend nccl --model tiny --local-batch-size 1 --seq-len 16 --pre-steps 2 --checkpoint-dir /kaggle/working/minitrain_checkpoints/fsdp2_step2
```

重新运行最终统一 benchmark（单行）：

```bash
python scripts/benchmark_unified.py --model tiny --world-size 2 --global-batch-size 2 --seq-len 128 --learning-rate 0.0003 --warmup-steps 5 --steps 20 --seed 42 --run-name kaggle_unified_tiny
```

## 8. 预期输出

正确恢复会打印：

```text
model_state_after_load=PASS
optimizer_state_after_resumed_step=PASS
training_step_resume=PASS
DDP DISTRIBUTED CHECKPOINT PASSED
```

FSDP2 命令最后一行相应为 `FSDP2 DISTRIBUTED CHECKPOINT PASSED`。目录中通常包含
`.metadata` 和一个或多个分片文件；文件数量是存储实现细节，不应作为 correctness。

## 9. Correctness 验证

当前已验证：

- 单进程 CPU：模型、optimizer、step 精确恢复。
- 两进程 Gloo DDP：恢复后继续一步与未中断分支精确对齐。
- 两进程 Gloo FSDP2：DTensor shard 恢复后继续一步精确对齐。
- 全项目 CPU 单元测试。

Kaggle NCCL checkpoint 命令已提供，但必须在目标 GPU 环境再次执行，不能用 Gloo
通过代替 NCCL/I/O 实测。

## 10. Benchmark 与 profiler 验证

最终性能结论只来自 `results/` 中已有真实 CSV，见 `report/final_report.md`。仓库中
没有提交 profiler trace，因此最终报告不会声称已经从 trace 证明 overlap。若用于
面试展示，应补跑 Step 14/18 并保存 trace 截图或事件摘要。

## 11. 本 Step Checkpoint

- [x] DDP/FSDP2 distributed checkpoint 实现。
- [x] 模型、AdamW、next step 恢复。
- [x] 恢复后继续训练 correctness。
- [x] CPU 和两进程 Gloo 验证。
- [x] README 与真实结果最终报告。
- [x] 复现命令保持单行。
- [ ] Kaggle 双 GPU NCCL checkpoint 再验证。
- [ ] 保存 DDP/FSDP2 profiler trace 并完成事件级分析。

状态：`IMPLEMENTED / CPU & GLOO VERIFIED / NOT YET NCCL CHECKPOINT-VERIFIED`。

## 12. 推荐 Git commit message

```text
feat: add distributed checkpoint and final project report
```

## 13. 对实习和面试的价值

这一 Step 证明你理解训练状态不只有参数，还包括 optimizer moments 和数据进度；
也能解释为什么 sharded training 需要 sharded checkpoint。配合前 23 步，项目形成
了 correctness、benchmark、memory、communication、profiling 和容错恢复的完整
LLM training systems 叙事。
