# Step 14 — DDP Profiling

## 1. 本 Step 要解决的问题

Step 13 告诉我们双卡训练快了多少，Step 14 解释为什么：梯度何时触发
AllReduce、通信是否与剩余 backward 重叠，以及 bucket 大小如何影响时间线。

## 2. 为什么需要它

一次 DDP step 变慢可能来自算子、数据传输、通信或同步等待。只有总耗时无法区分
这些原因。Profiler 把 CPU launch、CUDA kernel、NCCL collective 和人工标注区间放到
同一条时间线上，让性能结论具有可验证证据。

## 3. 核心原理

Autograd 按反向依赖顺序产生梯度。某个 gradient bucket 的梯度全部 ready 后，DDP
reducer 就可以启动该 bucket 的 AllReduce，而后面的 backward 计算仍可能继续。因此，
时间线上可能看到 NCCL AllReduce 与 backward CUDA kernels 重叠。

`bucket_cap_mb` 是 bucket 的目标上限，不保证所有 bucket 都等大。参数大小、梯度 ready
顺序和 DDP bucket rebuild 都会改变实际划分，因此必须同时查看 trace 和保存的 DDP
logging data，不能只根据命令行参数推断。

## 4. 修改文件

- `minitrain/distributed/profiling.py`：活动选择与 rank-local 输出路径。
- `scripts/profile_ddp.py`：warmup、profile、trace 与 reducer 元数据导出。
- `tests/test_ddp_profiling.py`：无需 GPU 的辅助逻辑测试。
- `README.md`：增加 Step 14 入口。

## 5. Kaggle/Linux 执行命令

正式双 GPU NCCL profile（单行）：

```bash
torchrun --standalone --nproc-per-node=2 scripts/profile_ddp.py --backend nccl --model tiny --local-batch-size 2 --seq-len 128 --warmup-steps 3 --profile-steps 3 --bucket-cap-mb 1 --output-dir results/raw/profiles/ddp_tiny
```

查看输出文件（单行）：

```bash
ls -lh results/raw/profiles/ddp_tiny
```

运行 CPU 单元测试（单行）：

```bash
conda run -n sglang python -m unittest discover -s tests -v
```

## 6. 预期输出

每个 rank 分别生成 `ddp_rankN.json`、`ddp_rankN_operators.txt` 和
`ddp_rankN_ddp_logging.json`。JSON trace 可以加载到 Perfetto UI；operator table 用于
快速排序热点；logging 文件记录 reducer 观察到的 bucket 信息。

## 7. Correctness 验证

本 Step 不用 profile 区间中的额外 collective 检查 loss，因为那会人为污染通信轨迹。
DDP 数值正确性已由 Step 12 独立验证。本 Step 的 correctness 是：两个 rank 都完成相同
数量的更新、各自产生互不覆盖的 trace，并且 trace 包含人工标注的 forward、backward、
optimizer 区间。

## 8. Profiler 验证问题

在 trace 中依次回答：

1. `minitrain/backward` 区间内能否找到 NCCL AllReduce？
2. AllReduce 是否在 backward 完全结束前启动？
3. NCCL kernel 与计算 kernel 是否在不同 CUDA stream 上发生时间重叠？
4. `ddp_logging.json` 中实际 bucket 大小是什么？
5. 将 `--bucket-cap-mb` 改为 25 后，collective 数量、启动时机和总 step 时间如何变化？

只有时间区间真正相交才能称为 overlap；相邻但不相交不算。第一次运行结果也不能直接
推广为性能结论，应重复实验并固定模型、batch、序列长度和 profiler 参数。

## 9. 本 Step Checkpoint

- IMPLEMENTED：per-rank trace、operator table、DDP logging 导出。
- CPU-VERIFIED：路径、活动选择与通信事件过滤测试。
- NOT YET GPU-VERIFIED：NCCL kernel、真实 overlap 和 bucket 行为必须在双 GPU 运行确认。

## 10. 推荐 Git commit message

`perf: add DDP communication profiling`

## 11. 对实习/面试的价值

你不仅能说“DDP 会 AllReduce”，还可以用 trace 指出 collective 的启动位置，解释
gradient bucket 为什么让通信提前发生，并用实验讨论 bucket 粒度与 overlap 的权衡。
