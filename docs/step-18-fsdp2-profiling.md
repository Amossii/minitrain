# Step 18 — FSDP2 Profiling

## 1. 本 Step 要解决的问题

Step 16/17 说明 FSDP2 能节省多少显存和扩大多少模型边界。本 Step 从时间线解释代价：
参数何时 AllGather、梯度何时 ReduceScatter、通信是否与相邻层计算重叠，以及为什么
FSDP2 即使省显存也可能比 DDP 慢。

## 2. 一层 FSDP2 的状态变化

```text
local parameter shard
  -> AllGather full parameter
  -> layer forward compute
  -> reshard/free full parameter
  -> backward 前再次 AllGather
  -> layer backward compute
  -> ReduceScatter full gradient
  -> local gradient shard
```

DDP 长期保留完整参数，backward 主要执行 gradient AllReduce；FSDP2 额外需要参数
AllGather，并用 ReduceScatter 直接产出本地 gradient shard。这是显存与通信的交换。

## 3. 为什么需要逐层 bottom-up group

Step 15 已将每个 Transformer Block 建成独立 FSDP group。这样下一层计算时，可以预取
相邻层参数；上一层 backward 结束后也可以尽快 ReduceScatter 并释放 full parameter。
如果整个模型只有一个 group，通信粒度过大，峰值显存和 overlap 空间都会变差。

## 4. 修改文件

- `minitrain/distributed/fsdp2_profiling.py`：rank-local profiler 输出路径。
- `minitrain/distributed/profiling.py`：识别 AllGather/ReduceScatter 事件。
- `scripts/profile_fsdp2.py`：FSDP2 warmup、profile 与四类产物导出。
- `tests/test_fsdp2_profiling.py`：路径与通信事件筛选测试。

## 5. Kaggle/Linux 执行命令

双 GPU NCCL profile（单行）：

```bash
torchrun --standalone --nproc-per-node=2 scripts/profile_fsdp2.py --backend nccl --model tiny --local-batch-size 2 --seq-len 128 --warmup-steps 3 --profile-steps 3 --seed 42 --output-dir results/raw/profiles/fsdp2_tiny
```

查看输出（单行）：

```bash
ls -lh results/raw/profiles/fsdp2_tiny
```

查看自动识别的通信事件（单行）：

```bash
python -c "import json; print(json.dumps(json.load(open('results/raw/profiles/fsdp2_tiny/fsdp2_rank0_communications.json')), indent=2))"
```

运行 CPU 测试（单行）：

```bash
conda run -n sglang python -m unittest discover -s tests -v
```

## 6. 输出文件

每个 rank 生成：

- `fsdp2_rankN.json`：可加载到 Perfetto 的完整 trace。
- `fsdp2_rankN_operators.txt`：按 CUDA/CPU self time 排序的算子表。
- `fsdp2_rankN_shards.json`：每个参数的 global/local shape 和 placement。
- `fsdp2_rankN_communications.json`：自动筛出的通信事件名称索引。

## 7. Trace 必须回答的问题

1. forward 每个 Block 前能否找到 AllGather？
2. `reshard_after_forward` 后，backward 前是否再次 AllGather？
3. 每层梯度完成后能否找到 ReduceScatter？
4. collective 与 GEMM/attention kernel 是否在不同 CUDA stream 上时间重叠？
5. 是否存在 GPU 空洞，即计算和通信都没有运行的等待区间？
6. 与 Step 14 DDP trace 相比，collective 类型和次数发生了什么变化？

只有 CUDA 时间区间真正相交才算 overlap。CPU launch 重叠、事件相邻或者不同 step 的
事件不能作为通信计算重叠的证据。

## 8. 为什么 FSDP2 可能更慢

FSDP2 在 forward 和 backward 都需要参数 AllGather，通信次数通常多于 DDP。小模型、
小 batch 或慢互联下，通信启动与同步开销可能大于分片收益。FSDP2 首要目标是降低
model-state 显存，并不保证吞吐一定高于 DDP。

## 9. Checkpoint

- IMPLEMENTED：rank-local trace、算子表、shard metadata、通信事件索引。
- CPU-VERIFIED：输出路径和事件筛选单测。
- NOT YET GPU-VERIFIED：真实 NCCL AllGather/ReduceScatter 和 overlap 需在 Kaggle 验证。

## 10. 推荐 Git commit message

`perf: add FSDP2 communication profiling`

## 11. 对实习/面试的价值

你能够从 trace 解释 FSDP2 的显存收益来自哪里、额外通信发生在哪里，并以实际 CUDA
时间线讨论通信计算 overlap，而不是只背诵 AllGather/ReduceScatter 名称。
