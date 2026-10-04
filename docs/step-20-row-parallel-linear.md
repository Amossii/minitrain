# Step 20 — RowParallelLinear

## 1. 本 Step 要解决的问题

Row Parallel 沿普通 `nn.Linear.weight=[out_features, in_features]` 的第 1 维，也就是
输入维切分。它通常直接消费 ColumnParallelLinear 未 Gather 的输出 shard，从而让两个
Linear 之间不需要额外通信。

两卡数据流：

```text
X0 [..., I/2]             X1 [..., I/2]
W0 [O, I/2]               W1 [O, I/2]
      |                          |
Y0 = X0 @ W0^T            Y1 = X1 @ W1^T
      \--------- AllReduce SUM --------/
                    Y [..., O]
                    + bias
```

## 2. 输入、状态、输出和下一次调用

- 输入：默认是当前 rank 的 `[..., I/TP]` shard。
- 内部状态：本地 `weight=[O, I/TP]`；bias 是 replicated `[O]`。
- 输出：AllReduce 后每个 rank 都持有完整 `[..., O]`。
- 下一次调用：optimizer 更新本地 weight；replicated bias 因各 rank 梯度相同而保持一致。

模块也支持 `input_is_parallel=False`：输入为完整 replicated tensor，forward 本地切片，
backward 用 AllGather 把各 rank 的局部输入梯度拼成完整 `dX`。

## 3. 为什么 bias 在 AllReduce 后添加

本地矩阵乘计算的是对完整输出的部分贡献：

```text
Y = sum(X_rank @ W_rank^T) + bias
```

如果每个 rank 在 AllReduce 前都添加 bias，两卡求和后会得到 `2 * bias`。因此本地
`F.linear` 必须使用 `bias=None`，通信完成后再添加一次 replicated bias。

## 4. 修改文件

- `minitrain/tensor_parallel/row_linear.py`：输入分片、本地 Linear、AllReduce 和 autograd。
- `scripts/verify_row_parallel.py`：与普通 Linear 的完整 forward/backward 对齐。
- `tests/test_row_parallel.py`：输入列分片范围测试。

## 5. Kaggle/Linux 执行命令

双 GPU NCCL correctness（单行）：

```bash
torchrun --standalone --nproc-per-node=2 scripts/verify_row_parallel.py --backend nccl --batch-size 2 --seq-len 3 --in-features 12 --out-features 8 --seed 42
```

本机双进程 CPU correctness（单行）：

```bash
conda run -n sglang python -m torch.distributed.run --standalone --nproc-per-node=2 scripts/verify_row_parallel.py --backend gloo --batch-size 2 --seq-len 3 --in-features 12 --out-features 8 --seed 42
```

完整 CPU 单测（单行）：

```bash
conda run -n sglang python -m unittest discover -s tests -v
```

## 6. Correctness 标准

必须验证：预分片输入输出、replicated 输入输出、完整输入梯度、本地 weight 梯度和完整
bias 梯度都与普通 `nn.Linear` 对齐。只检查最终 shape 或“程序没报错”不算完成。

## 7. 与 Column Parallel 的组合

```text
replicated X
  → ColumnParallelLinear(gather_output=False)
  → sharded hidden state
  → RowParallelLinear(input_is_parallel=True)
  → AllReduce
  → replicated Y
```

中间不需要 AllGather。这个 Column→Row 配对将在 Step 21 用于 Attention 和 SwiGLU。

## 8. Checkpoint

- IMPLEMENTED：输入列分片、本地 GEMM、输出 AllReduce、replicated bias。
- CPU-VERIFIED：分片范围单测。
- NOT YET DISTRIBUTED-VERIFIED：需要运行 Gloo 或 NCCL correctness。

## 9. 推荐 Git commit message

`feat: implement RowParallelLinear`

## 10. 对实习/面试的价值

你可以写出 Row Parallel 的矩阵分解，解释为什么输出必须 AllReduce、为什么 bias 只能
在通信后添加，以及 Column→Row 为什么能避免中间 AllGather。
