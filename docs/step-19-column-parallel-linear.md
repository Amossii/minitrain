# Step 19 — ColumnParallelLinear

## 1. 本 Step 要解决的问题

普通 Linear 使用 `weight=[out_features, in_features]`。Column Parallel 沿第 0 维，
也就是输出维切分 weight；每个 TP rank 只保存一部分输出行，从而减少单卡参数量和本地
矩阵乘规模。

两卡情况下：

```text
replicated X [..., in_features]
              |                 |
rank 0: W[0:O/2, :]    rank 1: W[O/2:O, :]
              |                 |
local Y0 [..., O/2]    local Y1 [..., O/2]
              \-------AllGather-------/
                     Y [..., O]
```

## 2. 输入、状态、输出和下一次调用

- 输入：所有 rank 相同的 replicated `X=[..., in_features]`。
- 内部状态：本 rank 的 `weight=[O/TP, I]` 和 `bias=[O/TP]`。
- 默认输出：本地 `Y_rank=[..., O/TP]`，不通信。
- 可选输出：`gather_output=True` 时执行 AllGather，得到完整 `[..., O]`。
- 下一次调用：参数 ownership 不变；optimizer 只更新当前 rank 的本地参数。

## 3. 反向为什么需要 AllReduce

每张卡只能计算自己的输入梯度贡献：

```text
dX_rank = dY_rank @ W_rank
dX = sum(dX_rank)
```

因此手写 autograd mapping 在 forward 对输入保持 identity，在 backward 对 `dX_rank`
执行 AllReduce。AllGather 的 backward 不需要再次通信，只需从完整 `dY` 中取回本 rank
对应的输出梯度切片。

## 4. 修改文件

- `minitrain/tensor_parallel/column_linear.py`：参数分片、本地 Linear 和 autograd collective。
- `scripts/verify_column_parallel.py`：与 `nn.Linear` 的完整前向/反向数值对齐。
- `tests/test_column_parallel.py`：无需分布式环境的分片范围测试。

## 5. Kaggle/Linux 执行命令

双 GPU NCCL correctness（单行）：

```bash
torchrun --standalone --nproc-per-node=2 scripts/verify_column_parallel.py --backend nccl --batch-size 2 --seq-len 3 --in-features 8 --out-features 12 --seed 42
```

本机双进程 CPU correctness（单行）：

```bash
conda run -n sglang python -m torch.distributed.run --standalone --nproc-per-node=2 scripts/verify_column_parallel.py --backend gloo --batch-size 2 --seq-len 3 --in-features 8 --out-features 12 --seed 42
```

完整 CPU 单测（单行）：

```bash
conda run -n sglang python -m unittest discover -s tests -v
```

## 6. Correctness 标准

必须同时看到：

```text
local_output=PASS
gathered_output=PASS
input_gradient=PASS
local_parameter_gradients=PASS
COLUMN PARALLEL CORRECTNESS CHECKS PASSED
```

验证内容包括：本地输出等于普通 Linear 对应输出切片；AllGather 后等于完整输出；输入
梯度等于 reference 完整 `dX`；每个 rank 的 weight/bias gradient 等于 reference 对应
参数切片。仅仅 shape 正确或程序不报错不算完成。

## 7. 通信与工程取舍

如果下一层能够直接消费输出 shard（例如后续 RowParallelLinear），就应保持
`gather_output=False`，避免一次 AllGather。只有调用者确实需要完整输出时才 Gather。

当前初始化会临时生成完整权重以保证各 rank 的 shard 共同构成标准 Linear 初始化，
但持久状态只有本地 shard。大规模生产系统通常在 CPU/checkpoint 侧直接切片，避免临时
完整 GPU 权重；本项目当前优先让初始化和 correctness 清晰可验证。

## 8. Checkpoint

- IMPLEMENTED：本地参数 shard、本地 GEMM、可选 AllGather、反向 dX AllReduce。
- CPU-VERIFIED：分片范围单测。
- NOT YET DISTRIBUTED-VERIFIED：需要运行 Gloo 或 NCCL correctness。

## 9. 推荐 Git commit message

`feat: implement ColumnParallelLinear`

## 10. 对实习/面试的价值

你可以准确写出 weight/input/output shape，说明 forward 何时需要 AllGather，并解释
为什么 Column Parallel 的输入梯度必须汇总所有输出分片的贡献。
