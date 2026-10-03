"""验证手写 ColumnParallelLinear 与 nn.Linear 的前向和反向一致。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.distributed as dist
from torch import nn

from minitrain.distributed.runtime import cleanup_distributed, init_distributed
from minitrain.tensor_parallel.column_linear import ColumnParallelLinear


# 输入指定一个小型 Linear workload；输出 PASS 标记，并验证本地 ownership、完整输出、
# 输入梯度和本地参数梯度。调用后 reference 和 TP 参数都保留一次 backward 的梯度。
def main() -> int:
    """用普通 nn.Linear 作为数值基准验证 Column Parallel。"""

    parser = argparse.ArgumentParser(description="Verify ColumnParallelLinear.")
    parser.add_argument("--backend", choices=("gloo", "nccl"), default="nccl")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=3)
    parser.add_argument("--in-features", type=int, default=8)
    parser.add_argument("--out-features", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if min(args.batch_size, args.seq_len, args.in_features, args.out_features) <= 0:
        parser.error("all tensor dimensions must be positive")

    context = None
    try:
        context = init_distributed(backend=args.backend)
        if args.out_features % context.world_size != 0:
            parser.error("--out-features must be divisible by world size")
        torch.manual_seed(args.seed)
        reference = nn.Linear(args.in_features, args.out_features).to(context.device)
        column = ColumnParallelLinear(
            args.in_features,
            args.out_features,
            bias=True,
            gather_output=False,
            device=context.device,
        )
        column.load_from_linear(reference)

        torch.manual_seed(args.seed + 1)
        source = torch.randn(
            args.batch_size,
            args.seq_len,
            args.in_features,
            device=context.device,
        )
        reference_input = source.detach().clone().requires_grad_(True)
        parallel_input = source.detach().clone().requires_grad_(True)

        reference_output = reference(reference_input)
        local_output = column(parallel_input, gather_output=False)
        expected_local = reference_output[
            ..., column.output_start : column.output_end
        ]
        torch.testing.assert_close(local_output, expected_local)
        print(
            f"rank={context.rank} weight_shape={tuple(column.weight.shape)} "
            f"local_output_shape={tuple(local_output.shape)} local_output=PASS",
            flush=True,
        )

        gathered_output = column(parallel_input, gather_output=True)
        torch.testing.assert_close(gathered_output, reference_output)
        if context.is_main_process:
            print("gathered_output=PASS", flush=True)

        # 两边使用同一个完整 loss。Gather 的 backward 先切分 dY，随后输入映射的
        # backward AllReduce 各列分片对 dX 的贡献，最终应与 reference 完全一致。
        reference_loss = reference_output.square().mean()
        parallel_loss = gathered_output.square().mean()
        reference_loss.backward()
        parallel_loss.backward()
        torch.testing.assert_close(parallel_input.grad, reference_input.grad)
        torch.testing.assert_close(
            column.weight.grad,
            reference.weight.grad[column.output_start : column.output_end],
        )
        assert column.bias is not None and reference.bias is not None
        torch.testing.assert_close(
            column.bias.grad,
            reference.bias.grad[column.output_start : column.output_end],
        )
        if context.is_main_process:
            print("input_gradient=PASS", flush=True)
            print("local_parameter_gradients=PASS", flush=True)
            print("COLUMN PARALLEL CORRECTNESS CHECKS PASSED", flush=True)
        dist.barrier()
    finally:
        if context is not None:
            cleanup_distributed()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
