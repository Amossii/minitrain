"""验证手写 RowParallelLinear 与 nn.Linear 的前向和反向一致。"""

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
from minitrain.tensor_parallel.row_linear import RowParallelLinear


# 输入指定一个小型 Linear workload；输出验证两种输入布局、完整输出、完整输入梯度、
# 本地 weight 梯度和 replicated bias 梯度，所有比较均以 nn.Linear 为基准。
def main() -> int:
    """用普通 nn.Linear 作为数值基准验证 Row Parallel。"""

    parser = argparse.ArgumentParser(description="Verify RowParallelLinear.")
    parser.add_argument("--backend", choices=("gloo", "nccl"), default="nccl")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=3)
    parser.add_argument("--in-features", type=int, default=12)
    parser.add_argument("--out-features", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if min(args.batch_size, args.seq_len, args.in_features, args.out_features) <= 0:
        parser.error("all tensor dimensions must be positive")

    context = None
    try:
        context = init_distributed(backend=args.backend)
        if args.in_features % context.world_size != 0:
            parser.error("--in-features must be divisible by world size")
        torch.manual_seed(args.seed)
        reference = nn.Linear(args.in_features, args.out_features).to(context.device)
        row = RowParallelLinear(
            args.in_features,
            args.out_features,
            bias=True,
            input_is_parallel=True,
            device=context.device,
        )
        row.load_from_linear(reference)

        torch.manual_seed(args.seed + 1)
        source = torch.randn(
            args.batch_size,
            args.seq_len,
            args.in_features,
            device=context.device,
        )
        local_input = source[..., row.input_start : row.input_end].contiguous()
        reference_output = reference(source)
        sharded_input_output = row(local_input, input_is_parallel=True)
        torch.testing.assert_close(sharded_input_output, reference_output)
        print(
            f"rank={context.rank} weight_shape={tuple(row.weight.shape)} "
            f"local_input_shape={tuple(local_input.shape)} sharded_input_output=PASS",
            flush=True,
        )

        # replicated 输入路径不仅验证 forward 的本地切片，也验证 backward 中将各 rank
        # 的局部 dX 通过 AllGather 恢复为完整输入梯度。
        reference_input = source.detach().clone().requires_grad_(True)
        parallel_input = source.detach().clone().requires_grad_(True)
        reference_output = reference(reference_input)
        parallel_output = row(parallel_input, input_is_parallel=False)
        torch.testing.assert_close(parallel_output, reference_output)
        if context.is_main_process:
            print("replicated_input_output=PASS", flush=True)

        reference_loss = reference_output.square().mean()
        parallel_loss = parallel_output.square().mean()
        reference_loss.backward()
        parallel_loss.backward()
        torch.testing.assert_close(parallel_input.grad, reference_input.grad)
        torch.testing.assert_close(
            row.weight.grad,
            reference.weight.grad[:, row.input_start : row.input_end],
        )
        assert row.bias is not None and reference.bias is not None
        torch.testing.assert_close(row.bias.grad, reference.bias.grad)
        if context.is_main_process:
            print("input_gradient=PASS", flush=True)
            print("local_weight_gradient=PASS", flush=True)
            print("replicated_bias_gradient=PASS", flush=True)
            print("ROW PARALLEL CORRECTNESS CHECKS PASSED", flush=True)
        dist.barrier()
    finally:
        if context is not None:
            cleanup_distributed()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
