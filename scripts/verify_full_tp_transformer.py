"""验证 vocabulary + block Tensor Parallel 与普通 Transformer 数值等价。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.distributed as dist

from minitrain.distributed.runtime import cleanup_distributed, init_distributed
from minitrain.model import MiniTransformer
from minitrain.tensor_parallel.full_transformer import (
    FullTensorParallelTransformer,
    gather_full_logits,
)
from minitrain.tensor_parallel.transformer import gather_tensor_parallel_tensors
from minitrain.training import causal_lm_loss
from scripts.verify_tp_transformer import (
    assert_named_tensors_match,
    correctness_config,
)


# CLI 输入定义一次确定性更新；输出验证完整 logits、分布式 loss、所有梯度和更新参数，
# 确保 vocabulary sharding 只改变 ownership/通信而不改变训练数学。
def main() -> int:
    """执行 Full TP 相对普通 Transformer 的端到端 correctness。"""

    parser = argparse.ArgumentParser(description="Verify full Transformer TP.")
    parser.add_argument("--backend", choices=("gloo", "nccl"), default="nccl")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.batch_size <= 0 or args.seq_len <= 0 or args.learning_rate <= 0:
        parser.error("batch size, sequence length, and learning rate must be positive")

    context = None
    try:
        context = init_distributed(backend=args.backend)
        config = correctness_config(args.seq_len)
        torch.manual_seed(args.seed)
        reference = MiniTransformer(config).to(context.device)
        full_tp = FullTensorParallelTransformer(config, device=context.device)
        full_tp.load_from_transformer(reference)

        torch.manual_seed(args.seed + 1)
        input_ids = torch.randint(
            0, config.vocab_size, (args.batch_size, args.seq_len), device=context.device
        )
        labels = torch.randint(
            0, config.vocab_size, (args.batch_size, args.seq_len), device=context.device
        )
        reference_logits = reference(input_ids)
        local_logits = full_tp(input_ids)
        observed_logits = gather_full_logits(local_logits)
        torch.testing.assert_close(
            observed_logits, reference_logits, atol=1e-5, rtol=1e-4
        )

        reference_loss = causal_lm_loss(reference_logits, labels)
        observed_loss = full_tp.loss(local_logits, labels)
        torch.testing.assert_close(
            observed_loss, reference_loss, atol=1e-6, rtol=1e-5
        )
        reference_loss.backward()
        observed_loss.backward()
        full_gradients = gather_tensor_parallel_tensors(full_tp, gradients=True)
        assert_named_tensors_match(full_gradients, reference, gradients=True)

        reference_optimizer = torch.optim.SGD(
            reference.parameters(), lr=args.learning_rate
        )
        tp_optimizer = torch.optim.SGD(full_tp.parameters(), lr=args.learning_rate)
        reference_optimizer.step()
        tp_optimizer.step()
        full_parameters = gather_tensor_parallel_tensors(full_tp)
        assert_named_tensors_match(full_parameters, reference, gradients=False)

        if context.is_main_process:
            print(f"local_logits_shape={tuple(local_logits.shape)}", flush=True)
            print("vocab_parallel_embedding=PASS", flush=True)
            print("vocab_parallel_lm_head=PASS", flush=True)
            print("vocab_parallel_cross_entropy=PASS", flush=True)
            print("full_model_gradients=PASS", flush=True)
            print("updated_parameters=PASS", flush=True)
            print("FULL TP CORRECTNESS CHECKS PASSED", flush=True)
        dist.barrier()
    finally:
        if context is not None:
            cleanup_distributed()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
