"""验证 Tensor Parallel Transformer 与普通模型的一次完整更新等价。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.distributed as dist

from minitrain.config import ModelConfig
from minitrain.distributed.runtime import cleanup_distributed, init_distributed
from minitrain.model import MiniTransformer
from minitrain.tensor_parallel.transformer import (
    TensorParallelTransformer,
    gather_tensor_parallel_tensors,
)
from minitrain.training import causal_lm_loss


# 输入是 seq_len，输出一个包含真实 Attention/SwiGLU 但足够小的配置，
# 让每个 rank 可以同时保存 TP 模型和完整 reference 模型。
def correctness_config(seq_len: int) -> ModelConfig:
    """返回 Transformer TP correctness 使用的小型模型配置。"""

    return ModelConfig(
        vocab_size=64,
        seq_len=seq_len,
        hidden_size=32,
        num_layers=2,
        num_heads=4,
        intermediate_size=96,
    )


# 输入是 Gather 后的 TP tensor 和 reference 模型；输出为空，任一参数或梯度不一致
# 都抛出带参数名的错误，从而定位具体 TP projection。
def assert_named_tensors_match(
    observed: dict[str, torch.Tensor],
    reference: MiniTransformer,
    *,
    gradients: bool,
) -> None:
    """比较 TP 完整参数/梯度和 reference 的同名 tensor。"""

    expected = {}
    for name, parameter in reference.named_parameters():
        value = parameter.grad if gradients else parameter
        if value is None:
            raise AssertionError(f"reference missing gradient for {name}")
        expected[name] = value.detach()
    if observed.keys() != expected.keys():
        raise AssertionError("TP and reference parameter names differ")
    for name in observed:
        torch.testing.assert_close(
            observed[name], expected[name], atol=1e-5, rtol=1e-4, msg=lambda _: name
        )


# CLI 输入定义一次确定性更新；输出依次验证 logits、loss、全模型梯度和更新后参数。
def main() -> int:
    """执行并验证一次完整 Transformer Tensor Parallel 更新。"""

    parser = argparse.ArgumentParser(description="Verify Transformer TP.")
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
        tp_model = TensorParallelTransformer(config, device=context.device)
        tp_model.load_from_transformer(reference)

        torch.manual_seed(args.seed + 1)
        input_ids = torch.randint(
            0,
            config.vocab_size,
            (args.batch_size, args.seq_len),
            device=context.device,
        )
        labels = torch.randint(
            0,
            config.vocab_size,
            (args.batch_size, args.seq_len),
            device=context.device,
        )
        reference_logits = reference(input_ids)
        tp_logits = tp_model(input_ids)
        torch.testing.assert_close(tp_logits, reference_logits, atol=1e-5, rtol=1e-4)
        if context.is_main_process:
            print(f"logits_shape={tuple(tp_logits.shape)} forward=PASS", flush=True)

        reference_loss = causal_lm_loss(reference_logits, labels)
        tp_loss = causal_lm_loss(tp_logits, labels)
        torch.testing.assert_close(tp_loss, reference_loss, atol=1e-6, rtol=1e-5)
        reference_loss.backward()
        tp_loss.backward()
        full_gradients = gather_tensor_parallel_tensors(tp_model, gradients=True)
        assert_named_tensors_match(full_gradients, reference, gradients=True)
        if context.is_main_process:
            print("loss=PASS", flush=True)
            print("full_model_gradients=PASS", flush=True)

        reference_optimizer = torch.optim.SGD(
            reference.parameters(), lr=args.learning_rate
        )
        tp_optimizer = torch.optim.SGD(tp_model.parameters(), lr=args.learning_rate)
        reference_optimizer.step()
        tp_optimizer.step()
        full_parameters = gather_tensor_parallel_tensors(tp_model)
        assert_named_tensors_match(full_parameters, reference, gradients=False)
        if context.is_main_process:
            print("updated_parameters=PASS", flush=True)
            print("TRANSFORMER TP CORRECTNESS CHECKS PASSED", flush=True)
        dist.barrier()
    finally:
        if context is not None:
            cleanup_distributed()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
