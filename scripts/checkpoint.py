import time

import torch
from torch.utils.checkpoint import checkpoint


# 创建一个较大的 MLP block。
#
# 输入 shape:
#   [batch_size, hidden_size]
#
# 输出 shape:
#   [batch_size, hidden_size]
#
# 这里故意使用两个较大的 Linear，
# 让中间 activation 占用更明显，
# 便于观察 checkpoint 前后的显存差异。
def build_block(
    hidden_size: int = 4096,
    intermediate_size: int = 16384,
) -> torch.nn.Module:
    return torch.nn.Sequential(
        torch.nn.Linear(
            hidden_size,
            intermediate_size,
        ),
        torch.nn.GELU(),
        torch.nn.Linear(
            intermediate_size,
            hidden_size,
        ),
    )


# 构造多个连续 block。
#
# num_blocks:
#   模拟 Transformer 的多层结构。
#
# 每个 block 的输入输出 shape 相同：
#
#   [B, H]
#
# 这样可以连续堆叠。
def build_model(
    num_blocks: int = 1,
    hidden_size: int = 4096,
) -> torch.nn.ModuleList:
    blocks = torch.nn.ModuleList()

    for _ in range(num_blocks):
        blocks.append(
            build_block(
                hidden_size=hidden_size,
            )
        )

    return blocks


# 普通 forward。
#
# 每个 block 都按正常 autograd 规则执行。
# 为了 backward，
# PyTorch 会保留需要的中间 activation。
def forward_normal(
    blocks: torch.nn.ModuleList,
    x: torch.Tensor,
) -> torch.Tensor:
    for block in blocks:
        x = block(x)

    return x


# Activation Checkpoint forward。
#
# 每个 block 都通过 checkpoint 执行。
#
# Forward:
#   block 内部的大部分中间 activation
#   不长期保存。
#
# Backward:
#   PyTorch 会从 block 输入重新执行 forward，
#   恢复 backward 所需要的中间结果。
#
# use_reentrant=False:
#   使用较新的非 reentrant checkpoint 实现。
def forward_checkpointed(
    blocks: torch.nn.ModuleList,
    x: torch.Tensor,
) -> torch.Tensor:
    for block in blocks:
        x = checkpoint(
            block,
            x,
            use_reentrant=False,
        )

    return x


# 执行若干训练 step，并测量：
#
# 1. 平均 step time
# 2. peak allocated GPU memory
#
# use_checkpoint:
#   False:
#       普通训练。
#
#   True:
#       使用 Activation Checkpointing。
def benchmark(
    blocks: torch.nn.ModuleList,
    optimizer: torch.optim.Optimizer,
    x: torch.Tensor,
    use_checkpoint: bool,
    warmup_steps: int = 3,
    measure_steps: int = 10,
) -> tuple[float, float]:

    # 先进行 warmup。
    #
    # 避免 CUDA lazy init、allocator 建立、
    # optimizer state 首次创建等因素污染正式结果。
    for _ in range(warmup_steps):
        optimizer.zero_grad(
            set_to_none=True,
        )

        if use_checkpoint:
            output = forward_checkpointed(
                blocks,
                x,
            )
        else:
            output = forward_normal(
                blocks,
                x,
            )

        loss = output.float().square().mean()

        loss.backward()

        optimizer.step()

    torch.cuda.synchronize()

    # 正式测量前重置峰值显存。
    torch.cuda.reset_peak_memory_stats()

    start = time.perf_counter()

    for _ in range(measure_steps):
        optimizer.zero_grad(
            set_to_none=True,
        )

        if use_checkpoint:
            output = forward_checkpointed(
                blocks,
                x,
            )
        else:
            output = forward_normal(
                blocks,
                x,
            )

        loss = output.float().square().mean()

        loss.backward()

        optimizer.step()

    # CUDA kernel launch 是异步的。
    # 等待正式 workload 全部完成后再停止计时。
    torch.cuda.synchronize()

    elapsed = time.perf_counter() - start

    avg_step_time = elapsed / measure_steps

    peak_memory_gb = torch.cuda.max_memory_allocated() / 1024**3

    return (
        avg_step_time,
        peak_memory_gb,
    )


# 主函数。
#
# 对比：
#
# 1. 普通训练
# 2. Activation Checkpointing
#
# 观察：
#
# checkpoint:
#   peak memory ↓
#
# 但通常：
#   step time ↑
def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("This demo requires a CUDA GPU.")

    device = torch.device("cuda")
    print(device)
    torch.manual_seed(42)

    blocks = build_model(
        num_blocks=2,
        hidden_size=4096,
    ).to(device)

    optimizer = torch.optim.AdamW(
        blocks.parameters(),
        lr=1e-4,
    )

    x = torch.randn(
        16,
        4096,
        device=device,
        requires_grad=True,
    )

    normal_time, normal_memory = benchmark(
        blocks=blocks,
        optimizer=optimizer,
        x=x,
        use_checkpoint=False,
    )

    checkpoint_time, checkpoint_memory = benchmark(
        blocks=blocks,
        optimizer=optimizer,
        x=x,
        use_checkpoint=True,
    )

    print("Normal training:")

    print(f"  step_time = {normal_time * 1000:.2f} ms")

    print(f"  peak_memory = {normal_memory:.2f} GB")

    print()

    print("Activation checkpointing:")

    print(f"  step_time = {checkpoint_time * 1000:.2f} ms")

    print(f"  peak_memory = {checkpoint_memory:.2f} GB")

    print()

    memory_reduction = (1.0 - checkpoint_memory / normal_memory) * 100

    slowdown = (checkpoint_time / normal_time - 1.0) * 100

    print(f"memory_reduction = {memory_reduction:.2f}%")

    print(f"step_time_increase = {slowdown:.2f}%")


if __name__ == "__main__":
    main()
