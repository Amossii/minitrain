import torch


# 打印当前 PyTorch CUDA allocator 的显存状态。
# allocated:
#   当前被活跃 Tensor 使用的显存。
#
# reserved:
#   PyTorch allocator 已向 CUDA 申请并保留的显存，
#   其中可能包含尚未被 Tensor 使用的缓存空间。
#
# peak_allocated:
#   从上次 reset_peak_memory_stats() 开始，
#   allocated 曾经达到的最大值。
def print_memory(tag: str) -> None:
    gb = 1024**3

    allocated = torch.cuda.memory_allocated() / gb
    reserved = torch.cuda.memory_reserved() / gb
    peak_allocated = torch.cuda.max_memory_allocated() / gb

    print(
        f"{tag:>20s} | "
        f"allocated={allocated:.3f} GB | "
        f"reserved={reserved:.3f} GB | "
        f"peak={peak_allocated:.3f} GB"
    )


# 构造一个简单 MLP，
# 用它观察 forward、backward 和 optimizer step
# 对显存状态的影响。
def main() -> None:
    device = "cuda"

    model = torch.nn.Sequential(
        torch.nn.Linear(8192, 8192),
        torch.nn.GELU(),
        torch.nn.Linear(8192, 8192),
    ).to(device=device, dtype=torch.float16)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    x = torch.randn(
        64,
        8192,
        device=device,
        dtype=torch.float16,
    )

    torch.cuda.reset_peak_memory_stats()

    print_memory("after init")

    y = model(x)

    print_memory("after forward")

    loss = y.float().square().mean()
    loss.backward()

    print_memory("after backward")

    optimizer.step()

    # CUDA 是异步执行的。
    # synchronize 保证前面提交到 GPU 的工作真正完成，
    # 再读取此时的显存统计。
    torch.cuda.synchronize()

    print_memory("after optimizer")


if __name__ == "__main__":
    main()
