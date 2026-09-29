import os

import torch
import torch.distributed as dist
from torch.distributed.fsdp import fully_shard


# 初始化分布式环境。
#
# torchrun 会为每个进程提供 LOCAL_RANK。
# 每个进程绑定一张 GPU，并使用 NCCL
# 加入默认 Process Group。
#
# 返回：
#   当前进程绑定的 local_rank。
def init_distributed() -> int:
    local_rank = int(os.environ["LOCAL_RANK"])

    torch.cuda.set_device(local_rank)

    dist.init_process_group(
        backend="nccl",
        init_method="env://",
    )

    return local_rank


# 构造一个简单 Transformer-like block。
#
# 输入:
#   [batch_size, hidden_size]
#
# 输出:
#   [batch_size, hidden_size]
#
# 这里使用 MLP 代替完整 Transformer，
# 目的是把重点放在 FSDP shard boundary
# 和训练流程，而不是模型结构本身。
def build_block(
    hidden_size: int,
    intermediate_size: int,
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


# 构造一个由多个 block 组成的模型。
#
# 每个 block 后续会单独成为一个 FSDP unit，
# 从而避免整个模型一次性 AllGather。
def build_model(
    hidden_size: int = 2048,
    intermediate_size: int = 8192,
    num_blocks: int = 4,
) -> torch.nn.Module:

    class ToyModel(torch.nn.Module):
        # 初始化模型结构。
        def __init__(self) -> None:
            super().__init__()

            self.blocks = torch.nn.ModuleList(
                [
                    build_block(
                        hidden_size=hidden_size,
                        intermediate_size=intermediate_size,
                    )
                    for _ in range(num_blocks)
                ]
            )

            self.output = torch.nn.Linear(
                hidden_size,
                hidden_size,
            )

        # 执行完整前向传播。
        #
        # FSDP 会在每个被 fully_shard 的 block
        # 使用之前自动处理参数 AllGather，
        # 使用之后根据策略处理 Reshard。
        def forward(
            self,
            x: torch.Tensor,
        ) -> torch.Tensor:
            for block in self.blocks:
                x = block(x)

            return self.output(x)

    return ToyModel()


# 对模型应用 FSDP2。
#
# 顺序：
#
# 1. 先对内部 block 调用 fully_shard。
# 2. 最后再对 root model 调用 fully_shard。
#
# 这样每个 block 成为独立 shard unit，
# 而 root 可以继续管理未被内部 unit
# 包含的参数，例如 output layer。
def apply_fsdp(
    model: torch.nn.Module,
) -> torch.nn.Module:

    for block in model.blocks:
        fully_shard(block)

    fully_shard(model)

    return model


# 完成一个 optimizer step。
#
# 从用户代码角度：
#
# Forward
# ↓
# Backward
# ↓
# Optimizer
#
# 与普通 PyTorch 基线非常相似。
#
# AllGather / ReduceScatter / Reshard
# 由 FSDP 在内部执行。
def train_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    x: torch.Tensor,
) -> float:

    optimizer.zero_grad(
        set_to_none=True,
    )

    # autocast 控制算子计算精度。
    #
    # 这里保持 BF16，
    # 使 FSDP 实验与假定的 BF16 baseline
    # 具有一致的计算精度条件。
    with torch.autocast(
        device_type="cuda",
        dtype=torch.bfloat16,
    ):
        output = model(x)

        loss = output.float().square().mean()

    loss.backward()

    optimizer.step()

    return loss.item()


# 打印当前 rank 的显存情况。
#
# allocated:
#   活跃 Tensor 当前占用。
#
# peak:
#   从上次 reset_peak_memory_stats
#   开始的最大 allocated memory。
def print_memory(
    rank: int,
    tag: str,
) -> None:

    gb = 1024**3

    allocated = torch.cuda.memory_allocated() / gb

    peak = torch.cuda.max_memory_allocated() / gb

    print(f"rank={rank} {tag} allocated={allocated:.3f} GB peak={peak:.3f} GB")


# 主程序。
#
# 推荐运行：
#
# torchrun \
#   --standalone \
#   --nproc-per-node=2 \
#   /kaggle/working/distributed_lab/fsdp2_baseline.py
def main() -> None:
    local_rank = init_distributed()

    rank = dist.get_rank()

    device = torch.device(f"cuda:{local_rank}")

    # 所有 rank 使用相同随机种子，
    # 使初始化可控。
    torch.manual_seed(42)

    model = build_model()

    model = model.to(device)

    # 先应用 FSDP。
    #
    # optimizer 必须在 shard 结构建立后再创建，
    # 避免 optimizer 持有不正确的参数引用或状态布局。
    model = apply_fsdp(model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
    )

    batch_size = 8
    hidden_size = 2048

    # 每个 rank 使用不同随机数据。
    #
    # rank 被加入 seed，
    # 避免不同 DP rank 完全处理相同输入。
    generator = torch.Generator(device=device)

    generator.manual_seed(1000 + rank)

    x = torch.randn(
        batch_size,
        hidden_size,
        device=device,
        generator=generator,
    )

    # 先跑一个 warmup step。
    #
    # Adam optimizer states
    # 往往在第一次 optimizer.step()
    # 附近才真正创建。
    train_step(
        model=model,
        optimizer=optimizer,
        x=x,
    )

    torch.cuda.synchronize()

    # 正式测量前重置峰值统计。
    torch.cuda.reset_peak_memory_stats()

    loss = train_step(
        model=model,
        optimizer=optimizer,
        x=x,
    )

    torch.cuda.synchronize()

    print(f"rank={rank} loss={loss:.6f}")

    print_memory(
        rank=rank,
        tag="after_train_step",
    )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
