import argparse
import os

import deepspeed
import torch


# 构造一个简单的 MLP block。
#
# 输入 shape:
#   [batch_size, hidden_size]
#
# 输出 shape:
#   [batch_size, hidden_size]
#
# 使用较大的 Linear，
# 便于观察不同 ZeRO stage 对模型状态显存的影响。
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


# 构造完整 toy model。
#
# 模型结构应与 DDP / FSDP baseline 保持一致，
# 避免把模型变化混入框架对比。
def build_model(
    hidden_size: int = 2048,
    intermediate_size: int = 8192,
    num_blocks: int = 4,
) -> torch.nn.Module:

    class ToyModel(torch.nn.Module):
        # 初始化模型。
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

        # 执行完整模型前向传播。
        def forward(
            self,
            x: torch.Tensor,
        ) -> torch.Tensor:
            for block in self.blocks:
                x = block(x)

            return self.output(x)

    return ToyModel()


# 执行一个 DeepSpeed microstep。
#
# DeepSpeed Engine 接管：
#
# backward
# gradient synchronization / partition
# gradient accumulation bookkeeping
# optimizer step
#
# 当 accumulation 尚未达到边界时，
# engine.step() 不一定真正更新参数。
def train_microstep(
    engine,
    x: torch.Tensor,
) -> float:

    output = engine(x)

    loss = output.float().square().mean()

    # 不再直接调用 loss.backward()。
    #
    # DeepSpeed 需要参与 backward 生命周期，
    # 才能正确执行 ZeRO gradient partitioning。
    engine.backward(loss)

    # DeepSpeed 根据 gradient accumulation 配置
    # 决定这一 microstep 是否真正进行 optimizer update。
    engine.step()

    return loss.item()


# 打印 GPU 显存统计。
#
# peak memory 要在正式测量前 reset，
# 否则 warmup 和 optimizer state 首次创建
# 会污染结果。
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
# DeepSpeed 推荐通过 launcher 启动，例如：
#
# deepspeed \
#   --num_gpus=2 \
#   deepspeed_baseline.py \
#   --deepspeed \
#   --deepspeed_config ds_zero2.json
def main() -> None:
    parser = argparse.ArgumentParser()

    # 向 argparse 注册 DeepSpeed 所需参数。
    parser = deepspeed.add_config_arguments(parser)

    args = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])

    torch.cuda.set_device(local_rank)

    device = torch.device(f"cuda:{local_rank}")

    rank = int(os.environ["RANK"])

    torch.manual_seed(42)

    model = build_model()

    # deepspeed.initialize 会根据配置：
    #
    # 1. 建立 distributed environment。
    # 2. 处理 ZeRO state partition。
    # 3. 创建或包装 optimizer。
    # 4. 返回 DeepSpeed Engine。
    engine, optimizer, _, _ = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=model.parameters(),
    )

    batch_size = 8
    hidden_size = 2048

    generator = torch.Generator(device=device)

    generator.manual_seed(1000 + rank)

    x = torch.randn(
        batch_size,
        hidden_size,
        device=device,
        generator=generator,
    )

    # 先执行 warmup。
    #
    # gradient_accumulation_steps = 2，
    # 所以两个 microsteps 才构成一次完整 update。
    for _ in range(2):
        train_microstep(
            engine=engine,
            x=x,
        )

    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()

    losses = []

    # 正式再完成一次 global update。
    for _ in range(2):
        loss = train_microstep(
            engine=engine,
            x=x,
        )

        losses.append(loss)

    torch.cuda.synchronize()

    mean_loss = sum(losses) / len(losses)

    print(f"rank={rank} mean_local_loss={mean_loss:.6f}")

    print_memory(
        rank=rank,
        tag="after_update",
    )


if __name__ == "__main__":
    main()
