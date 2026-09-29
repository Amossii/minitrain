import argparse
import os

import deepspeed
import torch


# 构造一个简单的 MLP block。
#
# 输入：
#   [batch_size, hidden_size]
#
# 输出：
#   [batch_size, hidden_size]
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
# 模型结构与 DDP / FSDP baseline 保持一致，
# 避免模型本身变化影响显存对比。
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
# 1. backward
# 2. gradient synchronization / partition
# 3. gradient accumulation bookkeeping
# 4. optimizer step
def train_microstep(
    engine,
    x: torch.Tensor,
    rank: int,
    step_name: str,
) -> float:

    # 用这些探针确认 ZeRO-3 到底停在 forward、backward
    # 还是 engine.step()。
    print(f"rank={rank} {step_name} before_forward")

    output = engine(x)

    print(f"rank={rank} {step_name} after_forward")

    loss = output.float().square().mean()

    engine.backward(loss)

    print(f"rank={rank} {step_name} after_backward")

    engine.step()

    print(f"rank={rank} {step_name} after_engine_step")

    return loss.item()


# 打印 GPU 显存统计。
#
# allocated：
#   当前 PyTorch tensor 实际占用的显存。
#
# peak：
#   从 reset_peak_memory_stats() 之后出现过的峰值显存。
def print_memory(
    rank: int,
    tag: str,
) -> None:

    gb = 1024**3

    allocated = torch.cuda.memory_allocated() / gb
    peak = torch.cuda.max_memory_allocated() / gb

    print(f"rank={rank} {tag} allocated={allocated:.3f} GB peak={peak:.3f} GB")


# 主程序。
def main() -> None:
    parser = argparse.ArgumentParser()

    # DeepSpeed launcher 会自动给不同进程传入：
    # --local_rank=0
    # --local_rank=1
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
    )

    parser = deepspeed.add_config_arguments(parser)

    args = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])

    torch.cuda.set_device(local_rank)

    device = torch.device(f"cuda:{local_rank}")

    rank = int(os.environ["RANK"])

    torch.manual_seed(42)

    print(f"rank={rank} START")

    model = build_model()

    print(f"rank={rank} before_deepspeed_initialize")

    # DeepSpeed 会根据 ds_zero2.json / ds_zero3.json
    # 自动建立相应的 ZeRO optimizer。
    engine, optimizer, _, _ = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=model.parameters(),
    )

    print(
        f"rank={rank} "
        f"after_deepspeed_initialize "
        f"zero_stage={engine.zero_optimization_stage()}"
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
        dtype=torch.float16,
    )

    print(f"rank={rank} input_ready")

    # warmup：
    # gradient_accumulation_steps=2，
    # 因此两个 microstep 构成一次 optimizer update。
    for i in range(2):
        train_microstep(
            engine=engine,
            x=x,
            rank=rank,
            step_name=f"warmup_{i}",
        )

    print(f"rank={rank} warmup_finished")

    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()

    print(f"rank={rank} peak_memory_reset")

    losses = []

    # 正式测量的一次 global update。
    for i in range(2):
        loss = train_microstep(
            engine=engine,
            x=x,
            rank=rank,
            step_name=f"measure_{i}",
        )

        losses.append(loss)

    print(f"rank={rank} measured_steps_finished")

    torch.cuda.synchronize()

    mean_loss = sum(losses) / len(losses)

    print(f"rank={rank} mean_local_loss={mean_loss:.6f}")

    print_memory(
        rank=rank,
        tag="after_update",
    )

    print(f"rank={rank} END")


if __name__ == "__main__":
    main()
