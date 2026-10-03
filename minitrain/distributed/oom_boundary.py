"""用于构造模型规模并分析 DDP/FSDP2 OOM 边界。"""

from __future__ import annotations

import csv
import math
from pathlib import Path

from minitrain.config import ModelConfig


BOUNDARY_FIELDS = (
    "strategy",
    "target_parameters_millions",
    "actual_parameters",
    "hidden_size",
    "num_layers",
    "num_heads",
    "intermediate_size",
    "vocab_size",
    "seq_len",
    "world_size",
    "local_batch_size",
    "global_batch_size",
    "precision",
    "optimizer",
    "steps",
    "status",
    "peak_memory_allocated",
    "peak_memory_reserved",
    "log_path",
)


# 输入是模型结构，输出是与 MiniTransformer 实际模块逐项对应的参数量；
# 该公式避免为了选择候选规模而真正分配数百 MB 参数。
def transformer_parameter_count(config: ModelConfig) -> int:
    """计算当前无 bias MiniTransformer 的精确参数量。"""

    hidden = config.hidden_size
    intermediate = config.intermediate_size
    embeddings_and_head = (
        2 * config.vocab_size * hidden + config.seq_len * hidden
    )
    per_block = 4 * hidden * hidden + 3 * hidden * intermediate + 2 * hidden
    return embeddings_and_head + config.num_layers * per_block + hidden


# 输入是目标参数量和固定架构族，输出选择最接近目标且可被 head 数整除的宽度；
# 层数等维度保持不变，确保 sweep 只沿一个可解释方向增长。
def build_boundary_config(
    target_parameters_millions: float,
    seq_len: int,
    *,
    vocab_size: int = 32_000,
    num_layers: int = 12,
    num_heads: int = 8,
    mlp_ratio: int = 3,
) -> ModelConfig:
    """为目标参数量生成同一架构族中最接近的合法配置。"""

    if target_parameters_millions <= 0:
        raise ValueError("target_parameters_millions must be positive")
    if min(seq_len, vocab_size, num_layers, num_heads, mlp_ratio) <= 0:
        raise ValueError("all architecture dimensions must be positive")

    target = target_parameters_millions * 1_000_000
    quadratic = num_layers * (4 + 3 * mlp_ratio)
    linear = 2 * vocab_size + seq_len + 2 * num_layers + 1
    root = (-linear + math.sqrt(linear * linear + 4 * quadratic * target)) / (
        2 * quadratic
    )
    center = max(num_heads, round(root / num_heads) * num_heads)
    candidates = {
        max(num_heads, center + offset * num_heads) for offset in range(-2, 3)
    }
    configs = [
        ModelConfig(
            vocab_size=vocab_size,
            seq_len=seq_len,
            hidden_size=hidden,
            num_layers=num_layers,
            num_heads=num_heads,
            intermediate_size=hidden * mlp_ratio,
        )
        for hidden in candidates
    ]
    return min(
        configs,
        key=lambda config: abs(transformer_parameter_count(config) - target),
    )


# 输入是 torchrun 的 stdout/stderr，输出只识别明确的 CUDA 分配失败；
# NCCL、shape、代码异常等错误必须继续抛出，不能被伪装成 OOM 结果。
def is_cuda_oom(output: str) -> bool:
    """判断子进程日志是否包含可信的 CUDA OOM 特征。"""

    normalized = output.lower()
    markers = (
        "cuda out of memory",
        "torch.cuda.outofmemoryerror",
        "cuda error: out of memory",
    )
    return any(marker in normalized for marker in markers)


# 输入是完整实验行，输出是某策略真实 PASS 过的最大参数量；
# 若该策略没有任何成功点则返回 None，而不是伪造零参数边界。
def maximum_passing_parameters(
    rows: list[dict[str, str | int | float]], strategy: str
) -> int | None:
    """返回指定策略已验证可训练的最大实际参数量。"""

    passing = [
        int(row["actual_parameters"])
        for row in rows
        if row["strategy"] == strategy and row["status"] == "PASS"
    ]
    return max(passing) if passing else None


# 输入是实验结果和目标文件，输出稳定 CSV，保留 PASS/OOM 以及对应日志路径。
def write_boundary_csv(
    path: Path, rows: list[dict[str, str | int | float]]
) -> None:
    """写入 OOM boundary 原始结果表。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=BOUNDARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
