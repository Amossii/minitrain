"""FSDP2 与 Tensor Parallel 公平比较所需的实验协议。"""

from __future__ import annotations


# 输入是并行策略、有效 global batch 和进程数，输出每个 rank 实际接收的 batch；
# 该差异来自数据并行与模型并行语义，不能为了让数字相同而改变 workload。
def per_rank_batch_size(strategy: str, global_batch_size: int, world_size: int) -> int:
    """计算固定有效 global batch 下每个 rank 的物理输入 batch。"""

    if strategy not in ("fsdp2", "tp"):
        raise ValueError(f"unsupported strategy: {strategy}")
    if global_batch_size <= 0 or world_size <= 0:
        raise ValueError("global_batch_size and world_size must be positive")
    if strategy == "fsdp2":
        if global_batch_size % world_size != 0:
            raise ValueError("FSDP2 global batch must be divisible by world size")
        return global_batch_size // world_size
    # TP ranks cooperate on the same samples, so every rank sees the full batch.
    return global_batch_size


# 输入是容量扫描结果、两个策略和吞吐目标规模，输出为空；只有两种策略在完全
# 相同的实际模型配置上都 PASS，吞吐比较才有意义。
def validate_throughput_target(
    rows: list[dict[str, str | int | float]], target_millions: float
) -> None:
    """确认吞吐目标已被两种策略的容量实验共同验证。"""

    expected = {"fsdp2", "tp"}
    passed = {
        str(row["strategy"])
        for row in rows
        if float(row["target_parameters_millions"]) == target_millions
        and row["status"] == "PASS"
    }
    missing = expected - passed
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(
            f"throughput target {target_millions:g}M was not PASS for: {names}"
        )
