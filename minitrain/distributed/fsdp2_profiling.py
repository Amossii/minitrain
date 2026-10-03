"""FSDP2 profiler 输出路径与通信事件辅助函数。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class FSDP2ProfilePaths:
    """保存一个 rank 独占的 trace、算子表和 FSDP2 元数据路径。"""

    trace: Path
    operators: Path
    shards: Path
    communications: Path


# 输入是输出目录和 rank，输出文件名包含 rank，避免多个进程并发覆盖；
# 这些文件分别服务于时间线、热点排序、静态分片和通信事件快速检查。
def fsdp2_profile_paths(output_dir: Path, rank: int) -> FSDP2ProfilePaths:
    """生成确定且互不冲突的 FSDP2 profiler 输出路径。"""

    if rank < 0:
        raise ValueError("rank must be non-negative")
    prefix = f"fsdp2_rank{rank}"
    return FSDP2ProfilePaths(
        trace=output_dir / f"{prefix}.json",
        operators=output_dir / f"{prefix}_operators.txt",
        shards=output_dir / f"{prefix}_shards.json",
        communications=output_dir / f"{prefix}_communications.json",
    )
