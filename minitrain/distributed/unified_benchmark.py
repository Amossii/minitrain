"""Step 23 统一 benchmark 的指标、策略语义与公平性校验。"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, fields
from pathlib import Path
import statistics

from minitrain.training import StepMeasurement


@dataclass(frozen=True, slots=True)
class StrategySemantics:
    """描述一个策略属于哪类并行，以及计时区间内的主要通信。"""

    parallelism_family: str
    communication: str


STRATEGY_SEMANTICS = {
    "single": StrategySemantics("single", "none"),
    "ddp": StrategySemantics("data_parallel", "gradient_all_reduce"),
    "fsdp2": StrategySemantics(
        "data_parallel", "parameter_all_gather+gradient_reduce_scatter"
    ),
    "zero1": StrategySemantics(
        "data_parallel", "gradient_all_reduce+updated_parameter_all_gather"
    ),
    "zero2": StrategySemantics(
        "data_parallel", "gradient_reduce_scatter+updated_parameter_all_gather"
    ),
    "zero3": StrategySemantics(
        "data_parallel", "parameter_all_gather+gradient_reduce_scatter"
    ),
    "tp": StrategySemantics("tensor_parallel", "block_output_all_reduce"),
}


# 输入是策略名，输出是并行类别和主要通信；拒绝未知名字可防止 CSV 中出现
# 无法解释的实验行。
def strategy_semantics(strategy: str) -> StrategySemantics:
    """返回统一 benchmark 支持的策略语义。"""

    try:
        return STRATEGY_SEMANTICS[strategy]
    except KeyError as exc:
        choices = ", ".join(STRATEGY_SEMANTICS)
        raise ValueError(f"unknown strategy {strategy!r}; choices: {choices}") from exc


@dataclass(frozen=True, slots=True)
class UnifiedStepMetrics:
    """记录一次更新的观测值和不可变 workload 元数据。

    `batch_size_per_process` 是每个进程看到的 batch；`global_batch_size` 是
    一次参数更新包含的不同样本数。数据并行中前者乘 world size 等于后者，
    Tensor Parallel 中各 rank 看到同一批样本，因此两者相等。
    """

    step: int
    loss: float
    step_time: float
    forward_time: float
    backward_time: float
    optimizer_time: float
    tokens_per_second: float
    samples_per_second: float
    peak_memory_allocated: int
    peak_memory_reserved: int
    world_size: int
    batch_size_per_process: int
    global_batch_size: int
    seq_len: int
    num_parameters: int
    warmup_steps: int
    model_name: str
    precision: str
    optimizer: str
    strategy: str
    parallelism_family: str
    communication: str

    @classmethod
    def from_measurement(
        cls,
        *,
        step: int,
        measurement: StepMeasurement,
        world_size: int,
        batch_size_per_process: int,
        global_batch_size: int,
        seq_len: int,
        num_parameters: int,
        warmup_steps: int,
        model_name: str,
        precision: str,
        optimizer: str,
        strategy: str,
    ) -> "UnifiedStepMetrics":
        """将一次实测结果转换为统一 CSV 行，并按真实 global batch 算吞吐。"""

        positive = (
            world_size,
            batch_size_per_process,
            global_batch_size,
            seq_len,
            num_parameters,
        )
        if step < 0 or warmup_steps < 0:
            raise ValueError("step and warmup_steps must be non-negative")
        if any(value <= 0 for value in positive):
            raise ValueError("world/batch/sequence/parameter values must be positive")
        if measurement.step_time <= 0:
            raise ValueError("step_time must be positive")
        semantics = strategy_semantics(strategy)
        if semantics.parallelism_family == "data_parallel":
            expected = batch_size_per_process * world_size
            if global_batch_size != expected:
                raise ValueError(
                    "data parallel global batch must equal batch per process times "
                    "world size"
                )
        if semantics.parallelism_family == "tensor_parallel":
            if global_batch_size != batch_size_per_process:
                raise ValueError(
                    "tensor parallel ranks share one batch, so global batch must "
                    "equal batch per process"
                )

        return cls(
            step=step,
            loss=measurement.loss,
            step_time=measurement.step_time,
            forward_time=measurement.forward_time,
            backward_time=measurement.backward_time,
            optimizer_time=measurement.optimizer_time,
            tokens_per_second=global_batch_size * seq_len / measurement.step_time,
            samples_per_second=global_batch_size / measurement.step_time,
            peak_memory_allocated=measurement.peak_memory_allocated,
            peak_memory_reserved=measurement.peak_memory_reserved,
            world_size=world_size,
            batch_size_per_process=batch_size_per_process,
            global_batch_size=global_batch_size,
            seq_len=seq_len,
            num_parameters=num_parameters,
            warmup_steps=warmup_steps,
            model_name=model_name,
            precision=precision,
            optimizer=optimizer,
            strategy=strategy,
            parallelism_family=semantics.parallelism_family,
            communication=semantics.communication,
        )

    def to_dict(self) -> dict[str, int | float | str]:
        """输出字段顺序稳定的字典，供 CSV writer 使用。"""

        return asdict(self)


# 输入是一组同构记录和目标路径，输出完整原始 CSV；原始逐 step 数据保留，
# 后续汇总规则改变时无需重跑昂贵 GPU 实验。
def write_unified_metrics_csv(path: Path, records: list[UnifiedStepMetrics]) -> None:
    """写入统一 benchmark 的逐 step 原始记录。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    names = [field.name for field in fields(UnifiedStepMetrics)]
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=names)
        writer.writeheader()
        writer.writerows(record.to_dict() for record in records)


UNIFIED_SUMMARY_FIELDS = (
    "model_name",
    "strategy",
    "parallelism_family",
    "communication",
    "precision",
    "optimizer",
    "world_size",
    "batch_size_per_process",
    "global_batch_size",
    "seq_len",
    "num_parameters",
    "warmup_steps",
    "measured_steps",
    "median_loss",
    "median_step_time",
    "median_forward_time",
    "median_backward_time",
    "median_optimizer_time",
    "median_tokens_per_second",
    "median_samples_per_second",
    "max_peak_memory_allocated",
    "max_peak_memory_reserved",
)


# 输入是一份逐 step CSV，输出一个中位数汇总行；所有 workload 字段必须在
# 文件内恒定，避免把不同实验误聚合成一个数字。
def summarize_unified_metrics(path: Path) -> dict[str, str | int | float]:
    """校验并汇总一次策略运行。"""

    with path.open(encoding="utf-8", newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))
    if not rows:
        raise ValueError(f"metrics file contains no measured steps: {path}")

    fixed_fields = UNIFIED_SUMMARY_FIELDS[:12]
    for field in fixed_fields:
        values = {row[field] for row in rows}
        if len(values) != 1:
            raise ValueError(f"{field} changes within one benchmark run: {path}")

    def median(field: str) -> float:
        return statistics.median(float(row[field]) for row in rows)

    first = rows[0]
    return {
        "model_name": first["model_name"],
        "strategy": first["strategy"],
        "parallelism_family": first["parallelism_family"],
        "communication": first["communication"],
        "precision": first["precision"],
        "optimizer": first["optimizer"],
        "world_size": int(first["world_size"]),
        "batch_size_per_process": int(first["batch_size_per_process"]),
        "global_batch_size": int(first["global_batch_size"]),
        "seq_len": int(first["seq_len"]),
        "num_parameters": int(first["num_parameters"]),
        "warmup_steps": int(first["warmup_steps"]),
        "measured_steps": len(rows),
        "median_loss": median("loss"),
        "median_step_time": median("step_time"),
        "median_forward_time": median("forward_time"),
        "median_backward_time": median("backward_time"),
        "median_optimizer_time": median("optimizer_time"),
        "median_tokens_per_second": median("tokens_per_second"),
        "median_samples_per_second": median("samples_per_second"),
        "max_peak_memory_allocated": max(
            int(row["peak_memory_allocated"]) for row in rows
        ),
        "max_peak_memory_reserved": max(
            int(row["peak_memory_reserved"]) for row in rows
        ),
    }


# 输入是同一张比较表的策略汇总行，输出为空；固定字段不一致时直接失败，
# 防止用更大 batch、不同模型或更少 warmup 换取看似更好的结果。
def validate_comparable_summaries(
    rows: list[dict[str, str | int | float]],
) -> None:
    """验证比较表中的 workload 和测量协议完全相同。"""

    if not rows:
        raise ValueError("at least one summary row is required")
    fixed_fields = (
        "model_name",
        "precision",
        "optimizer",
        "global_batch_size",
        "seq_len",
        "num_parameters",
        "warmup_steps",
        "measured_steps",
    )
    for field in fixed_fields:
        values = {row[field] for row in rows}
        if len(values) != 1:
            raise ValueError(f"unfair comparison: {field} differs: {sorted(values)}")


# 输入是已验证汇总行和目标路径，输出一个确定字段顺序的比较表。
def write_unified_summary_csv(
    path: Path, rows: list[dict[str, str | int | float]]
) -> None:
    """校验公平性并写入统一比较表。"""

    validate_comparable_summaries(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=UNIFIED_SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
