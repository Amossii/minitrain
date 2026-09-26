"""Runtime environment inspection used before GPU systems experiments."""

from __future__ import annotations

import platform
import shutil
import subprocess
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class EnvironmentReport:
    """A stable snapshot of software and accelerator availability."""

    python_version: str
    pytorch_version: str | None
    cuda_build_version: str | None
    cuda_available: bool
    nccl_available: bool
    nccl_version: str | None
    gpu_names: tuple[str, ...]
    gpu_topology: str | None
    error: str | None = None

    @property
    def gpu_count(self) -> int:
        """Return the number of CUDA devices visible to this process."""

        return len(self.gpu_names)


# This function normalizes version formats because PyTorch releases have
# returned NCCL versions as either integers or tuples.
def _format_nccl_version(version: object) -> str | None:
    """Convert a PyTorch NCCL version value into readable dotted notation."""

    if version is None:
        return None
    if isinstance(version, tuple):
        return ".".join(str(part) for part in version)
    if isinstance(version, int):
        major = version // 10_000
        minor = (version % 10_000) // 100
        patch = version % 100
        return f"{major}.{minor}.{patch}"
    return str(version)


# Input is an optional torch-like object for deterministic unit testing.
# Output is immutable environment data consumed by the CLI. In distributed
# training this prevents launching workers against an invalid CUDA/NCCL setup.
def inspect_environment(torch_module: Any | None = None) -> EnvironmentReport:
    """Inspect PyTorch, CUDA, NCCL, visible GPUs, and their topology."""

    if torch_module is None:
        try:
            import torch as torch_module
        except ImportError as exc:
            return EnvironmentReport(
                python_version=platform.python_version(),
                pytorch_version=None,
                cuda_build_version=None,
                cuda_available=False,
                nccl_available=False,
                nccl_version=None,
                gpu_names=(),
                gpu_topology=_read_gpu_topology(),
                error=f"PyTorch import failed: {exc}",
            )

    cuda_available = bool(torch_module.cuda.is_available())
    gpu_count = int(torch_module.cuda.device_count()) if cuda_available else 0
    gpu_names = tuple(
        str(torch_module.cuda.get_device_name(index)) for index in range(gpu_count)
    )

    distributed = getattr(torch_module, "distributed", None)
    nccl_available = bool(
        distributed is not None
        and hasattr(distributed, "is_nccl_available")
        and distributed.is_nccl_available()
    )
    nccl_version = None
    if nccl_available:
        nccl_api = getattr(torch_module.cuda, "nccl", None)
        if nccl_api is not None and hasattr(nccl_api, "version"):
            try:
                nccl_version = _format_nccl_version(nccl_api.version())
            except RuntimeError:
                # Availability is still useful even if the runtime cannot query
                # the dynamically loaded NCCL library version.
                nccl_version = None

    return EnvironmentReport(
        python_version=platform.python_version(),
        pytorch_version=str(torch_module.__version__),
        cuda_build_version=getattr(torch_module.version, "cuda", None),
        cuda_available=cuda_available,
        nccl_available=nccl_available,
        nccl_version=nccl_version,
        gpu_names=gpu_names,
        gpu_topology=_read_gpu_topology(),
    )


# nvidia-smi is intentionally kept outside the PyTorch probe: topology describes
# physical GPU links and is needed later when interpreting collective bandwidth.
def _read_gpu_topology() -> str | None:
    """Return `nvidia-smi topo -m` output, or None when it is unavailable."""

    executable = shutil.which("nvidia-smi")
    if executable is None:
        return None

    try:
        result = subprocess.run(
            [executable, "topo", "-m"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


# Input is the detected report; output is human-readable diagnostics. Keeping
# formatting separate lets correctness tests validate detection without GPUs.
def format_environment_report(report: EnvironmentReport) -> str:
    """Format an environment report for terminal inspection."""

    lines = [
        f"Python: {report.python_version}",
        f"PyTorch: {report.pytorch_version or 'NOT INSTALLED'}",
        f"CUDA build: {report.cuda_build_version or 'NOT AVAILABLE'}",
        f"CUDA available: {report.cuda_available}",
        f"NCCL available: {report.nccl_available}",
        f"NCCL version: {report.nccl_version or 'NOT AVAILABLE'}",
        f"GPU count: {report.gpu_count}",
    ]
    lines.extend(f"GPU {index}: {name}" for index, name in enumerate(report.gpu_names))
    if report.error:
        lines.append(f"Error: {report.error}")
    lines.append("GPU topology:")
    lines.append(report.gpu_topology or "NOT AVAILABLE")
    return "\n".join(lines)


# Strict validation is used on Kaggle before launching multi-process jobs.
# It returns all unmet requirements so the user can fix the environment once.
def validate_environment(report: EnvironmentReport, required_gpus: int) -> list[str]:
    """Return requirement failures for a CUDA/NCCL experiment environment."""

    failures: list[str] = []
    if report.pytorch_version is None:
        failures.append("PyTorch is not installed")
    if not report.cuda_available:
        failures.append("CUDA is not available to PyTorch")
    if not report.nccl_available:
        failures.append("the NCCL backend is not available")
    if report.gpu_count < required_gpus:
        failures.append(
            f"expected at least {required_gpus} GPU(s), found {report.gpu_count}"
        )
    return failures
