"""MiniTrain 手写 Tensor Parallel 组件。"""

from minitrain.tensor_parallel.column_linear import ColumnParallelLinear
from minitrain.tensor_parallel.row_linear import RowParallelLinear

__all__ = ["ColumnParallelLinear", "RowParallelLinear"]
