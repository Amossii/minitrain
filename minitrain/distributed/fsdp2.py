"""FSDP2 wrapping and shard-inspection helpers for MiniTrain."""

from __future__ import annotations

from dataclasses import dataclass

import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Shard
from torch.distributed.fsdp import FSDPModule, fully_shard

from minitrain.model import MiniTransformer


@dataclass(frozen=True, slots=True)
class ParameterShardInfo:
    """Describe one rank's local ownership of one global FSDP2 parameter."""

    name: str
    global_shape: tuple[int, ...]
    local_shape: tuple[int, ...]
    global_numel: int
    local_numel: int
    placements: tuple[str, ...]


# Inputs are an unsharded MiniTransformer and its device type. Output is the same
# object converted to FSDPModule; parameters become dim-0 DTensors across ranks.
def apply_fsdp2(model: MiniTransformer, device_type: str) -> FSDPModule:
    """Shard Transformer blocks bottom-up and then shard the root module."""

    if device_type not in ("cpu", "cuda"):
        raise ValueError("device_type must be cpu or cuda")
    if not dist.is_initialized():
        raise RuntimeError("apply_fsdp2 requires an initialized process group")
    # An explicit mesh prevents a Gloo/CPU job from accidentally selecting an
    # installed CUDA accelerator and maps the mesh to the current world group.
    mesh = init_device_mesh(device_type, (dist.get_world_size(),))

    # Each block becomes one communication group. Bottom-up order prevents the
    # root group from swallowing parameters that should be gathered per layer.
    for block in model.blocks:
        fully_shard(block, mesh=mesh)
    fully_shard(model, mesh=mesh)
    return model  # type: ignore[return-value]


# Input is a fully-sharded model. Output exposes global/local shapes and shard
# placements without gathering parameters, so ownership remains inspectable.
def inspect_parameter_shards(model: FSDPModule) -> list[ParameterShardInfo]:
    """Return rank-local metadata for every FSDP2 DTensor parameter."""

    shards: list[ParameterShardInfo] = []
    for name, parameter in model.named_parameters():
        if not isinstance(parameter, DTensor):
            raise TypeError(f"parameter {name} is not a DTensor after fully_shard")
        if not any(isinstance(placement, Shard) for placement in parameter.placements):
            raise AssertionError(f"parameter {name} has no sharded placement")
        local = parameter.to_local()
        shards.append(
            ParameterShardInfo(
                name=name,
                global_shape=tuple(parameter.shape),
                local_shape=tuple(local.shape),
                global_numel=parameter.numel(),
                local_numel=local.numel(),
                placements=tuple(str(item) for item in parameter.placements),
            )
        )
    return shards
