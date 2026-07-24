import logging
import os
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Literal

import torch
import torch.distributed as dist
from megatron.core import mpu

logger = logging.getLogger(__name__)


@dataclass
class DistInfo:
    """Information about the distributed environment."""

    rank: int
    world_size: int
    local_rank: int
    master_addr: str
    master_port: int


def load_dist_info_from_env() -> DistInfo:
    """Load distributed environment information from environment variables."""
    info = DistInfo(
        rank=int(os.environ["RANK"]),
        world_size=int(os.environ["WORLD_SIZE"]),
        local_rank=int(os.environ["LOCAL_RANK"]),
        master_addr=os.environ["MASTER_ADDR"],
        master_port=int(os.environ["MASTER_PORT"]),
    )
    return info


def set_env_dist_info(dist_info: DistInfo) -> None:
    """Set distributed environment information to environment variables."""
    os.environ["RANK"] = str(dist_info.rank)
    os.environ["WORLD_SIZE"] = str(dist_info.world_size)
    os.environ["LOCAL_RANK"] = str(dist_info.local_rank)
    os.environ["MASTER_ADDR"] = dist_info.master_addr
    os.environ["MASTER_PORT"] = str(dist_info.master_port)


@dataclass
class MCoreDistInfo(DistInfo):
    tp_rank: int
    cp_rank: int
    dp_rank: int
    pp_rank: int
    ep_rank: int
    vpp_rank: int | None
    tp_size: int
    cp_size: int
    dp_size: int
    pp_size: int
    ep_size: int
    vpp_size: int | None

    def __post_init__(self) -> None:
        num_gpus: int = torch.cuda.device_count()
        self.dist_info: str = "-".join(
            [
                f"rk{self.rank}_{self.world_size}",
                f"lrk{self.local_rank}_{num_gpus}",
                f"tp{self.tp_rank}_{self.tp_size}",
                f"cp{self.cp_rank}_{self.cp_size}",
                f"dp{self.dp_rank}_{self.dp_size}",
                f"pp{self.pp_rank}_{self.pp_size}",
                f"ep{self.ep_rank}_{self.ep_size}",
                f"vpp{self.vpp_rank}_{self.vpp_size}" if self.vpp_rank is not None else "vpp_NA",
            ]
        )

    def __str__(self) -> str:
        return self.dist_info


def load_mcore_dist_info_from_env() -> "MCoreDistInfo":
    """Load MegaTron Core distributed info from environment and MegaTron parallel state."""
    dist_info = load_dist_info_from_env()
    return MCoreDistInfo(
        # DistInfo fields
        rank=dist_info.rank,
        world_size=dist_info.world_size,
        local_rank=dist_info.local_rank,
        master_addr=dist_info.master_addr,
        master_port=dist_info.master_port,
        # MegaTron-specific fields
        tp_rank=mpu.get_tensor_model_parallel_rank(),
        cp_rank=mpu.get_context_parallel_rank(),
        dp_rank=mpu.get_data_parallel_rank(),
        pp_rank=mpu.get_pipeline_model_parallel_rank(),
        ep_rank=mpu.get_expert_model_parallel_rank(),
        vpp_rank=mpu.get_virtual_pipeline_model_parallel_rank(),
        tp_size=mpu.get_tensor_model_parallel_world_size(),
        cp_size=mpu.get_context_parallel_world_size(),
        dp_size=mpu.get_data_parallel_world_size(),
        pp_size=mpu.get_pipeline_model_parallel_world_size(),
        ep_size=mpu.get_expert_model_parallel_world_size(),
        vpp_size=mpu.get_virtual_pipeline_model_parallel_world_size(),
    )


def init_gloabal_process_group(
    dist_info: DistInfo | None = None,
    backend: Literal["nccl", "gloo"] = "nccl",
    timeout_seconds: int = 1800,
) -> DistInfo:
    """Initialize the global process group for distributed training."""
    dist_info = dist_info or load_dist_info_from_env()
    set_env_dist_info(dist_info)
    torch.cuda.set_device(dist_info.local_rank)
    dist.init_process_group(backend=backend, timeout=timedelta(seconds=timeout_seconds))
    return dist_info


def get_visible_devices() -> list[int]:
    """Get the list of visible CUDA devices."""
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        return [int(i) for i in os.environ["CUDA_VISIBLE_DEVICES"].split(",")]
    return list(range(torch.cuda.device_count()))


def set_visible_devices(devices: list[int]) -> None:
    """Set the visible CUDA devices."""
    if not devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(d) for d in devices)
    logger.info(f"Set visible devices: {os.environ['CUDA_VISIBLE_DEVICES']}")


def cleanup_distributed() -> None:
    """Clean up the distributed environment."""
    dist.destroy_process_group()


def all_gather_object(obj: Any, group: dist.ProcessGroup | None = None) -> list[Any]:
    group = group or dist.group.WORLD
    world_size = dist.get_world_size(group=group)
    gathered: list[Any] = [None] * world_size
    dist.all_gather_object(gathered, obj, group=group)
    return gathered


def all_gather_list(items: list, group: dist.ProcessGroup | None = None) -> list[Any]:
    group = group or dist.group.WORLD
    world_size = dist.get_world_size(group=group)
    gathered: list[Any] = [None] * world_size
    dist.all_gather_object(gathered, items, group=group)
    output: list[Any] = []
    for rank_list in gathered:
        if rank_list is not None:
            output.extend(rank_list)
    return output


def broadcast_object(obj: Any, src: int = 0, group: dist.ProcessGroup | None = None) -> Any:
    group = group or dist.group.WORLD
    items = [obj]
    dist.broadcast_object_list(items, src=src, group=group)
    return items[0]


def broadcast_object_list(items: list, src: int = 0, group: dist.ProcessGroup | None = None) -> list:
    group = group or dist.group.WORLD
    dist.broadcast_object_list(items, src=src, group=group)
    return items


def barrier(group: dist.ProcessGroup | None = None) -> None:
    group = group or dist.group.WORLD
    dist.barrier(group=group)
