import contextlib
import gc
import logging
import os
import time
from collections.abc import Callable, Generator
from dataclasses import asdict, dataclass
from functools import wraps
from typing import Literal

import torch

from axrl.utils.timer import Timer

logger = logging.getLogger(__name__)


def get_current_device() -> torch.device:
    """Get the current device."""
    if torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


def clear_cache(*, verbose: bool = False) -> None:
    with Timer("Clear Cache", verbose=verbose):
        gc.collect()
        torch.cuda.empty_cache()


@dataclass
class GpuUsageInfo:
    name: str
    cpu_time_s: float
    start_mem_gbs: float
    end_mem_gbs: float
    peak_mem_gbs: float
    start_mem_reserved_gbs: float
    end_mem_reserved_gbs: float
    peak_mem_reserved_gbs: float

    def to_metrics(self) -> dict[str, float]:
        kvs = asdict(self)
        del kvs["name"]
        return kvs

    def __repr__(self) -> str:
        s = (
            f"[{self.name}] CPU Time: {self.cpu_time_s:.4f}s | "
            f"Memory: {self.start_mem_gbs:.4f}GB -> {self.end_mem_gbs:.4f}GB (Peak: {self.peak_mem_gbs:.4f}GB) | "
            f"Reserved: {self.start_mem_reserved_gbs:.4f}GB -> "
            f"{self.end_mem_reserved_gbs:.4f}GB (Peak: {self.peak_mem_reserved_gbs:.4f}GB)"
        )
        return s


@dataclass
class InfoHandle:
    usage_info: GpuUsageInfo | None = None


@contextlib.contextmanager
def GpuUsageTracker(  # noqa: N802
    name: str = "gpu-usage", on_usage_info_ready: Callable[[GpuUsageInfo], None] | None = None
) -> Generator[InfoHandle, None, None]:
    """Context manager to record CPU time and GPU memory usage."""
    cpu_start = time.perf_counter()
    assert torch.cuda.is_available(), "CUDA is not available for profiling."
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start_mem = torch.cuda.memory_allocated()
    start_reserved = torch.cuda.memory_reserved()

    handle = InfoHandle()
    yield handle

    torch.cuda.synchronize()
    cpu_end = time.perf_counter()
    end_mem = torch.cuda.memory_allocated()
    reserved_mem = torch.cuda.memory_reserved()
    peak_mem = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()

    usage_info = GpuUsageInfo(
        name=name,
        cpu_time_s=cpu_end - cpu_start,
        start_mem_gbs=start_mem / (1024**3),
        end_mem_gbs=end_mem / (1024**3),
        peak_mem_gbs=peak_mem / (1024**3),
        start_mem_reserved_gbs=start_reserved / (1024**3),
        end_mem_reserved_gbs=reserved_mem / (1024**3),
        peak_mem_reserved_gbs=peak_reserved / (1024**3),
    )
    handle.usage_info = usage_info
    if on_usage_info_ready is not None:
        on_usage_info_ready(usage_info)


@dataclass
class GpuMemoryInfo:
    total_gb: float
    used_gb: float
    free_gb: float
    used_ratio: float

    def __repr__(self) -> str:
        return f"Used: {self.used_ratio:.2%} ({self.used_gb:.2f}GB / {self.total_gb:.2f}GB)"


def get_gpu_memory_info() -> list[GpuMemoryInfo]:
    """Get GPU memory information for all available GPUs."""
    assert torch.cuda.is_available(), "CUDA is not available."
    torch.cuda.synchronize()
    infos: list[GpuMemoryInfo] = []
    for i in range(torch.cuda.device_count()):
        free_mem, total_mem = torch.cuda.mem_get_info(torch.device("cuda", i))
        used_mem = total_mem - free_mem
        used_ratio = used_mem / total_mem if total_mem > 0 else 0.0
        info = GpuMemoryInfo(
            total_gb=total_mem / (1024**3),
            used_gb=used_mem / (1024**3),
            free_gb=free_mem / (1024**3),
            used_ratio=used_ratio,
        )
        infos.append(info)
    return infos


def assert_all_gpus_empty(max_used_gb: float = 2) -> None:
    assert torch.cuda.is_available(), "CUDA is not available."
    gpu_infos = get_gpu_memory_info()
    for rank, info in enumerate(gpu_infos):
        assert info.used_gb < max_used_gb, f"GPU {rank} is not empty. Used memory: {info.used_gb:.4f}GB"


def print_usage_info(usage_info: GpuUsageInfo) -> None:
    logger.info(
        f"[{usage_info.name}] CPU Time: {usage_info.cpu_time_s:.4f}s | "
        f"Memory: {usage_info.start_mem_gbs:.4f}GB -> {usage_info.end_mem_gbs:.4f}GB (Peak: {usage_info.peak_mem_gbs:.4f}GB) | "
        f"Reserved: {usage_info.start_mem_reserved_gbs:.4f}GB -> "
        f"{usage_info.end_mem_reserved_gbs:.4f}GB (Peak: {usage_info.peak_mem_reserved_gbs:.4f}GB)"
    )


def track_gpu_usage(
    func: Callable,
    name: str | None = None,
    on_usage_info_ready: Callable[[GpuUsageInfo], None] = print_usage_info,
) -> Callable:
    """Decorator that records CPU time and GPU memory for a function.

    Avoid profiling on very short functions to minimize profiling overhead.
    """

    @wraps(func)
    def wrapper(*args, **kwargs) -> None:  # noqa: ANN002, ANN003
        prof_name = name or func.__name__
        with GpuUsageTracker(name=prof_name, on_usage_info_ready=on_usage_info_ready):
            return func(*args, **kwargs)

    return wrapper


def log_gpu_memory_after_move(name: str, tags: list[str] | None, move_target: Literal["cpu", "gpu"], elapsed_seconds: float) -> None:
    local_rank_str = os.environ.get("LOCAL_RANK", "")
    if local_rank_str and int(local_rank_str) != 0:
        return
    memory_info = get_gpu_memory_info()
    logger.info(f"{name} moved {tags} to {move_target} in {elapsed_seconds:.2f}s. Memory Info: {memory_info}")
