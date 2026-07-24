"""System-level utilities (file descriptors, resource monitoring, etc.)."""

from __future__ import annotations

import logging
import os
import shutil
import sys
import time
from collections import Counter
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

SHM_PATH = "/dev/shm"  # noqa: S108 - system shared-memory mount.
_KIB_PER_GIB = 1024**2
_BYTES_PER_GIB = 1024**3
_PROC_ROOT = Path("/proc")

RUNTIME_PROCESS_GROUPS = (
    "openhands",
    "tmux",
    "raylet",
    "gcs_server",
    "sglang",
    "megatron_worker",
    "rollout_actor",
    "ray_worker",
    "orphan_multiprocessing",
)
_OPEN_FD_WARNING_THRESHOLD = 10_000

logger = logging.getLogger(__name__)


def get_open_fd_count() -> int:
    """Return the number of open file descriptors for the current process.

    Uses ``/proc/<pid>/fd`` on Linux.  Returns ``-1`` on non-Linux platforms
    or if the ``/proc`` filesystem is unavailable.
    """
    try:
        return len(list((_PROC_ROOT / str(os.getpid()) / "fd").iterdir()))
    except OSError:
        return -1


def get_open_fd_target_summary(*, limit: int = 12) -> dict[str, int]:
    """Return the most common open-FD target categories for this process."""
    fd_dir = _PROC_ROOT / str(os.getpid()) / "fd"
    counts: Counter[str] = Counter()
    try:
        entries = list(fd_dir.iterdir())
    except OSError:
        return {}
    for entry in entries:
        try:
            target = str(entry.readlink())
        except OSError:
            counts["<unreadable>"] += 1
            continue
        counts[_classify_fd_target(target)] += 1
    return dict(counts.most_common(limit))


def format_open_fd_target_summary(*, limit: int = 12) -> str:
    return format_fd_target_summary(get_open_fd_target_summary(limit=limit))


def format_fd_target_summary(summary: Mapping[str, int]) -> str:
    return ", ".join(f"{target}={count}" for target, count in summary.items()) or "unavailable"


def log_resource_usage_metrics(
    phase: str,
    *,
    metrics: dict[str, float] | None = None,
    prefix: str = "system",
    global_step: int | None = None,
    fd_target_limit: int = 24,
) -> dict[str, float]:
    metrics = get_resource_usage_metrics(prefix=prefix) if metrics is None else metrics
    fd_target_summary = format_fd_target_summary(get_open_fd_target_summary(limit=fd_target_limit))
    step_text = "" if global_step is None else f" global_step={global_step}"
    message = (
        f"Resource snapshot phase={phase}{step_text}: {format_resource_usage_summary(prefix=prefix, metrics=metrics)}; fd_targets={fd_target_summary}"
    )
    print(message, flush=True)
    logger.info(message)
    warn_resource_usage_metrics(metrics, phase=phase, prefix=prefix, global_step=global_step, fd_target_summary=fd_target_summary)
    return metrics


def warn_resource_usage_metrics(
    metrics: dict[str, float],
    *,
    phase: str,
    prefix: str = "system",
    global_step: int | None = None,
    fd_target_summary: str | None = None,
    open_fd_warning_threshold: int = _OPEN_FD_WARNING_THRESHOLD,
) -> None:
    global_step_text = "unknown" if global_step is None else str(global_step)
    if fd_target_summary is None:
        fd_target_summary = format_open_fd_target_summary(limit=24)

    open_fd_count = metrics.get(f"{prefix}/open_fd_count", -1.0)
    if open_fd_count >= open_fd_warning_threshold:
        logger.warning(
            "High open FD count phase=%s global_step=%s open_fd_count=%.0f fd_targets=%s.",
            phase,
            global_step_text,
            open_fd_count,
            fd_target_summary,
        )

    orphan_count = metrics.get(f"{prefix}/process_count__orphan_multiprocessing", 0.0)
    if orphan_count:
        logger.warning(
            "Orphan multiprocessing workers detected phase=%s global_step=%s count=%.0f rss=%.1fGiB fd_count=%.0f.",
            phase,
            global_step_text,
            orphan_count,
            metrics.get(f"{prefix}/process_rss_gib__orphan_multiprocessing", 0.0),
            metrics.get(f"{prefix}/process_fd_count__orphan_multiprocessing", 0.0),
        )

    zombie_count = metrics.get(f"{prefix}/process_zombie_count__all", 0.0)
    if zombie_count:
        logger.warning("Zombie processes detected phase=%s global_step=%s count=%.0f.", phase, global_step_text, zombie_count)

    for process_group in RUNTIME_PROCESS_GROUPS:
        process_fd_count = metrics.get(f"{prefix}/process_fd_count__{process_group}", 0.0)
        if process_fd_count >= open_fd_warning_threshold:
            logger.warning(
                "High grouped FD count phase=%s global_step=%s process_group=%s fd_count=%.0f process_count=%.0f.",
                phase,
                global_step_text,
                process_group,
                process_fd_count,
                metrics.get(f"{prefix}/process_count__{process_group}", 0.0),
            )


def get_resource_usage_metrics(*, prefix: str = "system") -> dict[str, float]:
    """Return driver-node resource metrics plus Ray cluster status."""
    start_time = time.perf_counter()
    unprefixed: dict[str, float] = {"open_fd_count": float(get_open_fd_count())}
    unprefixed.update(get_open_fd_target_metrics())
    unprefixed.update(get_process_memory_metrics())
    unprefixed.update(get_system_memory_metrics())
    unprefixed.update(get_disk_usage_metrics(SHM_PATH, name="shm"))
    unprefixed.update(get_runtime_process_metrics())
    unprefixed.update(get_ray_status_metrics())
    unprefixed["resource_snapshot_seconds"] = time.perf_counter() - start_time
    metrics = _prefix_metrics(prefix, unprefixed)
    return metrics


def get_open_fd_target_metrics(*, limit: int = 32) -> dict[str, float]:
    """Return scalar metrics for the most common open-FD target categories."""
    return {f"open_fd_target_count__{_metric_name(target)}": float(count) for target, count in get_open_fd_target_summary(limit=limit).items()}


def get_process_memory_metrics(*, pid: int | None = None) -> dict[str, float]:
    """Return RSS/virtual-memory metrics for one process in GiB."""
    pid = os.getpid() if pid is None else pid
    values = _read_status_kib(pid, ("VmRSS", "VmHWM", "VmSize"))
    return {
        "process_rss_gib": _kib_to_gib(values.get("VmRSS", 0)),
        "process_peak_rss_gib": _kib_to_gib(values.get("VmHWM", 0)),
        "process_vms_gib": _kib_to_gib(values.get("VmSize", 0)),
    }


def get_system_memory_metrics() -> dict[str, float]:
    """Return host memory metrics in GiB, separating pressure from page cache."""
    values = _read_meminfo_kib()
    total = values.get("MemTotal", 0)
    available = values.get("MemAvailable", 0)
    buffers = values.get("Buffers", 0)
    cached = values.get("Cached", 0)
    reclaimable = values.get("SReclaimable", 0)
    shmem = values.get("Shmem", 0)
    used_pressure = max(total - available, 0)
    buff_cache = buffers + cached + reclaimable
    return {
        "memory_total_gib": _kib_to_gib(total),
        "memory_available_gib": _kib_to_gib(available),
        "memory_pressure_used_gib": _kib_to_gib(used_pressure),
        "memory_buff_cache_gib": _kib_to_gib(buff_cache),
        "memory_shmem_gib": _kib_to_gib(shmem),
    }


def get_disk_usage_metrics(path: str, *, name: str) -> dict[str, float]:
    try:
        usage = shutil.disk_usage(path)
    except OSError:
        return {
            f"{name}_total_gib": -1.0,
            f"{name}_used_gib": -1.0,
            f"{name}_free_gib": -1.0,
        }
    return {
        f"{name}_total_gib": _bytes_to_gib(usage.total),
        f"{name}_used_gib": _bytes_to_gib(usage.used),
        f"{name}_free_gib": _bytes_to_gib(usage.free),
    }


def get_runtime_process_metrics() -> dict[str, float]:
    """Return local-node runtime process counts/RSS grouped by the names we care about."""
    summary = {name: _ProcessGroupStats() for name in RUNTIME_PROCESS_GROUPS}
    total_process_count = 0
    total_zombie_count = 0
    total_orphan_count = 0
    for process in _iter_process_infos():
        total_process_count += 1
        total_zombie_count += int(process.is_zombie)
        total_orphan_count += int(process.ppid == 1)
        name = _classify_runtime_process(process)
        if name is None:
            continue
        process_fd_count = get_process_fd_count(process.pid)
        summary[name].add(process, fd_count=process_fd_count)

    metrics: dict[str, float] = {
        "process_count__all": float(total_process_count),
        "process_zombie_count__all": float(total_zombie_count),
        "process_orphan_count__all": float(total_orphan_count),
    }
    for name, stats in summary.items():
        metrics.update(stats.to_metrics(name))
    return metrics


def _classify_runtime_process(process: _ProcessInfo) -> str | None:
    if process.ppid == 1 and ("multiprocessing.spawn" in process.cmdline or "multiprocessing.resource_tracker" in process.cmdline):
        return "orphan_multiprocessing"
    if process.comm == "openhands":
        return "openhands"
    if process.comm.startswith("tmux"):
        return "tmux"
    if process.comm == "raylet":
        return "raylet"
    if process.comm == "gcs_server":
        return "gcs_server"
    text = f"{process.comm} {process.cmdline}"
    if "RemoteMegatronWorker" in text:
        return "megatron_worker"
    if "RemoteRolloutActor" in text:
        return "rollout_actor"
    if process.comm.startswith("sglang") or "RemoteSGLangWorker" in text:
        return "sglang"
    if process.comm.startswith("ray::") or "default_worker.py" in process.cmdline:
        return "ray_worker"
    return None


def get_ray_status_metrics() -> dict[str, float]:
    """Return Ray cluster status if the driver is connected to Ray."""
    ray = sys.modules.get("ray")
    if ray is None or not ray.is_initialized():
        return {"ray_initialized": 0.0}

    try:
        cluster_resources = ray.cluster_resources()
        available_resources = ray.available_resources()
        nodes = ray.nodes()
    except Exception:
        return {"ray_initialized": 1.0, "ray_status_query_failed": 1.0}

    metrics = {
        "ray_initialized": 1.0,
        "ray_status_query_failed": 0.0,
        "ray_total_nodes": float(len(nodes)),
        "ray_alive_nodes": float(sum(1 for node in nodes if node.get("Alive"))),
    }
    for resource in ("CPU", "GPU"):
        total = float(cluster_resources.get(resource, 0.0))
        available = float(available_resources.get(resource, 0.0))
        key = resource.lower()
        metrics[f"ray_total_{key}"] = total
        metrics[f"ray_available_{key}"] = available
        metrics[f"ray_used_{key}"] = max(total - available, 0.0)
    for resource in ("memory", "object_store_memory"):
        total = float(cluster_resources.get(resource, 0.0))
        available = float(available_resources.get(resource, 0.0))
        key = f"ray_{resource}"
        metrics[f"{key}_total_gib"] = _bytes_to_gib(total)
        metrics[f"{key}_available_gib"] = _bytes_to_gib(available)
        metrics[f"{key}_used_gib"] = _bytes_to_gib(max(total - available, 0.0))
    return metrics


def format_resource_usage_summary(*, prefix: str = "system", metrics: dict[str, float] | None = None) -> str:
    metrics = get_resource_usage_metrics(prefix=prefix) if metrics is None else metrics
    return (
        f"fd={metrics.get(f'{prefix}/open_fd_count', -1):.0f} "
        f"snapshot={metrics.get(f'{prefix}/resource_snapshot_seconds', 0):.3f}s "
        f"rss={metrics.get(f'{prefix}/process_rss_gib', -1):.2f}GiB "
        f"mem_pressure={metrics.get(f'{prefix}/memory_pressure_used_gib', -1):.1f}GiB "
        f"mem_available={metrics.get(f'{prefix}/memory_available_gib', -1):.1f}GiB "
        f"shm_used={metrics.get(f'{prefix}/shm_used_gib', -1):.1f}GiB "
        f"ray_init={metrics.get(f'{prefix}/ray_initialized', 0):.0f} "
        f"ray_query_failed={metrics.get(f'{prefix}/ray_status_query_failed', 0):.0f} "
        f"ray_nodes={metrics.get(f'{prefix}/ray_alive_nodes', 0):.0f}/{metrics.get(f'{prefix}/ray_total_nodes', 0):.0f} "
        f"ray_gpu={metrics.get(f'{prefix}/ray_used_gpu', 0):.1f}/{metrics.get(f'{prefix}/ray_total_gpu', 0):.1f} "
        f"ray_store={metrics.get(f'{prefix}/ray_object_store_memory_used_gib', 0):.1f}GiB "
        f"openhands={metrics.get(f'{prefix}/process_count__openhands', 0):.0f} "
        f"tmux={metrics.get(f'{prefix}/process_count__tmux', 0):.0f} "
        f"zombie={metrics.get(f'{prefix}/process_zombie_count__all', 0):.0f} "
        f"orphan={metrics.get(f'{prefix}/process_orphan_count__all', 0):.0f} "
        f"orphan_mp={metrics.get(f'{prefix}/process_count__orphan_multiprocessing', 0):.0f} "
        f"orphan_mp_rss={metrics.get(f'{prefix}/process_rss_gib__orphan_multiprocessing', 0):.1f}GiB "
        f"ray_workers={metrics.get(f'{prefix}/process_count__ray_worker', 0):.0f} "
        f"ray_worker_fd={metrics.get(f'{prefix}/process_fd_count__ray_worker', 0):.0f}"
    )


def _classify_fd_target(target: str) -> str:
    if target.startswith("socket:"):
        return "socket"
    if target.startswith("pipe:"):
        return "pipe"
    if target.startswith("anon_inode:"):
        return target.split("]", 1)[0] + "]"
    if target.startswith("/tmp/ray/"):  # noqa: S108 - Ray diagnostic path.
        return "/tmp/ray"  # noqa: S108 - Ray diagnostic path.
    if target.startswith(f"{SHM_PATH}/"):
        return SHM_PATH
    if target.endswith(" (deleted)"):
        return "deleted-file"
    if target.startswith("/"):
        path = Path(target)
        if len(path.parts) >= 3:
            return str(Path(*path.parts[:3]))
        return str(path.parent)
    return target


def _metric_name(name: str) -> str:
    return name.strip("/").replace("/", "_").replace("[", "").replace("]", "").replace(":", "_").replace("-", "_") or "unknown"


@dataclass(frozen=True)
class _ProcessInfo:
    pid: int
    ppid: int
    comm: str
    cmdline: str
    state: str
    rss_kib: int

    @property
    def is_zombie(self) -> bool:
        return self.state.startswith("Z")


@dataclass
class _ProcessGroupStats:
    count: int = 0
    rss_kib: int = 0
    zombie_count: int = 0
    orphan_count: int = 0
    fd_count: int = 0
    fd_max: int = 0

    def add(self, process: _ProcessInfo, *, fd_count: int) -> None:
        self.count += 1
        self.rss_kib += process.rss_kib
        self.zombie_count += int(process.is_zombie)
        self.orphan_count += int(process.ppid == 1)
        if fd_count >= 0:
            self.fd_count += fd_count
            self.fd_max = max(self.fd_max, fd_count)

    def to_metrics(self, name: str) -> dict[str, float]:
        return {
            f"process_count__{name}": float(self.count),
            f"process_rss_gib__{name}": _kib_to_gib(self.rss_kib),
            f"process_zombie_count__{name}": float(self.zombie_count),
            f"process_orphan_count__{name}": float(self.orphan_count),
            f"process_fd_count__{name}": float(self.fd_count),
            f"process_fd_max__{name}": float(self.fd_max),
        }


def _iter_process_infos() -> Iterable[_ProcessInfo]:
    try:
        entries = list(_PROC_ROOT.iterdir())
    except OSError:
        return
    for entry in entries:
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        try:
            comm = (entry / "comm").read_text(encoding="utf-8").strip()
            cmdline = (entry / "cmdline").read_bytes().replace(b"\x00", b" ").decode(errors="replace")
            status_values = _read_status_fields(pid, ("PPid", "State", "VmRSS"))
        except OSError:
            continue
        ppid = int(status_values.get("PPid", 0))
        state = str(status_values.get("State", ""))
        rss_kib = int(status_values.get("VmRSS", 0))
        yield _ProcessInfo(pid=pid, ppid=ppid, comm=comm, cmdline=cmdline, state=state, rss_kib=rss_kib)


def get_process_fd_count(pid: int) -> int:
    try:
        return len(list((_PROC_ROOT / str(pid) / "fd").iterdir()))
    except OSError:
        return -1


def _read_meminfo_kib() -> dict[str, int]:
    values: dict[str, int] = {}
    try:
        lines = (_PROC_ROOT / "meminfo").read_text(encoding="utf-8").splitlines()
    except OSError:
        return values
    for line in lines:
        name, _, rest = line.partition(":")
        parts = rest.strip().split()
        if parts:
            with suppress(ValueError):
                values[name] = int(parts[0])
    return values


def _read_status_kib(pid: int, keys: tuple[str, ...]) -> dict[str, int]:
    values = _read_status_fields(pid, keys)
    return {key: int(value) for key, value in values.items() if isinstance(value, int)}


def _read_status_fields(pid: int, keys: tuple[str, ...]) -> dict[str, int | str]:
    wanted = set(keys)
    values: dict[str, int | str] = {}
    try:
        lines = (_PROC_ROOT / str(pid) / "status").read_text(encoding="utf-8").splitlines()
    except OSError:
        return values
    for line in lines:
        name, _, rest = line.partition(":")
        if name not in wanted:
            continue
        text = rest.strip()
        parts = text.split()
        if parts and parts[0].isdigit():
            values[name] = int(parts[0])
        else:
            values[name] = text
    return values


def _prefix_metrics(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}/{name}": value for name, value in metrics.items()}


def _kib_to_gib(value: int) -> float:
    return value / _KIB_PER_GIB


def _bytes_to_gib(value: float) -> float:
    return value / _BYTES_PER_GIB


if __name__ == "__main__":
    print(get_open_fd_count())
