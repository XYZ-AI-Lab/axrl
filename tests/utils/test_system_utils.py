from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from axrl.utils.system_utils import (
    format_open_fd_target_summary,
    format_resource_usage_summary,
    get_resource_usage_metrics,
    warn_resource_usage_metrics,
)

if TYPE_CHECKING:
    import pytest


def test_resource_usage_metrics_include_sigterm_diagnostics() -> None:
    metrics = get_resource_usage_metrics()

    required_keys = {
        "system/open_fd_count",
        "system/resource_snapshot_seconds",
        "system/process_rss_gib",
        "system/memory_available_gib",
        "system/memory_pressure_used_gib",
        "system/shm_used_gib",
        "system/process_count__openhands",
        "system/process_count__tmux",
        "system/process_zombie_count__all",
        "system/process_orphan_count__all",
        "system/process_count__orphan_multiprocessing",
        "system/process_fd_count__ray_worker",
        "system/ray_initialized",
    }

    assert required_keys <= metrics.keys()
    assert all(isinstance(metrics[key], float) for key in required_keys)


def test_resource_usage_summary_uses_reviewable_field_names() -> None:
    metrics = {
        "system/open_fd_count": 12.0,
        "system/resource_snapshot_seconds": 0.25,
        "system/process_rss_gib": 1.5,
        "system/memory_pressure_used_gib": 100.0,
        "system/memory_available_gib": 900.0,
        "system/shm_used_gib": 2.0,
        "system/ray_initialized": 1.0,
        "system/ray_status_query_failed": 0.0,
        "system/ray_alive_nodes": 2.0,
        "system/ray_total_nodes": 2.0,
        "system/ray_used_gpu": 16.0,
        "system/ray_total_gpu": 16.0,
        "system/ray_object_store_memory_used_gib": 3.0,
        "system/process_count__openhands": 4.0,
        "system/process_count__tmux": 1.0,
        "system/process_zombie_count__all": 0.0,
        "system/process_orphan_count__all": 3.0,
        "system/process_count__orphan_multiprocessing": 0.0,
        "system/process_rss_gib__orphan_multiprocessing": 0.0,
        "system/process_count__ray_worker": 8.0,
        "system/process_fd_count__ray_worker": 128.0,
    }

    summary = format_resource_usage_summary(metrics=metrics)

    for field in (
        "fd=12",
        "snapshot=0.250s",
        "rss=1.50GiB",
        "mem_pressure=100.0GiB",
        "mem_available=900.0GiB",
        "shm_used=2.0GiB",
        "ray_init=1",
        "ray_query_failed=0",
        "ray_nodes=2/2",
        "ray_gpu=16.0/16.0",
        "openhands=4",
        "tmux=1",
        "zombie=0",
        "orphan=3",
        "orphan_mp=0",
        "ray_workers=8",
        "ray_worker_fd=128",
    ):
        assert field in summary


def test_open_fd_target_summary_is_available() -> None:
    assert isinstance(format_open_fd_target_summary(), str)


def test_resource_usage_warning_flags_debuggable_resource_risks(caplog: pytest.LogCaptureFixture) -> None:
    metrics = {
        "system/open_fd_count": 10_001.0,
        "system/process_count__orphan_multiprocessing": 1.0,
        "system/process_rss_gib__orphan_multiprocessing": 2.0,
        "system/process_fd_count__orphan_multiprocessing": 3.0,
        "system/process_zombie_count__all": 1.0,
        "system/process_fd_count__ray_worker": 10_001.0,
        "system/process_count__ray_worker": 2.0,
    }
    caplog.set_level(logging.WARNING, logger="axrl.utils.system_utils")

    warn_resource_usage_metrics(metrics, phase="unit-test", global_step=7, fd_target_summary="socket=10001")

    messages = "\n".join(record.getMessage() for record in caplog.records)
    assert "High open FD count" in messages
    assert "Orphan multiprocessing workers detected" in messages
    assert "Zombie processes detected" in messages
    assert "High grouped FD count" in messages
