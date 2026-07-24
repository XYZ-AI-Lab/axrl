import contextlib
import logging
import os
import shutil
import subprocess
import time
from pathlib import Path

import ray
from torch_memory_saver.utils import get_binary_path_from_package

logger = logging.getLogger(__name__)


def _torch_memory_saver_preload_path() -> Path:
    return Path(get_binary_path_from_package("torch_memory_saver_hook_mode_preload"))


def get_runtime_envs(cuda_visible_devices: str, *, set_torch_memory_saver_envs: bool = False) -> dict[str, dict[str, str]]:
    runtime_env = {
        "env_vars": {
            # Manually control visible devices
            "CUDA_VISIBLE_DEVICES": cuda_visible_devices,
            "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
            # NCCL settings that help reduce memory fragmentation during multi-process workloads.
            "NCCL_CUMEM_ENABLE": "0",
            "NCCL_NVLS_ENABLE": "0",
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
            "NCCL_DEBUG": "WARN",
        }
    }
    if set_torch_memory_saver_envs:
        dynlib_path = _torch_memory_saver_preload_path()
        assert dynlib_path.exists(), f"Dynamic library {dynlib_path} does not exist"
        runtime_env["env_vars"].update(
            {
                "LD_PRELOAD": str(dynlib_path),
                "TMS_INIT_ENABLE": "1",
                "TMS_INIT_ENABLE_CPU_BACKUP": "1",
            }
        )
    return runtime_env


def set_ld_preload_for_current_process() -> None:
    import os

    dynlib_path = _torch_memory_saver_preload_path()
    assert dynlib_path.exists(), f"Dynamic library {dynlib_path} does not exist"
    os.environ["LD_PRELOAD"] = str(dynlib_path)


def kill_remote_workers(remote_workers: list) -> None:
    for worker in remote_workers:
        with contextlib.suppress(Exception):
            ray.kill(worker, no_restart=True)
    time.sleep(0.5)


def restart() -> None:
    if ray.is_initialized():
        ray.shutdown()
    time.sleep(1)
    ray.init()


def init_or_connect() -> None:
    if ray.is_initialized():
        return
    ray_address = os.environ.get("RAY_ADDRESS", "").strip()
    if ray_address:
        logger.info("Connecting to Ray cluster at RAY_ADDRESS=%s.", ray_address)
        ray.init(address=ray_address)
        return
    logger.warning("Ray is not initialized and RAY_ADDRESS is not set; initializing a local Ray runtime.")
    ray.init()


def stop() -> None:
    """Shut down the Ray client and force-kill the local Ray cluster.

    ``ray.shutdown()`` only disconnects this Python client from Ray; the
    raylet / GCS / ``ray::IDLE`` worker processes keep running under the
    test process's PID until it exits. So we also run
    ``ray stop --force`` to actually kill those processes. Use at the end
    of a test that called ``ray.init()`` (via ``restart()`` or otherwise)
    so the conftest leak assertion sees a clean state.
    """
    if ray.is_initialized():
        ray.shutdown()
    if shutil.which("ray") is not None:
        subprocess.run(
            ["ray", "stop", "--force"],  # noqa: S607
            check=False,
            capture_output=True,
            timeout=30,
        )
