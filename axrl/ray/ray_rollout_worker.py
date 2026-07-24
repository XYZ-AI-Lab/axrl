from __future__ import annotations

import asyncio
import contextlib
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast, override

import ray

from axrl.data import GenerationInput, GenerationOutput
from axrl.ray import ray_utils
from axrl.utils.timer import Timer
from axrl.worker.infer_router import InferenceRouter, _warn_if_event_timing_slow
from axrl.worker.infer_worker import InferWorker
from axrl.worker.sglang_worker import SGLangWorker

if TYPE_CHECKING:
    from ray.actor import ActorHandle

    from axrl.configs import RolloutWorkerConfig
    from axrl.ray.resource_group import ResourceGroup
    from axrl.utils.tensor_store import TensorHandle

logger = logging.getLogger(__name__)


@ray.remote
class RemoteSGLangWorker(SGLangWorker):
    pass


@dataclass(frozen=True)
class RolloutWorkerGroupLayout:
    gpu_counts_per_bundle: list[int]
    gpus_per_engine: int
    bundles_per_engine: int


class LocalRolloutWorkerGroup(InferenceRouter[GenerationInput, GenerationOutput]):
    """Local router that owns the remote SGLang engine actors."""

    def __init__(
        self,
        config: RolloutWorkerConfig,
        resource_group: ResourceGroup,
    ) -> None:
        super().__init__(max_imbalance=config.max_imbalance)
        self.config = config
        self.resource_group = resource_group
        self.num_engines = self.config.num_workers
        self.layout = self._build_worker_group_layout(config, resource_group)
        self.gpu_counts_per_bundle = self.layout.gpu_counts_per_bundle
        self.gpus_per_engine = self.layout.gpus_per_engine
        self.bundles_per_engine = self.layout.bundles_per_engine

    @staticmethod
    def _build_worker_group_layout(config: RolloutWorkerConfig, resource_group: ResourceGroup) -> RolloutWorkerGroupLayout:
        gpu_counts_per_bundle = [int(x.gpu) for x in resource_group.requests]
        assert gpu_counts_per_bundle, "At least one resource bundle is required to start rollout workers."
        num_engines = config.num_workers
        assert num_engines > 0, "`num_workers` must be greater than zero."
        total_gpus = sum(gpu_counts_per_bundle)
        assert total_gpus % num_engines == 0, f"Total GPUs ({total_gpus}) must be divisible by num_engines ({num_engines})"
        gpus_per_engine = total_gpus // num_engines
        assert gpus_per_engine == config.gpus_per_worker(), f"GPUs per engine ({gpus_per_engine}) must match model ({config.gpus_per_worker()=})"
        assert all(x > 0 for x in gpu_counts_per_bundle), f"Each bundle must have at least one GPU, got {gpu_counts_per_bundle}"
        assert len(set(gpu_counts_per_bundle)) == 1, f"All bundles must have the same number of GPUs, got {gpu_counts_per_bundle}"
        gpus_per_bundle = gpu_counts_per_bundle[0]
        assert gpus_per_engine % gpus_per_bundle == 0, f"GPUs per engine ({gpus_per_engine}) must be divisible by GPUs per bundle ({gpus_per_bundle})"
        return RolloutWorkerGroupLayout(
            gpu_counts_per_bundle=gpu_counts_per_bundle,
            gpus_per_engine=gpus_per_engine,
            bundles_per_engine=gpus_per_engine // gpus_per_bundle,
        )

    @override
    def initialize(self) -> None:
        rollout_workers: list[ActorHandle] = []
        init_handles = []
        for i in range(self.num_engines):
            bundle_indices = list(range(i * self.bundles_per_engine, (i + 1) * self.bundles_per_engine))
            init_handle, worker = self._start_remote_worker(bundle_indices)
            init_handles.append(init_handle)
            rollout_workers.append(worker)
        ray.get(init_handles)
        self._set_workers(rollout_workers)  # type: ignore[arg-type]
        logger.info(f"Started {len(self._remote_workers)} remote RolloutWorker engines.")

    def get_engine_handles(self) -> list[ActorHandle]:
        return list(self._remote_workers)

    def get_engine_handle(self, engine_index: int) -> ActorHandle:
        return cast("ActorHandle", self._remote_workers[engine_index])

    def get_config(self) -> RolloutWorkerConfig:
        return self.config

    def get_resource_group(self) -> ResourceGroup:
        return self.resource_group

    def get_gpus(self) -> list[int]:
        return list(self.gpu_counts_per_bundle)

    def get_gpus_per_engine(self) -> int:
        return self.gpus_per_engine

    def _start_remote_worker(self, bundle_indices: list[int]) -> tuple[ray.ObjectRef[None], ActorHandle]:
        master_addr, master_port = self._get_master_endpoint(bundle_indices[0])
        logger.info(f"Master endpoint for engine {master_addr}:{master_port}")
        nnodes = len(bundle_indices)
        remote_workers: list[ActorHandle] = []
        for node_rank, bundle_index in enumerate(bundle_indices):
            logger.info(f"Bundle index for engine: {bundle_index}, node rank: {node_rank}, nnodes: {nnodes}")
            cuda_visible_devices: str = self.resource_group.bundle_infos[bundle_index].cuda_visible_devices
            config = self.config.model_copy()
            config.master_addr = master_addr
            config.master_port = master_port
            config.nnodes = nnodes
            config.node_rank = node_rank
            config.name = f"{self.config.name}_engine_node_rank_{node_rank}"
            logger.info(f"Starting remote RolloutWorker with config: {config}")

            remote_worker = cast(
                "ActorHandle",
                RemoteSGLangWorker.options(  # type: ignore[attr-defined]
                    max_concurrency=max(1, config.max_running_requests),
                    runtime_env=ray_utils.get_runtime_envs(cuda_visible_devices),
                    scheduling_strategy=self.resource_group.get_scheduling_strategy(bundle_index),
                ).remote(config),
            )
            remote_workers.append(remote_worker)

        init_handles: list[ray.ObjectRef[None]] = [w.initialize.remote() for w in remote_workers]  # type: ignore[misc]
        init_handle: ray.ObjectRef[None] = init_handles[0]
        representative_worker = remote_workers[0]
        return init_handle, representative_worker

    def _get_master_endpoint(self, bundle_index: int) -> tuple[str, int]:
        address = self.resource_group.run_task(func=self.get_available_address, bundle_index=bundle_index)
        host, port_str = address.split(":")
        return host, int(port_str)

    async def pause_generation(self) -> None:
        with Timer("Paused generation on all rollout workers", verbose=True):
            refs = [w.pause_generation.remote() for w in self._remote_workers]
            await asyncio.gather(*refs)

    async def resume_generation(self) -> None:
        with Timer("Resumed generation on all rollout workers", verbose=True):
            refs = [w.resume_generation.remote() for w in self._remote_workers]
            await asyncio.gather(*refs)

    async def release_gpu_memory(self, *, backup_weights_on_cpu: bool = True) -> None:
        with Timer("Released GPU memory on all rollout workers", verbose=True):
            refs = [w.release_gpu_memory.remote(backup_weights_on_cpu=backup_weights_on_cpu) for w in self._remote_workers]
            await asyncio.gather(*refs)
        await self.clear_session_worker_mapping()

    async def resume_gpu_memory(self, tags: list[str] | None = None) -> None:
        with Timer("Resumed GPU memory on all rollout workers", verbose=True):
            refs = [w.resume_gpu_memory.remote(tags=tags) for w in self._remote_workers]
            await asyncio.gather(*refs)

    async def is_gpu_memory_released(self) -> bool:
        refs = [w.is_gpu_memory_released.remote() for w in self._remote_workers]
        results: list[bool] = await asyncio.gather(*refs)
        return all(results)

    async def warmup_tensor_store(self) -> list[TensorHandle]:
        """Ask every rollout worker to produce a warmup handle."""
        refs = [w.warmup_tensor_store.remote() for w in self._remote_workers]
        return list(await asyncio.gather(*refs))


@ray.remote(num_cpus=1)
class RemoteRayRolloutWorker(LocalRolloutWorkerGroup):
    """Ray actor wrapper around the local rollout worker group."""


class RayRolloutWorker(InferWorker[GenerationInput, GenerationOutput]):
    """Typed rollout worker wrapper over an initialized remote Ray actor.

    The remote actor owns routing state and SGLang engine handles. This local
    object exposes the rollout-worker API and forwards calls to that actor.
    """

    def __init__(
        self,
        actor: ActorHandle,
    ) -> None:
        super().__init__()
        self._actor = actor

    def get_engine_handles(self) -> list[ActorHandle]:
        return cast("list[ActorHandle]", ray.get(self._actor.get_engine_handles.remote()))

    def get_engine_handle(self, engine_index: int) -> ActorHandle:
        return cast("ActorHandle", ray.get(self._actor.get_engine_handle.remote(engine_index)))

    def get_config(self) -> RolloutWorkerConfig:
        return cast("RolloutWorkerConfig", ray.get(self._actor.get_config.remote()))

    def get_resource_group(self) -> ResourceGroup:
        return cast("ResourceGroup", ray.get(self._actor.get_resource_group.remote()))

    def get_gpus(self) -> list[int]:
        return cast("list[int]", ray.get(self._actor.get_gpus.remote()))

    def get_gpus_per_engine(self) -> int:
        return cast("int", ray.get(self._actor.get_gpus_per_engine.remote()))

    @override
    async def generate(self, req: GenerationInput) -> GenerationOutput:
        req.event_timing.mark_scheduled()
        result: GenerationOutput = await self._actor.generate.remote(req)
        result.event_timing.mark_driver_received()
        _warn_if_event_timing_slow(result.session_id, result.event_timing)
        return result

    async def pause_generation(self) -> None:
        await self._actor.pause_generation.remote()

    async def resume_generation(self) -> None:
        await self._actor.resume_generation.remote()

    async def release_gpu_memory(self, *, backup_weights_on_cpu: bool = True) -> None:
        await self._actor.release_gpu_memory.remote(backup_weights_on_cpu=backup_weights_on_cpu)

    async def resume_gpu_memory(self, tags: list[str] | None = None) -> None:
        await self._actor.resume_gpu_memory.remote(tags=tags)

    async def is_gpu_memory_released(self) -> bool:
        return await self._actor.is_gpu_memory_released.remote()

    async def flush_cache(self) -> None:
        await self._actor.flush_cache.remote()

    async def warmup_tensor_store(self) -> list[TensorHandle]:
        return await self._actor.warmup_tensor_store.remote()

    @override
    def shutdown(self) -> None:
        with contextlib.suppress(Exception):
            ray.get(self._actor.shutdown.remote(), timeout=30)
        ray.kill(self._actor, no_restart=True)

    @staticmethod
    def initialize_remote_actor(config: RolloutWorkerConfig, resource_group: ResourceGroup) -> ActorHandle:
        actor = cast(
            "ActorHandle",
            RemoteRayRolloutWorker.options(max_concurrency=max(1, config.max_running_requests * config.num_workers + 8)).remote(  # type: ignore[attr-defined]
                config,
                resource_group,
            ),
        )
        ray.get(actor.initialize.remote())
        return actor
