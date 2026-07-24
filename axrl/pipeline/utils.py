from __future__ import annotations

import contextlib
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import ray

from axrl.data import RolloutResult
from axrl.ray.resource_group import Request, ResourceGroup
from axrl.trainer.grpo_trainer import GrpoTrainer
from axrl.trainer.sft_trainer import SftTrainer
from axrl.trainer.value_trainer import ValueTrainer

if TYPE_CHECKING:
    from axrl.agent.base_agent import BaseAgent
    from axrl.configs import MegatronWorkerConfig, RolloutWorkerConfig, SamplingConfig
    from axrl.envs.base_env import BaseEnv
    from axrl.pipeline.config import PipelineExperimentConfig, PipelineRunMode
    from axrl.processor.base_processor import BaseProcessor
    from axrl.ray.ray_infer_worker import RayInferWorker
    from axrl.ray.ray_megatron_worker import RayMegatronWorker
    from axrl.ray.ray_rollout_worker import RayRolloutWorker
    from axrl.trainer.base_trainer import BaseTrainer
    from axrl.utils.ray_task_queue import RayTaskQueue
    from axrl.worker.infer_worker import InferWorker


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PipelineWorkerPlacement:
    rollout: ResourceGroup | None
    megatron: ResourceGroup | None
    value: ResourceGroup | None = None


def needs_rollout_worker(run_mode: PipelineRunMode) -> bool:
    return run_mode in ("online_rl_train", "eval_only", "mismatch_test")


def needs_megatron_worker(run_mode: PipelineRunMode) -> bool:
    return run_mode in ("online_rl_train", "replay_rl_train", "sft_train", "mismatch_test")


def needs_weight_sync(run_mode: PipelineRunMode) -> bool:
    return run_mode in ("online_rl_train", "mismatch_test")


def needs_value_worker(config: PipelineExperimentConfig) -> bool:
    return config.controller.run_mode == "online_rl_train" and config.grpo.ppo_value is not None


def is_rollout_only_mode(run_mode: PipelineRunMode) -> bool:
    return run_mode == "eval_only"


def build_trainer(config: PipelineExperimentConfig) -> BaseTrainer:
    if config.controller.run_mode == "sft_train":
        return SftTrainer(config.sft)
    return GrpoTrainer(config.grpo)


def build_value_trainer(config: PipelineExperimentConfig) -> BaseTrainer:
    assert config.grpo.ppo_value is not None, "PPO value trainer requires grpo.ppo_value."
    return ValueTrainer(config.grpo.ppo_value)


def get_rollout_worker_config(config: PipelineExperimentConfig) -> RolloutWorkerConfig:
    rollout_config = config.rollout_worker.model_copy(deep=True)
    # Keep eval concurrency controlled by the controller for now.
    rollout_config.max_running_requests = config.controller.max_running_requests
    rollout_config.max_running_requests_eval = None
    if config.controller.run_mode == "eval_only" and config.eval_only.model is not None:
        rollout_config.model = config.eval_only.model
        logger.info("Using eval-only rollout model override: %s.", rollout_config.model.name)
    logger.info(
        "Using controller.max_running_requests=%d for rollout worker request concurrency.",
        rollout_config.max_running_requests,
    )
    return rollout_config


async def rollout_from_env(env: BaseEnv, agent: BaseAgent, sampling_config: SamplingConfig) -> RolloutResult:
    observation = env.conv
    assert observation is not None and observation.gen_state.input_ids is not None
    while True:
        generation_output = await agent.act(observation, sampling_config)
        observation, _, done, sample, response_metric = await env.step(generation_output)
        if not done:
            continue
        return RolloutResult(conversation=observation, trace=sample, metric=response_metric)


def shutdown_local_workers(local_workers: dict[str, InferWorker[Any, Any]]) -> None:
    seen: set[int] = set()
    for worker in local_workers.values():
        worker_id = id(worker)
        if worker_id in seen:
            continue
        seen.add(worker_id)
        worker.shutdown()


def shutdown_shared_workers(shared_workers: dict[str, RayRolloutWorker | RayInferWorker[Any, Any]]) -> None:
    seen: set[int] = set()
    for worker in shared_workers.values():
        worker_id = id(worker)
        if worker_id in seen:
            continue
        seen.add(worker_id)
        worker.shutdown()


def shutdown_pipeline_workers(
    *,
    shared_workers: dict[str, RayRolloutWorker | RayInferWorker[Any, Any]] | None,
    rollout_worker: RayRolloutWorker | None,
    megatron_worker: RayMegatronWorker | None,
    value_worker: RayMegatronWorker | None = None,
) -> None:
    with contextlib.suppress(Exception):
        if shared_workers is not None:
            shutdown_shared_workers(shared_workers)

    rollout_group = rollout_worker.get_resource_group() if rollout_worker is not None else None
    megatron_group = megatron_worker.resource_group if megatron_worker is not None else None
    value_group = value_worker.resource_group if value_worker is not None else None
    with contextlib.suppress(Exception):
        if rollout_worker is not None:
            rollout_worker.shutdown()
    with contextlib.suppress(Exception):
        if megatron_worker is not None:
            megatron_worker.shutdown()
    with contextlib.suppress(Exception):
        if value_worker is not None:
            value_worker.shutdown()

    seen: set[str] = set()
    for group in (rollout_group, megatron_group, value_group):
        if group is None:
            continue
        group_id = str(group.pg.id)
        if group_id in seen:
            continue
        seen.add(group_id)
        with contextlib.suppress(Exception):
            group.shutdown()


def shutdown_task_queue(queue: RayTaskQueue[Any] | None) -> None:
    with contextlib.suppress(Exception):
        if queue is not None:
            ray.kill(queue.get_actor_handle(), no_restart=True)


def start_ray_infer_worker[InT, OutT](
    processor_cls: type[BaseProcessor[InT, OutT]],
    config: Any,
    *,
    num_processes: int,
    num_cpus: int | None = None,
    timeout_seconds: float = 60.0,
) -> RayInferWorker[InT, OutT]:
    from axrl.ray.ray_infer_worker import RayInferWorker

    remote_actor = RayInferWorker.initialize_remote_actor(
        processor_cls,
        config=config,
        num_processes=num_processes,
        num_cpus=num_cpus,
        timeout_seconds=timeout_seconds,
    )
    worker = RayInferWorker[InT, OutT](remote_actor)
    logger.info("%s Ray infer worker started with %d processes.", processor_cls.__name__, num_processes)
    return worker


async def create_rollout_worker(
    config: RolloutWorkerConfig,
    resource_group: ResourceGroup,
    *,
    move_to_cpu_after_init: bool,
) -> RayRolloutWorker:
    from axrl.ray.ray_rollout_worker import RayRolloutWorker

    rollout_worker = RayRolloutWorker(RayRolloutWorker.initialize_remote_actor(config, resource_group))
    if move_to_cpu_after_init:
        await rollout_worker.release_gpu_memory(backup_weights_on_cpu=False)
        logger.info("Moved rollout worker GPU memory to CPU/offloaded state after initialization.")
    return rollout_worker


def create_megatron_worker(config: MegatronWorkerConfig, resource_group: ResourceGroup, *, move_to_cpu_after_init: bool) -> RayMegatronWorker:
    from axrl.ray.ray_megatron_worker import RayMegatronWorker

    megatron_worker = RayMegatronWorker(config, resource_group=resource_group)
    megatron_worker.initialize()
    megatron_worker.copy_weights_to_cpu(name="init_weights")
    if move_to_cpu_after_init:
        megatron_worker.to_cpu()
    return megatron_worker


def rollout_total_gpus(config: RolloutWorkerConfig) -> int:
    return config.gpus_per_worker() * config.num_workers


def rollout_resource_requests(config: RolloutWorkerConfig) -> list[Request]:
    gpu_per_worker = config.gpus_per_worker()
    return [Request(cpu=1, gpu=gpu_per_worker) for _ in range(config.num_workers)]


def megatron_resource_requests(config: MegatronWorkerConfig) -> list[Request]:
    return [Request(cpu=1, gpu=1) for _ in range(config.world_size())]


def required_cluster_gpus(
    *,
    rollout_worker: RolloutWorkerConfig,
    megatron_worker: MegatronWorkerConfig,
    colocated: bool,
    needs_rollout: bool,
    needs_megatron: bool,
) -> int:
    rollout_gpus = rollout_total_gpus(rollout_worker) if needs_rollout else 0
    megatron_gpus = megatron_worker.world_size() if needs_megatron else 0
    if needs_rollout and needs_megatron and colocated:
        return max(rollout_gpus, megatron_gpus)
    return rollout_gpus + megatron_gpus


def assert_cluster_has_gpus(required_gpus: int, *, run_mode: str) -> None:
    assert ray.is_initialized(), "Pipeline worker placement requires an initialized Ray driver."
    cluster_gpus = int(ray.cluster_resources().get("GPU", 0))
    assert cluster_gpus >= required_gpus, f"Need at least {required_gpus} GPUs for run_mode={run_mode!r}, but Ray reports {cluster_gpus} GPUs."
    logger.info(f"Worker placement for {run_mode}: required_gpus={required_gpus}, cluster_gpus={cluster_gpus}.")


def get_worker_placement(
    *,
    rollout_worker: RolloutWorkerConfig,
    megatron_worker: MegatronWorkerConfig,
    run_mode: str,
    colocated: bool,
    needs_rollout: bool,
    needs_megatron: bool,
) -> PipelineWorkerPlacement:
    assert needs_rollout or needs_megatron, f"run_mode={run_mode!r} does not require any workers."

    rollout_gpus = rollout_total_gpus(rollout_worker)
    megatron_gpus = megatron_worker.world_size()
    if needs_rollout and needs_megatron:
        assert rollout_gpus == megatron_gpus, (
            f"Rollout and Megatron workers must use the same number of GPUs for weight-sync compatibility, "
            f"got rollout_gpus={rollout_gpus}, megatron_gpus={megatron_gpus}."
        )

    required_gpus = required_cluster_gpus(
        rollout_worker=rollout_worker,
        megatron_worker=megatron_worker,
        colocated=colocated,
        needs_rollout=needs_rollout,
        needs_megatron=needs_megatron,
    )
    assert_cluster_has_gpus(required_gpus, run_mode=run_mode)

    if needs_rollout and needs_megatron and colocated:
        resource_group = ResourceGroup(rollout_resource_requests(rollout_worker))
        return PipelineWorkerPlacement(rollout=resource_group, megatron=resource_group)

    rollout_resource_group = ResourceGroup(rollout_resource_requests(rollout_worker)) if needs_rollout else None
    if not needs_megatron:
        return PipelineWorkerPlacement(rollout=rollout_resource_group, megatron=None)

    megatron_requests = rollout_resource_requests(rollout_worker) if needs_rollout else megatron_resource_requests(megatron_worker)
    megatron_resource_group = ResourceGroup(megatron_requests)
    return PipelineWorkerPlacement(rollout=rollout_resource_group, megatron=megatron_resource_group)


def get_pipeline_worker_placement(config: PipelineExperimentConfig) -> PipelineWorkerPlacement:
    controller_config = config.controller
    rollout_worker = get_rollout_worker_config(config)
    placement = get_worker_placement(
        rollout_worker=rollout_worker,
        megatron_worker=config.megatron_worker,
        run_mode=controller_config.run_mode,
        colocated=controller_config.colocated,
        needs_rollout=needs_rollout_worker(controller_config.run_mode),
        needs_megatron=needs_megatron_worker(controller_config.run_mode),
    )
    if needs_value_worker(config):
        assert config.value_worker is not None, "PPO value_worker config must be populated before placement."
        value_world_size = config.value_worker.world_size()
        assert value_world_size == config.megatron_worker.world_size(), (
            f"PPO value_worker world_size ({value_world_size}) must match actor world_size ({config.megatron_worker.world_size()})."
        )
        assert placement.megatron is not None
        return PipelineWorkerPlacement(rollout=placement.rollout, megatron=placement.megatron, value=placement.megatron)
    return placement


async def initialize_pipeline_workers(
    config: PipelineExperimentConfig,
) -> tuple[RayRolloutWorker | None, RayMegatronWorker | None, RayMegatronWorker | None]:
    placement = get_pipeline_worker_placement(config)
    run_mode = config.controller.run_mode
    rollout_config = get_rollout_worker_config(config)

    rollout_worker: RayRolloutWorker | None = None
    if needs_rollout_worker(run_mode):
        assert placement.rollout is not None
        rollout_worker = await create_rollout_worker(
            rollout_config,
            placement.rollout,
            move_to_cpu_after_init=needs_megatron_worker(run_mode),
        )

    megatron_worker: RayMegatronWorker | None = None
    if needs_megatron_worker(run_mode):
        assert placement.megatron is not None
        megatron_worker = create_megatron_worker(
            config.megatron_worker,
            placement.megatron,
            move_to_cpu_after_init=needs_rollout_worker(run_mode),
        )
        megatron_worker.set_trainer(build_trainer(config))

    value_worker: RayMegatronWorker | None = None
    if needs_value_worker(config):
        assert config.value_worker is not None
        assert placement.value is not None
        value_worker = create_megatron_worker(
            config.value_worker,
            placement.value,
            move_to_cpu_after_init=needs_rollout_worker(run_mode),
        )
        value_worker.set_trainer(build_value_trainer(config))

    if needs_weight_sync(run_mode):
        assert rollout_worker is not None
        assert megatron_worker is not None
        megatron_worker.build_weight_updater(rollout_worker, bucket_size_gb=1.0)
        megatron_worker.connect_rollout_worker()

    return rollout_worker, megatron_worker, value_worker
