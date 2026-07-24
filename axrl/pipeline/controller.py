from __future__ import annotations

import asyncio
import gc
import logging
import math
import random
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast, override

import numpy as np
import ray
import torch
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
from tqdm import tqdm

from axrl.controller.base_controller import BaseController
from axrl.controller.stage_manager import ColocatedStageManager, DisaggregatedStageManager, StageManager
from axrl.data import Conversation, RolloutResult
from axrl.data.sample import _concat_sample_tensor_dicts
from axrl.metrics.response_metric import aggregate_response_metrics, aggregate_response_metrics_by_subset
from axrl.opd.teacher_logprobs import aggregate_teacher_metrics
from axrl.pipeline.rollout_actor import RolloutActor
from axrl.pipeline.rollout_data import GroupFilterType, RolloutGroup, RolloutRuntime, TrainGroupBatch
from axrl.pipeline.utils import (
    assert_cluster_has_gpus,
    get_pipeline_worker_placement,
    initialize_pipeline_workers,
    needs_megatron_worker,
    needs_rollout_worker,
    required_cluster_gpus,
    rollout_resource_requests,
    rollout_total_gpus,
    shutdown_pipeline_workers,
    shutdown_task_queue,
)
from axrl.ray import ray_utils
from axrl.ray.resource_group import ResourceGroup
from axrl.trainer.ppo_utils import build_terminal_token_rewards, compute_gae, normalize_over_valid_tokens_in_batch
from axrl.utils import zst_utils
from axrl.utils.logger import get_metric_logger
from axrl.utils.ray_task_queue import RayTaskQueue
from axrl.utils.system_utils import log_resource_usage_metrics
from axrl.utils.timer import Timer

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, AsyncIterator, Sequence

    from axrl.configs import DatasetConfig, GrpoTrainerConfig, MegatronWorkerConfig
    from axrl.data import SampleTensorDict
    from axrl.datasets.base_dataset import BaseDataset
    from axrl.pipeline.config import PipelineExperimentConfig
    from axrl.pipeline.utils import PipelineWorkerPlacement
    from axrl.ray.ray_infer_worker import RayInferWorker
    from axrl.ray.ray_megatron_worker import RayMegatronWorker
    from axrl.ray.ray_rollout_worker import RayRolloutWorker
    from axrl.recipe.base_recipe import BaseRecipe
    from axrl.utils.logger import MetricLogger
    from axrl.utils.tensor_store import TensorHandle
    from axrl.worker.infer_worker import InferWorker


logger = logging.getLogger(__name__)
_RESOURCE_MONITOR_INTERVAL_SECONDS = 300.0
# Health-check cadence while waiting for rollout results; not a per-rollout deadline.
_ROLLOUT_HEALTH_CHECK_INTERVAL_SECONDS = 30.0


class PipelineController(BaseController):
    """Generic pipeline driver backed by a required recipe object."""

    def __init__(self, config: PipelineExperimentConfig, recipe: BaseRecipe) -> None:
        super().__init__()
        self.config = config
        self.recipe = recipe
        self.rollout_worker: RayRolloutWorker | None = None
        self.megatron_worker: RayMegatronWorker | None = None
        self.value_worker: RayMegatronWorker | None = None
        self.train_dataset: BaseDataset | None = None
        self.test_datasets: list[BaseDataset] = []
        self.eval_dataset_configs: list[DatasetConfig] = []
        self.shared_workers: dict[str, RayRolloutWorker | RayInferWorker[Any, Any]] | None = None
        self.recipe_services: dict[str, Any] = {}
        self.rollout_queue: RayTaskQueue[Conversation] | None = None
        self.result_queue: RayTaskQueue[RolloutResult] | None = None
        self.rollout_runtime: RolloutRuntime | None = None
        self.rollout_actors: list[RolloutActor] = []
        self.global_step = 0
        self.value_step = 0
        self.output_dir: Path | None = None
        self.checkpoint_dir: Path | None = None
        self.metric_logger: MetricLogger | None = None
        self._packing_pool: InferWorker[Any, SampleTensorDict] | None = None
        self.stage_manager: StageManager | None = None
        self.prev_eval_global_step = 0
        self._rollout_schedule_id = 0
        self._resource_monitor_stop_event = threading.Event()
        self._resource_monitor_thread: threading.Thread | None = None
        self._latest_resource_metrics: dict[str, float] | None = None
        self._opd_teacher_weights_initialized = False

    @override
    async def initialize(self) -> None:
        logger.info("Pipeline initialize begin: run_mode=%s.", self.config.controller.run_mode)
        with Timer("Pipeline initialize: ray init/connect", verbose=True):
            ray_utils.init_or_connect()
        self.global_step = 0
        self.value_step = 0
        self.prev_eval_global_step = 0
        self._rollout_schedule_id = 0
        with Timer("Pipeline initialize: eval-only model override", verbose=True):
            self._apply_eval_only_model_override()
        with Timer("Pipeline initialize: register datasets", verbose=True):
            await self.recipe.register_datasets()
        with Timer("Pipeline initialize: check configs", verbose=True):
            self.check_configs()
        with Timer("Pipeline initialize: output dir", verbose=True):
            self._initialize_output_dir()
        with Timer("Pipeline initialize: metric logger", verbose=True):
            self._initialize_logger()
        with Timer("Pipeline initialize: recipe services", verbose=True):
            self.recipe_services = await self.recipe.start_services()
        with Timer("Pipeline initialize: OPD teacher services", verbose=True):
            await self.start_opd_teacher_services()
        with Timer("Pipeline initialize: pipeline workers", verbose=True):
            await self.initialize_workers()
        with Timer("Pipeline initialize: Megatron teacher weights", verbose=True):
            self.initialize_megatron_teacher_weights()
        with Timer("Pipeline initialize: stage manager", verbose=True):
            self.initialize_stage_manager()
        with Timer("Pipeline initialize: datasets", verbose=True):
            self.initialize_datasets()
        with Timer("Pipeline initialize: rollout runtime", verbose=True):
            await self.initialize_rollout_runtime()
        self.checkpoint_dir = self.config.megatron_worker.get_checkpoint_dir()
        with Timer("Pipeline initialize: resource monitor", verbose=True):
            self.start_resource_monitor()
        logger.info("Pipeline initialize done: run_mode=%s global_step=%d.", self.config.controller.run_mode, self.global_step)

    def _apply_eval_only_model_override(self) -> None:
        if self.config.controller.run_mode != "eval_only" or self.config.eval_only.model is None:
            return
        rollout_config = self.config.rollout_worker.model_copy(deep=True)
        rollout_config.model = self.config.eval_only.model
        self.config.rollout_worker = rollout_config

    @staticmethod
    def _check_ppo_value_configs(
        *,
        grpo_config: GrpoTrainerConfig,
        megatron_config: MegatronWorkerConfig,
        value_config: MegatronWorkerConfig | None,
    ) -> None:
        ppo_value_config = grpo_config.ppo_value
        if grpo_config.loss_type == "ppo":
            assert ppo_value_config is not None, "grpo.loss_type='ppo' requires grpo.ppo_value."
            assert value_config is not None, "grpo.loss_type='ppo' requires value_worker."
            assert value_config.model_role == "value", "PPO value_worker.model_role must be 'value'."
            assert megatron_config.model_role == "actor", "PPO megatron_worker.model_role must be 'actor'."
            assert value_config.world_size() == megatron_config.world_size(), (
                f"PPO value_worker world_size ({value_config.world_size()}) must match actor world_size ({megatron_config.world_size()})."
            )
            assert value_config.model.seq_length == megatron_config.model.seq_length, (
                f"PPO value_worker seq_length ({value_config.model.seq_length}) must match actor seq_length ({megatron_config.model.seq_length})."
            )
            assert value_config.checkpoint_dir != megatron_config.checkpoint_dir, (
                "PPO actor and value workers must use different checkpoint_dir values."
            )
            assert grpo_config.is_base_logprobs == "old_logprobs", "PPO actor loss must use old_logprobs as the policy-ratio base."

        if ppo_value_config is not None:
            assert grpo_config.loss_type == "ppo", "grpo.ppo_value is only supported with grpo.loss_type='ppo'."
            assert value_config is not None, "grpo.ppo_value requires value_worker."
            assert not ppo_value_config.use_stateless_value_model, "grpo.ppo_value.use_stateless_value_model is reserved and not implemented yet."

    def check_configs(self) -> None:
        controller_config = self.config.controller
        run_mode = controller_config.run_mode
        if self.config.grpo.loss_type == "ppo" or self.config.grpo.ppo_value is not None:
            assert run_mode == "online_rl_train", "Pipeline PPO/value_worker currently supports controller.run_mode='online_rl_train' only."
        if needs_rollout_worker(run_mode):
            assert controller_config.max_running_requests >= controller_config.num_rollout_actors, (
                f"max_running_requests ({controller_config.max_running_requests}) must be greater than or equal to "
                f"num_rollout_actors ({controller_config.num_rollout_actors}) so every rollout actor can block on the queue."
            )

        if run_mode in ("online_rl_train", "mismatch_test"):
            online_config = self.config.online_rl_train
            grpo_config = self.config.grpo
            megatron_config = self.config.megatron_worker
            value_config = self.config.value_worker
            assert online_config.model_sync_every_n_global_updates > 0, "model_sync_every_n_global_updates must be positive."
            assert online_config.batch_rollout_for_n_global_updates > 0, "batch_rollout_for_n_global_updates must be positive."
            assert online_config.reward_mean_type == "group", "Pipeline GRPO currently supports reward_mean_type='group'."
            assert online_config.reward_std_type == "group", "Pipeline GRPO currently supports reward_std_type='group'."
            assert not grpo_config.normalize_advantage_by_batch_std, "Batch advantage std normalization is not supported in the pipeline controller."
            assert grpo_config.turn_reward_alpha == 0.0, "Turn reward normalization is disabled for packed rollout samples."
            checkpoint_interval = online_config.checkpoint_every_n_global_updates
            if checkpoint_interval is not None:
                assert checkpoint_interval > 0, "checkpoint_every_n_global_updates must be positive when checkpointing is enabled."
                assert checkpoint_interval % online_config.model_sync_every_n_global_updates == 0
            if online_config.rollout_save_every_n_global_updates is not None:
                assert online_config.rollout_save_every_n_global_updates % online_config.model_sync_every_n_global_updates == 0, (
                    f"rollout_save_every_n_global_updates ({online_config.rollout_save_every_n_global_updates}) must be divisible by "
                    f"model_sync_every_n_global_updates ({online_config.model_sync_every_n_global_updates})."
                )
            if megatron_config.reset_init_weights_every_k_steps is not None:
                assert megatron_config.reset_init_weights_every_k_steps % online_config.model_sync_every_n_global_updates == 0, (
                    f"reset_init_weights_every_k_steps ({megatron_config.reset_init_weights_every_k_steps}) must be divisible by "
                    f"model_sync_every_n_global_updates ({online_config.model_sync_every_n_global_updates})."
                )
            if online_config.strict_on_policy:
                assert online_config.model_sync_every_n_global_updates == 1, "strict_on_policy requires model_sync_every_n_global_updates=1."
                assert online_config.batch_rollout_for_n_global_updates == 1, "strict_on_policy requires batch_rollout_for_n_global_updates=1."

            self._check_ppo_value_configs(
                grpo_config=grpo_config,
                megatron_config=megatron_config,
                value_config=value_config,
            )

            opd_config = grpo_config.opd
            if opd_config.enabled:
                assert opd_config.teacher_model is not None, "OPD requires grpo.opd.teacher_model."
                if opd_config.backend == "sglang":
                    assert needs_rollout_worker(run_mode), f"OPD backend='sglang' requires rollout workers for run_mode={run_mode!r}."
                elif opd_config.backend == "megatron":
                    assert needs_megatron_worker(run_mode), f"OPD backend='megatron' requires a Megatron worker for run_mode={run_mode!r}."
                else:
                    raise AssertionError(f"Unsupported OPD backend: {opd_config.backend!r}.")

        test_dataset_configs = self.get_test_dataset_configs()
        if run_mode == "eval_only":
            assert test_dataset_configs, "eval_only requires non-empty test_datasets."
        for dataset_config in test_dataset_configs or []:
            if dataset_config.eval_num_rollouts_per_prompt is not None:
                assert dataset_config.eval_num_rollouts_per_prompt > 0, (
                    f"eval_num_rollouts_per_prompt must be positive when set for dataset {dataset_config.name!r}."
                )

    def get_train_dataset_configs(self) -> Sequence[DatasetConfig] | None:
        recipe_configs = self.recipe.get_train_dataset_configs()
        return self.config.train_datasets if recipe_configs is None else recipe_configs

    def get_test_dataset_configs(self) -> Sequence[DatasetConfig] | None:
        recipe_configs = self.recipe.get_test_dataset_configs()
        return self.config.test_datasets if recipe_configs is None else recipe_configs

    def initialize_datasets(self) -> None:
        self.train_dataset = None
        self.test_datasets = []
        self.eval_dataset_configs = []
        train_dataset_configs = self.get_train_dataset_configs()
        if train_dataset_configs is not None and self.megatron_worker is not None:
            from axrl.datasets.base_dataset import BaseDataset

            self.train_dataset = BaseDataset.concat(self._prepare_datasets(train_dataset_configs))
            if self.config.online_rl_train.max_prompt_length is not None:
                self.train_dataset.filter_by_max_prompt_length(self.config.online_rl_train.max_prompt_length)
        test_dataset_configs = self.get_test_dataset_configs()
        if test_dataset_configs and (self.rollout_worker is not None or self.config.controller.run_mode == "sft_train"):
            self._initialize_eval_datasets(test_dataset_configs)

    def _prepare_datasets(self, items: Sequence[DatasetConfig] | None) -> list[BaseDataset]:
        """Create datasets with recipe-specific pre/post initialization hooks."""
        from axrl.datasets import get_dataset

        datasets: list[BaseDataset] = []
        for cfg in items or []:
            dataset = get_dataset(cfg)
            # Some recipes need to inject prompt/tool metadata before
            # initialize() builds Conversation objects; search-agent Hermes
            # tool schemas are the motivating case.
            self.recipe.prepare_dataset(dataset, cfg)
            dataset.initialize()
            self.recipe.validate_dataset(dataset, cfg)
            datasets.append(dataset)
        return datasets

    async def initialize_workers(self) -> None:
        self.rollout_worker, self.megatron_worker, self.value_worker = await initialize_pipeline_workers(self.config)

    async def start_opd_teacher_services(self) -> None:
        opd_config = self.config.grpo.opd
        if not opd_config.enabled or opd_config.backend != "sglang":
            return
        from axrl.utils.sglang_launch_utils import start_sglang_router

        teacher_gpus = rollout_total_gpus(opd_config.sglang_worker)
        base_gpus = required_cluster_gpus(
            rollout_worker=self.config.rollout_worker,
            megatron_worker=self.config.megatron_worker,
            colocated=self.config.controller.colocated,
            needs_rollout=needs_rollout_worker(self.config.controller.run_mode),
            needs_megatron=needs_megatron_worker(self.config.controller.run_mode),
        )
        assert_cluster_has_gpus(base_gpus + teacher_gpus, run_mode=f"{self.config.controller.run_mode}+opd_sglang_teacher")
        teacher_resource_group = ResourceGroup(rollout_resource_requests(opd_config.sglang_worker))
        handle = await start_sglang_router(
            teacher_resource_group,
            opd_config.sglang_worker,
            router_host=opd_config.sglang_host,
            router_port=opd_config.sglang_port,
        )
        opd_config.sglang_host = handle.host
        opd_config.sglang_port = handle.port
        self.recipe_services["opd_teacher_sglang"] = (handle, teacher_resource_group)
        logger.info("Started OPD SGLang teacher at %s.", handle.base_url)

    async def stop_opd_teacher_services(self) -> None:
        service = self.recipe_services.pop("opd_teacher_sglang", None)
        if service is None:
            return
        handle, resource_group = service
        await handle.terminate()
        resource_group.shutdown()
        logger.info("Stopped OPD SGLang teacher service.")

    def initialize_stage_manager(self) -> None:
        if self.rollout_worker is None or self.megatron_worker is None:
            self.stage_manager = None
            return
        if self.config.controller.colocated:
            self.stage_manager = ColocatedStageManager(
                rollout_worker=self.rollout_worker,
                megatron_worker=self.megatron_worker,
                value_worker=self.value_worker,
            )
        else:
            self.stage_manager = DisaggregatedStageManager(
                rollout_worker=self.rollout_worker,
                megatron_worker=self.megatron_worker,
                value_worker=self.value_worker,
            )

    async def initialize_rollout_runtime(self) -> None:
        if self.rollout_worker is None:
            logger.info("Skipping rollout runtime initialization because rollout_worker is None.")
            return
        controller_config = self.config.controller
        assert controller_config.max_running_requests > 0, "max_running_requests must be greater than zero."
        logger.info(
            "Initializing rollout runtime: max_running_requests=%d num_rollout_actors=%d num_cpus_per_actor=%d.",
            controller_config.max_running_requests,
            controller_config.num_rollout_actors,
            controller_config.num_cpus_per_actor,
        )
        with Timer("Pipeline rollout runtime: shared workers", verbose=True):
            self.shared_workers = self.recipe.initialize_shared_workers(self.recipe_services)
        logger.info("Initialized shared rollout workers: names=%s.", sorted(self.shared_workers))
        with Timer("Pipeline rollout runtime: rollout queue actor", verbose=True):
            self.rollout_queue = RayTaskQueue[Conversation](
                RayTaskQueue.initialize_remote_actor(
                    max_running_tasks=controller_config.max_running_requests,
                )
            )
        logger.info("Created rollout task queue actor.")
        with Timer("Pipeline rollout runtime: result queue actor", verbose=True):
            self.result_queue = RayTaskQueue[RolloutResult](
                RayTaskQueue.initialize_remote_actor(
                    max_running_tasks=controller_config.max_running_requests,
                )
            )
        logger.info("Created rollout result queue actor.")
        with Timer("Pipeline rollout runtime: runtime handles", verbose=True):
            self.rollout_runtime = RolloutRuntime(
                rollout_worker=self.rollout_worker,
                rollout_queue=self.rollout_queue,
                result_queue=self.result_queue,
                shared_workers=self.shared_workers,
            )
        with Timer("Pipeline rollout runtime: actor concurrency", verbose=True):
            max_running_tasks_per_actor = self._initial_max_running_tasks_per_actor()
        logger.info(
            "Starting rollout actors: count=%d max_running_tasks_per_actor=%d.",
            controller_config.num_rollout_actors,
            max_running_tasks_per_actor,
        )
        with Timer("Pipeline rollout runtime: rollout actors", verbose=True):
            self.rollout_actors = await self._start_rollout_actors(
                self.rollout_runtime,
                self.recipe,
                max_running_tasks_per_actor,
            )
        logger.info("Started rollout actors: count=%d.", len(self.rollout_actors))

    def get_worker_placement(self) -> PipelineWorkerPlacement:
        return get_pipeline_worker_placement(self.config)

    async def start(self) -> None:
        try:
            await self.initialize()
            run_mode = self.config.controller.run_mode
            if run_mode == "online_rl_train":
                await self.run_online_rl_train()
            elif run_mode == "replay_rl_train":
                await self.run_replay_rl_train()
            elif run_mode == "eval_only":
                await self.run_eval_only()
            elif run_mode == "sft_train":
                await self.run_sft_train()
            elif run_mode == "mismatch_test":
                await self.run_mismatch_test()
            else:
                raise NotImplementedError(f"PipelineController.start is not implemented for run_mode={run_mode!r}.")  # noqa: TRY301
        except Exception:
            logger.exception("PipelineController failed; starting cleanup.")
            raise
        finally:
            self.shutdown()
            await self.shutdown_recipe()

    async def shutdown_recipe(self) -> None:
        await self.stop_opd_teacher_services()
        await self.recipe.stop_services(self.recipe_services)
        self.recipe_services = {}
        await self.recipe.shutdown()

    @override
    async def switch_to_train(self) -> None:
        assert self.stage_manager is not None, "switch_to_train requires a stage manager."
        await self.stage_manager.switch_to_train()

    async def switch_to_value_train(self) -> None:
        assert self.stage_manager is not None, "switch_to_value_train requires a stage manager."
        await self.stage_manager.switch_to_value_train()

    async def switch_to_online_train_phase(self) -> None:
        if self.is_ppo_training():
            await self.switch_to_value_train()
            return
        await self.switch_to_train()

    @override
    async def prepare_for_weight_updates(self) -> None:
        assert self.stage_manager is not None, "prepare_for_weight_updates requires a stage manager."
        await self.stage_manager.switch_to_weight_sync()

    @override
    async def switch_to_rollout(self) -> None:
        assert self.stage_manager is not None, "switch_to_rollout requires a stage manager."
        await self.stage_manager.switch_to_rollout()

    def shutdown(self) -> None:
        self._stop_resource_monitor()
        if self._packing_pool is not None:
            self._packing_pool.shutdown()
            self._packing_pool = None
        self._shutdown_rollout_actors()
        shutdown_pipeline_workers(
            shared_workers=self.shared_workers,
            rollout_worker=self.rollout_worker,
            megatron_worker=self.megatron_worker,
            value_worker=self.value_worker,
        )
        shutdown_task_queue(self.rollout_queue)
        shutdown_task_queue(self.result_queue)
        if self.metric_logger is not None:
            self.metric_logger.close()
        self.shared_workers = None
        self.rollout_worker = None
        self.megatron_worker = None
        self.value_worker = None
        self.rollout_queue = None
        self.result_queue = None
        self.rollout_runtime = None
        self.rollout_actors = []
        self.metric_logger = None
        self.stage_manager = None

    def _shutdown_rollout_actors(self) -> None:
        if not self.rollout_actors:
            return
        max_workers = min(32, len(self.rollout_actors))
        logger.info("Shutting down %d rollout actors with %d threads.", len(self.rollout_actors), max_workers)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            list(executor.map(lambda actor: actor.shutdown(), self.rollout_actors))

    def _initialize_eval_datasets(self, dataset_configs: Sequence[DatasetConfig]) -> None:
        assert dataset_configs, "Pipeline eval rollout requires test_datasets."
        self.test_datasets = self._prepare_datasets(dataset_configs)
        self.eval_dataset_configs = list(dataset_configs)

    def _initialize_output_dir(self) -> None:
        from axrl.configs import AXRL_DIR

        output_dir = AXRL_DIR.output / self.config.controller.output_dir_name
        output_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir = output_dir

    def _initialize_logger(self) -> None:
        self.metric_logger = get_metric_logger(self.config.logger)
        self.metric_logger.log_config(self.config)
        self.config.megatron_worker.metric_logger_config = self.config.logger.model_copy()
        if self.config.value_worker is not None:
            self.config.value_worker.metric_logger_config = self.config.logger.model_copy()

    async def load_checkpoint_if_existed(self) -> None:
        assert self.megatron_worker is not None, "Checkpoint loading requires a Megatron worker."
        assert self.checkpoint_dir is not None, "PipelineController.initialize() must set checkpoint_dir before checkpoint loading."
        if self.checkpoint_dir.is_dir():
            if self.train_dataset is None:
                self.global_step = self.megatron_worker.load_checkpoint()
                self.load_value_checkpoint_if_exists()
                return
            self.megatron_worker, self.global_step, self.train_dataset = self.load_checkpoint(
                self.megatron_worker,
                self.train_dataset,
                checkpoint_dir=self.checkpoint_dir,
            )
            self.load_value_checkpoint_if_exists()
            return

    def load_value_checkpoint_if_exists(self) -> None:
        if self.value_worker is None or self.config.value_worker is None:
            return
        value_checkpoint_dir = self.config.value_worker.get_checkpoint_dir()
        if not value_checkpoint_dir.is_dir():
            return
        self.value_step = self.value_worker.load_checkpoint()
        logger.info("Loaded value worker checkpoint at value_step=%d from %s.", self.value_step, value_checkpoint_dir)

    def initialize_megatron_teacher_weights(self) -> None:
        opd_config = self.config.grpo.opd
        if self._opd_teacher_weights_initialized or not opd_config.enabled or opd_config.backend != "megatron":
            return
        assert self.megatron_worker is not None, "Megatron OPD teacher weights require a Megatron worker."
        assert opd_config.teacher_model is not None, "Megatron OPD requires teacher_model."
        current_weight_name = "__opd_current_student_weights"
        self.megatron_worker.copy_weights_to_cpu(current_weight_name)
        try:
            self.megatron_worker.load_hf_weights(opd_config.teacher_model.get_full_path(), reset_optimizer=False)
            self.megatron_worker.copy_weights_to_cpu(opd_config.teacher_weight_name)
            logger.info("Initialized OPD Megatron teacher weight snapshot %r.", opd_config.teacher_weight_name)
        finally:
            self.megatron_worker.apply_weights_from_cpu(current_weight_name)
            self.megatron_worker.remove_cpu_weight_copy(current_weight_name)
        self._opd_teacher_weights_initialized = True

    async def run_eval_only(self) -> list[RolloutResult]:
        results = await self.run_eval_rollouts()
        logger.info("Eval-only completed at global_step=%d.", self.global_step)
        return results

    async def run_evals(self) -> list[RolloutResult]:
        if not self.test_datasets:
            logger.info("No test_datasets configured; skipping evaluation.")
            return []
        if self.megatron_worker is not None:
            await self.switch_to_rollout()
        return await self.run_eval_rollouts()

    async def run_evals_if_needed(self) -> None:
        eval_interval = self.config.online_rl_train.eval_every_n_global_updates
        if self.global_step - self.prev_eval_global_step < eval_interval:
            return
        await self.run_evals()
        self.prev_eval_global_step = self.global_step

    async def run_eval_rollouts(self) -> list[RolloutResult]:
        assert self.test_datasets, "PipelineController.initialize() must initialize test_datasets before eval rollouts."
        assert self.eval_dataset_configs, "PipelineController.initialize() must initialize eval_dataset_configs before eval rollouts."

        all_results: list[RolloutResult] = []
        remaining_smoke_rollouts = self.config.controller.smoke_eval_rollouts
        for dataset, dataset_config in zip(self.test_datasets, self.eval_dataset_configs, strict=True):
            if remaining_smoke_rollouts is not None and remaining_smoke_rollouts <= 0:
                break
            results = await self._run_eval_dataset_rollouts(dataset, dataset_config, max_rollouts=remaining_smoke_rollouts)
            all_results.extend(results)
            if remaining_smoke_rollouts is not None:
                remaining_smoke_rollouts -= len(results)

        assert all_results, "No rollout conversations were built for eval rollouts."
        return all_results

    async def _run_eval_dataset_rollouts(
        self,
        dataset: BaseDataset,
        dataset_config: DatasetConfig,
        *,
        max_rollouts: int | None,
    ) -> list[RolloutResult]:
        rollout_queue, result_queue = self._check_rollout_ready()
        conversations = self._build_eval_rollout_conversations(dataset, dataset_config, max_rollouts=max_rollouts)
        assert conversations, "No rollout conversations were built for eval rollouts."
        self._check_rollout_conversations(conversations)
        dataset_name = dataset.__class__.__name__
        with Timer(f"Pipeline eval rollouts for {dataset_name} ({len(conversations)} total)", verbose=True) as timer:
            await rollout_queue.put_many(conversations)
            results = await self._collect_expected_rollout_results(result_queue, expected_result_count=len(conversations))
        prefix = f"eval_{dataset_name}"
        self._log_rollout_metrics(results, elapsed_seconds=timer.elapsed_seconds, prefix=prefix)
        self.log_scalars({f"{prefix}/eval_time_sec": timer.elapsed_seconds})
        if dataset_config.subset_key is not None:
            subset_metrics = aggregate_response_metrics_by_subset(results, dataset_config.subset_key, prefix)
            if subset_metrics:
                self.log_scalars(subset_metrics)
        self._save_eval_rollouts_if_needed(results, dataset_name=dataset_name)
        return results

    def _check_rollout_ready(self) -> tuple[RayTaskQueue[Conversation], RayTaskQueue[RolloutResult]]:
        assert self.rollout_worker is not None, "PipelineController.initialize() must create rollout_worker before rollouts."
        assert self.rollout_queue is not None, "PipelineController.initialize() must create rollout_queue before rollouts."
        assert self.result_queue is not None, "PipelineController.initialize() must create result_queue before rollouts."
        assert self.rollout_runtime is not None, "PipelineController.initialize() must create rollout_runtime before rollouts."
        assert self.rollout_actors, "PipelineController.initialize() must create rollout_actors before rollouts."
        return self.rollout_queue, self.result_queue

    def _build_eval_rollout_conversations(
        self,
        dataset: BaseDataset,
        dataset_config: DatasetConfig,
        *,
        max_rollouts: int | None,
    ) -> list[Conversation]:
        conversations: list[Conversation] = []
        rollouts_per_prompt = 1 if dataset_config.eval_num_rollouts_per_prompt is None else dataset_config.eval_num_rollouts_per_prompt
        for sample_index in range(len(dataset)):
            for _rollout_index in range(rollouts_per_prompt):
                if max_rollouts is not None and len(conversations) >= max_rollouts:
                    return conversations
                conv_index = len(conversations)
                conv = dataset.get_conv(sample_index).deep_copy()
                conv.extra["answer"] = dataset.get_label(sample_index)
                conv.gen_state.capture_routing = False
                conv.gen_state.captured_routing_rows = 0
                conv.gen_state.sampling_config = self.config.eval_sampling_config
                conv.gen_state.session_id = f"{conv.conversation_id}:eval:{conv_index}"
                conversations.append(conv)
        return conversations

    def build_train_rollout_conversations(self, num_groups: int) -> list[Conversation]:
        assert self.train_dataset is not None, "PipelineController.initialize() must initialize train_dataset before train rollouts."
        assert num_groups > 0, "num_groups must be greater than zero."
        online_config = self.config.online_rl_train
        group_size = online_config.num_rollouts_per_conversation
        assert group_size > 1, "num_rollouts_per_conversation must be greater than one for GRPO train rollouts."
        sample_indices = self.train_dataset.sample(num_groups, sample_type=online_config.sample_type)
        if online_config.sort_sampled_prompts_by_response_length:
            sample_indices = self.train_dataset.sort_by_mean_lengths(sample_indices)

        conversations: list[Conversation] = []
        schedule_id = self._rollout_schedule_id
        self._rollout_schedule_id += 1
        for group_index, sample_index in enumerate(sample_indices):
            base_conv = self.train_dataset.get_conv(sample_index)
            label = self.train_dataset.get_label(sample_index)
            group_id = f"train:{self.global_step}:s{schedule_id}:g{group_index}:{base_conv.conversation_id}"
            for rollout_index in range(group_size):
                conv = base_conv.deep_copy()
                conv.extra["answer"] = label
                conv.extra["group_id"] = group_id
                conv.extra["rollout_index"] = rollout_index
                conv.extra["pack_rollout_trace"] = True
                conv.gen_state.capture_routing = self.config.rollout_worker.enable_routing_replay
                conv.gen_state.captured_routing_rows = 0
                conv.gen_state.sampling_config = self.config.train_sampling_config
                conv.gen_state.session_id = f"{conv.conversation_id}:train:{self.global_step}:s{schedule_id}:g{group_index}:r{rollout_index}"
                conversations.append(conv)
        return conversations

    @staticmethod
    def _check_rollout_conversations(conversations: Sequence[Conversation]) -> None:
        for task_index, conv in enumerate(conversations):
            assert conv.gen_state.session_id, f"Rollout conversation at index {task_index} must have gen_state.session_id."

    async def enqueue_rollout_conversations(self, conversations: Sequence[Conversation]) -> None:
        rollout_queue, _ = self._check_rollout_ready()
        assert conversations, "Cannot enqueue an empty rollout conversation list."
        self._check_rollout_conversations(conversations)
        # Future refactor path: large traces/conversations may move through
        # object refs or bounded chunks if queue payloads become a bottleneck.
        await rollout_queue.put_many(list(conversations))

    async def _start_rollout_actors(
        self,
        runtime: RolloutRuntime,
        recipe: BaseRecipe,
        max_running_tasks_per_actor: int,
    ) -> list[RolloutActor]:
        scheduling_strategies = [
            self._get_rollout_actor_node_affinity_strategy(runtime, actor_index) for actor_index in range(self.config.controller.num_rollout_actors)
        ]
        remote_actor_tasks = [
            RolloutActor.initialize_remote_actor(
                worker_id=f"pipeline-rollout-{actor_index}",
                num_cpus_per_actor=self.config.controller.num_cpus_per_actor,
                max_running_tasks=max_running_tasks_per_actor,
                runtime=runtime,
                recipe=recipe,
                scheduling_strategy=scheduling_strategies[actor_index],
            )
            for actor_index in range(self.config.controller.num_rollout_actors)
        ]
        remote_actors = await asyncio.gather(*remote_actor_tasks)
        return [
            RolloutActor(
                remote_actor,
                max_running_tasks=max_running_tasks_per_actor,
            )
            for remote_actor in remote_actors
        ]

    @staticmethod
    def _get_rollout_actor_node_affinity_strategy(runtime: RolloutRuntime, actor_index: int) -> NodeAffinitySchedulingStrategy:
        """Place rollout actors evenly across rollout-worker bundle nodes.

        Ray CPU actors are node-affinity scheduled rather than bundle-affinity
        scheduled here. The goal is to cycle actors over the rollout-worker
        bundles so local rollout helpers stay close to SGLang workers and do
        not all land on the head node.
        """
        resource_group = runtime.rollout_worker.get_resource_group()
        assert resource_group.bundle_infos, "Rollout worker resource group must have at least one bundle."
        bundle_info = resource_group.bundle_infos[actor_index % len(resource_group.bundle_infos)]
        alive_nodes = [node for node in ray.nodes() if node.get("Alive")]
        alive_node_ids = {node["NodeID"] for node in alive_nodes if node.get("NodeID") is not None}
        if bundle_info.node_id is not None and bundle_info.node_id in alive_node_ids:
            return NodeAffinitySchedulingStrategy(
                node_id=bundle_info.node_id,
                soft=False,
            )

        node_id_by_ip = {
            node["NodeManagerAddress"]: node["NodeID"]
            for node in alive_nodes
            if node.get("NodeManagerAddress") is not None and node.get("NodeID") is not None
        }
        if bundle_info.ip not in node_id_by_ip:
            raise AssertionError(
                "Cannot place rollout actor on bundle "
                f"index={bundle_info.index} specs={bundle_info.specs} ip={bundle_info.ip} node_id={bundle_info.node_id}; "
                f"alive Ray node IDs are {sorted(alive_node_ids)}; alive Ray node IPs are {sorted(node_id_by_ip)}."
            )
        logger.warning(
            "Bundle index %d has missing/stale Ray node_id=%s; falling back to NodeManagerAddress match for ip=%s.",
            bundle_info.index,
            bundle_info.node_id,
            bundle_info.ip,
        )
        return NodeAffinitySchedulingStrategy(
            node_id=node_id_by_ip[bundle_info.ip],
            soft=False,
        )

    def _initial_max_running_tasks_per_actor(self) -> int:
        controller_config = self.config.controller
        assert controller_config.num_rollout_actors > 0, "num_rollout_actors must be greater than zero."
        assert controller_config.max_running_requests > 0, "max_running_requests must be greater than zero."
        max_running_tasks_per_actor = max(1, math.ceil(controller_config.max_running_requests / controller_config.num_rollout_actors))
        logger.info(
            "Initial rollout actor concurrency: max_running_requests=%d, num_rollout_actors=%d, max_running_tasks_per_actor=%d.",
            controller_config.max_running_requests,
            controller_config.num_rollout_actors,
            max_running_tasks_per_actor,
        )
        return max_running_tasks_per_actor

    async def _collect_expected_rollout_results(
        self,
        result_queue: RayTaskQueue[RolloutResult],
        *,
        expected_result_count: int,
    ) -> list[RolloutResult]:
        assert expected_result_count > 0, "expected_result_count must be greater than zero when collecting rollout results."
        results: list[RolloutResult] = []
        with tqdm(total=expected_result_count, desc="Collecting rollout results", unit="rollout") as progress:
            while len(results) < expected_result_count:
                assignment = await result_queue.get("pipeline-controller", timeout_seconds=_ROLLOUT_HEALTH_CHECK_INTERVAL_SECONDS)
                if assignment is None:
                    await self._log_rollout_wait(
                        result_queue,
                        context="eval",
                        completed_count=len(results),
                        expected_count=expected_result_count,
                    )
                    await self._check_rollout_actor_health()
                    continue
                results.append(assignment.task)
                await result_queue.complete(assignment.assignment_id)
                progress.update(1)
        return results

    async def _check_rollout_actor_health(self) -> None:
        await asyncio.gather(*(actor.check_health() for actor in self.rollout_actors))

    async def _log_rollout_wait(
        self,
        result_queue: RayTaskQueue[RolloutResult],
        *,
        context: str,
        completed_count: int | None = None,
        expected_count: int | None = None,
    ) -> None:
        rollout_stats = await self.rollout_queue.stats() if self.rollout_queue is not None else None
        result_stats = await result_queue.stats()
        progress = "" if completed_count is None or expected_count is None else f" progress={completed_count}/{expected_count}"
        rollout_queue_text = (
            "rollout_queue=unavailable"
            if rollout_stats is None
            else (
                f"rollout_queue=pending:{rollout_stats.pending_count} running:{rollout_stats.running_count} completed:{rollout_stats.completed_count}"
            )
        )
        logger.warning(
            "Still waiting for rollout results: context=%s%s %s result_queue=pending:%d running:%d completed:%d.",
            context,
            progress,
            rollout_queue_text,
            result_stats.pending_count,
            result_stats.running_count,
            result_stats.completed_count,
        )

    async def _log_train_rollout_group_progress(
        self,
        result_queue: RayTaskQueue[RolloutResult],
        *,
        generated_groups: int,
        scheduled_groups: int,
        valid_groups: int,
        target_valid_groups: int,
        skipped_groups: int,
        filter_counts: Counter[GroupFilterType],
    ) -> None:
        result_stats = await result_queue.stats()
        logger.info(
            "Train rollout group progress: generated=%d/%d valid=%d/%d skipped=%d filter_counts=%s result_queue=pending:%d running:%d completed:%d.",
            generated_groups,
            scheduled_groups,
            valid_groups,
            target_valid_groups,
            skipped_groups,
            dict(filter_counts),
            result_stats.pending_count,
            result_stats.running_count,
            result_stats.completed_count,
        )

    def get_num_groups_per_batch_rollout(self) -> int:
        online_config = self.config.online_rl_train
        global_batch_size = self.config.megatron_worker.global_batch_size
        rollouts_per_group = online_config.num_rollouts_per_conversation
        assert global_batch_size % rollouts_per_group == 0, (
            f"global_batch_size ({global_batch_size}) must be divisible by num_rollouts_per_conversation ({rollouts_per_group})."
        )
        num_groups_per_global_update = global_batch_size // rollouts_per_group
        return num_groups_per_global_update * online_config.batch_rollout_for_n_global_updates

    def get_num_groups_per_model_sync(self) -> int:
        online_config = self.config.online_rl_train
        global_batch_size = self.config.megatron_worker.global_batch_size
        rollouts_per_group = online_config.num_rollouts_per_conversation
        total_rollouts_per_model_sync = global_batch_size * online_config.model_sync_every_n_global_updates
        assert total_rollouts_per_model_sync % rollouts_per_group == 0, (
            f"global_batch_size * model_sync_every_n_global_updates ({total_rollouts_per_model_sync}) must be divisible by "
            f"num_rollouts_per_conversation ({rollouts_per_group})."
        )
        return total_rollouts_per_model_sync // rollouts_per_group

    def normalize_group_rewards(self, group: Sequence[RolloutResult]) -> None:
        """Normalize rewards for one completed GRPO rollout group in place."""
        metrics = [result.metric for result in group]
        conversation_ids = [result.conversation.conversation_id for result in group]
        assert len(set(conversation_ids)) == 1, f"Rollout group must share one conversation_id, got {conversation_ids}."

        group_scores = [metric.score for metric in metrics if metric.score is not None]
        assert len(group_scores) == len(metrics), "All rollout group metrics must have a valid score."
        assert len(group_scores) > 1, "At least two rollout results are required to compute group reward stats."
        if self.train_dataset is not None:
            self.train_dataset.update_score_history(conversation_ids[0], group_scores)

        group_mean = float(np.mean(group_scores))
        group_std = float(np.std(group_scores, ddof=1))
        for metric in metrics:
            metric.score_mean = group_mean
            metric.score_std = group_std

        for result in group:
            assert result.metric.score is not None
            if group_std < 1e-8:
                advantage = 0.0
            else:
                advantage = float(np.clip((result.metric.score - group_mean) / (group_std + 1e-6), -5.0, 5.0))
            result.reward = result.metric.score
            result.reward_baseline = group_mean
            result.scalar_advantage = advantage
            if result.trace is not None:
                for sample in result.trace.turn_samples:
                    sample.reward = result.metric.score
                    sample.reward_baseline = group_mean
                    sample.advantage = np.where(sample.loss_mask, np.float32(advantage), np.float32(0.0)).astype(np.float32, copy=False)
            self.finalize_packed_result_reward(
                result,
                score=result.metric.score,
                group_mean=group_mean,
                advantage=advantage,
            )

    @staticmethod
    def finalize_packed_result_reward(
        result: RolloutResult,
        *,
        score: float,
        group_mean: float,
        advantage: float,
    ) -> None:
        if result.packed_samples is None:
            return
        PipelineController.finalize_packed_sample_batch_reward(
            result.packed_samples,
            score=score,
            group_mean=group_mean,
            advantage=advantage,
        )

    @staticmethod
    def finalize_packed_sample_batch_reward(
        sample_batch: SampleTensorDict,
        *,
        score: float,
        group_mean: float,
        advantage: float,
    ) -> None:
        reward = cast("torch.Tensor", sample_batch["reward"])
        reward_baseline = cast("torch.Tensor", sample_batch["reward_baseline"])
        advantage_tensor = cast("torch.Tensor", sample_batch["advantage"])
        loss_mask = cast("torch.Tensor", sample_batch["loss_mask"])
        reward.fill_(score)
        reward_baseline.fill_(group_mean)
        sample_batch["advantage"] = loss_mask.to(advantage_tensor.dtype) * advantage

    def get_group_filter_type(self, group: Sequence[RolloutResult]) -> GroupFilterType:
        if not self.config.online_rl_train.filter_zero_std:
            return "pass"

        metrics = [result.metric for result in group]
        reward_std = metrics[0].score_std
        if reward_std is not None and reward_std > 1e-2:
            return "pass"

        if all(metric.score is not None and np.isclose(metric.score, 1) for metric in metrics):
            return "zero_std_all_success"

        if all(metric.score is not None and np.isclose(metric.score, 0) for metric in metrics):
            return "zero_std_all_fail"

        return "pass"

    async def cleanup_skipped_rollout_group(self, group: Sequence[RolloutResult], filter_type: GroupFilterType) -> None:
        """Hook for recipe-specific cleanup when a rollout group is filtered."""
        del filter_type
        await self.delete_r3_handles_and_caches(group, clear_trainer_caches=False)

    def _get_packing_pool(self) -> InferWorker[Any, SampleTensorDict]:
        if self._packing_pool is None:
            from axrl.data.rollout_trace_packing import RolloutTracePackingProcessor
            from axrl.processor.processor_pool import ProcessorPool

            num_workers = self.config.online_rl_train.num_rollouts_per_conversation
            self._packing_pool = ProcessorPool(
                RolloutTracePackingProcessor,
                config=None,
                num_processors=num_workers,
                timeout_seconds=600,
            )
            logger.info("Async rollout packing enabled: workers=%d task_granularity=trace.", num_workers)
        return self._packing_pool

    async def pack_rollout_group(self, group: Sequence[RolloutResult]) -> list[RolloutResult]:
        from axrl.data.rollout_trace_packing import RolloutTracePackRequest
        from axrl.data.sample import collect_unique_handles_from_sample_tensor_dict

        allow_prefix_sharing = self.config.controller.allow_prefix_merging and self.config.megatron_worker.use_magi_merged_forward
        requests = []
        results_to_pack = []
        for trajectory_id, result in enumerate(group):
            if result.trace is None and (result.packed_samples is not None or result.packed_samples_ref is not None):
                continue
            assert result.trace is not None, "RolloutResult.trace is required before packing rollout groups."
            results_to_pack.append(result)
            requests.append(
                RolloutTracePackRequest(
                    trajectory_id=trajectory_id,
                    turn_samples=result.trace.turn_samples,
                    max_pack_length=self.config.megatron_worker.model.seq_length,
                    allow_prefix_sharing=allow_prefix_sharing,
                )
            )
        if not requests:
            return list(group)
        packed_traces = await self._get_packing_pool().batch_generate(requests)
        # Tensors returned through multiprocessing queues are backed by
        # /dev/shm file descriptors. Clone them immediately so retaining a
        # large train batch does not retain one FD per tensor storage.
        local_packed_traces = [sample_batch.clone() for sample_batch in packed_traces]
        del packed_traces
        for result, sample_batch in zip(results_to_pack, local_packed_traces, strict=True):
            result.packed_samples = sample_batch
            result.trainable_token_count = int(sample_batch["loss_mask"].sum().item())
            result.routing_handles = collect_unique_handles_from_sample_tensor_dict(sample_batch)
        return list(group)

    @staticmethod
    def get_rollout_group_id(result: RolloutResult) -> str:
        group_id = result.conversation.extra.get(
            "group_id",
            result.conversation.extra.get("rollout_group_id", result.conversation.conversation_id),
        )
        assert isinstance(group_id, str) and group_id, f"Rollout result must have a non-empty group id, got {group_id!r}."
        return group_id

    async def stream_rollout_groups(self, input_queue: RayTaskQueue[RolloutResult]) -> AsyncIterator[RolloutGroup]:
        group_size = self.config.online_rl_train.num_rollouts_per_conversation
        assert group_size > 1, "num_rollouts_per_conversation must be greater than one for group reward normalization."
        pending_groups: dict[str, list[RolloutResult]] = {}
        while True:
            assignment = await input_queue.get("pipeline-group-stream", timeout_seconds=_ROLLOUT_HEALTH_CHECK_INTERVAL_SECONDS)
            if assignment is None:
                await self._check_rollout_actor_health()
                await self._log_rollout_wait(input_queue, context="train")
                continue
            result = assignment.task
            await input_queue.complete(assignment.assignment_id)
            group_id = self.get_rollout_group_id(result)
            group = pending_groups.setdefault(group_id, [])
            group.append(result)
            if len(group) < group_size:
                continue

            assert len(group) == group_size, f"Rollout group {group_id!r} has more than {group_size} results."
            del pending_groups[group_id]
            self.normalize_group_rewards(group)
            filter_type = self.get_group_filter_type(group)
            if filter_type != "pass":
                await self.cleanup_skipped_rollout_group(group, filter_type)
                yield RolloutGroup(results=list(group), filter_type=filter_type)
                continue
            yield RolloutGroup(results=list(group), filter_type="pass")

    async def collect_scheduled_rollout_groups(
        self,
        input_queue: RayTaskQueue[RolloutResult],
        *,
        scheduled_groups: int,
        max_valid_groups: int | None = None,
    ) -> TrainGroupBatch:
        assert scheduled_groups > 0, "scheduled_groups must be greater than zero."
        if max_valid_groups is not None:
            assert max_valid_groups > 0, "max_valid_groups must be greater than zero when set."
        valid_groups: list[RolloutGroup] = []
        skipped_groups: list[RolloutGroup] = []
        filter_type_counts: Counter[GroupFilterType] = Counter()
        group_stream = self.stream_rollout_groups(input_queue)
        with tqdm(total=scheduled_groups, desc="Collecting train rollout groups", unit="group") as progress:
            for _ in range(scheduled_groups):
                group = await anext(group_stream)
                filter_type_counts.update([group.filter_type])
                if group.is_valid and (max_valid_groups is None or len(valid_groups) < max_valid_groups):
                    valid_groups.append(group)
                else:
                    skipped_groups.append(group)
                    if group.is_valid:
                        await self.delete_r3_handles_and_caches(group.results, clear_trainer_caches=False)
                progress.update(1)
        return TrainGroupBatch(
            valid_groups=valid_groups,
            skipped_groups=skipped_groups,
            filter_type_counts=filter_type_counts,
        )

    async def _prepare_online_train_group_batch(
        self,
        valid_rollouts: list[RolloutGroup],
        skipped_rollouts: list[RolloutGroup],
        group_filter_types: Counter[GroupFilterType],
        *,
        target_valid_groups: int,
    ) -> TrainGroupBatch:
        random.shuffle(valid_rollouts)
        if len(valid_rollouts) > target_valid_groups:
            extra_rollouts = valid_rollouts[target_valid_groups:]
            del valid_rollouts[target_valid_groups:]
            skipped_rollouts.extend(extra_rollouts)
            await self.delete_r3_handles_and_caches(
                [result for rollout_group in extra_rollouts for result in rollout_group.results],
                clear_trainer_caches=False,
            )
        return TrainGroupBatch(
            valid_groups=list(valid_rollouts),
            skipped_groups=list(skipped_rollouts),
            filter_type_counts=Counter(group_filter_types),
        )

    async def stream_online_train_group_batches(self) -> AsyncGenerator[TrainGroupBatch, None]:
        _, result_queue = self._check_rollout_ready()
        online_config = self.config.online_rl_train
        target_valid_groups = self.get_num_groups_per_model_sync()
        scheduled_groups = self.get_num_groups_per_batch_rollout()
        group_filter_types: Counter[GroupFilterType] = Counter()
        valid_rollouts: list[RolloutGroup] = []
        skipped_rollouts: list[RolloutGroup] = []
        if gc.isenabled():
            gc.disable()
            logger.info("Disabled Python cyclic GC inside stream_online_train_group_batches.")

        pbar: tqdm = tqdm(total=target_valid_groups, desc="Valid groups collected", unit="group")
        while self.global_step < online_config.max_global_updates:
            await self.switch_to_rollout()
            conversations = self.build_train_rollout_conversations(scheduled_groups)
            logger.info(
                "Scheduling train rollout groups: groups=%d, rollouts=%d.",
                scheduled_groups,
                len(conversations),
            )
            await self.enqueue_rollout_conversations(conversations)

            group_stream = self.stream_rollout_groups(result_queue)
            generated_groups = 0
            last_progress_log_time = time.monotonic()
            while generated_groups < scheduled_groups:
                group = await anext(group_stream)
                generated_groups += 1
                group_filter_types.update([group.filter_type])
                now = time.monotonic()
                if now - last_progress_log_time >= 30:
                    await self._log_train_rollout_group_progress(
                        result_queue,
                        generated_groups=generated_groups,
                        scheduled_groups=scheduled_groups,
                        valid_groups=len(valid_rollouts),
                        target_valid_groups=target_valid_groups,
                        skipped_groups=len(skipped_rollouts),
                        filter_counts=group_filter_types,
                    )
                    last_progress_log_time = now
                if not group.is_valid:
                    skipped_rollouts.append(group)
                    continue

                valid_rollouts.append(group)
                pbar.update(1)
                if len(valid_rollouts) < target_valid_groups:
                    continue
                if online_config.strict_on_policy and generated_groups < scheduled_groups:
                    logger.info(
                        "Strict on-policy, wait current batch to finish: %d/%d groups.",
                        generated_groups,
                        scheduled_groups,
                    )
                    continue

                batch = await self._prepare_online_train_group_batch(
                    valid_rollouts,
                    skipped_rollouts,
                    group_filter_types,
                    target_valid_groups=target_valid_groups,
                )
                if self.stage_manager is not None:
                    await self.switch_to_online_train_phase()
                yield batch

                valid_rollouts.clear()
                skipped_rollouts.clear()
                group_filter_types.clear()
                gc.collect()
                pbar.close()
                pbar = tqdm(total=target_valid_groups, desc="Valid groups collected", unit="group")
            await self.run_evals_if_needed()
        pbar.close()

    def collect_packed_samples(self, batch: TrainGroupBatch) -> SampleTensorDict:
        assert batch.valid_groups, "TrainGroupBatch must contain at least one valid rollout group."
        sample_batches: list[SampleTensorDict] = []
        total_original_trainable_tokens = 0
        total_packed_trainable_tokens = 0
        trajectory_id = 0
        for group in batch.valid_groups:
            assert group.is_valid, f"Cannot collect packed samples from skipped group with filter_type={group.filter_type!r}."
            for result in group.results:
                sample_batch = self._take_packed_samples(result)
                self._apply_result_reward_to_packed_samples(result, sample_batch)
                sample_batch["trajectory_id"] = torch.full_like(sample_batch["trajectory_id"], trajectory_id)
                sample_batches.append(sample_batch)
                packed_trainable_tokens = int(sample_batch["loss_mask"].sum().item())
                expected_trainable_tokens = self._expected_trainable_token_count(result)
                if expected_trainable_tokens is not None:
                    assert packed_trainable_tokens == expected_trainable_tokens
                total_original_trainable_tokens += expected_trainable_tokens if expected_trainable_tokens is not None else packed_trainable_tokens
                total_packed_trainable_tokens += packed_trainable_tokens
                trajectory_id += 1

        assert total_original_trainable_tokens > 0, "Packed rollout groups must contain trainable tokens."
        assert total_packed_trainable_tokens == total_original_trainable_tokens
        samples = _concat_sample_tensor_dicts(sample_batches)
        samples["index"] = torch.arange(len(samples), dtype=samples["index"].dtype, device=samples["index"].device)
        total_trainable_tokens = int(samples["loss_mask"].sum().item())
        assert total_trainable_tokens == total_original_trainable_tokens
        return samples

    @staticmethod
    def _take_packed_samples(result: RolloutResult) -> SampleTensorDict:
        if result.packed_samples is not None:
            return result.packed_samples
        if result.packed_samples_ref is not None:
            from ray._private.internal_api import free as free_ray_refs

            packed_samples_ref = result.packed_samples_ref
            sample_batch = ray.get(packed_samples_ref)
            free_ray_refs([packed_samples_ref])
            result.packed_samples_ref = None
            return sample_batch
        raise AssertionError("RolloutResult must carry packed_samples or packed_samples_ref before collection.")

    @staticmethod
    def _expected_trainable_token_count(result: RolloutResult) -> int | None:
        if result.trainable_token_count is not None:
            return result.trainable_token_count
        if result.trace is None:
            return None
        return sum(sum(sample.loss_mask) for sample in result.trace.turn_samples)

    @staticmethod
    def _apply_result_reward_to_packed_samples(result: RolloutResult, sample_batch: SampleTensorDict) -> None:
        if result.reward is None or result.reward_baseline is None or result.scalar_advantage is None:
            return
        PipelineController.finalize_packed_sample_batch_reward(
            sample_batch,
            score=result.reward,
            group_mean=result.reward_baseline,
            advantage=result.scalar_advantage,
        )

    @staticmethod
    def aggregate_filter_type_metrics(filter_type_counts: Counter[GroupFilterType]) -> dict[str, float]:
        total = sum(filter_type_counts.values())
        if total == 0:
            return {}
        return {f"group_filter_type__{filter_type}": count / total for filter_type, count in filter_type_counts.items()}

    def log_scalars(self, scalars: dict[str, float]) -> None:
        assert self.metric_logger is not None, "PipelineController.initialize() must create metric_logger before logging metrics."
        self.metric_logger.log_scalars(scalars, step=self.global_step)

    def start_resource_monitor(self) -> None:
        if self._resource_monitor_thread is not None:
            return
        self._resource_monitor_stop_event.clear()
        self._resource_monitor_thread = threading.Thread(
            target=self._monitor_resource_usage,
            name="axrl-resource-monitor",
            daemon=True,
        )
        self._resource_monitor_thread.start()

    def _stop_resource_monitor(self) -> None:
        thread = self._resource_monitor_thread
        if thread is None:
            return
        self._resource_monitor_stop_event.set()
        thread.join(timeout=10.0)
        if thread.is_alive():
            logger.warning("Resource monitor thread did not stop within 10 seconds.")
            return
        self._resource_monitor_thread = None

    def _monitor_resource_usage(self) -> None:
        while not self._resource_monitor_stop_event.wait(_RESOURCE_MONITOR_INTERVAL_SECONDS):
            try:
                self.log_resource_usage("periodic")
            except Exception:
                logger.warning("Periodic resource snapshot failed.", exc_info=True)

    def log_resource_usage(self, phase: str) -> None:
        self._latest_resource_metrics = log_resource_usage_metrics(phase, global_step=self.global_step)

    def log_latest_resource_metrics(self, phase: str) -> None:
        if self._latest_resource_metrics is None:
            self.log_resource_usage(phase)
        assert self._latest_resource_metrics is not None
        self.log_scalars(self._latest_resource_metrics)

    def log_training_advantages(self, samples: SampleTensorDict) -> None:
        advantage = samples["advantage"].abs()
        loss_mask = samples["loss_mask"]
        per_sample_sum = (advantage * loss_mask).sum(dim=1)
        per_sample_count = loss_mask.sum(dim=1).clamp(min=1)
        abs_advantages = per_sample_sum / per_sample_count
        self.log_scalars(
            {
                "training_samples/min_abs_advantage": float(abs_advantages.min().item()),
                "training_samples/mean_abs_advantage": float(abs_advantages.mean().item()),
                "training_samples/max_abs_advantage": float(abs_advantages.max().item()),
                "training_samples/num_non_zero_advantage": int((~torch.isclose(abs_advantages, torch.zeros_like(abs_advantages))).sum().item()),
            }
        )

    def is_ppo_training(self) -> bool:
        return self.config.grpo.loss_type == "ppo"

    def prefix_metrics(self, metrics: dict[str, float], prefix: str) -> dict[str, float]:
        return {f"{prefix}/{key}": value for key, value in metrics.items()}

    def prepare_ppo_training_samples(self, samples: SampleTensorDict) -> list[Any]:
        assert self.value_worker is not None, "PPO sample preparation requires value_worker."
        assert self.megatron_worker is not None, "PPO sample preparation requires megatron_worker."
        ppo_value_config = self.config.grpo.ppo_value
        assert ppo_value_config is not None, "PPO sample preparation requires grpo.ppo_value."
        self.megatron_worker.to_cpu()
        self.value_worker.to_gpu()

        # Frozen critic predictions from before the value update; Slime calls
        # the same tensor "old_values" inside its clipped value loss.
        old_values, value_gpu_usage_infos = self.value_worker.compute_values(samples)
        old_values = old_values.to(device=samples["loss_mask"].device, dtype=torch.float32)
        loss_mask = samples["loss_mask"].bool()
        token_rewards = build_terminal_token_rewards(
            cast("torch.Tensor", samples["reward"]),
            loss_mask,
        )
        advantages, returns = compute_gae(
            rewards=token_rewards,
            values=old_values,
            loss_mask=loss_mask,
            gamma=ppo_value_config.gamma,
            gae_lambda=ppo_value_config.gae_lambda,
        )
        if self.config.grpo.normalize_advantages_over_valid_tokens_in_batch:
            advantages = normalize_over_valid_tokens_in_batch(advantages, loss_mask)
        samples["old_values"] = old_values
        samples["returns"] = returns
        samples["advantage"] = advantages.to(dtype=samples["advantage"].dtype)
        return list(value_gpu_usage_infos)

    def save_rollouts_snapshot(
        self,
        valid_rollouts: list[list[RolloutResult]],
        skipped_rollouts: list[list[RolloutResult]] | None = None,
    ) -> None:
        online_config = self.config.online_rl_train
        save_interval = online_config.rollout_save_every_n_global_updates
        if save_interval is None:
            return
        assert self.output_dir is not None, "PipelineController.initialize() must create output_dir before saving rollout snapshots."
        if self.global_step % save_interval != 0:
            return
        suffix = f"step{self.global_step}"
        rollouts_to_save = valid_rollouts + (skipped_rollouts or []) if online_config.save_all_rollouts else valid_rollouts
        rollout_path = self.output_dir / f"{online_config.rollout_save_filename}-{suffix}.zst"
        packed_payloads = [
            (result, result.packed_samples, result.packed_samples_ref)
            for group in rollouts_to_save
            for result in group
            if result.packed_samples is not None or result.packed_samples_ref is not None
        ]
        try:
            for result, _, _ in packed_payloads:
                result.packed_samples = None
                result.packed_samples_ref = None
            zst_utils.save_zst(rollouts_to_save, rollout_path, verbose=True)
        finally:
            for result, packed_samples, packed_samples_ref in packed_payloads:
                result.packed_samples = packed_samples
                result.packed_samples_ref = packed_samples_ref

    def save_training_samples_snapshot(self, sample_dataset: SampleTensorDict) -> None:
        online_config = self.config.online_rl_train
        save_interval = online_config.rollout_save_every_n_global_updates
        if save_interval is None:
            return
        assert self.output_dir is not None, "PipelineController.initialize() must create output_dir before saving training sample snapshots."
        if self.global_step % save_interval != 0:
            return
        sample_path = self.output_dir / f"training_samples-step{self.global_step}.zst"
        logger.info("Saving %d packed training samples to %s before Megatron train.", len(sample_dataset), sample_path)
        zst_utils.save_zst(sample_dataset, sample_path, verbose=True)

    def select_replay_train_rollouts(self, rollout_groups: list[list[RolloutResult]]) -> list[list[RolloutResult]]:
        target_groups = self.get_num_groups_per_model_sync()
        if len(rollout_groups) <= target_groups:
            return rollout_groups
        logger.info(
            "Replay rollout snapshot has %d groups; using the first %d training-valid groups and ignoring %d extra groups.",
            len(rollout_groups),
            target_groups,
            len(rollout_groups) - target_groups,
        )
        return rollout_groups[:target_groups]

    async def prepare_packed_sample_tensor_dict(self, group_results: list[list[RolloutResult]]) -> SampleTensorDict:
        num_trajectories = sum(len(group_result) for group_result in group_results)
        global_batch_size = self.config.megatron_worker.global_batch_size
        assert num_trajectories % global_batch_size == 0, (
            f"Number of rollout trajectories ({num_trajectories}) must be divisible by global_batch_size ({global_batch_size})."
        )
        packed_results = await asyncio.gather(*(self.pack_rollout_group(group_result) for group_result in group_results))
        groups = [RolloutGroup(results=group, filter_type="pass") for group in packed_results]
        return self.collect_packed_samples(TrainGroupBatch(valid_groups=groups))

    async def delete_r3_handles_and_caches(self, rollout_results: Sequence[RolloutResult], *, clear_trainer_caches: bool = True) -> None:
        if not self.config.rollout_worker.enable_routing_replay:
            return
        from axrl.data.sample import collect_unique_handles_from_sample_tensor_dict, collect_unique_handles_from_samples
        from axrl.utils import tensor_store as store

        seen: set[TensorHandle] = set()
        handles: list[TensorHandle] = []

        def add_handles(new_handles: list[TensorHandle]) -> None:
            for handle in new_handles:
                if handle not in seen:
                    seen.add(handle)
                    handles.append(handle)

        for result in rollout_results:
            if result.routing_handles is not None:
                add_handles(result.routing_handles)
            if result.trace is not None:
                add_handles(collect_unique_handles_from_samples(result.trace.turn_samples))
            if result.packed_samples is not None:
                add_handles(collect_unique_handles_from_sample_tensor_dict(result.packed_samples))
        if handles:
            store.delete_batch(handles)
        if clear_trainer_caches:
            assert self.megatron_worker is not None, "Trainer cache cleanup requires a Megatron worker."
            self.megatron_worker.clear_r3_caches()

    def build_sft_samples(self, dataset: BaseDataset | None = None) -> SampleTensorDict:
        """Build SFT samples from the configured dataset."""
        if dataset is None:
            assert self.train_dataset is not None, "PipelineController.initialize() must initialize train_dataset before SFT training."
            dataset = self.train_dataset
        from axrl.data.sample import SampleTensorDict
        from axrl.data.sft_sample_converter import SftSampleConverter

        converter = SftSampleConverter(self.config.megatron_worker.model)
        samples = [converter.process(conv) for conv in dataset._conversations]
        sample_dataset = SampleTensorDict.from_samples(samples, max_length=self.config.megatron_worker.model.seq_length)
        sample_dataset["trajectory_id"] = torch.arange(len(sample_dataset), dtype=torch.long)
        logger.info("Built %d SFT samples from %s.", len(sample_dataset), dataset.__class__.__name__)
        return sample_dataset

    def build_sft_eval_samples(self) -> SampleTensorDict | None:
        if not self.test_datasets:
            return None
        sample_dataset = _concat_sample_tensor_dicts([self.build_sft_samples(dataset) for dataset in self.test_datasets])
        sample_dataset["trajectory_id"] = torch.arange(len(sample_dataset), dtype=torch.long)
        return sample_dataset

    async def run_sft_train(self) -> list[dict[str, float]]:
        assert self.megatron_worker is not None, "sft_train requires a Megatron worker."
        await self.load_checkpoint_if_existed()
        sample_dataset = self.build_sft_samples()
        eval_dataset = self.build_sft_eval_samples()
        curve: list[dict[str, float]] = []
        for _epoch in range(self.config.megatron_worker.num_epochs):
            with Timer("Pipeline SFT train", verbose=True) as timer:
                self.global_step, train_metrics = self.megatron_worker.train(
                    self.global_step,
                    sample_dataset,
                    data_shuffle_seed=self.global_step,
                    compute_logprobs=False,
                )
            scalars = {"timing/train_seconds": timer.elapsed_seconds, **train_metrics}
            train_tag = f"{self.config.megatron_worker.model_role}_train"
            row = {
                "global_step": float(self.global_step),
                "train_loss": float(train_metrics[f"{train_tag}/loss"]),
                "train_grad_norm": float(train_metrics[f"{train_tag}/grad_norm"]),
            }
            if eval_dataset is not None:
                eval_metrics = self.megatron_worker.eval(self.global_step, eval_dataset)
                scalars.update(eval_metrics)
                row["eval_loss"] = float(eval_metrics["eval/loss"])
            self.log_scalars(scalars)
            curve.append(row)
        return curve

    async def run_replay_rl_train(self) -> None:
        assert self.megatron_worker is not None, "replay_rl_train requires a Megatron worker."
        await self.load_checkpoint_if_existed()
        replay_config = self.config.replay_rl_train
        assert (replay_config.sample_dict_path is None) != (replay_config.rollout_groups_path is None), (
            "replay_rl_train requires exactly one of sample_dict_path or rollout_groups_path."
        )
        if replay_config.sample_dict_path is not None:
            sample_dataset: SampleTensorDict = zst_utils.load_zst(Path(replay_config.sample_dict_path), verbose=True)
        else:
            assert replay_config.rollout_groups_path is not None
            loaded_rollouts: list[list[RolloutResult]] = zst_utils.load_zst(Path(replay_config.rollout_groups_path), verbose=True)
            valid_rollouts = self.select_replay_train_rollouts(loaded_rollouts)
            sample_dataset = await self.prepare_packed_sample_tensor_dict(valid_rollouts)
        if not self.config.megatron_worker.use_magi_merged_forward and "merge_info" in sample_dataset.keys():  # noqa: SIM118
            logger.info("Dropping replay merge_info because use_magi_merged_forward=False.")
            sample_dataset = sample_dataset.exclude("merge_info")  # type: ignore[assignment]
        self.log_training_advantages(sample_dataset)
        with Timer("Pipeline replay RL train", verbose=True) as timer:
            self.global_step, train_metrics = self.megatron_worker.train(
                self.global_step,
                sample_dataset,
                data_shuffle_seed=self.global_step,
            )
        self.log_scalars({"timing/train_seconds": timer.elapsed_seconds, **train_metrics})

    def log_online_rollout_batch_metrics(
        self,
        batch: TrainGroupBatch,
        valid_rollouts: list[list[RolloutResult]],
        skipped_rollouts: list[list[RolloutResult]],
        *,
        rollout_elapsed_seconds: float,
    ) -> None:
        self.log_scalars(self.aggregate_filter_type_metrics(batch.filter_type_counts))
        valid_metrics = [result.metric for group in valid_rollouts for result in group]
        all_metrics = valid_metrics + [result.metric for group in skipped_rollouts for result in group]
        all_results = [result for group in valid_rollouts + skipped_rollouts for result in group]
        self.log_scalars(aggregate_response_metrics(valid_metrics, rollout_elapsed_seconds, prefix="Valid-Rollouts"))
        self.log_scalars(aggregate_response_metrics(all_metrics, rollout_elapsed_seconds, prefix="All-Rollouts"))
        self.log_scalars(aggregate_teacher_metrics(all_results))
        if self.train_dataset is not None:
            self.log_scalars(self.train_dataset.get_hist_score_metrics(num_latest_scores=32))

    def prepare_online_training_samples(self, batch: TrainGroupBatch) -> SampleTensorDict:
        sample_dataset = self.collect_packed_samples(batch)
        if self.is_ppo_training():
            with Timer("Pipeline PPO sample preparation", verbose=True) as ppo_prepare_timer:
                value_forward_gpu_usage_infos = self.prepare_ppo_training_samples(sample_dataset)
            self.log_scalars({"timing/ppo_prepare_seconds": ppo_prepare_timer.elapsed_seconds})
            if self.config.megatron_worker.log_gpu_usaegs:
                for gpu_info in value_forward_gpu_usage_infos:
                    self.log_scalars(self.prefix_metrics(gpu_info.to_metrics(), f"ppo-value-forward-GPU/{gpu_info.name}"))
        return sample_dataset

    def train_online_sample_dataset(self, sample_dataset: SampleTensorDict) -> tuple[bool, dict[str, float]]:
        assert self.megatron_worker is not None, "online_rl_train requires a Megatron worker."
        metrics: dict[str, float] = {}
        should_train_actor = True
        train_timer_name = "Pipeline online train"
        train_timing_key = "timing/train_seconds"

        if self.is_ppo_training():
            assert self.value_worker is not None, "PPO training requires value_worker."
            ppo_value_config = self.config.grpo.ppo_value
            assert ppo_value_config is not None, "PPO training requires grpo.ppo_value."
            should_train_actor = self.value_step >= ppo_value_config.num_value_only_updates

            with Timer("Pipeline PPO value train", verbose=True) as value_timer:
                next_value_step, value_metrics = self.value_worker.train(
                    self.value_step,
                    sample_dataset,
                    data_shuffle_seed=self.global_step,
                    compute_logprobs=False,
                )
            self.value_step = next_value_step
            metrics.update({"timing/value_train_seconds": value_timer.elapsed_seconds})
            metrics.update(value_metrics)

            self.value_worker.to_cpu()
            if not should_train_actor:
                logger.info(
                    "Skipping PPO actor train for value-only warmup: value_step=%d num_value_only_updates=%d.",
                    self.value_step,
                    ppo_value_config.num_value_only_updates,
                )
                return False, metrics

            self.megatron_worker.to_gpu()
            train_timer_name = "Pipeline PPO actor train"
            train_timing_key = "timing/actor_train_seconds"

        with Timer(train_timer_name, verbose=True) as train_timer:
            self.global_step, train_metrics = self.megatron_worker.train(
                self.global_step,
                sample_dataset,
                data_shuffle_seed=self.global_step,
            )
        metrics.update({train_timing_key: train_timer.elapsed_seconds})
        metrics.update(train_metrics)
        return should_train_actor, metrics

    def save_checkpoint_if_needed(self, *, actor_trained: bool) -> None:
        online_config = self.config.online_rl_train
        checkpoint_interval = online_config.checkpoint_every_n_global_updates
        if checkpoint_interval is None or not actor_trained or self.global_step % checkpoint_interval != 0:
            return
        assert self.megatron_worker is not None, "Checkpointing requires a Megatron worker."
        assert self.train_dataset is not None, "Checkpointing requires train_dataset."
        assert self.checkpoint_dir is not None, "PipelineController.initialize() must set checkpoint_dir before checkpointing."
        self.save_checkpoint(
            self.megatron_worker,
            self.global_step,
            self.train_dataset,
            checkpoint_dir=self.checkpoint_dir,
            save_hf=self.config.megatron_worker.save_hf_checkpoint,
            most_recent_k=self.config.megatron_worker.most_recent_checkpoint_k,
        )
        if self.value_worker is not None:
            self.value_worker.save_checkpoint(self.global_step)

    async def sync_rollout_after_online_train(self, *, actor_trained: bool) -> None:
        if not actor_trained:
            await self.switch_to_rollout()
            return
        assert self.megatron_worker is not None, "online_rl_train requires a Megatron worker."
        with Timer("Pipeline weight sync", verbose=True) as sync_timer:
            await self.prepare_for_weight_updates()
            self.megatron_worker.update_rollout_model_weights()
            await self.switch_to_rollout()
        self.log_scalars({"timing/sync_seconds": sync_timer.elapsed_seconds})

    async def run_online_rl_train(self) -> None:
        assert self.rollout_worker is not None, "online_rl_train requires a rollout worker."
        assert self.megatron_worker is not None, "online_rl_train requires a Megatron worker."
        await self.switch_to_train()
        await self.load_checkpoint_if_existed()
        await self.prepare_for_weight_updates()
        self.megatron_worker.update_rollout_model_weights()
        online_config = self.config.online_rl_train
        self.prev_eval_global_step = self.global_step
        if online_config.eval_on_start:
            await self.run_evals()
            self.prev_eval_global_step = self.global_step

        train_group_batches = self.stream_online_train_group_batches()
        while self.global_step < online_config.max_global_updates:
            with Timer("Pipeline online rollout", verbose=True) as rollout_timer:
                batch = await anext(train_group_batches)
            valid_rollouts = [group.results for group in batch.valid_groups]
            skipped_rollouts = [group.results for group in batch.skipped_groups]
            self.log_scalars({"timing/rollout_seconds": rollout_timer.elapsed_seconds})
            self.log_online_rollout_batch_metrics(
                batch,
                valid_rollouts,
                skipped_rollouts,
                rollout_elapsed_seconds=rollout_timer.elapsed_seconds,
            )

            sample_dataset = self.prepare_online_training_samples(batch)
            self.log_training_advantages(sample_dataset)
            self.log_latest_resource_metrics("before_megatron_train")
            self.save_training_samples_snapshot(sample_dataset)
            actor_trained, train_metrics = self.train_online_sample_dataset(sample_dataset)
            self.log_scalars(train_metrics)
            await self.delete_r3_handles_and_caches([result for group in valid_rollouts for result in group])
            self.save_rollouts_snapshot(valid_rollouts, skipped_rollouts)
            self.save_checkpoint_if_needed(actor_trained=actor_trained)
            await self.sync_rollout_after_online_train(actor_trained=actor_trained)

    async def load_or_generate_mismatch_rollouts(
        self,
        rollout_path: Path,
        *,
        override: bool,
    ) -> list[list[RolloutResult]]:
        if rollout_path.is_file() and not override:
            logger.info("Loading mismatch rollouts from %s.", rollout_path)
            return zst_utils.load_zst(rollout_path, verbose=True)

        assert self.rollout_worker is not None, "mismatch_test requires a rollout worker when generating rollouts."
        assert self.megatron_worker is not None, "mismatch_test requires a Megatron worker when generating rollouts."
        await self.prepare_for_weight_updates()
        self.megatron_worker.update_rollout_model_weights()
        _, result_queue = self._check_rollout_ready()
        scheduled_groups = self.get_num_groups_per_batch_rollout()
        target_valid_groups = self.get_num_groups_per_model_sync()
        await self.switch_to_rollout()
        conversations = self.build_train_rollout_conversations(scheduled_groups)
        logger.info("Scheduling mismatch rollout groups: groups=%d, rollouts=%d.", scheduled_groups, len(conversations))
        await self.enqueue_rollout_conversations(conversations)
        original_filter_zero_std = self.config.online_rl_train.filter_zero_std
        self.config.online_rl_train.filter_zero_std = False
        batch = await self.collect_scheduled_rollout_groups(
            result_queue,
            scheduled_groups=scheduled_groups,
            max_valid_groups=target_valid_groups,
        )
        self.config.online_rl_train.filter_zero_std = original_filter_zero_std
        assert len(batch.valid_groups) == target_valid_groups, (
            f"Mismatch rollout should collect {target_valid_groups} valid groups when zero-std filtering is disabled, got {len(batch.valid_groups)}."
        )
        valid_rollouts = [group.results for group in batch.valid_groups]
        rollout_path.parent.mkdir(parents=True, exist_ok=True)
        zst_utils.save_zst(valid_rollouts, rollout_path, verbose=True)
        return valid_rollouts

    def compute_ref_and_old_logprobs_for_mismatch(self, sample_dataset: SampleTensorDict) -> list[Any]:
        assert self.megatron_worker is not None, "mismatch_test requires a Megatron worker before computing logprobs."
        self.megatron_worker.copy_weights_to_cpu("cur_weights")
        self.megatron_worker.apply_weights_from_cpu("init_weights")
        sample_dataset["ref_logprobs"], _ = self.megatron_worker.compute_logprobs(sample_dataset)
        self.megatron_worker.apply_weights_from_cpu("cur_weights")
        sample_dataset["old_logprobs"], old_gpu_usage_infos = self.megatron_worker.compute_logprobs(sample_dataset)
        return list(old_gpu_usage_infos)

    async def run_mismatch_test(self) -> None:
        from axrl.metrics.report_mismatch import MCoreRunMetrics, MismatchReportTask, MismatchRunResult, report_mismatch

        assert self.megatron_worker is not None, "mismatch_test requires a Megatron worker."
        mismatch_config = self.config.mismatch_test
        logger.info("Starting pipeline mismatch test: %s", mismatch_config)
        name = mismatch_config.name
        filename = mismatch_config._sanitize_name(name)
        output_dir = mismatch_config.get_output_dir()
        result_dir = mismatch_config.get_result_dir()
        data_dir = mismatch_config.get_exp_data_dir(name)
        log_dir = mismatch_config.get_log_dir()
        fig_dir = mismatch_config.get_fig_dir()
        for path in (output_dir, result_dir, data_dir, log_dir, fig_dir):
            path.mkdir(parents=True, exist_ok=True)

        result_path = mismatch_config.get_result_path(name)
        zst_utils.save_zst(MismatchRunResult(success=False, grpo_config=self.config.grpo.model_copy()), result_path)
        valid_rollout_path = mismatch_config.get_valid_rollouts_path(name)
        sample_dataset_path = mismatch_config.get_training_samples_path(name)

        with Timer("Pipeline mismatch rollouts", verbose=True) as rollout_timer:
            valid_rollouts = await self.load_or_generate_mismatch_rollouts(
                valid_rollout_path,
                override=mismatch_config.override_rollouts_if_exists,
            )
        response_metrics = [result.metric for group in valid_rollouts for result in group]
        response_metrics_agg = aggregate_response_metrics(response_metrics, None, prefix="Mismatch-Rollout")
        total_rollout_tokens = sum(metric.token_count for metric in response_metrics)
        rollout_throughput = total_rollout_tokens / rollout_timer.elapsed_seconds if rollout_timer.elapsed_seconds > 0 else None

        await self.switch_to_train()
        sample_dataset = await self.prepare_packed_sample_tensor_dict(valid_rollouts)
        logger.info("Generated %d training samples from mismatch rollouts.", len(sample_dataset))
        with Timer("Pipeline mismatch MCore logprobs", verbose=True) as mcore_timer:
            old_gpu_usage_infos = self.compute_ref_and_old_logprobs_for_mismatch(sample_dataset)
        await self.delete_r3_handles_and_caches([result for group in valid_rollouts for result in group])
        zst_utils.save_zst(sample_dataset, sample_dataset_path, verbose=True)

        baseline_path: Path | None = None
        if mismatch_config.baseline_samples_path is not None:
            baseline_path = Path(mismatch_config.baseline_samples_path)
        elif mismatch_config.baseline_name is not None:
            baseline_path = mismatch_config.get_training_samples_path(mismatch_config.baseline_name)
        if baseline_path is not None and not baseline_path.exists():
            raise FileNotFoundError(f"Baseline sample dataset not found at {baseline_path}.")

        time_str = datetime.now(tz=UTC).strftime("%Y%m%d-%H%M%S")
        mismatch_metrics = report_mismatch(
            MismatchReportTask(
                exp_name=name,
                exp_sample_data_path=str(sample_dataset_path),
                baseline_sample_data_path=str(baseline_path) if baseline_path is not None else None,
                baseline_name=mismatch_config.baseline_name or "baseline",
                output_fig_path=str(fig_dir / f"mismatch_report-{filename}-{time_str}.png"),
                output_txt_path=str(log_dir / f"top_mismatch-{filename}-{time_str}.log"),
                model=self.config.rollout_worker.model,
            ),
            metric_logger=None,
            step=self.global_step,
        )
        result = MismatchRunResult(
            success=True,
            grpo_config=self.config.grpo.model_copy(),
            metadata={
                "pipeline_config": self.config.model_dump(mode="json"),
                "controller": self.config.controller.model_dump(mode="json"),
                "online_rl_train": self.config.online_rl_train.model_dump(mode="json"),
            },
            response_metrics=response_metrics_agg,
            rollout_throughput=rollout_throughput,
            mcore=MCoreRunMetrics(
                end_to_end_time_sec=mcore_timer.elapsed_seconds,
                old_logprobs_gpu_usage=old_gpu_usage_infos,
            ),
            mismatch_metrics=mismatch_metrics,
        )
        zst_utils.save_zst(result, result_path)
        logger.info("Saved pipeline mismatch result to %s.", result_path)

    def _save_eval_rollouts_if_needed(self, results: Sequence[RolloutResult], *, dataset_name: str) -> None:
        if not self.config.controller.save_eval_rollouts:
            return
        assert self.output_dir is not None, "PipelineController.initialize() must create output_dir before saving eval rollouts."
        rollout_path = self.output_dir / f"eval_rollouts-{dataset_name}-step{self.global_step}.zst"
        zst_utils.save_zst(list(results), rollout_path, verbose=True)
        logger.info(f"Saved pipeline eval rollouts to {rollout_path}.")

    def _log_rollout_metrics(self, results: Sequence[RolloutResult], *, elapsed_seconds: float, prefix: str) -> None:
        assert self.metric_logger is not None, "PipelineController.initialize() must create metric_logger before rollouts."
        response_metrics = [result.metric for result in results]
        metrics = aggregate_response_metrics(response_metrics, elapsed_seconds, prefix=prefix)
        metrics.update(aggregate_teacher_metrics(results))
        metrics[f"{prefix}/rollout_time_sec"] = elapsed_seconds
        self.metric_logger.log_scalars(metrics, step=self.global_step)
