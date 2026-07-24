from __future__ import annotations

import asyncio
import gc
import logging
import random
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Literal, override

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from axrl.agent.rollout_agent import RolloutAgent
from axrl.configs import AXRL_DIR, DatasetConfig, SamplingConfig
from axrl.controller.base_controller import BaseController
from axrl.controller.stage_manager import ColocatedStageManager
from axrl.data import RolloutResult
from axrl.data.sample import SampleTensorDict, _concat_sample_tensor_dicts
from axrl.datasets.base_dataset import BaseDataset
from axrl.envs.math_env import MathEnv
from axrl.metrics.response_metric import (
    ResponseMetric,
    ResponseMetricCalculator,
    aggregate_response_metrics,
    aggregate_response_metrics_by_subset,
)
from axrl.processor.conversation_tokenizer import ConversationTokenizer
from axrl.processor.processor_pool import ProcessorPool
from axrl.ray.resource_group import Request, ResourceGroup
from axrl.trainer.grpo_trainer import GrpoTrainer
from axrl.utils import zst_utils
from axrl.utils.logger import get_metric_logger
from axrl.utils.system_utils import get_open_fd_count
from axrl.utils.timer import Timer

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Sequence

    from axrl.agent.base_agent import BaseAgent
    from axrl.data import Conversation, GenerationInput, GenerationOutput, Sample
    from axrl.data.rollout_trace import RolloutTrace
    from axrl.data.rollout_trace_packing import RolloutTracePackRequest
    from axrl.envs.base_env import BaseEnv
    from axrl.trainer.grpo_exp_config import GrpoExperimentConfig
    from axrl.utils.gpu_utils import GpuUsageInfo
    from axrl.verifier.base_verifier import VerifierInput, VerifierOutput
    from axrl.worker.infer_worker import InferWorker

logger = logging.getLogger(__name__)

GroupFilterType = Literal["pass", "zero_std_all_fail", "zero_std_all_success"]


class GrpoController(BaseController):
    def __init__(self, config: GrpoExperimentConfig) -> None:
        super().__init__()
        self.config = config
        self.train_dataset: BaseDataset
        self.test_datasets: list[BaseDataset] = []
        self.eval_dataset_configs: list[DatasetConfig] = []
        self.metric_calculator: InferWorker[GenerationOutput, ResponseMetric]
        self.conv_tokenizer: InferWorker[Conversation, GenerationInput]
        self.score_providers: dict[str, InferWorker[VerifierInput, VerifierOutput]]
        self._packing_pool: ProcessorPool[RolloutTracePackRequest, SampleTensorDict] | None = None

    @override
    async def initialize(self) -> None:
        self.check_configs()
        self._initialize_output_dir("grpo")
        self._initialize_logger()
        self._initialize_datasets()
        self.source_to_verifier_class = self.get_source_to_verifier_class()
        self._set_default_verifier_class_if_not_specified()
        self._initialize_score_providers()
        self._initialize_model(self.config.rollout_worker.model)
        await self._initialize_workers()
        self._initialize_stage_manager()
        self.metric_calculator = ProcessorPool(ResponseMetricCalculator, config=None, num_processors=2)
        self.conv_tokenizer = ProcessorPool(ConversationTokenizer, config=self.config.rollout_worker.model, num_processors=2)
        await self.tokenize_conversations()
        self.max_tokens = self.config.rollout_worker.model.seq_length
        self.checkpoint_dir = self.config.megatron_worker.get_checkpoint_dir()
        self.global_step = 0
        self._initialize_packing_pool()

    def shutdown(self) -> None:
        if self._packing_pool is not None:
            self._packing_pool.close()
            self._packing_pool = None
        for worker in self.score_providers.values():
            worker.shutdown()
        self.metric_calculator.shutdown()
        self.conv_tokenizer.shutdown()
        worker_resource_group = self.rollout_worker.get_resource_group()
        try:
            self.rollout_worker.shutdown()
            self.megatron_worker.shutdown()
        finally:
            worker_resource_group.shutdown()
        self.metric_logger.close()

    def _all_initialized_datasets(self) -> list[BaseDataset]:
        datasets: list[BaseDataset] = list(self.test_datasets)
        if not self.config.eval_only:
            datasets.append(self.train_dataset)
        return datasets

    def _set_default_verifier_class_if_not_specified(self) -> None:
        # populate default source to verifier mapping if not specified
        for dataset in self._all_initialized_datasets():
            verifier_class = dataset.get_verifier()
            if verifier_class is None:
                continue
            source = dataset._conversations[0].source
            if source not in self.source_to_verifier_class:
                self.source_to_verifier_class[source] = verifier_class
                logger.info(f"Auto-registered verifier {verifier_class.__name__} for dataset source: {source}.")

    def _initialize_score_providers(self) -> None:
        # gather all sources from datasets
        all_sources: set[str] = {conv.source for dataset in self._all_initialized_datasets() for conv in dataset._conversations}
        score_providers: dict[str, InferWorker[VerifierInput, VerifierOutput]] = {}
        for source in all_sources:
            assert source in self.source_to_verifier_class, f"No verifier constructor found for dataset source: {source}."
            verifier_class = self.source_to_verifier_class[source]
            score_providers[source] = ProcessorPool(verifier_class, config=None, num_processors=2)
            logger.info(f"Initialized score provider for dataset source: {source} using verifier {verifier_class.__name__}.")
        self.score_providers = score_providers

    def _initialize_output_dir(self, name: str) -> None:
        output_dir = AXRL_DIR.output / name
        output_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir = output_dir

    def _initialize_logger(self) -> None:
        self.metric_logger = get_metric_logger(self.config.logger)
        self.metric_logger.log_config(self.config)
        self.config.megatron_worker.metric_logger_config = self.config.logger.model_copy()

    async def _initialize_workers(self) -> None:
        config = self.config
        assert config.colocated, "Only colocated=True is supported currently."
        num_gpus = config.rollout_worker.gpus_per_worker()
        resource_group = ResourceGroup([Request(cpu=1, gpu=num_gpus) for _ in range(config.rollout_worker.num_workers)])
        self.megatron_worker = self.get_megatron_worker(config.megatron_worker, resource_group)
        self.rollout_worker = await self.get_rollout_worker(config.rollout_worker, resource_group)

        # build weight updater and connect
        self.megatron_worker.build_weight_updater(self.rollout_worker, bucket_size_gb=1.0)
        self.megatron_worker.connect_rollout_worker()

        # set grpo trainer
        self.megatron_worker.set_trainer(GrpoTrainer(config.grpo))

    def _initialize_datasets(self) -> None:
        if not self.config.eval_only:
            assert self.config.train_datasets is not None
            self.train_dataset_configs = list(self.config.train_datasets)
            self.train_dataset = BaseDataset.concat(self._prepare_datasets(self.train_dataset_configs))
            if self.config.grpo.max_prompt_length is not None:
                self.train_dataset.filter_by_max_prompt_length(self.config.grpo.max_prompt_length)
        if self.config.test_datasets:
            self.test_datasets = self._prepare_datasets(self.config.test_datasets)
            self.eval_dataset_configs = self.config.test_datasets

    def _initialize_stage_manager(self) -> None:
        assert self.rollout_worker is not None
        assert self.megatron_worker is not None
        assert self.config.colocated, "Only colocated=True is supported currently."
        self.stage_manager = ColocatedStageManager(rollout_worker=self.rollout_worker, megatron_worker=self.megatron_worker)

    async def _delete_r3_handles_and_caches(self, rollout_results: Sequence[RolloutResult], *, clear_trainer_caches: bool = True) -> None:
        """Delete R3 routing tensors from rollout traces and optionally clear trainer-side caches."""
        if not self.config.rollout_worker.enable_routing_replay:
            return
        from axrl.data.sample import collect_unique_handles_from_samples
        from axrl.utils import tensor_store as store

        samples = [sample for result in rollout_results if result.trace is not None for sample in result.trace.turn_samples]
        handles = collect_unique_handles_from_samples(samples)
        if handles:
            store.delete_batch(handles)
        if clear_trainer_caches:
            self.megatron_worker.clear_r3_caches()

    async def _warmup_tensor_store(self) -> None:
        """Warm plasma paths between each sglang producer and every megatron consumer.

        Seeds the directory + forces the first cross-node plasma
        transfer so step 0's ``R3 routing materialise`` timing stays
        apples-to-apples with later steps.
        """
        from axrl.utils import tensor_store as store

        with Timer("TensorStore warmup (producer put + consumer get)", verbose=True):
            handles = await self.rollout_worker.warmup_tensor_store()
            self.megatron_worker.warmup_tensor_store(handles)
            store.delete_batch(handles)

    @override
    async def rollout_from_conv(
        self,
        conv: Conversation,
        label: str | list[str],
        sampling_config: SamplingConfig,
        score_provider: InferWorker[VerifierInput, VerifierOutput],
        *,
        return_sample: bool,
    ) -> RolloutResult:
        max_length = sampling_config.max_total_tokens
        if max_length <= 0:
            max_length = self.config.rollout_worker.model.seq_length

        env = MathEnv(
            score_provider=score_provider,
            metric_calculator=self.metric_calculator,
            conv_tokenizer=self.conv_tokenizer,
            conv=conv,
            label=label,
            max_length=max_length,
            return_sample=return_sample,
        )
        agent = RolloutAgent(self.rollout_worker)
        return await self.rollout_from_env(env, agent, sampling_config)

    async def rollout_from_env(self, env: BaseEnv, agent: BaseAgent, sampling_config: SamplingConfig) -> RolloutResult:
        observation = env.conv
        assert observation is not None and observation.gen_state.input_ids is not None
        while True:
            generation_output = await agent.act(observation, sampling_config)
            observation, _, done, sample, response_metric = await env.step(generation_output)
            if not done:
                continue
            return RolloutResult(conversation=observation, trace=sample, metric=response_metric)

    async def eval_dataset(self, dataset: BaseDataset, eval_config: DatasetConfig, sampling_config: SamplingConfig) -> None:
        name = dataset.__class__.__name__
        num_convs = len(dataset)
        rollouts_per_prompt = eval_config.eval_num_rollouts_per_prompt
        assert rollouts_per_prompt is not None, f"eval_num_rollouts_per_prompt must be set for test dataset {eval_config.name}"
        total_rollouts = num_convs * rollouts_per_prompt
        await self.stage_manager.switch_to_rollout()
        max_concurrent = self.config.rollout_worker.max_running_requests_eval or self.config.rollout_worker.max_running_requests
        logger.info(
            f"Starting evaluation on dataset {name} with {num_convs} conversations, "
            f"{rollouts_per_prompt} rollouts per prompt, {total_rollouts} total rollouts, "
            f"max_concurrent={max_concurrent}."
        )

        # Each rollout needs its own Conversation; see ``sample_groups``.
        conv_label_groups: list[list[tuple[Conversation, str | list[str]]]] = []
        for i in range(num_convs):
            conv, label = dataset._conversations[i], dataset.get_label(i)
            conv_label_groups.append([(conv.deep_copy(), label) for _ in range(rollouts_per_prompt)])
        results: list[RolloutResult] = []
        with (
            Timer(f"Evaluation rollouts ({total_rollouts} total)", verbose=True) as timer,
            tqdm(total=total_rollouts, desc="Evaluation rollouts", unit="rollout") as pbar,
        ):
            async for _, group_result in self.stream_group_results(
                conv_label_groups=conv_label_groups,
                sampling_config=sampling_config,
                score_providers=self.score_providers,
                max_concurrent_requests=max_concurrent,
                return_sample=False,
                capture_routing=False,
            ):
                results.extend(group_result)
                pbar.update(len(group_result))

        all_metrics = [result.metric for result in results]
        prefix = f"eval_{name}"
        agg_metrics = aggregate_response_metrics(all_metrics, timer.elapsed_seconds, prefix=prefix)
        agg_metrics[f"{prefix}/eval_time_sec"] = timer.elapsed_seconds
        self.log_scalars(agg_metrics)

        if eval_config.subset_key is not None:
            subset_metrics = aggregate_response_metrics_by_subset(results, eval_config.subset_key, prefix)
            if subset_metrics:
                self.log_scalars(subset_metrics)

        # save evaluation rollouts
        eval_rollout_path = self.output_dir / f"eval_rollouts-{name}-step{self.global_step}.zst"
        zst_utils.save_zst(results, eval_rollout_path, verbose=True)

    async def load_checkpoint_if_existed(self, *, strict: bool = False) -> None:
        await self.stage_manager.switch_to_train()

        if self.checkpoint_dir.is_dir():
            if self.config.eval_only:
                self.global_step = self.megatron_worker.load_checkpoint()
            else:
                self.megatron_worker, self.global_step, self.train_dataset = self.load_checkpoint(
                    self.megatron_worker, self.train_dataset, checkpoint_dir=self.checkpoint_dir
                )
        elif strict:
            raise FileNotFoundError(f"Checkpoint directory {self.checkpoint_dir} does not exist. A valid checkpoint is required for eval_only mode.")

    async def run_evals(self) -> None:
        if not self.test_datasets:
            logger.info("No test_datasets configured; skipping evaluation.")
            return
        sampling_config = self.config.eval_sampling_config.model_copy()
        for dataset, eval_config in zip(self.test_datasets, self.eval_dataset_configs, strict=True):
            await self.eval_dataset(dataset, eval_config, sampling_config)

    async def tokenize_conversations(self) -> None:
        for dataset in self._all_initialized_datasets():
            convs = dataset._conversations
            dataset_name = dataset.__class__.__name__
            with Timer(f"Tokenizing {len(convs)} conversations for dataset {dataset_name}", verbose=True):
                results = await self.conv_tokenizer.batch_generate(convs)
                for conv, gen_input in zip(convs, results, strict=True):
                    conv.gen_state.input_ids = gen_input.input_ids

    def sample_groups(self, dataset: BaseDataset, num_convs: int, group_size: int) -> list[list[tuple[Conversation, str | list[str]]]]:
        sample_indices = dataset.sample(num_convs, sample_type=self.config.grpo.sample_type)
        if self.config.grpo.sort_sampled_prompts_by_response_length:
            sample_indices = dataset.sort_by_mean_lengths(sample_indices)
        groups: list[list[tuple[Conversation, str | list[str]]]] = []
        for i in sample_indices:
            conv, label = dataset.get_conv(i), dataset.get_label(i)
            groups.append([(conv.deep_copy(), label) for _ in range(group_size)])
        return groups

    def get_num_groups_per_batch_rollout(self) -> int:
        grpo_config = self.config.grpo
        global_batch_size = self.config.megatron_worker.global_batch_size
        assert global_batch_size % grpo_config.num_rollouts_per_conversation == 0
        num_convs_per_global_update = global_batch_size // grpo_config.num_rollouts_per_conversation
        num_convs_to_sample = num_convs_per_global_update * grpo_config.batch_rollout_for_n_global_updates
        return num_convs_to_sample

    def get_num_groups_per_model_sync(self) -> int:
        grpo_config = self.config.grpo
        global_batch_size = self.config.megatron_worker.global_batch_size
        total_rollouts_per_model_sync = global_batch_size * grpo_config.model_sync_every_n_global_updates
        assert total_rollouts_per_model_sync % grpo_config.num_rollouts_per_conversation == 0
        num_convs_per_model_sync = total_rollouts_per_model_sync // grpo_config.num_rollouts_per_conversation
        return num_convs_per_model_sync

    def get_filter_type(self, group_metrics: Sequence[ResponseMetric]) -> GroupFilterType:
        """Determine the filter type for a group of metrics."""
        if self.config.grpo.filter_zero_std is False:
            return "pass"

        reward_std = group_metrics[0].score_std
        if reward_std is not None and reward_std > 1e-2:
            return "pass"

        if all(metric.score is not None and np.isclose(metric.score, 1) for metric in group_metrics):
            return "zero_std_all_success"

        if all(metric.score is not None and np.isclose(metric.score, 0) for metric in group_metrics):
            return "zero_std_all_fail"

        return "pass"

    def normalize_group_rewards(
        self,
        dataset: BaseDataset,
        group_result: Sequence[RolloutResult],
    ) -> None:
        """Update score stats and normalize one completed rollout group in place."""
        logger.info(f"Normalizing rewards for a group of {len(group_result)} rollouts.")
        traces = [result.trace for result in group_result]
        assert all(trace is not None for trace in traces), "return_sample=True must produce RolloutTrace objects."
        rollout_traces: list[RolloutTrace] = [trace for trace in traces if trace is not None]
        metrics = [result.metric for result in group_result]
        conversation_ids = [result.conversation.conversation_id for result in group_result]
        assert len(set(conversation_ids)) == 1

        group_scores = [metric.score for metric in metrics if metric.score is not None]
        assert len(group_scores) == len(metrics), "All metrics must have a valid score."
        assert len(group_scores) > 1, "At least two metrics are required to compute group reward stats."
        dataset.update_score_history(conversation_ids[0], group_scores)
        group_mean = float(np.mean(group_scores))
        group_std = float(np.std(group_scores, ddof=1))
        for metric in metrics:
            metric.score_mean = group_mean
            metric.score_std = group_std

        scalar_advantages: list[float] = []
        for trace, metric in zip(rollout_traces, metrics, strict=True):
            assert metric.score is not None
            if group_std < 1e-8:
                advantage = 0.0
            else:
                group_std_for_adv = group_std + 1e-6
                if group_std_for_adv < 0.1:
                    logger.warning(f"Score std is too small: {group_std_for_adv}.")
                advantage = float(np.clip((metric.score - group_mean) / group_std_for_adv, -5.0, 5.0))
            scalar_advantages.append(advantage)

            for sample in trace.turn_samples:
                sample.reward = metric.score
                sample.reward_baseline = group_mean
                sample.advantage = np.where(sample.loss_mask, np.float32(advantage), np.float32(0.0)).astype(np.float32, copy=False)

        adv_mean = float(np.mean(scalar_advantages))
        adv_std = float(np.std(scalar_advantages, ddof=1)) + 1e-6
        logger.info(f"Normalized {len(rollout_traces)} rollout traces with advantage mean: {adv_mean}, std: {adv_std}.")

    def prepare_packed_samples(self, group_results: list[list[RolloutResult]]) -> list[Sample]:
        """Export normalized rollout traces as packed training samples."""
        traces: list[RolloutTrace] = [result.trace for group_result in group_results for result in group_result if result.trace is not None]
        assert traces, "No rollout traces to prepare."
        global_batch_size = self.config.megatron_worker.global_batch_size
        assert len(traces) % global_batch_size == 0
        max_pack_length = self.config.megatron_worker.model.seq_length
        from axrl.data.rollout_trace import pack_rollout_traces_for_train_batches

        samples = pack_rollout_traces_for_train_batches(
            traces,
            max_pack_length=max_pack_length,
            global_batch_size=global_batch_size,
            allow_prefix_sharing=self.config.megatron_worker.use_magi_merged_forward,
        )
        total_trainable_tokens = sum(sum(sample.loss_mask) for sample in samples)
        assert total_trainable_tokens > 0, "Packed samples must contain trainable tokens."
        logger.info(
            f"Prepared {len(samples)} packed training samples from {len(traces)} rollout traces with {total_trainable_tokens} trainable tokens."
        )
        return samples

    def _initialize_packing_pool(self) -> None:
        from axrl.data.rollout_trace_packing import RolloutTracePackingProcessor

        num_workers = self.config.grpo.num_rollouts_per_conversation

        logger.info(
            "Async rollout packing enabled: workers=%d task_granularity=trace.",
            num_workers,
        )
        self._packing_pool = ProcessorPool(
            RolloutTracePackingProcessor,
            config=None,
            num_processors=num_workers,
            timeout_seconds=600,
        )

    async def _pack_rollout_group(self, group_result: Sequence[RolloutResult]) -> None:
        start = time.perf_counter()
        pool = self._packing_pool
        assert pool is not None
        from axrl.data.rollout_trace_packing import RolloutTracePackRequest

        traces = [result.trace for result in group_result if result.trace is not None]
        assert len(traces) == len(group_result), "return_sample=True must produce RolloutTrace objects."
        requests = [
            RolloutTracePackRequest(
                trajectory_id=offset,
                turn_samples=trace.turn_samples,
                max_pack_length=self.config.megatron_worker.model.seq_length,
                allow_prefix_sharing=self.config.megatron_worker.use_magi_merged_forward,
            )
            for offset, trace in enumerate(traces)
        ]
        packed_traces = await pool.batch_generate(requests)
        for rollout_result, sample_batch in zip(group_result, packed_traces, strict=True):
            rollout_result.packed_samples = sample_batch
        logger.debug(
            "Rollout-trace pack job finished in %.2f seconds: traces=%d samples=%d.",
            time.perf_counter() - start,
            len(packed_traces),
            sum(len(sample_batch) for sample_batch in packed_traces),
        )

    def _collect_async_packed_sample_tensor_dict(
        self,
        group_results: list[list[RolloutResult]],
    ) -> SampleTensorDict:
        sample_batches: list[SampleTensorDict] = []
        total_original_trainable_tokens = 0
        total_packed_trainable_tokens = 0
        trajectory_id = 0
        for group_result in group_results:
            for result in group_result:
                assert result.trace is not None
                assert result.packed_samples is not None, "packing task completed without attaching packed samples"
                sample_batch = result.packed_samples
                sample_batch["trajectory_id"] = torch.full_like(sample_batch["trajectory_id"], trajectory_id)
                sample_batches.append(sample_batch)
                total_original_trainable_tokens += sum(sum(sample.loss_mask) for sample in result.trace.turn_samples)
                total_packed_trainable_tokens += int(sample_batch["loss_mask"].sum().item())
                trajectory_id += 1

        assert total_original_trainable_tokens > 0, "Packed traces must contain trainable tokens."
        assert total_packed_trainable_tokens == total_original_trainable_tokens
        sample_dataset = _concat_sample_tensor_dicts(sample_batches)
        sample_dataset["index"] = torch.arange(len(sample_dataset), dtype=sample_dataset["index"].dtype, device=sample_dataset["index"].device)
        total_trainable_tokens = int(sample_dataset["loss_mask"].sum().item())
        assert total_trainable_tokens == total_original_trainable_tokens
        logger.info(
            "Prepared %d async-packed tensorized training samples from %d rollout traces with %d trainable tokens and width %d.",
            len(sample_dataset),
            trajectory_id,
            total_trainable_tokens,
            int(sample_dataset["input_ids"].shape[1]),
        )
        return sample_dataset

    def aggregate_filter_type_metrics(self, group_filter_types: Counter[GroupFilterType]) -> dict[str, float]:
        total_skips = sum(group_filter_types.values())
        metrics = {}
        assert total_skips > 0
        for reason, count in group_filter_types.items():
            metrics[f"group_filter_type__{reason}"] = count / total_skips
        return metrics

    def log_scalars(self, scalars: dict[str, float]) -> None:
        assert self.metric_logger is not None
        self.metric_logger.log_scalars(scalars, step=self.global_step)

    async def run_evals_if_needed(self) -> None:
        if self.global_step - self.prev_eval_global_step < self.config.grpo.eval_every_n_global_updates:
            return
        await self.run_evals()
        self.prev_eval_global_step = self.global_step

    def check_configs(self) -> None:
        grpo_config = self.config.grpo
        assert grpo_config.reward_mean_type == "group", "GRPO controller now supports only group reward mean normalization."
        assert grpo_config.reward_std_type == "group", "GRPO controller now supports only group reward std normalization."
        assert not grpo_config.normalize_advantage_by_batch_std, "Batch advantage std normalization is no longer done in the controller."
        assert grpo_config.turn_reward_alpha == 0.0, "Turn reward normalization is disabled for packed rollout samples."
        assert grpo_config.checkpoint_every_n_global_updates % grpo_config.model_sync_every_n_global_updates == 0
        if grpo_config.rollout_save_every_n_global_updates is not None:
            assert grpo_config.rollout_save_every_n_global_updates % grpo_config.model_sync_every_n_global_updates == 0, (
                f"rollout_save_every_n_global_updates ({grpo_config.rollout_save_every_n_global_updates}) "
                f"must be a multiple of model_sync_every_n_global_updates ({grpo_config.model_sync_every_n_global_updates})."
            )
        if self.config.eval_only:
            assert self.config.test_datasets, "eval_only=True requires at least one test_datasets entry."
        megatron_config = self.config.megatron_worker
        if megatron_config.reset_init_weights_every_k_steps is not None:
            assert megatron_config.reset_init_weights_every_k_steps % grpo_config.model_sync_every_n_global_updates == 0, (
                f"reset_init_weights_every_k_steps ({megatron_config.reset_init_weights_every_k_steps}) "
                f"must be a multiple of model_sync_every_n_global_updates ({grpo_config.model_sync_every_n_global_updates})."
            )

    async def start(self) -> None:
        if self.config.mismatch_test.enabled:
            logger.info("Mismatch test enabled, starting mismatch test without training.")
            await self.run_mismatch_test()
            return

        if self.config.debug_train:
            logger.info("Debug train enabled, starting training from snapshot rollouts without streaming.")
            await self.train_from_snapshot_rollouts()
            return

        if self.config.eval_only:
            logger.info("Eval-only mode enabled, running evaluation without training.")
            await self.run_eval_only()
            return

        await self.run_train()

    async def _rollout_for_first_updates(
        self,
        rollout_path: Path,
        *,
        override: bool,
    ) -> list[list[RolloutResult]]:
        if rollout_path.is_file() and not override:
            logger.info(f"Rollout file {rollout_path} already exists, skipping first updates rollout.")
            return zst_utils.load_zst(rollout_path, verbose=True)

        await self.stage_manager.switch_to_weight_sync()
        self.megatron_worker.update_rollout_model_weights()
        await self.stage_manager.switch_to_rollout()
        num_groups_per_batch_rollout = self.get_num_groups_per_batch_rollout()
        num_groups_per_model_sync = self.get_num_groups_per_model_sync()

        async for valid_rollouts, _ in self.stream_rollouts(
            num_groups_per_model_sync=num_groups_per_model_sync,
            num_groups_per_batch_rollout=num_groups_per_batch_rollout,
        ):
            zst_utils.save_zst(valid_rollouts, rollout_path, verbose=True)
            return valid_rollouts

        raise RuntimeError("Reached max global updates without completing first rollouts.")

    def _refresh_mismatch_ref_and_old_logprobs(self, sample_dataset: SampleTensorDict) -> list[GpuUsageInfo]:
        self.megatron_worker.copy_weights_to_cpu("cur_weights")
        self.megatron_worker.apply_weights_from_cpu("init_weights")
        sample_dataset["ref_logprobs"], _ = self.megatron_worker.compute_logprobs(sample_dataset)
        self.megatron_worker.apply_weights_from_cpu("cur_weights")
        sample_dataset["old_logprobs"], old_gpu_usage_infos = self.megatron_worker.compute_logprobs(sample_dataset)
        return old_gpu_usage_infos

    async def run_mismatch_test(self) -> None:
        from axrl.metrics.report_mismatch import MCoreRunMetrics, MismatchReportTask, MismatchRunResult, report_mismatch

        mismatch_test_config = self.config.mismatch_test
        logger.info(f"Starting mismatch test: {mismatch_test_config}")
        assert not self.config.grpo.strict_on_policy, "strict_on_policy should be False for mismatch test."
        await self.initialize()
        name = mismatch_test_config.name
        filename = mismatch_test_config._sanitize_name(name)
        output_dir = mismatch_test_config.get_output_dir()
        result_dir = mismatch_test_config.get_result_dir()
        data_dir = mismatch_test_config.get_exp_data_dir(name)
        log_dir = mismatch_test_config.get_log_dir()
        fig_dir = mismatch_test_config.get_fig_dir()
        output_dir.mkdir(parents=True, exist_ok=True)
        result_dir.mkdir(parents=True, exist_ok=True)
        data_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)
        fig_dir.mkdir(parents=True, exist_ok=True)
        result_path = mismatch_test_config.get_result_path(name)
        zst_utils.save_zst(MismatchRunResult(success=False, grpo_config=self.config.grpo.model_copy()), result_path)
        valid_rollout_path = mismatch_test_config.get_valid_rollouts_path(name)
        sample_dataset_path = mismatch_test_config.get_training_samples_path(name)
        with Timer() as rollout_timer:
            valid_rollouts = await self._rollout_for_first_updates(
                valid_rollout_path,
                override=mismatch_test_config.override_rollouts_if_exists,
            )
        response_metrics = [result.metric for group_result in valid_rollouts for result in group_result]
        response_metrics_agg = aggregate_response_metrics(response_metrics, None, prefix="Mismatch-Rollout")
        total_rollout_tokens = sum(metric.token_count for metric in response_metrics)
        rollout_throughput = total_rollout_tokens / rollout_timer.elapsed_seconds if rollout_timer.elapsed_seconds > 0 else None
        await self.stage_manager.switch_to_train()
        samples = self.prepare_packed_samples(valid_rollouts)
        sample_dataset = SampleTensorDict.from_samples(samples, max_length=self.config.megatron_worker.model.seq_length)
        logger.info(f"Generated {len(sample_dataset)} training samples from rollouts.")
        await self._warmup_tensor_store()
        with Timer() as mcore_timer:
            old_gpu_usage_infos = self._refresh_mismatch_ref_and_old_logprobs(sample_dataset)
        await self._delete_r3_handles_and_caches([result for group_result in valid_rollouts for result in group_result])
        logger.info(f"Updated reference and old logprobs for {len(sample_dataset)} samples.")
        zst_utils.save_zst(sample_dataset, sample_dataset_path, verbose=True)
        baseline_path: Path | None = None
        if mismatch_test_config.baseline_samples_path is not None:
            baseline_path = Path(mismatch_test_config.baseline_samples_path)
        elif mismatch_test_config.baseline_name is not None:
            baseline_path = mismatch_test_config.get_training_samples_path(mismatch_test_config.baseline_name)

        if baseline_path is not None and not baseline_path.exists():
            raise FileNotFoundError(f"Baseline sample dataset not found at {baseline_path}.")

        time_str = pd.Timestamp.now().strftime("%Y%m%d-%H%M%S")
        mismatch_metrics = report_mismatch(
            MismatchReportTask(
                exp_name=name,
                exp_sample_data_path=str(sample_dataset_path),
                baseline_sample_data_path=str(baseline_path) if baseline_path is not None else None,
                baseline_name=mismatch_test_config.baseline_name or "baseline",
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
            response_metrics=response_metrics_agg,
            rollout_throughput=rollout_throughput,
            mcore=MCoreRunMetrics(
                end_to_end_time_sec=mcore_timer.elapsed_seconds,
                old_logprobs_gpu_usage=list(old_gpu_usage_infos),
            ),
            mismatch_metrics=mismatch_metrics,
        )
        zst_utils.save_zst(result, result_path)
        logger.info(f"Saved mismatch result to {result_path}: {asdict(result)}")
        self.shutdown()

    async def stream_rollouts(  # noqa: PLR0915
        self,
        num_groups_per_model_sync: int,
        num_groups_per_batch_rollout: int,
    ) -> AsyncGenerator[
        tuple[
            list[list[RolloutResult]],
            list[list[RolloutResult]],
        ],
        None,
    ]:
        """Yield ``(valid_rollouts, skipped_rollouts)`` per training step.

        ``valid_rollouts`` has length ``num_groups_per_model_sync`` and
        drives training. ``skipped_rollouts`` contains the filter-rejected
        groups accumulated during the same step. Each rollout result carries
        its conversation, trace, metric, and optional packed samples.
        """
        sampling_config = self.config.rollout_worker.sampling_config
        self.prev_eval_global_step = self.global_step
        group_filter_types: Counter[GroupFilterType] = Counter()
        valid_rollouts: list[list[RolloutResult]] = []
        skipped_rollouts: list[list[RolloutResult]] = []
        packing_tasks: list[asyncio.Task[None]] = []
        assert self._packing_pool is not None, "initialize() must create the rollout packing pool before stream_rollouts()."
        if gc.isenabled():
            gc.disable()
            logger.info("Disabled Python cyclic GC inside stream_rollouts.")
        rollout_timer = Timer()
        rollout_timer.start()
        pbar: tqdm = tqdm(total=num_groups_per_model_sync, desc="Valid groups collected", unit="group")
        while self.global_step < self.config.grpo.max_global_updates:
            conv_label_groups = self.sample_groups(
                dataset=self.train_dataset,
                num_convs=num_groups_per_batch_rollout,
                group_size=self.config.grpo.num_rollouts_per_conversation,
            )
            group_result: Sequence[RolloutResult]
            await self.stage_manager.switch_to_rollout()
            generated_groups = 0
            logger.info(f"Starting rollout for {len(conv_label_groups)} groups.")
            async for _, group_result in self.stream_group_results(
                conv_label_groups=conv_label_groups,
                sampling_config=sampling_config,
                score_providers=self.score_providers,
                max_concurrent_requests=self.config.rollout_worker.max_running_requests,
                return_sample=True,
                capture_routing=self.config.rollout_worker.enable_routing_replay,
            ):
                generated_groups += 1
                metrics = [result.metric for result in group_result]
                self.normalize_group_rewards(self.train_dataset, group_result)
                filter_type = self.get_filter_type(metrics)
                group_filter_types.update([filter_type])
                if filter_type != "pass":
                    logger.debug(f"Skipping group due to filter type: {filter_type}.")
                    await self._delete_r3_handles_and_caches(group_result, clear_trainer_caches=False)
                    skipped_rollouts.append(list(group_result))
                    continue
                rollout_group = list(group_result)
                valid_rollouts.append(rollout_group)
                packing_tasks.append(asyncio.create_task(self._pack_rollout_group(rollout_group)))
                pbar.update(1)
                if len(valid_rollouts) < num_groups_per_model_sync:
                    continue
                if self.config.grpo.strict_on_policy and generated_groups < num_groups_per_batch_rollout:
                    # Waste rollouts, only for testing strict on-policy training
                    assert self.config.grpo.model_sync_every_n_global_updates == 1
                    assert self.config.grpo.batch_rollout_for_n_global_updates == 1
                    logger.info(f"Strict on-policy, wait current batch to finish: {generated_groups}/{num_groups_per_batch_rollout} groups.")
                    continue
                with tqdm(total=len(packing_tasks), desc="Collecting packed rollout groups", unit="group") as packing_pbar:
                    for task in asyncio.as_completed(packing_tasks):
                        await task
                        packing_pbar.update(1)
                random.shuffle(valid_rollouts)
                if len(valid_rollouts) > num_groups_per_model_sync:
                    extra_rollouts = valid_rollouts[num_groups_per_model_sync:]
                    valid_rollouts = valid_rollouts[:num_groups_per_model_sync]
                    skipped_rollouts.extend(extra_rollouts)
                    await self._delete_r3_handles_and_caches([result for group in extra_rollouts for result in group], clear_trainer_caches=False)

                # rollout finished, log rollout metrics
                rollout_timer.stop()
                self.log_scalars({"timing/rollout_seconds": rollout_timer.elapsed_seconds})
                valid_metrics = [result.metric for group in valid_rollouts for result in group]
                all_metrics = valid_metrics + [result.metric for group in skipped_rollouts for result in group]
                self.log_scalars(aggregate_response_metrics(valid_metrics, rollout_timer.elapsed_seconds, prefix="Valid-Rollouts"))
                self.log_scalars(aggregate_response_metrics(all_metrics, rollout_timer.elapsed_seconds, prefix="All-Rollouts"))
                self.log_scalars(self.aggregate_filter_type_metrics(group_filter_types))
                self.log_scalars(self.train_dataset.get_hist_score_metrics(num_latest_scores=32))
                self.log_scalars({"system/open_fd_count": get_open_fd_count()})
                if self.config.colocated:
                    await self.rollout_worker.pause_generation()

                yield valid_rollouts, skipped_rollouts

                valid_rollouts.clear()
                skipped_rollouts.clear()
                group_filter_types.clear()
                packing_tasks.clear()
                # Future improvement: reduce rollout/sample object churn so this explicit
                # collection can be removed or moved out of the rollout/training boundary.
                gc.collect()
                rollout_timer.start()
                pbar = tqdm(total=num_groups_per_model_sync, desc="Valid groups collected", unit="group")

            # No rollouts are currently in flight; run evaluation if needed.
            await self.run_evals_if_needed()

    def save_rollouts_snapshot(self, valid_rollouts: list[list[RolloutResult]]) -> None:
        """Save a rollouts snapshot without serializing async-packed tensors."""
        save_interval = self.config.grpo.rollout_save_every_n_global_updates
        if save_interval is not None and self.global_step % save_interval == 0:
            suffix = f"step{self.global_step}"
        else:
            suffix = "latest"
        rollout_path = self.output_dir / f"{self.config.grpo.rollout_save_filename}-{suffix}.zst"
        packed_samples = [(result, result.packed_samples) for group in valid_rollouts for result in group if result.packed_samples is not None]
        try:
            for result, _ in packed_samples:
                result.packed_samples = None
            zst_utils.save_zst(valid_rollouts, rollout_path, verbose=True)
        finally:
            for result, sample_batch in packed_samples:
                result.packed_samples = sample_batch

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

    async def run_eval_only(self) -> None:
        """Load a checkpoint and run evaluation only, then exit.

        Requires a valid checkpoint to exist; raises FileNotFoundError otherwise.
        """
        await self.initialize()
        await self.load_checkpoint_if_existed(strict=False)

        await self.stage_manager.switch_to_weight_sync()
        self.megatron_worker.update_rollout_model_weights()
        await self.run_evals()
        logger.info(f"Eval-only completed at global_step={self.global_step}.")
        self.shutdown()

    async def run_train(self) -> None:
        await self.initialize()
        await self.load_checkpoint_if_existed()
        await self.stage_manager.switch_to_weight_sync()
        self.megatron_worker.update_rollout_model_weights()
        if self.config.eval_on_start:
            await self.run_evals()

        await self._warmup_tensor_store()
        logger.info(f"Starting training loop from step {self.global_step}.")
        num_groups_per_batch_rollout = self.get_num_groups_per_batch_rollout()
        num_groups_per_model_sync = self.get_num_groups_per_model_sync()
        async for valid_rollouts, skipped_rollouts in self.stream_rollouts(
            num_groups_per_model_sync=num_groups_per_model_sync,
            num_groups_per_batch_rollout=num_groups_per_batch_rollout,
        ):
            # start to train model
            await self.stage_manager.switch_to_train()
            sample_dataset = self._collect_async_packed_sample_tensor_dict(valid_rollouts)
            self.log_training_advantages(sample_dataset)
            with Timer() as train_timer:
                self.global_step, _ = self.megatron_worker.train(
                    self.global_step,
                    sample_dataset,
                    data_shuffle_seed=self.global_step,
                )
            self.log_scalars({"timing/train_seconds": train_timer.elapsed_seconds})
            await self._delete_r3_handles_and_caches([result for group in valid_rollouts for result in group])
            rollouts_to_save = valid_rollouts + skipped_rollouts if self.config.grpo.save_all_rollouts else valid_rollouts
            self.save_rollouts_snapshot(rollouts_to_save)
            if self.global_step % self.config.grpo.checkpoint_every_n_global_updates == 0:
                self.save_checkpoint(
                    self.megatron_worker,
                    self.global_step,
                    self.train_dataset,
                    checkpoint_dir=self.checkpoint_dir,
                    save_hf=self.config.megatron_worker.save_hf_checkpoint,
                    most_recent_k=self.config.megatron_worker.most_recent_checkpoint_k,
                )
            valid_rollouts.clear()
            skipped_rollouts.clear()
            del sample_dataset

            # sync model weights from megatron to rollout worker
            with Timer() as sync_timer:
                await self.stage_manager.switch_to_weight_sync()
                self.megatron_worker.update_rollout_model_weights()
                await self.stage_manager.switch_to_rollout()
            self.log_scalars({"timing/sync_seconds": sync_timer.elapsed_seconds})

        self.shutdown()

    async def train_from_snapshot_rollouts(self) -> None:
        self.check_configs()
        self._initialize_output_dir("grpo")
        self._initialize_logger()
        self._initialize_model(self.config.rollout_worker.model)
        # initialize megatron worker only
        config = self.config
        num_gpus = config.rollout_worker.gpus_per_worker()
        resource_group = ResourceGroup([Request(cpu=1, gpu=num_gpus) for _ in range(config.rollout_worker.num_workers)])
        self.megatron_worker = self.get_megatron_worker(config.megatron_worker, resource_group)
        self.megatron_worker.set_trainer(GrpoTrainer(config.grpo))
        self.megatron_worker.to_gpu()
        self.global_step = 0
        path = self.output_dir / f"{self.config.grpo.rollout_save_filename}-latest.zst"
        valid_rollouts: list[list[RolloutResult]] = zst_utils.load_zst(path)
        logger.info(f"Loaded {len(valid_rollouts)} valid rollouts from {path}.")
        samples = self.prepare_packed_samples(valid_rollouts)
        with Timer() as train_timer:
            self.global_step, _ = self.megatron_worker.train(
                self.global_step,
                SampleTensorDict.from_samples(samples, max_length=self.config.megatron_worker.model.seq_length),
                data_shuffle_seed=self.global_step,
            )
        self.log_scalars({"timing/train_seconds": train_timer.elapsed_seconds})
