from __future__ import annotations

import asyncio
import logging
import shutil
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypeVar

from transformers import AutoTokenizer

from axrl.datasets import get_dataset
from axrl.ray.ray_cpu_worker import RayCPUWorker
from axrl.ray.resource_group import Request, ResourceGroup
from axrl.utils import zst_utils
from axrl.utils.hf.download_model_from_hf import download_model
from axrl.utils.timer import Timer

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Sequence
    from pathlib import Path

    from axrl.configs import DatasetConfig, MegatronWorkerConfig, ModelConfig, RolloutWorkerConfig, SamplingConfig
    from axrl.data import Conversation, RolloutResult
    from axrl.datasets.base_dataset import BaseDataset
    from axrl.processor.base_processor import BaseProcessor
    from axrl.ray.ray_megatron_worker import RayMegatronWorker
    from axrl.ray.ray_rollout_worker import RayRolloutWorker
    from axrl.verifier.base_verifier import BaseVerifier, VerifierInput, VerifierOutput
    from axrl.worker.infer_worker import InferWorker

logger = logging.getLogger(__name__)

InT = TypeVar("InT")
OutT = TypeVar("OutT")


@dataclass(frozen=True, slots=True)
class RequestTaskMetadata:
    group_index: int
    item_index: int


@dataclass(slots=True)
class PartialGroupState:
    group_size: int
    results: list[RolloutResult | None] = field(init=False)
    finished_count: int = 0

    def __post_init__(self) -> None:
        self.results = [None] * self.group_size

    def set_result(self, item_index: int, result: RolloutResult) -> None:
        assert self.results[item_index] is None, f"Duplicate result for item_index={item_index}."
        self.results[item_index] = result
        self.finished_count += 1

    def is_complete(self) -> bool:
        return self.finished_count == self.group_size

    def get_results(self) -> list[RolloutResult]:
        assert self.is_complete(), "Group results are not complete yet."
        completed_results: list[RolloutResult] = []
        for item_index, result in enumerate(self.results):
            assert result is not None, f"Missing result for item_index={item_index}."
            completed_results.append(result)
        return completed_results


class BaseController:
    def __init__(self) -> None:
        self._rollout_session_counter = 0

    def get_source_to_verifier_class(self) -> dict[str, type[BaseVerifier]]:
        """Get a mapping of data source names to their corresponding verifier classes."""
        from axrl.verifier.dapo_verifier import DapoVerifier
        from axrl.verifier.gsm8k import GSM8KVerifier

        return {
            "openai/gsm8k": GSM8KVerifier,
            "BytedTsinghua-SIA/DAPO-Math-17k": DapoVerifier,
            "BytedTsinghua-SIA/AIME-2024": DapoVerifier,
        }

    async def initialize(self) -> None:
        pass

    async def switch_to_train(self) -> None:
        raise NotImplementedError

    async def prepare_for_weight_updates(self) -> None:
        raise NotImplementedError

    async def switch_to_rollout(self) -> None:
        raise NotImplementedError

    async def get_rollout_worker(self, config: RolloutWorkerConfig, resource_group: ResourceGroup) -> RayRolloutWorker:
        from axrl.ray.ray_rollout_worker import RayRolloutWorker

        remote_rollout_worker = RayRolloutWorker.initialize_remote_actor(config, resource_group)
        rollout_worker = RayRolloutWorker(remote_rollout_worker)
        await rollout_worker.release_gpu_memory(backup_weights_on_cpu=False)
        return rollout_worker

    def get_megatron_worker(self, config: MegatronWorkerConfig, resource_group: ResourceGroup) -> RayMegatronWorker:
        from axrl.ray.ray_megatron_worker import RayMegatronWorker

        megatron_worker = RayMegatronWorker(config, resource_group=resource_group)
        megatron_worker.initialize()
        megatron_worker.copy_weights_to_cpu(name="init_weights")
        megatron_worker.to_cpu()
        return megatron_worker

    def _initialize_model(self, model_config: ModelConfig) -> None:
        model_dir = model_config.get_full_path()
        if model_dir.exists():
            logger.info(f"Model already exists at {model_dir}.")
            return
        download_model(config=model_config)
        logger.info(f"Model downloaded to {model_dir}.")

    @staticmethod
    def start_cpu_worker(
        processor_cls: type[BaseProcessor[InT, OutT]],
        config: Any,
        num_cpus: int = 2,
        timeout_seconds: float = 60.0,
    ) -> RayCPUWorker[InT, OutT]:
        resource_group = ResourceGroup([Request(gpu=0, cpu=num_cpus)])
        worker = RayCPUWorker(processor_cls, config=config, resource_group=resource_group, timeout_seconds=timeout_seconds)
        worker.initialize()
        logger.info(f"{processor_cls.__name__} CPU worker started.")
        return worker

    def get_pad_token_id(self, model_config: ModelConfig) -> int:
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_config.get_full_path(),
            trust_remote_code=True,
        )
        pad_token_id = self.tokenizer.pad_token_id
        assert pad_token_id is not None
        logger.info(f"Pad token ID for model {model_config.name}: {pad_token_id}")
        return pad_token_id

    def _prepare_datasets(self, items: Sequence[DatasetConfig] | None) -> list[BaseDataset]:
        """Create and initialize datasets from a typed list of DatasetConfig."""
        datasets: list[BaseDataset] = []
        for cfg in items or []:
            dataset = get_dataset(cfg)
            dataset.initialize()
            datasets.append(dataset)
        return datasets

    def save_checkpoint(
        self,
        megatron_worker: RayMegatronWorker,
        global_step: int,
        train_dataset: BaseDataset,
        checkpoint_dir: Path,
        *,
        save_hf: bool = False,
        most_recent_k: int = 2,
        max_retries: int = 5,
    ) -> None:
        logger.info(f"Checkpoint training state at step {global_step} to {checkpoint_dir}.")
        extra_info_dir = checkpoint_dir.parent / "checkpoint_extra_info"
        extra_info_dir.mkdir(parents=True, exist_ok=True)
        extra_info_path = extra_info_dir / f"{global_step}.zst"
        extra_info = train_dataset._score_history
        with Timer("Checkpoint training state", verbose=True):
            for attempt in range(1, max_retries + 1):
                try:
                    megatron_worker.save_checkpoint(global_step)
                    zst_utils.save_zst(extra_info, extra_info_path, verbose=True)
                    break
                except Exception:
                    logger.exception(f"Failed to save checkpoint at step {global_step}, attempt {attempt}/{max_retries}.")
            else:
                raise RuntimeError(f"Failed to save checkpoint at step {global_step} after {max_retries} attempts.")
        logger.info(f"Checkpoint training state at step {global_step} completed.")

        if save_hf:
            self.save_hf_checkpoint(megatron_worker, global_step, checkpoint_dir, most_recent_k=most_recent_k)

    def save_hf_checkpoint(
        self,
        megatron_worker: RayMegatronWorker,
        global_step: int,
        checkpoint_dir: Path,
        *,
        most_recent_k: int = 2,
    ) -> None:
        hf_parent = checkpoint_dir.parent / "hf"
        hf_dir = hf_parent / f"step_{global_step}"
        hf_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving HF checkpoint at step {global_step} to {hf_dir}.")
        with Timer(f"Save HF checkpoint at step {global_step}", verbose=True):
            try:
                megatron_worker.save_hf_pretrained(hf_dir)
            except Exception:
                logger.exception(f"Failed to save HF checkpoint at step {global_step}, skipping.")
                return
        logger.info(f"HF checkpoint saved to {hf_dir}.")
        self._cleanup_old_hf_checkpoints(hf_parent, most_recent_k)

    @staticmethod
    def _cleanup_old_hf_checkpoints(hf_parent: Path, most_recent_k: int) -> None:
        if most_recent_k <= 0:
            return
        step_dirs = sorted(
            (d for d in hf_parent.glob("step_*") if d.is_dir()),
            key=lambda d: int(d.name.split("_", 1)[1]),
        )
        for old_dir in step_dirs[:-most_recent_k]:
            logger.info(f"Removing old HF checkpoint: {old_dir}")
            try:
                shutil.rmtree(old_dir)
            except Exception:
                logger.exception(f"Failed to remove old HF checkpoint {old_dir}; continuing.")

    def load_checkpoint(
        self, megatron_worker: RayMegatronWorker, dataset: BaseDataset, checkpoint_dir: Path
    ) -> tuple[RayMegatronWorker, int, BaseDataset]:
        logger.info(f"Load training state from {checkpoint_dir}.")
        extra_info_dir = checkpoint_dir.parent / "checkpoint_extra_info"

        with Timer("Load training state", verbose=True):
            assert checkpoint_dir.exists() and checkpoint_dir.is_dir()
            for retry in range(3):
                try:
                    global_step = megatron_worker.load_checkpoint()
                    logger.info(f"Load training state from {checkpoint_dir} completed.")

                    extra_info_path = extra_info_dir / f"{global_step}.zst"
                    assert extra_info_path.exists(), f"Extra info file {extra_info_path} does not exist."
                    extra_info = zst_utils.load_zst(extra_info_path, verbose=True)
                    dataset._score_history = extra_info
                    logger.info(f"Loaded score history for training data: {dataset.get_hist_score_metrics(num_latest_scores=16)}")

                    return megatron_worker, global_step, dataset
                except Exception as _:
                    logger.exception(f"Failed to load checkpoint from {checkpoint_dir}, retry: {retry}.")
                    continue

        raise RuntimeError("Failed to load checkpoint after multiple attempts.")

    def _assign_rollout_session_id(self, conv: Conversation, metadata: RequestTaskMetadata) -> None:
        assert conv.conversation_id, "Conversation ID must be set before rollout scheduling."
        self._rollout_session_counter += 1
        conv.gen_state.session_id = f"{conv.conversation_id}:rollout:{self._rollout_session_counter}:g{metadata.group_index}:i{metadata.item_index}"

    async def stream_group_results(
        self,
        conv_label_groups: list[list[tuple[Conversation, str | list[str]]]],
        sampling_config: SamplingConfig,
        max_concurrent_requests: int,
        score_providers: dict[str, InferWorker[VerifierInput, VerifierOutput]],
        *,
        return_sample: bool,
        capture_routing: bool,
    ) -> AsyncGenerator[tuple[int, Sequence[RolloutResult]], None]:
        """Generate responses for groups of inference requests with streaming results.

        Schedules one task per request while preserving the external interface of
        yielding complete groups. Request results are buffered per group and
        emitted only after the full group has finished, in original group order.

        Yields:
            Tuples of (group index, completed rollout results) for each completed group.
        """
        assert len(conv_label_groups) > 0
        pending_requests = [
            (RequestTaskMetadata(group_index=group_index, item_index=item_index), conv, label)
            for group_index, conv_labels in enumerate(conv_label_groups)
            for item_index, (conv, label) in enumerate(conv_labels)
        ]
        partial_groups = {group_index: PartialGroupState(group_size=len(conv_labels)) for group_index, conv_labels in enumerate(conv_label_groups)}
        active_tasks: dict[asyncio.Task[RolloutResult], RequestTaskMetadata] = {}
        next_request_index = 0
        completed_groups = 0
        total_groups = len(conv_label_groups)

        while completed_groups < total_groups:
            while next_request_index < len(pending_requests) and len(active_tasks) < max_concurrent_requests:
                metadata, conv, label = pending_requests[next_request_index]
                next_request_index += 1
                self._assign_rollout_session_id(conv, metadata)
                conv.gen_state.capture_routing = capture_routing
                logger.debug(f"Scheduling rollout for group: {metadata.group_index}, session id: {conv.gen_state.session_id}.")
                task = asyncio.create_task(
                    self.rollout_from_conv(
                        conv,
                        label,
                        sampling_config,
                        score_provider=score_providers[conv.source],
                        return_sample=return_sample,
                    )
                )
                active_tasks[task] = metadata

            if not active_tasks:
                break

            done_tasks, _ = await asyncio.wait(active_tasks.keys(), return_when=asyncio.FIRST_COMPLETED)
            for task in done_tasks:
                metadata = active_tasks.pop(task)
                result = task.result()
                group_state = partial_groups[metadata.group_index]
                group_state.set_result(metadata.item_index, result)
                logger.debug(f"Completed rollout for group: {metadata.group_index}, session id: {result.conversation.gen_state.session_id}.")
                if not group_state.is_complete():
                    continue
                completed_groups += 1
                completed_group = group_state.get_results()
                del partial_groups[metadata.group_index]
                yield metadata.group_index, completed_group

    async def rollout_from_conv(
        self,
        conv: Conversation,
        label: str | list[str],
        sampling_config: SamplingConfig,
        score_provider: InferWorker[VerifierInput, VerifierOutput],
        *,
        return_sample: bool,
    ) -> RolloutResult:
        raise NotImplementedError
