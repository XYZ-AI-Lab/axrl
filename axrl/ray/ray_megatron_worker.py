from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, cast, override

import ray
import torch
from torch import Tensor

from axrl.data.global_batch_iterator import assert_trajectory_count_divisible, build_global_batches_for_dp_rank
from axrl.ray import ray_utils
from axrl.ray.resource_group import Request, ResourceGroup
from axrl.utils import setup_logger
from axrl.utils.timer import Timer
from axrl.worker.megatron_worker import MegatronWorker
from axrl.worker.trainer_worker import TrainerWorker

if TYPE_CHECKING:
    from pathlib import Path

    from tensordict import TensorDict

    from axrl.configs import MegatronWorkerConfig
    from axrl.data import SampleTensorDict
    from axrl.ray.ray_rollout_worker import RayRolloutWorker
    from axrl.trainer.base_trainer import BaseTrainer
    from axrl.utils.gpu_utils import GpuUsageInfo
    from axrl.utils.tensor_store import TensorHandle

logger = logging.getLogger(__name__)


@ray.remote
class RemoteMegatronWorker(MegatronWorker):
    pass


class RayMegatronWorker(TrainerWorker):
    def __init__(
        self,
        config: MegatronWorkerConfig,
        resource_group: ResourceGroup,
    ) -> None:
        super().__init__()
        self.config = config
        self.resource_group = resource_group
        self.gpus = [int(x.gpu) for x in self.resource_group.requests]
        self.total_gpus = sum(self.gpus)
        self._check_configs()
        self._remote_workers: list[RemoteMegatronWorker] = []
        self._padding_routing_handle: TensorHandle | None = None

    def _check_configs(self) -> None:
        config = self.config
        world_size = config.world_size()
        expert_group_size = config.expert_parallel_group_size()
        assert self.total_gpus == world_size, f"Total GPUs ({self.total_gpus}) must match Megatron world_size ({world_size})"
        # Related Megatron-LM code (commit-pinned permalink):
        # https://github.com/NVIDIA/Megatron-LM/blob/3bec9aa97dda898d16ff5a89bac0ed2b6682b172/megatron/core/parallel_state.py#L787-L794
        assert self.total_gpus % expert_group_size == 0, (
            f"Total GPUs ({self.total_gpus}) must be divisible by expert parallel group size ({expert_group_size})"
        )

    @override
    def initialize(self) -> None:
        super().initialize()
        remote_workers: list[RemoteMegatronWorker] = []

        master_ip = self.resource_group.run_task(func=self.get_ip, bundle_index=0)
        master_port = self.resource_group.run_task(func=self.get_port, bundle_index=0)
        init_handles = []
        world_size = self.total_gpus
        rank: int = 0
        for bundle_index, bundle_info in enumerate(self.resource_group.bundle_infos):
            num_gpus = int(bundle_info.specs.gpu)
            cuda_visible_devices = bundle_info.cuda_visible_devices
            for local_rank in range(num_gpus):
                init_handle, worker = self._start_remote_worker(
                    bundle_index=bundle_index,
                    rank=rank,
                    world_size=world_size,
                    master_addr=master_ip,
                    master_port=master_port,
                    local_rank=local_rank,
                    cuda_visible_devices=cuda_visible_devices,
                )
                init_handles.append(init_handle)
                remote_workers.append(worker)
                rank += 1
        ray.get(init_handles)  # Ensure all workers are initialized
        self._remote_workers = remote_workers
        logger.info(f"Started {len(self._remote_workers)} remote MegatronWorker workers.")

    def _split_samples_for_dp(
        self,
        samples: SampleTensorDict,
        *,
        shuffle: bool,
        shuffle_seed: int,
    ) -> list[list[SampleTensorDict]]:
        """Build DP-local global batches on the driver.

        The Ray API keeps the full sample tensor dict out of remote worker
        calls. The driver splits samples by DP rank, stores each local batch in
        Ray plasma, and each Megatron rank receives only the refs for its own
        DP rank.

        When routing replay is enabled, the driver creates a padding routing
        handle once and embeds it only in padding rows. This keeps the Ray
        boundary DP-local: workers still fetch only refs for their own DP rank.
        """
        padding_routing_handle = self._get_padding_routing_handle() if self._needs_padding_routing_handle(samples) else None
        dp_batches = [
            build_global_batches_for_dp_rank(
                samples,
                global_batch_size=self.config.global_batch_size,
                dp_size=self.config.dp_size,
                dp_rank=dp_rank,
                padding_sample_length=self.config.padding_sample_length,
                padding_routing_handle=padding_routing_handle,
                shuffle=shuffle,
                shuffle_seed=shuffle_seed,
            )
            for dp_rank in range(self.config.dp_size)
        ]
        self._log_dp_split_summary(samples, dp_batches)
        return dp_batches

    def _log_dp_split_summary(self, samples: SampleTensorDict, dp_batches: list[list[SampleTensorDict]]) -> None:
        trajectory_id_tensor = samples.get("trajectory_id", None)
        if not isinstance(trajectory_id_tensor, torch.Tensor):
            logger.info("DP split summary: packed_samples=%d trajectory_id=missing.", len(samples))
            return

        trajectory_ids = trajectory_id_tensor.tolist()
        if not trajectory_ids:
            logger.info("DP split summary: packed_samples=0 trajectories=0 global_batches=0.")
            return
        if all(tid == -1 for tid in trajectory_ids):
            trajectory_ids = list(range(len(trajectory_ids)))

        num_trajectories = max(trajectory_ids) + 1
        num_global_batches = (num_trajectories + self.config.global_batch_size - 1) // self.config.global_batch_size
        real_rows_by_global_batch = [0] * num_global_batches
        for tid in trajectory_ids:
            real_rows_by_global_batch[tid // self.config.global_batch_size] += 1

        local_rows_by_global_batch = [len(batch) for batch in dp_batches[0]] if dp_batches else []
        padded_rows_by_global_batch = [local_rows * self.config.dp_size for local_rows in local_rows_by_global_batch]
        padding_rows_by_global_batch = [padded - real for padded, real in zip(padded_rows_by_global_batch, real_rows_by_global_batch, strict=True)]
        logger.info(
            "DP split summary: packed_samples=%d trajectories=%d global_batch_size=%d global_batches=%d dp_size=%d "
            "real_rows_per_global_batch=%s padded_rows_per_global_batch=%s padding_rows_per_global_batch=%s "
            "local_rows_per_dp_rank=%s.",
            len(samples),
            num_trajectories,
            self.config.global_batch_size,
            num_global_batches,
            self.config.dp_size,
            real_rows_by_global_batch,
            padded_rows_by_global_batch,
            padding_rows_by_global_batch,
            local_rows_by_global_batch,
        )

    def _needs_padding_routing_handle(self, samples: SampleTensorDict) -> bool:
        return self.config.enable_routing_replay and "routing_handles_per_path" in samples

    def _get_padding_routing_handle(self) -> TensorHandle:
        if self._padding_routing_handle is None:
            self._padding_routing_handle = MegatronWorker.create_padding_routing_handle(self.config, self.config.model.get_full_path())
        return self._padding_routing_handle

    @staticmethod
    def _put_dp_batches(
        dp_batches: list[list[SampleTensorDict]],
    ) -> list[ray.ObjectRef[list[SampleTensorDict]]]:
        return [ray.put(batches) for batches in dp_batches]

    def _remote_dp_ranks(self) -> list[int]:
        dp_ranks: list[int] = ray.get([worker.get_dp_rank.remote() for worker in self._remote_workers])
        assert len(dp_ranks) == len(self._remote_workers)
        assert set(dp_ranks) <= set(range(self.config.dp_size)), (
            f"Remote DP ranks must be within 0..{self.config.dp_size - 1}, got {sorted(set(dp_ranks))}."
        )
        return dp_ranks

    @staticmethod
    def _merge_tensor_outputs(
        samples: SampleTensorDict,
        outputs: list[TensorDict],
        *,
        key: str,
    ) -> Tensor:
        """Drop padding rows and restore output chunks to sample-index order."""
        num_samples = len(samples)
        merged: Tensor | None = None
        filled = torch.zeros(num_samples, dtype=torch.bool)
        for output in outputs:
            assert key in output
            values = cast("Tensor", output[key])
            indices = cast("Tensor", output["index"]).flatten().to(device=values.device)
            real_mask = indices >= 0
            if not bool(real_mask.any()):
                continue
            if merged is None:
                merged = values.new_empty((num_samples, *values.shape[1:]))
            real_indices = indices[real_mask].long()
            assert int(real_indices.max().item()) < num_samples, (
                f"{key} sample index {int(real_indices.max().item())} is out of range for {num_samples} samples."
            )
            assert not bool(filled[real_indices.cpu()].any()), f"Duplicate {key} values for sample indices {real_indices.tolist()}."
            merged[real_indices] = values[real_mask]
            filled[real_indices.cpu()] = True
        assert merged is not None, f"No real sample {key} values were produced."
        assert bool(filled.all()), f"Missing {key} values for sample indices {torch.nonzero(~filled).flatten().tolist()}."
        return merged

    @staticmethod
    def _merge_logprob_outputs(
        samples: SampleTensorDict,
        outputs: list[TensorDict],
    ) -> Tensor:
        return RayMegatronWorker._merge_tensor_outputs(samples, outputs, key="log_prob")

    @staticmethod
    def _merge_value_outputs(
        samples: SampleTensorDict,
        outputs: list[TensorDict],
    ) -> Tensor:
        return RayMegatronWorker._merge_tensor_outputs(samples, outputs, key="values")

    def load_hf_weights(self, hf_model_dir: Path, *, reset_optimizer: bool = True) -> None:
        with Timer(f"Loaded weights from {hf_model_dir}, {reset_optimizer=}", verbose=True):
            refs = [w.load_hf_weights.remote(hf_model_dir, reset_optimizer=reset_optimizer) for w in self._remote_workers]
            ray.get(refs)

    def clear_r3_caches(self) -> None:
        refs = [w.clear_r3_caches.remote() for w in self._remote_workers]
        ray.get(refs)

    def warmup_tensor_store(self, handles: list[TensorHandle]) -> None:
        """Fan ``handles`` to every megatron worker for a warmup fetch."""
        refs = [w.warmup_tensor_store.remote(handles) for w in self._remote_workers]
        ray.get(refs)

    def save_hf_pretrained(self, hf_model_dir: Path) -> None:
        with Timer(f"Saved weights to {hf_model_dir}", verbose=True):
            refs = [w.save_hf_pretrained.remote(hf_model_dir) for w in self._remote_workers]
            ray.get(refs)

    def copy_weights_to_cpu(self, name: str) -> None:
        with Timer(f"Saved snapshot '{name}'", verbose=True):
            refs = [w.copy_weights_to_cpu.remote(name) for w in self._remote_workers]
            ray.get(refs)

    def remove_cpu_weight_copy(self, name: str) -> None:
        with Timer(f"Removed snapshot '{name}'", verbose=True):
            refs = [w.remove_cpu_weight_copy.remote(name) for w in self._remote_workers]
            ray.get(refs)

    def apply_weights_from_cpu(self, name: str) -> None:
        with Timer(f"Applied snapshot '{name}'", verbose=True):
            refs = [w.apply_weights_from_cpu.remote(name) for w in self._remote_workers]
            ray.get(refs)

    def save_checkpoint(self, global_step: int) -> None:
        with Timer(f"Saved checkpoint at step {global_step}", verbose=True):
            refs = [w.save_checkpoint.remote(global_step) for w in self._remote_workers]
            ray.get(refs)

    def load_checkpoint(self) -> int:
        with Timer("Loaded checkpoint", verbose=True):
            refs = [w.load_checkpoint.remote() for w in self._remote_workers]
            results = ray.get(refs)
            # all workers return the same global_step
            return results[0]

    def _start_remote_worker(
        self,
        bundle_index: int,
        rank: int,
        world_size: int,
        master_addr: str,
        master_port: int,
        local_rank: int,
        cuda_visible_devices: str,
    ) -> tuple[Any, RemoteMegatronWorker]:
        config = self.config.model_copy()
        worker = RemoteMegatronWorker.options(
            scheduling_strategy=self.resource_group.get_scheduling_strategy(bundle_index),
            runtime_env=ray_utils.get_runtime_envs(cuda_visible_devices, set_torch_memory_saver_envs=True),
        ).remote(
            config=config,
            rank=rank,
            world_size=world_size,
            local_rank=local_rank,
            master_ip=master_addr,
            master_port=master_port,
        )
        init_handle = worker.initialize.remote()  # type: ignore
        return init_handle, worker

    def train(
        self, global_step: int, samples: SampleTensorDict, data_shuffle_seed: int = 0, *, compute_logprobs: bool = True
    ) -> tuple[int, dict[str, float]]:
        """Train with driver-built DP-local batches.

        ``samples`` is split into DP-aware global batches on the Ray driver.
        Each local batch is stored once in Ray plasma, and every remote
        Megatron rank receives only the batch refs for its own DP rank.

        Args:
            global_step: Current trainer step.
            samples: Packed training samples with dense, non-negative ``index``
                values for real samples.
            data_shuffle_seed: Seed used when ``config.shuffle_train_data`` is
                enabled.
            compute_logprobs: When true, each actor computes ``ref_logprobs``
                and ``old_logprobs`` on its local DP batches before training.
        """
        assert_trajectory_count_divisible(samples, self.config.global_batch_size)
        dp_batches = self._split_samples_for_dp(
            samples,
            shuffle=self.config.shuffle_train_data,
            shuffle_seed=data_shuffle_seed,
        )
        return self.train_from_dp_batches(global_step, dp_batches, compute_logprobs=compute_logprobs)

    def train_from_dp_batches(
        self,
        global_step: int,
        dp_batches: list[list[SampleTensorDict]],
        *,
        compute_logprobs: bool = True,
    ) -> tuple[int, dict[str, float]]:
        """Train from DP-aware batches.

        ``dp_batches[dp_rank]`` contains the local global batches for that DP
        rank. This method stores one list of batches per DP rank in Ray plasma
        and sends each Megatron rank only the object ref for its own DP rank.
        """
        assert len(dp_batches) == self.config.dp_size, f"Expected {self.config.dp_size} DP batch lists, got {len(dp_batches)}."
        dp_batch_refs = self._put_dp_batches(dp_batches)
        remote_dp_ranks = self._remote_dp_ranks()
        refs = [
            worker._train_from_local_batches.remote(
                global_step,
                dp_batch_refs[dp_rank],
                update_logprobs=compute_logprobs,
            )
            for worker, dp_rank in zip(self._remote_workers, remote_dp_ranks, strict=True)
        ]
        results = ray.get(refs)
        return results[0]

    def eval(self, global_step: int, samples: SampleTensorDict) -> dict[str, float]:
        """Evaluate with driver-built DP-local batches."""
        dp_batches = self._split_samples_for_dp(samples, shuffle=False, shuffle_seed=0)
        dp_batch_refs = self._put_dp_batches(dp_batches)
        remote_dp_ranks = self._remote_dp_ranks()
        refs = [
            worker._eval_from_local_batches.remote(global_step, dp_batch_refs[dp_rank])
            for worker, dp_rank in zip(self._remote_workers, remote_dp_ranks, strict=True)
        ]
        results = ray.get(refs)
        return results[0]

    def record_memory_history(self) -> None:
        refs = [w.record_memory_history.remote() for w in self._remote_workers]
        ray.get(refs)
        logger.info("Started recording memory history on all workers.")

    def save_memory_profile(self, log_dir: Path) -> None:
        refs = [w.save_memory_profile.remote(log_dir) for w in self._remote_workers]
        ray.get(refs)
        logger.info(f"Saved memory profiles to {log_dir}.")

    @override
    def compute_logprobs(self, samples: SampleTensorDict, batch_size: int | None = None) -> tuple[Tensor, list[GpuUsageInfo]]:
        """Compute logprobs with driver-built DP-local batches.

        The returned tensor is restored to ``samples["index"]`` order. Padding
        rows inside DP batches use ``index == -1`` and are dropped.
        """
        del batch_size
        dp_batches = self._split_samples_for_dp(samples, shuffle=False, shuffle_seed=0)
        dp_batch_refs = self._put_dp_batches(dp_batches)
        remote_dp_ranks = self._remote_dp_ranks()
        refs = [
            worker._compute_logprob_outputs_from_local_batches.remote(dp_batch_refs[dp_rank])
            for worker, dp_rank in zip(self._remote_workers, remote_dp_ranks, strict=True)
        ]
        results: list[tuple[list[TensorDict] | None, GpuUsageInfo]] = ray.get(refs)
        outputs: list[TensorDict] = []
        for output_chunks, _ in results:
            if output_chunks is None:
                continue
            outputs.extend(output_chunks)
        merged = self._merge_logprob_outputs(samples, outputs)
        gpu_usage_infos = [gpu_usage_info for _, gpu_usage_info in results]
        assert len(merged) == len(samples)
        assert len(gpu_usage_infos) == len(self._remote_workers)
        return merged, gpu_usage_infos

    def compute_values(self, samples: SampleTensorDict, batch_size: int | None = None) -> tuple[Tensor, list[GpuUsageInfo]]:
        """Compute value predictions with driver-built DP-local batches."""
        del batch_size
        dp_batches = self._split_samples_for_dp(samples, shuffle=False, shuffle_seed=0)
        dp_batch_refs = self._put_dp_batches(dp_batches)
        remote_dp_ranks = self._remote_dp_ranks()
        refs = [
            worker._compute_value_outputs_from_local_batches.remote(dp_batch_refs[dp_rank])
            for worker, dp_rank in zip(self._remote_workers, remote_dp_ranks, strict=True)
        ]
        results: list[tuple[list[TensorDict] | None, GpuUsageInfo]] = ray.get(refs)
        outputs: list[TensorDict] = []
        for output_chunks, _ in results:
            if output_chunks is None:
                continue
            outputs.extend(output_chunks)
        merged = self._merge_value_outputs(samples, outputs)
        gpu_usage_infos = [gpu_usage_info for _, gpu_usage_info in results]
        assert len(merged) == len(samples)
        assert len(gpu_usage_infos) == len(self._remote_workers)
        return merged, gpu_usage_infos

    def magi_prefix_merging_layer_diff(self, samples: SampleTensorDict) -> dict[str, Any]:
        refs = [w.magi_prefix_merging_layer_diff.remote(samples) for w in self._remote_workers]
        return ray.get(refs)[0]

    def reproduce_spike(self, snapshot_dir: Path) -> dict[str, Any]:
        """Reproduce a gradient spike from a saved debug snapshot across all workers."""
        refs = [w.reproduce_spike.remote(snapshot_dir) for w in self._remote_workers]
        results = ray.get(refs)
        # All workers should agree — return rank 0's result
        return results[0]

    def shutdown(self) -> None:
        results_ref = [w.shutdown.remote() for w in self._remote_workers]
        results = ray.get(results_ref)
        assert len(results) == len(self._remote_workers)
        end_mem_reserved_gbs = [info.end_mem_reserved_gbs for info in results]
        logger.info(f"RayMegatronWorker shutdown. end_mem_reserved_gbs: {end_mem_reserved_gbs}")
        ray_utils.kill_remote_workers(self._remote_workers)
        self._remote_workers = []

    def _is_colocated(self, rollout_worker: RayRolloutWorker) -> bool:
        return self.resource_group.pg.id == rollout_worker.get_resource_group().pg.id

    def build_weight_updater(self, rollout_worker: RayRolloutWorker, bucket_size_gb: float = 1.0) -> None:
        colocated = self._is_colocated(rollout_worker)

        logger.info(
            "Building rollout weight updater on %d Megatron workers: colocated=%s, bucket_size_gb=%s.",
            len(self._remote_workers),
            colocated,
            bucket_size_gb,
        )
        refs = [
            worker.set_rollout_weight_updater.remote(
                rollout_worker,
                bucket_size_gb=bucket_size_gb,
                colocated=colocated,
            )
            for worker in self._remote_workers
        ]
        try:
            ray.get(refs)
        except Exception:
            ready_refs, pending_refs = ray.wait(refs, num_returns=len(refs), timeout=0)
            logger.exception(
                "Failed to build rollout weight updater: ready=%d, pending=%d, total=%d.",
                len(ready_refs),
                len(pending_refs),
                len(refs),
            )
            raise
        logger.info(f"Built rollout weight updater with colocated={colocated}.")

    def set_trainer(self, trainer: BaseTrainer) -> None:
        refs = [w.set_trainer.remote(trainer) for w in self._remote_workers]
        ray.get(refs)
        logger.info(f"Set trainer for with type: {trainer.__class__.__name__}.")

    def connect_rollout_worker(self) -> None:
        logger.info("Connecting rollout weight updater on %d Megatron workers.", len(self._remote_workers))
        refs = [worker.connect_rollout_worker.remote() for worker in self._remote_workers]
        try:
            ray.get(refs)
        except Exception:
            ready_refs, pending_refs = ray.wait(refs, num_returns=len(refs), timeout=0)
            logger.exception(
                "Failed to connect rollout weight updater: ready=%d, pending=%d, total=%d.",
                len(ready_refs),
                len(pending_refs),
                len(refs),
            )
            raise
        logger.info("Connected rollout weight updater.")

    def update_rollout_model_weights(self) -> list[GpuUsageInfo]:
        refs = [worker.update_rollout_model_weights.remote() for worker in self._remote_workers]
        results: list[GpuUsageInfo] = ray.get(refs)
        logger.info("Updated rollout weights.")
        return results

    def to_cpu(self) -> None:
        refs = [w.to_cpu.remote() for w in self._remote_workers]
        ray.get(refs)

    def to_gpu(self) -> None:
        refs = [w.to_gpu.remote() for w in self._remote_workers]
        ray.get(refs)


def _test_ray_megatron_worker() -> None:
    if not ray.is_initialized():
        ray.init()
    from axrl.example.config_examples import get_megatron_trainer_config

    config = get_megatron_trainer_config(pp_size=2, vpp_size=None, dp_size=2)
    resource_group = ResourceGroup(requests=[Request(cpu=1, gpu=4)] * 1)
    worker = RayMegatronWorker(config, resource_group)
    worker.initialize()
    logger.info("RayMegatronWorker initialized successfully.")


if __name__ == "__main__":
    setup_logger("info")
    _test_ray_megatron_worker()
