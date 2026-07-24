"""Weight updater that synchronizes weights from Megatron to SGLang engines.

Colocated mode (SGLangTensorWeightUpdater):
    Uses FlattenedTensorBucket with per-rank CUDA IPC and Gloo gather, following
    slime's colocated weight sync pattern. This achieves ~2s weight sync for a 30B
    MoE model. The key optimizations:

    1. FlattenedTensorBucket: packs all tensors in a bucket into a SINGLE contiguous
       GPU buffer, producing only ONE CUDA IPC handle per bucket instead of hundreds.

    2. Per-rank buckets + Gloo gather: each Megatron rank creates its FlattenedTensorBucket
       on its OWN GPU, then Gloo CPU gather_object collects per-rank serialized buckets
       to the engine's source rank. This ensures each SGLang TP worker receives an IPC
       handle valid for its own physical GPU. Without this, sharing one IPC handle across
       TP ranks fails because CUDA IPC handles are per-device -- a handle for GPU 0 cannot
       be opened by a process on GPU 1.

    3. Synchronous per-bucket processing with Gloo barrier: each bucket is sent to the
       engine and waited on before proceeding to the next. A Gloo barrier ensures all ranks
       keep their GPU tensors alive until the engine consumes the IPC handles. This bounds
       GPU memory to ~1 bucket (~2GB) instead of all buckets at once.

    4. torch.cuda.ipc_collect(): cleans up closed CUDA IPC handle entries after each bucket.
       Without this, the handle table grows indefinitely across updates, causing deallocation
       slowdown.


References:
- https://github.com/THUDM/slime/blob/675ca6e75a6ea125289a6af4c1666574dc896121/slime/backends/megatron_utils/update_weight_utils.py
- https://github.com/volcengine/verl/blob/1fe5daf7f15499e10b75f792a65efe5988c8ff04/verl/workers/rollout/sglang_rollout/utils.py
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any, cast, override

import ray
import torch
import torch.distributed as dist
from sglang.srt.utils import MultiprocessingSerializer, init_custom_process_group
from sglang.srt.weight_sync.tensor_bucket import FlattenedTensorBucket
from torch_memory_saver import torch_memory_saver

from axrl.utils import gpu_utils
from axrl.utils.megatron.fp8_quantizer import (
    load_quantization_config,
    quantize_params_for_fp8,
)
from axrl.utils.megatron.weight_update import WeightUpdater

if TYPE_CHECKING:
    from torch.distributed import ProcessGroup


logger = logging.getLogger(__name__)


class SGLangTensorWeightUpdater(WeightUpdater):
    """Colocated weight updater using per-rank CUDA IPC with Gloo gather.

    During connect(), sets up:
    - _ipc_engine: the Ray actor for the SGLang engine colocated with this rank
    - _ipc_gather_group: a Gloo process group containing all Megatron ranks that
      map to the same SGLang engine (e.g., ranks [0,1,2,3] for engine 0)
    - _ipc_gather_src: the first rank in the group, responsible for sending to the engine

    During update_weights(), each rank:
    1. Exports HF weights via the Megatron bridge (lazy generator)
    2. Packs each bucket into a FlattenedTensorBucket on its own GPU
    3. Serializes (single IPC handle) and Gloo-gathers to the source rank
    4. Source rank sends per-TP-rank data to the engine via Ray and waits
    5. All ranks barrier, then free GPU memory and call ipc_collect()
    """

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self._ipc_gather_group: dist.ProcessGroup | None = None
        self._ipc_gather_src: int = -1
        self._ipc_engine: Any | None = None
        self._fp8_quantization_config: dict | None = None

    @override
    def connect(self) -> None:
        from axrl.ray.ray_rollout_worker import RayRolloutWorker

        self._group_name = "weight_update_group"
        ray_sglang_worker = self.ray_rollout_worker
        assert isinstance(ray_sglang_worker, RayRolloutWorker)

        # --- Step 1: Find which engine this Megatron rank is colocated with ---
        # The original code only matched ranks whose device == engine's HEAD device,
        # meaning only 1 of 4 ranks per engine was detected. We now check if this
        # rank's device falls within ANY of the engine's CUDA_VISIBLE_DEVICES, so
        # all 4 ranks per engine are correctly matched.
        cur_ip_and_device = self.cur_megatron_worker.get_ip_and_current_device()
        cur_ip, cur_device = cur_ip_and_device
        engine_handles = ray_sglang_worker.get_engine_handles()
        my_engine_idx = -1

        for i, engine in enumerate(engine_handles):
            engine_ip = cast("str", ray.get(engine.get_ip.remote()))
            engine_devices_str = str(ray.get(engine.get_cuda_visible_devices.remote()))
            engine_devices = set(engine_devices_str.split(","))
            if cur_ip == engine_ip and cur_device in engine_devices:
                self._ipc_engine = engine
                my_engine_idx = i
                logger.info(
                    f"Mcore {self.cur_megatron_worker.mcore_dist_info} colocated with engine {i}: "
                    f"device={cur_device}, engine_devices={engine_devices_str}"
                )
                break

        # --- Step 2: Create per-engine Gloo gather groups ---
        # Following slime's pattern: one Gloo group per engine, containing all Megatron
        # ranks whose GPUs belong to that engine. This enables gather_object() to
        # collect per-rank FlattenedTensorBuckets during weight updates.
        # We use Gloo (CPU backend) because gather_object needs CPU-side pickle transfer.
        cur_rank = self.cur_megatron_worker.mcore_dist_info.rank
        world_size = self.cur_megatron_worker.mcore_dist_info.world_size

        # A world-scope Gloo group for the initial all_gather_object to exchange assignments
        if not hasattr(SGLangTensorWeightUpdater, "_gloo_world_group"):
            SGLangTensorWeightUpdater._gloo_world_group = dist.new_group(ranks=list(range(world_size)), backend="gloo")

        # Exchange engine assignments across all ranks
        engine_assignments: list[int | None] = [None] * world_size
        dist.all_gather_object(
            engine_assignments,
            my_engine_idx,
            group=SGLangTensorWeightUpdater._gloo_world_group,
        )

        # Build per-engine rank lists and create Gloo subgroups
        engine_rank_map: dict[int, list[int]] = {}
        for rank_idx, engine_idx in enumerate(engine_assignments):
            if engine_idx is not None and engine_idx >= 0:
                engine_rank_map.setdefault(engine_idx, []).append(rank_idx)

        for engine_idx, ranks in sorted(engine_rank_map.items()):
            new_group = dist.new_group(ranks=ranks, backend="gloo")
            if cur_rank in ranks:
                self._ipc_gather_group = new_group  # pyright: ignore[reportAttributeAccessIssue]
                # The first rank in the group is the gather destination and
                # the one that sends per-TP-rank data to the engine via Ray.
                self._ipc_gather_src = ranks[0]
                logger.info(f"Mcore rank {cur_rank}: IPC gather group for engine {engine_idx}, ranks={ranks}, gather_src={self._ipc_gather_src}")

        if self._ipc_engine is not None:
            logger.info(f"Mcore {self.cur_megatron_worker.mcore_dist_info} weight updater connected.")
        else:
            logger.info(f"Mcore {self.cur_megatron_worker.mcore_dist_info} no colocated engine (gather participant only).")

        # Load FP8 quantization config from rollout model if it's an FP8 checkpoint.
        rollout_config = self.ray_rollout_worker.get_config()
        rollout_model_name = rollout_config.model.name
        rollout_model_path = rollout_config.model.get_full_path()
        self._fp8_quantization_config = load_quantization_config(rollout_model_path)
        if self._fp8_quantization_config is not None:
            logger.info(
                f"FP8 weight sync enabled for rollout model {rollout_model_name}: "
                f"fmt={self._fp8_quantization_config.get('fmt')}, "
                f"block_size={self._fp8_quantization_config.get('weight_block_size')}"
            )

    @override
    def update_weights(self) -> None:
        modules: list[torch.nn.Module] = self.cur_megatron_worker.model  # type: ignore

        per_tensor_param = self.cur_megatron_worker.export_hf_weights(modules)
        # Note: export_hf_weights is lazy (generator), actual work happens during iteration

        assert self.ray_rollout_worker.get_config().engine_type == "sglang"

        cur_rank = self.cur_megatron_worker.mcore_dist_info.rank

        with torch_memory_saver.disable():
            t_start = time.perf_counter()
            t_bucket = 0.0
            t_serialize = 0.0
            t_gather = 0.0
            n_buckets = 0

            for params_batch in self.get_named_tensor_buckets(per_tensor_param, self.bucket_size_gb):
                n_buckets += 1

                if self._ipc_gather_group is None:
                    del params_batch
                    continue

                # Quantize BF16 weights to FP8 before packing if rollout model uses FP8.
                if self._fp8_quantization_config is not None:
                    params_batch = quantize_params_for_fp8(params_batch, self._fp8_quantization_config)  # noqa: PLW2901

                # --- Key optimization 1: FlattenedTensorBucket ---
                # Packs ALL tensors in this bucket into a single contiguous uint8 GPU buffer.
                # When serialized via MultiprocessingSerializer (which uses ForkingPickler),
                # this produces only ONE CUDA IPC handle for the entire bucket (~2GB),
                # instead of hundreds of individual handles (one per tensor).
                t0 = time.perf_counter()
                bucket = FlattenedTensorBucket(named_tensors=params_batch)
                flattened_tensor_data = {
                    "flattened_tensor": bucket.get_flattened_tensor(),
                    "metadata": bucket.get_metadata(),
                }
                t1 = time.perf_counter()
                t_bucket += t1 - t0

                # Serialize the flattened bucket. output_str=True encodes as base64 string,
                # which is needed for Gloo gather_object (it uses pickle internally).
                serialized = MultiprocessingSerializer.serialize(flattened_tensor_data, output_str=True)
                t2 = time.perf_counter()
                t_serialize += t2 - t1

                # --- Key optimization 2: Gloo gather for per-rank IPC ---
                # Each Megatron rank created its FlattenedTensorBucket on its OWN GPU.
                # Gloo gather_object collects all per-rank serialized buckets to the
                # source rank. The source rank then sends them to the SGLang engine,
                # where each TP worker receives data from the Megatron rank on the
                # SAME physical GPU -- ensuring the CUDA IPC handle is valid.
                # (CUDA IPC handles are per-device: a handle for GPU 0 can only be
                # opened by processes with access to GPU 0.)
                gather_list = [None] * dist.get_world_size(self._ipc_gather_group) if cur_rank == self._ipc_gather_src else None
                dist.gather_object(
                    serialized,
                    object_gather_list=gather_list,
                    dst=self._ipc_gather_src,
                    group=self._ipc_gather_group,
                )
                t3 = time.perf_counter()
                t_gather += t3 - t2

                # Source rank sends per-TP-rank data to the SGLang engine and waits synchronously.
                refs = []
                if cur_rank == self._ipc_gather_src and self._ipc_engine is not None:
                    # gather_list[i] = serialized bucket from rank i in the group.
                    # SGLang's TP worker i will deserialize gather_list[i], which
                    # contains an IPC handle for rank i's GPU -- matching TP worker i's GPU.
                    # load_format="flattened_bucket" tells SGLang to use the efficient
                    # _update_weights_from_flattened_bucket path that reconstructs
                    # tensors as zero-copy views into the flattened buffer.
                    ref = self._ipc_engine.update_weights_from_tensor.remote(
                        gather_list,
                        "flattened_bucket",
                    )
                    refs.append(ref)

                if refs:
                    ray.get(refs)

                # --- Key optimization 3: Barrier + ipc_collect per bucket ---
                # All ranks must wait until the engine has consumed this bucket before
                # freeing GPU tensors. Non-source ranks' GPU memory is still referenced
                # by IPC handles in the gathered data -- freeing early causes stale reads.
                dist.barrier(group=self._ipc_gather_group)

                # Release this bucket's GPU memory and clean up closed IPC handle entries.
                # ipc_collect() is essential: without it, PyTorch's IPC handle table grows
                # across updates, making deallocation O(n) per handle -- the root cause of
                # the "1000 memory blocks" warning and exponential slowdown.
                del flattened_tensor_data, params_batch
                torch.cuda.ipc_collect()

            t_total = time.perf_counter() - t_start
            logger.info(
                f"Weight update: {n_buckets} buckets in {t_total:.2f}s "
                f"(bucket_create={t_bucket:.2f}s, serialize={t_serialize:.2f}s, "
                f"gather={t_gather:.2f}s)"
            )
            gpu_utils.clear_cache()


class SGLangDistributedWeightUpdater(WeightUpdater):
    @override
    def connect(self) -> None:
        from axrl.ray.ray_rollout_worker import RayRolloutWorker

        self._group_name = f"weight_update_group_{self.cur_megatron_worker.mcore_dist_info.rank}"
        self._update_group: ProcessGroup | None = None
        rank = self.cur_megatron_worker.mcore_dist_info.rank
        if rank != 0:
            return
        ray_sglang_worker = self.ray_rollout_worker
        assert isinstance(ray_sglang_worker, RayRolloutWorker)
        master_ip = self.cur_megatron_worker.get_ip()
        master_port = self.cur_megatron_worker.get_port()
        world_size = sum(ray_sglang_worker.get_gpus()) + 1
        refs = [
            engine.init_weights_update_group.remote(
                master_address=master_ip,
                master_port=master_port,
                rank_offset=1 + i * ray_sglang_worker.get_gpus_per_engine(),
                world_size=world_size,
                group_name=self._group_name,
                backend="nccl",
            )
            for i, engine in enumerate(ray_sglang_worker.get_engine_handles())
        ]
        self._update_group = init_custom_process_group(
            backend="nccl",
            init_method=f"tcp://{master_ip}:{master_port}",
            rank=0,
            world_size=world_size,
            group_name=self._group_name,
        )
        ray.get(refs)

    @override
    def update_weights(self) -> None:
        modules: list[torch.nn.Module] = self.cur_megatron_worker.model  # type: ignore
        per_tensor_param = self.cur_megatron_worker.export_hf_weights(modules)
        assert self.ray_rollout_worker.get_config().engine_type == "sglang"
        ray_sglang_worker = self.ray_rollout_worker
        for params_batch in self.get_named_tensor_buckets(per_tensor_param, self.bucket_size_gb):
            # simply broadcast weights from rank 0 to sglang engines
            if self.cur_megatron_worker.mcore_dist_info.rank != 0:
                continue
            names = [name for name, _ in params_batch]
            dtypes = [param.dtype for _, param in params_batch]
            shapes = [param.shape for _, param in params_batch]
            refs = [
                engine.update_weights_from_distributed.remote(
                    names=names,
                    dtypes=dtypes,
                    shapes=shapes,
                    group_name=self._group_name,
                )
                for engine in ray_sglang_worker.get_engine_handles()
            ]

            handles = []
            for _, param in params_batch:
                handles.append(dist.broadcast(tensor=param.data, src=0, group=self._update_group, async_op=True))
            for handle in handles:
                handle.wait()
            ray.get(refs)
        gpu_utils.clear_cache()
