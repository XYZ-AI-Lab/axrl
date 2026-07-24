from __future__ import annotations

import collections
import json
import logging
import os
from functools import partial
from pathlib import Path
from types import MethodType
from typing import TYPE_CHECKING, Any, cast, override

import numpy as np
import torch
import torch.distributed as dist
from megatron.core import mpu
from megatron.core.distributed.finalize_model_grads import finalize_model_grads
from megatron.core.num_microbatches_calculator import (
    destroy_num_microbatches_calculator,
    init_num_microbatches_calculator,
)
from megatron.core.optimizer import MegatronOptimizer, get_megatron_optimizer
from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler
from megatron.core.pipeline_parallel import get_forward_backward_func
from megatron.core.utils import get_model_config
from torch import Tensor
from torch_memory_saver import torch_memory_saver
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer

from axrl.configs import MegatronWorkerConfig, SftTrainerConfig
from axrl.trainer.grpo_trainer import GrpoTrainer
from axrl.trainer.sft_trainer import SftTrainer
from axrl.utils import dist_utils, gpu_utils, setup_logger
from axrl.utils.gpu_utils import GpuUsageInfo, GpuUsageTracker
from axrl.utils.logger import LoggerBuffer, MetricLogger, get_metric_logger
from axrl.utils.megatron.fp32_head import cast_output_layer_to_fp32
from axrl.utils.megatron.router_replay import clear_router_replay_state
from axrl.utils.megatron.seqlen_balancing import realign_non_tensor_keys_after_split, split_into_balanced_microbatches
from axrl.utils.megatron.spike_snapshot_routing import (
    collect_unique_routing_handles_from_batch,
    restore_spike_snapshot_routing,
    save_spike_snapshot_routing,
)
from axrl.utils.megatron.utils import (
    apply_deterministic_flags,
    get_model_forward_fn,
    init_distributed,
    unwrap_model,
)
from axrl.utils.megatron.value_head import replace_output_layer_with_value_head
from axrl.utils.megatron.weight_update import WeightUpdater
from axrl.utils.moe_utils import get_routing_info_shape
from axrl.utils.timer import Timer
from axrl.worker.trainer_worker import TrainerWorker

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator

    from megatron.bridge import AutoBridge
    from megatron.bridge.models.conversion.model_bridge import HFWeightTuple
    from megatron.bridge.models.gpt_provider import GPTModelProvider
    from megatron.bridge.models.qwen_vl.qwen35_vl_bridge import Qwen35VLBridge, Qwen35VLMoEBridge
    from megatron.bridge.models.qwen_vl.qwen35_vl_provider import Qwen35VLModelProvider, Qwen35VLMoEModelProvider
    from megatron.core.models.gpt import GPTModel
    from tensordict import TensorDict

    from axrl.data.sample import SampleTensorDict
    from axrl.ray.ray_rollout_worker import RayRolloutWorker
    from axrl.trainer import BaseTrainer
    from axrl.utils.megatron.model_forward import GPTModelForwardFn
    from axrl.utils.megatron.routing_materialiser import RoutingMaterialiser
    from axrl.utils.tensor_store import TensorHandle

    type Qwen35VLProvider = Qwen35VLMoEModelProvider | Qwen35VLModelProvider
    type Qwen35VLModelBridge = Qwen35VLMoEBridge | Qwen35VLBridge

logger = logging.getLogger(__name__)


def _is_qwen36_model_path(model_path: Path) -> bool:
    return "qwen3.6" in str(model_path).lower()


def _get_weight_loader_bridge(bridge: AutoBridge, provider: Qwen35VLProvider) -> Qwen35VLModelBridge:
    from megatron.bridge.models.qwen_vl.qwen35_vl_bridge import Qwen35VLBridge, Qwen35VLMoEBridge

    for hook in provider._pre_wrap_hooks:
        assert isinstance(hook, partial)
        assert isinstance(hook.func, MethodType)
        model_bridge = hook.func.__self__
        if isinstance(model_bridge, (Qwen35VLMoEBridge, Qwen35VLBridge)):
            return model_bridge

    model_bridge = bridge._model_bridge
    assert isinstance(model_bridge, (Qwen35VLMoEBridge, Qwen35VLBridge))
    return model_bridge


def _configure_language_model_only_provider(bridge: AutoBridge, provider: GPTModelProvider) -> Qwen35VLModelBridge:
    from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
    from megatron.bridge.models.qwen.qwen35_bridge import Qwen35Bridge, Qwen35MoEBridge
    from megatron.bridge.models.qwen_vl.qwen35_vl_bridge import Qwen35VLBridge, Qwen35VLMoEBridge
    from megatron.bridge.models.qwen_vl.qwen35_vl_provider import Qwen35VLModelProvider, Qwen35VLMoEModelProvider

    if not isinstance(provider, (Qwen35VLMoEModelProvider, Qwen35VLModelProvider)):
        raise TypeError(f"use_language_model_only currently supports Qwen35 VLM providers, got {type(provider).__name__}.")

    assert isinstance(provider, (Qwen35VLMoEModelProvider, Qwen35VLModelProvider))
    vl_provider: Qwen35VLProvider = provider
    model_bridge = _get_weight_loader_bridge(bridge, vl_provider)
    bridge_name = type(model_bridge).__name__

    if isinstance(model_bridge, Qwen35VLMoEBridge):

        def moe_mapping_registry(_self: Qwen35VLMoEBridge) -> MegatronMappingRegistry:
            mapping_list = []
            mapping_list.extend(Qwen35MoEBridge._get_moe_lm_mappings(hf_prefix="model.language_model.", megatron_prefix=""))
            return MegatronMappingRegistry(*mapping_list)

        mapping_registry = moe_mapping_registry

    elif isinstance(model_bridge, Qwen35VLBridge):

        def dense_mapping_registry(_self: Qwen35VLBridge) -> MegatronMappingRegistry:
            mapping_list = []
            mapping_list.extend(Qwen35Bridge._get_dense_lm_mappings(hf_prefix="model.language_model.", megatron_prefix=""))
            return MegatronMappingRegistry(*mapping_list)

        mapping_registry = dense_mapping_registry

    else:
        raise TypeError(f"use_language_model_only currently supports Qwen35 VLM bridges, got {bridge_name}.")

    model_bridge.mapping_registry = MethodType(mapping_registry, model_bridge)
    # Upstream Qwen VL annotates `provide()` as returning the full VL wrapper.
    # For language-only mode we intentionally override it with the provider's
    # typed language-model builder.
    vl_provider.provide = vl_provider.provide_language_model  # type: ignore[method-assign]
    if vl_provider.position_embedding_type == "mrope":
        vl_provider.position_embedding_type = "rope"
    # SGLang's normal Qwen3.5/3.6 target-model path skips checkpoint `mtp.*`
    # weights; keep Megatron text-only logprob on the same non-draft target path.
    vl_provider.mtp_num_layers = None
    if vl_provider.sequence_parallel:
        vl_provider.scatter_embedding_sequence_parallel = True
    logger.info(f"Configured {bridge_name} to load and build the language model only.")
    return model_bridge


def _force_tms_preload_initialization_if_needed(context: str) -> None:
    """Initialize torch_memory_saver in preload mode before bridge imports.

    Megatron Core 0.16.0's dynamic_context import path sets
    torch_memory_saver.hook_mode = "torch" at import time. Our Ray Megatron
    workers are launched in LD_PRELOAD mode, so if bridge imports happen before
    torch_memory_saver initializes, the Python wrapper can bind the wrong hook
    mode and later produce CUDA IPC-incompatible storage. Force initialization
    while preload mode is still selected.
    """
    ld_preload = os.environ.get("LD_PRELOAD", "")
    if "torch_memory_saver" not in ld_preload:
        return

    if getattr(torch_memory_saver, "_impl", None) is not None:
        return

    torch_memory_saver.hook_mode = "preload"
    torch_memory_saver._ensure_initialized()
    torch_memory_saver._impl_ctor_kwargs = {}
    logger.info("Initialized torch_memory_saver in preload mode before bridge import (%s).", context)


class MegatronWorker(TrainerWorker):
    def __init__(
        self,
        config: MegatronWorkerConfig,
        rank: int,
        world_size: int,
        local_rank: int,
        master_ip: str,
        master_port: int,
    ) -> None:
        super().__init__()
        self.config = config
        setup_logger(self.config.log_level)
        if config.deterministic_mode:
            apply_deterministic_flags()
        self.model_forward_fn: GPTModelForwardFn = get_model_forward_fn(
            use_magi_merged_forward=config.use_magi_merged_forward,
            use_magi_flat_forward=config.use_magi_flat_forward,
        )
        self._padding_routing_handle: TensorHandle | None = None
        self.trainer: BaseTrainer = SftTrainer(config=SftTrainerConfig())
        self.local_rank: int = local_rank
        self.dist_info = dist_utils.DistInfo(
            rank=rank,
            world_size=world_size,
            local_rank=local_rank,
            master_addr=master_ip,
            master_port=master_port,
        )
        self.config.metric_logger_config.name = f"MegatronWorker-{self.config.model.get_full_path().name}-{self.dist_info.rank}"
        self.metric_logger: MetricLogger = get_metric_logger(self.config.metric_logger_config)
        self.logger_buffer: LoggerBuffer = LoggerBuffer(rank=self.dist_info.rank)
        dist_utils.set_env_dist_info(self.dist_info)
        self.optimizer: MegatronOptimizer | None = None
        self.scheduler: OptimizerParamScheduler | None = None
        self.rollout_weight_updater: WeightUpdater | None = None
        self.name = f"MegatronWorker-{self.config.model.name}-{self.dist_info.rank}"
        self._cpu_snapshots: dict[str, Any] = {}
        self._checkpointing: Any | None = None
        self._distributed_data_parallel_config_cls: type[Any] | None = None
        self._logprob_output_gather_group: dist.ProcessGroup | None = None
        self.is_on_gpu: bool = False
        self.routing_materialiser: RoutingMaterialiser | None = None
        self._te_runtime_attention_backend_logged = False

        # Gradient spike detection state
        self._grad_norm_history: collections.deque[float] = collections.deque(
            maxlen=self.config.spike_debug.history_window,
        )
        self._last_median_norm: float | None = None
        logger.info(f"Created MegatronWorker with dist info: {self.dist_info}")
        logger.info(f"Using CUDA_VISIBLE_DEVICES={self.get_cuda_visible_devices()}")

    def _import_bridge_modules(self) -> type[AutoBridge]:
        _force_tms_preload_initialization_if_needed("MegatronWorker._import_bridge_modules")
        from megatron.bridge import AutoBridge
        from megatron.bridge.training import checkpointing
        from megatron.bridge.training.config import DistributedDataParallelConfig

        self._checkpointing = checkpointing
        self._distributed_data_parallel_config_cls = DistributedDataParallelConfig
        return AutoBridge

    def get_ip_and_current_device(self) -> tuple[str, str]:
        ip = self.get_ip()
        cuda_visible_devices = self.get_cuda_visible_devices()
        devices = cuda_visible_devices.split(",")
        assert self.local_rank < len(devices)
        return ip, devices[self.local_rank]

    @override
    def initialize(self) -> None:
        super().initialize()
        init_distributed(self.config)
        self.mcore_dist_info = dist_utils.load_mcore_dist_info_from_env()
        model_config = self.config.model
        logger.info(f"Worker {self.mcore_dist_info} loading model from {model_config.get_full_path()}.")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_config.get_full_path(),
            trust_remote_code=model_config.trust_remote_code,
        )
        self.model_path: Path = self.config.model.get_full_path()
        assert not _is_qwen36_model_path(self.model_path) or self.config.use_language_model_only, (
            "Qwen3.6 checkpoints must set megatron_worker.use_language_model_only=True."
        )
        AutoBridge = self._import_bridge_modules()
        self.bridge = AutoBridge.from_hf_pretrained(self.model_path)
        self.language_model_only_model_bridge: Qwen35VLModelBridge | None = None
        self.model_provider, self.model = self.get_model()
        if self.config.enable_fp32_lm_head and self.config.model_role != "value":
            cast_output_layer_to_fp32(self.model)
        self._enable_finalize_model_grads()
        logger.info("Initialized model.")
        self.initialize_checkpoint_configs()
        if not self.config.inference_only:
            self.optimizer = self.get_optimizer(self.model)
            self.scheduler = self.get_scheduler(self.optimizer)
            assert self.optimizer is not None and self.scheduler is not None
            logger.info("Initialized optimizer and scheduler.")
        gpu_utils.clear_cache()
        self.is_on_gpu = True

        if self.config.enable_routing_replay:
            from axrl.utils.megatron.routing_materialiser import RoutingMaterialiser

            self.routing_materialiser = RoutingMaterialiser()
            self._padding_routing_handle = self.create_padding_routing_handle(self.config, self.model_path)
        logger.info(f"Initialized MegatronWorker, mcore dist info: {self.mcore_dist_info}.")

    @staticmethod
    def create_padding_routing_handle(config: MegatronWorkerConfig, model_path: Path) -> TensorHandle:
        from axrl.utils import tensor_store as store

        assert 0 < config.padding_sample_length <= config.model.seq_length
        hf_config = AutoConfig.from_pretrained(
            model_path,
            trust_remote_code=config.model.trust_remote_code,
        )
        num_layers, topk = get_routing_info_shape(hf_config)
        routing_rows = config.padding_sample_length - 1
        padding_routing = np.broadcast_to(
            np.arange(topk, dtype=np.int16).reshape(1, 1, topk),
            (routing_rows, num_layers, topk),
        ).copy()
        # Keep the padding object owned by the caller. Broadcasting a rank-0
        # Ray ObjectRef can make other ranks depend on a busy actor while
        # collectives are in flight.
        return store.put(padding_routing)

    def clear_r3_caches(self) -> None:
        if self.routing_materialiser is not None:
            self.routing_materialiser.clear()

    def warmup_tensor_store(self, handles: list[TensorHandle]) -> None:
        """Fetch ``handles`` to prime plasma transfer from each owning sglang worker."""
        if not handles:
            return
        from axrl.utils import tensor_store as store

        store.get_batch(handles)

    def _enable_finalize_model_grads(self) -> None:
        """Enable Megatron's finalize_model_grads() hook."""
        for model_chunk in self.model:
            cfg = get_model_config(model_chunk)
            cfg.calculate_per_token_loss = True
            cfg.finalize_model_grads_func = finalize_model_grads

    def initialize_checkpoint_configs(self) -> None:
        from megatron.bridge.training.config import (
            CheckpointConfig,
            ConfigContainer,
            LoggerConfig,
            TrainingConfig,
        )
        from megatron.bridge.training.state import GlobalState

        assert self._checkpointing is not None

        self.state = GlobalState()
        self.checkpoing_config = CheckpointConfig(
            save=str(self.config.get_checkpoint_dir()),
            load=str(self.config.get_checkpoint_dir()),
            ckpt_step=self.config.checkpoint_step,
            most_recent_k=self.config.most_recent_checkpoint_k,
            save_tokenizer_assets=False,
            load_rng=False,  # need test?
            load_optim=not self.config.inference_only,
        )
        self.state.cfg = ConfigContainer(
            model=self.model_provider,
            optimizer=self.config.optimizer.to_megatron_config(),
            checkpoint=self.checkpoing_config,
            logger=LoggerConfig(),
            train=TrainingConfig(
                micro_batch_size=self.config.train_micro_batch_size,
                global_batch_size=self.config.global_batch_size,
            ),
            scheduler=None,  # type: ignore
            tokenizer=None,  # type: ignore
            dataset=None,  # type: ignore
        )  # used by checkpointing
        init_num_microbatches_calculator(
            self.mcore_dist_info.rank,
            global_batch_size=self.config.global_batch_size,
            micro_batch_size=self.config.train_micro_batch_size,
            data_parallel_size=self.mcore_dist_info.dp_size,
            rampup_batch_size=None,
        )  # needed by checkpointing

        self.ckpt_ctx = self._checkpointing.init_checkpointing_context(self.checkpoing_config)
        logger.info(f"Initialized checkpointing context, checkpoint dir: {self.checkpoing_config.save}.")

        # Separate checkpoint config for spike snapshots — avoids swapping fields at runtime.
        if self.config.spike_debug.enabled:
            spike_ckpt_dir = str(self.config.get_checkpoint_dir().parent / "spike_snapshots")
            self.spike_checkpoing_config = CheckpointConfig(
                save=spike_ckpt_dir,
                load=spike_ckpt_dir,
                most_recent_k=self.config.spike_debug.max_snapshots,
                save_tokenizer_assets=False,
            )

    def record_memory_history(self) -> None:
        torch.cuda.memory._record_memory_history()
        logger.info("Started recording memory history.")

    def save_memory_profile(self, log_dir: Path) -> None:
        log_dir.mkdir(parents=True, exist_ok=True)
        path = log_dir / f"mem_snapshot_rank{self.dist_info.rank}.pickle"
        torch.cuda.memory._dump_snapshot(str(path))
        logger.info(f"Saved memory snapshot to {path}.")

    def set_trainer(self, trainer: BaseTrainer) -> None:
        self.trainer = trainer
        self.trainer.set_metric_agg_type(self.logger_buffer)

    def set_rollout_weight_updater(
        self,
        ray_rollout_worker: RayRolloutWorker,
        bucket_size_gb: float = 1.0,
        *,
        colocated: bool = False,
    ) -> None:
        logger.info(
            "Starting rollout weight updater setup: dist=%s, colocated=%s, bucket_size_gb=%s.",
            self.mcore_dist_info,
            colocated,
            bucket_size_gb,
        )
        try:
            self.rollout_weight_updater = WeightUpdater.create_weight_updater(
                cur_megatron_worker=self,
                ray_rollout_worker=ray_rollout_worker,
                bucket_size_gb=bucket_size_gb,
                colocated=colocated,
            )
        except BaseException:
            logger.exception(
                "Failed rollout weight updater setup: dist=%s, colocated=%s, bucket_size_gb=%s.",
                self.mcore_dist_info,
                colocated,
                bucket_size_gb,
            )
            raise
        logger.info(f"Set rollout weight updater: {type(self.rollout_weight_updater).__name__}, colocated: {colocated}.")

    def connect_rollout_worker(self) -> None:
        assert self.rollout_weight_updater is not None
        with Timer("Connecting rollout weight updater", verbose=True):
            self.rollout_weight_updater.connect()

    def update_rollout_model_weights(self) -> GpuUsageInfo:
        assert self.rollout_weight_updater is not None
        with Timer("Updating rollout weights", verbose=True), GpuUsageTracker() as tracker:
            self.rollout_weight_updater.update_weights()
        gpu_info = tracker.usage_info
        assert gpu_info is not None
        logger.info(f"Rollout weight update GPU usage ({self.mcore_dist_info}): {gpu_info}")
        return gpu_info

    def export_hf_weights(self, modules: list[torch.nn.Module]) -> Iterable[HFWeightTuple]:
        if self.language_model_only_model_bridge is None:
            return self.bridge.export_hf_weights(modules)
        return self.language_model_only_model_bridge.stream_weights_megatron_to_hf(
            modules,
            self.bridge.hf_pretrained,
            cpu=False,
        )

    def load_hf_weights(self, hf_model_dir: Path, *, reset_optimizer: bool = True) -> None:
        self.bridge.load_hf_weights(self.model, hf_path=str(hf_model_dir))  # type: ignore
        if reset_optimizer:
            self.optimizer = self.get_optimizer(self.model)
            self.scheduler = self.get_scheduler(self.optimizer)
            assert self.optimizer is not None and self.scheduler is not None
            logger.info("Initialized optimizer and scheduler.")

        gpu_utils.clear_cache()

    def save_hf_pretrained(self, hf_model_dir: Path) -> None:
        self.bridge.save_hf_pretrained(self.model, path=str(hf_model_dir))

    def save_checkpoint(self, global_step: int) -> None:
        assert self._checkpointing is not None
        self.state.train_state.step = global_step
        self._checkpointing.save_checkpoint(
            state=self.state,
            model=self.model,  # type: ignore
            optimizer=self.optimizer,
            opt_param_scheduler=self.scheduler,
            num_floating_point_operations_so_far=self.state.train_state.floating_point_operations_so_far,
            checkpointing_context=self.ckpt_ctx,
        )

    def _get_sub_optimizers(self) -> list[MegatronOptimizer]:
        """Return the list of sub-optimizers to iterate over.

        For regular optimizers returns a single-element list.
        For ChainedOptimizer (MoE) returns all chained sub-optimizers.
        """
        assert self.optimizer is not None
        chained = getattr(self.optimizer, "chained_optimizers", None)
        if chained is not None:
            return chained  # type: ignore[return-value]
        return [self.optimizer]

    def _clip_grad_by_total_norm(self, total_norm: float) -> None:
        """Clip gradients using a pre-computed total norm.

        Iterates sub-optimizers and clips each one's parameters using the
        shared total_norm. This avoids calling the top-level clip_grad_norm()
        which recomputes the norm and doesn't work on ChainedOptimizer.

        This replicates the clipping logic from:
        - ChainedOptimizer.step():
          https://github.com/NVIDIA/Megatron-LM/blob/3bec9aa97dda898d16ff5a89bac0ed2b6682b172/megatron/core/optimizer/optimizer.py#L1316-L1331
        - MixedPrecisionOptimizer.clip_grad_norm():
          https://github.com/NVIDIA/Megatron-LM/blob/3bec9aa97dda898d16ff5a89bac0ed2b6682b172/megatron/core/optimizer/optimizer.py#L203-L221
        - clip_grad_by_total_norm_fp32():
          https://github.com/NVIDIA/Megatron-LM/blob/3bec9aa97dda898d16ff5a89bac0ed2b6682b172/megatron/core/optimizer/clip_grads.py#L138-L177
        """
        from megatron.core.optimizer.clip_grads import clip_grad_by_total_norm_fp32

        for opt in self._get_sub_optimizers():
            # Skip stub optimizers (ranks with no trainable params, e.g. MoE expert parallelism)
            # Ref: https://github.com/NVIDIA/Megatron-LM/blob/3bec9aa97dda898d16ff5a89bac0ed2b6682b172/megatron/core/optimizer/optimizer.py#L1317-L1319
            if getattr(opt, "is_stub_optimizer", False):
                continue
            params = opt.get_parameters()
            if params and opt.config.clip_grad > 0.0:
                clip_grad_by_total_norm_fp32(
                    params,  # type: ignore[arg-type]  # Parameter is a Tensor subclass
                    opt.config.clip_grad,
                    total_norm,
                    opt.config.use_precision_aware_optimizer_no_fp8_or_ds_fp8,
                )

    def _detect_spike(self, grad_norm: float, global_step: int) -> bool:
        """Detect gradient spike as grad_norm > spike_ratio * median(last N steps).

        Skips detection during warmup (first K steps) to avoid false positives
        while grad norms are still unstable. The history deque is always updated
        so the median is ready when warmup ends.
        """
        cfg = self.config.spike_debug

        # Always record history (even during warmup)
        self._grad_norm_history.append(grad_norm)

        if global_step < cfg.warmup_steps:
            return False

        # Need at least 1 prior value to compute median
        if len(self._grad_norm_history) < 2:
            return False

        # Median of history excluding the current step (last element)
        history = list(self._grad_norm_history)[:-1]
        self._last_median_norm = sorted(history)[len(history) // 2]

        return self._last_median_norm > 0 and grad_norm > cfg.spike_ratio * self._last_median_norm

    def _snapshot_spike_debug_info(self, global_step: int, batch: TensorDict, grad_norm: float) -> None:
        """Save a debug snapshot when a gradient spike is detected (before update).

        Saves model+optimizer checkpoint, input batch, per-param grad norms,
        and replay metadata into a dedicated ``spike_snapshots/`` directory.
        Megatron's ``most_recent_k`` rotation automatically cleans up old snapshots.
        """
        rank = torch.distributed.get_rank()

        # 1. Save model + optimizer checkpoint using the spike checkpoint config.
        # Swap state.cfg.checkpoint to the spike config, save, then restore.
        # The spike config has its own save dir and most_recent_k for rotation.
        self.state.cfg.checkpoint = self.spike_checkpoing_config  # type: ignore[union-attr]
        try:
            self.save_checkpoint(global_step)
        finally:
            self.state.cfg.checkpoint = self.checkpoing_config  # type: ignore[union-attr]

        # Extra files go inside the iter_* dir so rotation cleans them up together.
        iter_dir = Path(self.spike_checkpoing_config.save) / f"iter_{global_step:07d}"  # type: ignore[arg-type]
        iter_dir.mkdir(parents=True, exist_ok=True)

        # 2. Save input batch
        torch.save(batch, iter_dir / f"batch_rank{rank}.pt")
        routing_payload_count = save_spike_snapshot_routing(batch, iter_dir / f"routing_payload_rank{rank}.pt")
        if routing_payload_count:
            logger.info("Saved %d routing payloads for spike replay on rank %d.", routing_payload_count, rank)

        # 3. Save per-parameter grad norms (identifies which layers spiked)
        if self.config.spike_debug.save_per_param_grads:
            grad_info = {}
            for name, param in self.model[0].named_parameters():
                main_grad = getattr(param, "main_grad", None)
                if main_grad is not None:
                    grad_info[name] = {
                        "grad_norm": main_grad.norm().item(),
                        "grad_max": main_grad.abs().max().item(),
                        "param_norm": param.data.norm().item(),
                    }
            torch.save(grad_info, iter_dir / f"grad_info_rank{rank}.pt")

        # 4. Save replay metadata (rank 0 only — global state)
        if rank == 0:
            torch.save(self.trainer, iter_dir / "trainer.pt")
            with (iter_dir / "megatron_worker_config.json").open("w") as f:
                json.dump(self.config.model_dump(), f, indent=2)

            median_norm = getattr(self, "_last_median_norm", None)
            metadata = {
                "metadata_version": 2,
                "global_step": global_step,
                "grad_norm": grad_norm,
                "grad_norm_median": median_norm,
                "spike_ratio_actual": grad_norm / median_norm if median_norm else None,
                "config": self.config.model_dump(),
                "megatron_worker_config_file": "megatron_worker_config.json",
                "trainer_file": "trainer.pt",
                "routing_payload_count_rank0": routing_payload_count,
            }
            with (iter_dir / "axrl_metadata.json").open("w") as f:
                json.dump(metadata, f, indent=2)

    def _compare_per_param_grads(self, snapshot_dir: Path, rank: int, rtol: float, result: dict[str, Any]) -> None:
        """Compare per-parameter grad norms between saved snapshot and current replay."""
        saved_grad_info_path = snapshot_dir / f"grad_info_rank{rank}.pt"
        if not saved_grad_info_path.exists():
            return
        saved_grad_info = torch.load(saved_grad_info_path, weights_only=False)
        per_param_diff: dict[str, dict[str, float]] = {}
        for name, param in self.model[0].named_parameters():
            main_grad = getattr(param, "main_grad", None)
            if main_grad is not None and name in saved_grad_info:
                replayed_norm = main_grad.norm().item()
                original_norm = saved_grad_info[name]["grad_norm"]
                per_param_diff[name] = {
                    "original": original_norm,
                    "replayed": replayed_norm,
                    "relative_diff": abs(replayed_norm - original_norm) / max(original_norm, 1e-8),
                }
        result["per_param_diff"] = per_param_diff

        # Count mismatched params (skip near-zero grad norms where relative diff is meaningless)
        atol = 0.01  # absolute tolerance floor — ignore tiny grad norms
        mismatched = [k for k, v in per_param_diff.items() if v["relative_diff"] > rtol and max(v["original"], v["replayed"]) > atol]
        result["mismatched_params"] = len(mismatched)
        result["total_params"] = len(per_param_diff)
        if mismatched:
            logger.warning(f"Reproduction mismatch in {len(mismatched)}/{len(per_param_diff)} params")
            for name in mismatched[:5]:  # log first 5
                d = per_param_diff[name]
                logger.warning(f"  {name}: original={d['original']:.6f}, replayed={d['replayed']:.6f}")

    def _restore_spike_debug_routing_payloads(self, snapshot_dir: Path, rank: int, batch: TensorDict) -> None:
        routing_payload_path = snapshot_dir / f"routing_payload_rank{rank}.pt"
        if routing_payload_path.exists():
            restored_routing_payloads = restore_spike_snapshot_routing(batch, routing_payload_path)
            if self.routing_materialiser is not None:
                self.routing_materialiser.clear()
            logger.info("Restored %d routing payloads for spike replay on rank %d.", restored_routing_payloads, rank)
        elif collect_unique_routing_handles_from_batch(batch):
            logger.warning(
                "Spike snapshot contains routing handles but no routing payload file at %s. "
                "Replay will rely on the original Ray object refs, which may not exist.",
                routing_payload_path,
            )

    def _load_spike_debug_metadata(self, snapshot_dir: Path) -> dict[str, Any] | None:
        metadata_path = snapshot_dir / "axrl_metadata.json"
        if not metadata_path.exists():
            metadata_path = snapshot_dir / "metadata.json"
        if not metadata_path.exists():
            return None
        with metadata_path.open() as f:
            return json.load(f)

    def _restore_spike_debug_trainer(self, snapshot_dir: Path) -> None:
        trainer_path = snapshot_dir / "trainer.pt"
        if not trainer_path.exists():
            return
        trainer = cast("BaseTrainer", torch.load(trainer_path, weights_only=False))
        self.set_trainer(trainer)
        logger.info("Loaded trainer for spike replay from %s.", trainer_path)

    def reproduce_spike(self, snapshot_dir: Path) -> dict[str, Any]:
        """Reproduce a gradient spike from a saved debug snapshot.

        Loads the model+optimizer checkpoint and input batch from the snapshot,
        re-runs forward-backward and gradient clipping, then compares the
        resulting grad norms with the saved values to verify reproduction.

        Args:
            snapshot_dir: Path to the spike snapshot directory, e.g.
                ``spike_snapshots/iter_0000042/``.

        Returns:
            A dict containing reproduction diagnostics:
                - reproduced: bool — whether grad norms match within tolerance
                - original_grad_norm: float — grad norm from the snapshot
                - replayed_grad_norm: float — grad norm from the replay
                - per_param_diff: dict — per-parameter grad norm comparison
                  (only if save_per_param_grads was enabled)
        """
        rank = torch.distributed.get_rank()

        # Ensure spike checkpoint config exists (may not if spike_debug was disabled)
        if not hasattr(self, "spike_checkpoing_config"):
            from megatron.bridge.training.config import CheckpointConfig

            self.spike_checkpoing_config = CheckpointConfig(
                save=str(snapshot_dir.parent),
                load=str(snapshot_dir),
                save_tokenizer_assets=False,
            )

        # 1. Load metadata/trainer. Legacy snapshots may not have these files.
        metadata = self._load_spike_debug_metadata(snapshot_dir)
        original_grad_norm = None if metadata is None else metadata.get("grad_norm")
        self._restore_spike_debug_trainer(snapshot_dir)

        # 2. Load model + optimizer checkpoint
        # Point the spike checkpoint config's load dir at the iter_* directory.
        # Megatron-Bridge detects it's an iteration dir and loads directly
        # via _DIRECT_ITERATION_DIR_SENTINEL (checkpointing.py:2478-2480),
        # skipping tracker file resolution.
        self.spike_checkpoing_config.load = str(snapshot_dir)
        self.spike_checkpoing_config.load_optim = False
        self.spike_checkpoing_config.load_rng = False
        self.state.cfg.checkpoint = self.spike_checkpoing_config  # type: ignore[union-attr]
        assert self._checkpointing is not None
        self._checkpointing.load_checkpoint(
            state=self.state,
            model=self.model,  # type: ignore[arg-type]
            optimizer=self.optimizer,
            opt_param_scheduler=self.scheduler,
            checkpointing_context=self.ckpt_ctx,
        )
        self.state.cfg.checkpoint = self.checkpoing_config  # type: ignore[union-attr]
        logger.info(f"Loaded spike snapshot checkpoint from {snapshot_dir}")

        # 3. Load input batch
        batch_path = snapshot_dir / f"batch_rank{rank}.pt"
        batch: TensorDict = torch.load(batch_path, map_location="cuda", weights_only=False)
        logger.info(f"Loaded batch from {batch_path}, batch_size={batch.batch_size}")
        self._restore_spike_debug_routing_payloads(snapshot_dir, rank, batch)

        # 4. Re-run forward-backward
        assert self.optimizer is not None
        for model_chunk in self.model:
            model_chunk.train()
            model_chunk.zero_grad_buffer()  # type: ignore[union-attr]

        self.forward_backward_batch(
            batch,
            forward_only=False,
            use_dynamic_batch_size=self.config.use_dynamic_batch_size,
            forward_step=self._forward_step_with_loss_postprocess,
        )

        has_inf_or_nan = self.optimizer.prepare_grads()
        if has_inf_or_nan:
            logger.warning("NaN/Inf detected during replay — cannot reproduce grad norm.")
            self.optimizer.zero_grad()
            return {
                "reproduced": False,
                "original_grad_norm": original_grad_norm,
                "replayed_grad_norm": float("nan"),
                "reason": "NaN/Inf in replayed gradients",
            }

        # 5. Compute grad norm (clip without stepping)
        replayed_grad_norm = self.optimizer.get_grad_norm()
        self._clip_grad_by_total_norm(replayed_grad_norm)

        # 6. Compare with original (only meaningful on rank 0 where metadata exists)
        # Without deterministic mode, CUDA non-determinism causes ~5-10% variance.
        # Since we detect spikes at 5x+ the median, 20% tolerance is sufficient
        # to confirm it's the same spike.
        rtol = 0.20
        if original_grad_norm is not None:
            relative_diff = abs(replayed_grad_norm - original_grad_norm) / max(original_grad_norm, 1e-8)
            reproduced = relative_diff < rtol
        else:
            relative_diff = 0.0
            reproduced = True  # non-rank-0 workers skip comparison

        result: dict[str, Any] = {
            "reproduced": reproduced,
            "original_grad_norm": original_grad_norm,
            "replayed_grad_norm": replayed_grad_norm,
            "relative_diff": relative_diff,
        }

        # 7. Per-parameter comparison if available
        self._compare_per_param_grads(snapshot_dir, rank, rtol, result)

        # Don't apply the update — leave model in pre-update state for debugging
        self.optimizer.zero_grad()

        if original_grad_norm is not None:
            logger.info(
                f"Spike reproduction {'PASSED' if reproduced else 'FAILED'}: "
                f"original={original_grad_norm:.6f}, replayed={replayed_grad_norm:.6f}, "
                f"relative_diff={result['relative_diff']:.6f}"
            )
        return result

    def load_checkpoint(self) -> int:
        assert self._checkpointing is not None
        self._checkpointing.load_checkpoint(
            state=self.state,
            model=self.model,  # type: ignore
            optimizer=self.optimizer,
            opt_param_scheduler=self.scheduler,
            checkpointing_context=self.ckpt_ctx,
        )
        global_step: int = self.state.train_state.step
        if self.mcore_dist_info.rank == 0:
            logger.info(
                f"Loaded checkpoint at global step {global_step}, Scheduler: {self.scheduler.__dict__ if self.scheduler is not None else None}"
            )
        return global_step

    def copy_weights_to_cpu(self, name: str) -> None:
        assert self.model is not None
        with torch.no_grad():
            chunk_states: list[dict[str, torch.Tensor]] = []
            for model_chunk in self.model:
                module = unwrap_model(model_chunk)
                state = module.state_dict()
                cpu_state = {k: (v.detach().cpu().clone() if torch.is_tensor(v) else v) for k, v in state.items()}
                chunk_states.append(cpu_state)
        self._cpu_snapshots[name] = {"states": chunk_states}

    def remove_cpu_weight_copy(self, name: str) -> None:
        if name in self._cpu_snapshots:
            del self._cpu_snapshots[name]

    def apply_weights_from_cpu(self, name: str) -> None:
        assert name in self._cpu_snapshots, f"Snapshot {name} not found"
        payload = self._cpu_snapshots[name]
        states: list[dict[str, torch.Tensor]] = payload["states"]
        assert len(states) == len(self.model), f"Snapshot chunk count {len(states)} != current model chunks {len(self.model)}"
        with torch.no_grad():
            for model_chunk, state in zip(self.model, states, strict=True):
                module = unwrap_model(model_chunk)
                module.load_state_dict(state, strict=True)

    def _configure_provider_parallelism_and_routing(self, provider: GPTModelProvider) -> None:
        provider.tensor_model_parallel_size = self.config.tp_size
        provider.sequence_parallel = self.config.tp_size > 1
        provider.pipeline_model_parallel_size = self.config.pp_size
        provider.virtual_pipeline_model_parallel_size = self.config.vpp_size
        provider.context_parallel_size = self.config.cp_size
        provider.expert_model_parallel_size = self.config.ep_size
        provider.expert_tensor_parallel_size = self.config.etp_size
        provider.variable_seq_lengths = True
        provider.moe_enable_routing_replay = self.config.enable_routing_replay
        provider.moe_token_dispatcher_type = "alltoall"
        provider.attention_softmax_in_fp32 = True
        provider.moe_router_dtype = "fp32"
        provider.moe_aux_loss_coeff = self.config.moe_aux_loss_coeff
        provider.moe_router_load_balancing_type = self.config.moe_router_load_balancing_type

    def _configure_provider_memory_saving(self, provider: GPTModelProvider) -> None:
        provider.recompute_granularity = self.config.recompute_granularity
        provider.recompute_method = self.config.recompute_method
        provider.recompute_num_layers = self.config.recompute_num_layers
        provider.recompute_modules = self.config.recompute_modules
        provider.cpu_offloading = self.config.cpu_offloading
        provider.cpu_offloading_num_layers = self.config.cpu_offloading_num_layers
        provider.cpu_offloading_double_buffering = self.config.cpu_offloading_double_buffering

    def _configure_provider_misc_options(self, provider: GPTModelProvider) -> None:
        if self.config.apply_rope_fusion is not None:
            provider.apply_rope_fusion = self.config.apply_rope_fusion

        if self.config.batch_invariant_mode:
            provider.batch_invariant_mode = True

        if self.config.use_language_model_only:
            self.language_model_only_model_bridge = _configure_language_model_only_provider(self.bridge, provider)

    def _configure_provider_attention_backend(self, provider: GPTModelProvider) -> None:
        if self.config.attention_backend is not None:
            from megatron.core.transformer.enums import AttnBackend

            backend_map = {
                "auto": AttnBackend.auto,
                "flash": AttnBackend.flash,
                "fused": AttnBackend.fused,
                "local": AttnBackend.local,
                "unfused": AttnBackend.unfused,
            }
            provider.attention_backend = backend_map[self.config.attention_backend]
        provider.calculate_per_token_loss = True

    def _configure_provider_fp8(self, provider: GPTModelProvider) -> None:
        if self.config.fp8 is not None:
            provider.fp8 = self.config.fp8
            provider.fp8_recipe = self.config.fp8_recipe
            if self.config.fp8_recipe == "blockwise":
                os.environ.setdefault("NVTE_FP8_BLOCK_SCALING_FP32_SCALES", "1")
            logger.info(f"MCore FP8 enabled: fp8={self.config.fp8}, fp8_recipe={self.config.fp8_recipe}")

    def _configure_provider(self, provider: GPTModelProvider) -> None:
        self._configure_provider_parallelism_and_routing(provider)
        self._configure_provider_memory_saving(provider)
        self._configure_provider_misc_options(provider)
        self._configure_provider_attention_backend(provider)
        self._configure_provider_fp8(provider)

    def _log_provider_config(self, provider: GPTModelProvider) -> None:
        if self.dist_info.rank != 0:
            return
        selected_attention_backend = self.config.attention_backend or "auto"
        resolved_attention_backend = getattr(provider.attention_backend, "name", provider.attention_backend)
        logger.info(f"Megatron attention backend: selected={selected_attention_backend}, resolved={resolved_attention_backend}")
        logger.info(f"GPTModelProvider configs: {provider.__dict__}")

    def get_model(self) -> tuple[GPTModelProvider, list[GPTModel]]:
        assert self._distributed_data_parallel_config_cls is not None
        provider: GPTModelProvider = self.bridge.to_megatron_provider(load_weights=True, hf_path=self.model_path)
        self._configure_provider(provider)
        if self.config.model_role == "value":
            provider.share_embeddings_and_output_weights = False
        provider.finalize()
        self._log_provider_config(provider)
        if self.config.model_role == "value":
            provider.register_pre_wrap_hook(replace_output_layer_with_value_head)
        model = provider.provide_distributed_model(
            fp16=self.config.fp16,
            bf16=self.config.bf16,
            wrap_with_ddp=True,
            ddp_config=self._distributed_data_parallel_config_cls(
                use_distributed_optimizer=self.config.optimizer.use_distributed_optimizer,
                grad_reduce_in_fp32=not self.config.inference_only,
                overlap_grad_reduce=False,
                average_in_collective=False,
            ),
        )
        for model_chunk in model:
            get_model_config(model_chunk).moe_replay_routing_for_loss_tokens_only = self.config.replay_routing_for_loss_tokens_only
        return provider, model

    def _train_metric_tag(self) -> str:
        return f"{self.config.model_role}_train"

    def get_optimizer(self, model: list[GPTModel]) -> MegatronOptimizer:
        optimizer_config = self.config.optimizer.to_megatron_config()
        optimizer = get_megatron_optimizer(
            config=optimizer_config,
            model_chunks=model,  # type: ignore
            use_gloo_process_groups=self.config.use_gloo_process_groups,
        )
        return optimizer

    def get_scheduler(self, optimizer: MegatronOptimizer) -> OptimizerParamScheduler:
        scheduler = OptimizerParamScheduler(
            optimizer=optimizer,
            **self.config.lr_scheduler.model_dump(),
        )
        return scheduler

    def is_output_owner_rank(self) -> bool:
        info = self.mcore_dist_info
        is_tp_leader = info.tp_rank == 0
        is_cp_leader = info.cp_rank == 0
        is_last_pp_stage = info.pp_rank == info.pp_size - 1
        is_vpp_leader = info.vpp_rank in (None, 0)
        return is_tp_leader and is_cp_leader and is_last_pp_stage and is_vpp_leader

    def get_dp_rank(self) -> int:
        return self.mcore_dist_info.dp_rank

    def make_iterator(self, micro_batches: list[TensorDict]) -> list[Iterator[TensorDict]] | Iterator[TensorDict]:
        """Wrap ``micro_batches`` in the iterator Megatron's forward_backward_func expects.

        R3 routing materialise is lazy in every case: each yielded
        microbatch carries ``routed_experts`` resolved just-in-time
        while the next ``prefetch_depth`` fetches run in background
        threads. Peak routing-tensor memory is bounded by
        ``prefetch_depth``, never the full microbatch list.

        With vpp, each stage gets its own prefetch generator. The
        first stage pays the real fetch cost; subsequent stages hit
        the materialiser's per-trajectory cache (fast dict lookup).
        """
        if self.routing_materialiser is not None:
            from axrl.utils.megatron.routing_materialiser import iter_microbatches_with_prefetched_routing

            if self.mcore_dist_info.vpp_size is not None:
                return [
                    iter_microbatches_with_prefetched_routing(
                        micro_batches,
                        self.routing_materialiser,
                        prefetch_depth=2,
                    )
                    for _ in range(self.mcore_dist_info.vpp_size)
                ]
            return iter_microbatches_with_prefetched_routing(
                micro_batches,
                self.routing_materialiser,
                prefetch_depth=2,
            )

        if self.mcore_dist_info.vpp_size is not None:
            data = [micro_batches] * self.mcore_dist_info.vpp_size
            return [iter(d) for d in data]
        return iter(micro_batches)

    def forward_backward_batch(
        self,
        batch: TensorDict,
        forward_step: Callable[[Iterator[TensorDict], GPTModel], Any],
        *,
        use_dynamic_batch_size: bool,
        forward_only: bool,
    ) -> list:
        micro_batch_size = self.config.eval_micro_batch_size if forward_only else self.config.train_micro_batch_size
        if use_dynamic_batch_size:
            max_micro_batch_total_tokens = micro_batch_size * self.config.model.seq_length
            microbatch_group_size = None
            if self.config.vpp_size is not None:
                microbatch_group_size = get_model_config(self.model[0]).microbatch_group_size_per_vp_stage
            micro_batches, _ = split_into_balanced_microbatches(
                batch=batch,
                max_token_len=max_micro_batch_total_tokens,
                dp_group=mpu.get_data_parallel_group(),  # pyright: ignore[reportArgumentType]
                vpp_size=self.config.vpp_size,
                microbatch_group_size=microbatch_group_size,
                verbose=self.config.log_microbatches,
            )
        else:
            micro_batches = batch.split(micro_batch_size)
            realign_non_tensor_keys_after_split(batch, micro_batches)
        logger.debug(f"Split batch of size {batch.batch_size} into {len(micro_batches)} micro-batches.")
        if "merge_info" in batch.keys():  # noqa: SIM118 - explicit keys() for tensordict semantics
            assert self.config.use_magi_merged_forward, "samples carry merge_info but use_magi_merged_forward=False"

        num_micro_batches = len(micro_batches)
        forward_backward_func = get_forward_backward_func()
        with torch.set_grad_enabled(not forward_only):
            losses_reduced: list[dict[str, float | TensorDict]] | list[dict[str, float]] = forward_backward_func(
                forward_step_func=forward_step,
                data_iterator=self.make_iterator(micro_batches),  # type: ignore
                model=self.model,  # type: ignore
                num_microbatches=num_micro_batches,
                seq_length=1,
                micro_batch_size=1,
                forward_only=forward_only,
            )
        if self.config.enable_routing_replay:
            clear_router_replay_state()
        self._log_te_runtime_attention_backend()
        gpu_utils.clear_cache()
        return losses_reduced

    def _log_te_runtime_attention_backend(self) -> None:
        if self.dist_info.rank != 0 or self._te_runtime_attention_backend_logged:
            return
        from transformer_engine.pytorch.attention.dot_product_attention import _attention_backends

        logger.info(f"Transformer Engine runtime attention backend: {_attention_backends}")
        self._te_runtime_attention_backend_logged = True

    def _forward_step_with_loss_postprocess(self, data_iterator: Iterator[TensorDict], model: GPTModel) -> Any:
        """Wrap `trainer.forward_step` and rescale loss for Megatron backward.

        Trainers return a mean loss; with `calculate_per_token_loss=True`, Megatron expects a
        summed loss tensor to keep gradient magnitude consistent.
        """
        outputs, loss_func = self.trainer.forward_step(data_iterator, model, model_forward_fn=self.model_forward_fn)  # type: ignore[misc]

        def wrapped_loss_func(forward_outputs: dict[str, torch.Tensor]) -> Any:
            result = loss_func(forward_outputs)
            assert isinstance(result, tuple) and len(result) == 3
            loss, denom, metrics = result
            scaled_loss = loss * denom
            scaled_loss = scaled_loss * self.mcore_dist_info.cp_size
            return scaled_loss, denom, metrics

        return outputs, wrapped_loss_func

    @override
    def train_step(self, global_step: int, batch: TensorDict) -> dict[str, float] | None:
        for model_chunk in self.model:
            model_chunk.train()
            model_chunk.zero_grad_buffer()  # type: ignore
        assert self.optimizer is not None and self.scheduler is not None
        with GpuUsageTracker(f"GPU-{self.mcore_dist_info}") as gpu_tracker, Timer() as timer:
            local_metrics: list[dict[str, float]] = self.forward_backward_batch(
                batch,
                forward_only=False,
                use_dynamic_batch_size=self.config.use_dynamic_batch_size,
                forward_step=self._forward_step_with_loss_postprocess,
            )
            has_inf_or_nan = self.optimizer.prepare_grads()
            valid_step = not has_inf_or_nan

            grad_norm: float = 0.0
            if valid_step:
                # Decompose optimizer.step() into get_grad_norm + clip + step_with_ready_grads
                # so we can inspect grad_norm BEFORE the model weights are updated.
                # Uses get_grad_norm() + _clip_grad_by_total_norm() instead of clip_grad_norm()
                # to support both regular optimizers and ChainedOptimizer (MoE).
                # Ref: https://github.com/NVIDIA/Megatron-LM/blob/3bec9aa97dda898d16ff5a89bac0ed2b6682b172/megatron/core/optimizer/optimizer.py#L1308-L1338
                grad_norm = self.optimizer.get_grad_norm()
                self._clip_grad_by_total_norm(grad_norm)

                # Gradient spike detection and snapshotting (before weight update)
                if self.config.spike_debug.enabled and self._detect_spike(grad_norm, global_step):
                    logger.warning(f"Gradient spike detected at step {global_step}, grad_norm={grad_norm:.4f}, saving debug snapshot.")
                    self._snapshot_spike_debug_info(global_step, batch, grad_norm)

                update_successful = self.optimizer.step_with_ready_grads()
                self.optimizer.zero_grad()
                assert update_successful
                self.scheduler.step(increment=1)
                logger.debug(f"Completed optimizer step at global step {global_step}, grad norm: {grad_norm:.4f}.")
            else:
                # NaN/Inf detected in gradients — optimizer step skipped.
                # Don't record in spike history — a 0.0 placeholder would lower
                # the median and make future detection overly sensitive.
                logger.warning(f"NaN/Inf in gradients at step {global_step}, optimizer step skipped.")
        update_info: dict[str, float] = {
            "is_valid_step": float(valid_step),
            "grad_norm": grad_norm,
            "step_time_sec": timer.elapsed_seconds,
        }
        # add lr info
        for param_group_id, param_group in enumerate(self.optimizer.param_groups):
            update_info[f"lr-pg_{param_group_id}"] = self.scheduler.get_lr(param_group)
            update_info[f"wd-pg_{param_group_id}"] = self.scheduler.get_wd(param_group)
        gpu_usage_info = gpu_tracker.usage_info
        assert gpu_usage_info is not None
        train_tag = self._train_metric_tag()
        self.logger_buffer.update_metrics(global_step, train_tag, local_metrics)
        self.logger_buffer.update_metrics(global_step, train_tag, [update_info], gather_group="dp")
        if self.config.log_gpu_usaegs:
            self.logger_buffer.update_metrics(global_step, f"{train_tag}-GPU/{gpu_usage_info.name}", [gpu_usage_info.to_metrics()])

        # Clear GPU cache to release memory after training step
        gpu_utils.clear_cache()

        if (global_step + 1) % self.config.log_every_k_steps == 0:
            _step, merged_metrics = self.logger_buffer.sync_and_flush()
            return merged_metrics
        return None

    @override
    def eval_step(self, global_step: int, batch: TensorDict) -> dict[str, float] | None:
        for model_chunk in self.model:
            model_chunk.eval()
        with GpuUsageTracker(f"GPU-{self.mcore_dist_info}") as gpu_tracker:
            local_metrics = self.forward_backward_batch(
                batch,
                forward_only=True,
                forward_step=self._forward_step_with_loss_postprocess,
                use_dynamic_batch_size=self.config.use_dynamic_batch_size,
            )
        assert gpu_tracker.usage_info is not None
        self.logger_buffer.update_metrics(global_step, "eval", local_metrics)
        if self.config.log_gpu_usaegs:
            self.logger_buffer.update_metrics(global_step, "eval-GPU", [gpu_tracker.usage_info.to_metrics()])
        return None

    def forward_logprobs(self, batch: TensorDict) -> list[TensorDict]:
        for model_chunk in self.model:
            model_chunk.eval()
        metrics: list = self.forward_backward_batch(
            batch,
            forward_only=True,
            forward_step=partial(self.trainer.logprob_forward_step, model_forward_fn=self.model_forward_fn),
            use_dynamic_batch_size=self.config.use_dynamic_batch_size,
        )
        outputs: list[TensorDict] = [m["output"] for m in metrics]
        return outputs

    def forward_values(self, batch: TensorDict) -> list[TensorDict]:
        from axrl.trainer.value_trainer import ValueTrainer

        assert isinstance(self.trainer, ValueTrainer), f"forward_values requires ValueTrainer, got {type(self.trainer).__name__}."
        for model_chunk in self.model:
            model_chunk.eval()
        metrics: list = self.forward_backward_batch(
            batch,
            forward_only=True,
            forward_step=partial(self.trainer.value_forward_step, model_forward_fn=self.model_forward_fn),
            use_dynamic_batch_size=self.config.use_dynamic_batch_size,
        )
        outputs: list[TensorDict] = [m["output"] for m in metrics]
        return outputs

    @staticmethod
    def _assert_unique_real_sample_indices(batches: list[TensorDict], *, context: str) -> None:
        assert batches, "batches must not be empty."
        indices: list[int] = []
        for batch in batches:
            batch_indices: Tensor = batch["index"]
            batch_indices = batch_indices.flatten()
            assert batch_indices is not None
            indices.extend(batch_indices[batch_indices >= 0].tolist())
        assert len(indices) == len(set(indices)), f"Duplicate local sample indices on {context}: {indices}"

    def _compute_logprob_outputs_from_local_batches(self, batches: list[SampleTensorDict]) -> tuple[list[TensorDict] | None, GpuUsageInfo]:
        """RayMegatronWorker-only entrypoint for DP-local logprob batches.

        Ray resolves the object ref produced by ``ray.put`` before this actor
        method runs, so ``batches`` is already the local DP batch list. All
        ranks participate in the distributed forward pass, but only the
        output-owner rank for each DP group returns output chunks to the driver.
        """
        train_batches = cast("list[TensorDict]", batches)
        self._assert_unique_real_sample_indices(train_batches, context=str(self.mcore_dist_info))
        logger.info(f"Computing logprobs for {len(batches)} local DP batches on {self.mcore_dist_info}.")
        for model_chunk in self.model:
            model_chunk.eval()
        local_outputs: list[TensorDict] = []
        should_collect_outputs = self.is_output_owner_rank()
        with GpuUsageTracker(f"GPU-{self.mcore_dist_info}") as usage_tracker:
            for batch in tqdm(
                train_batches,
                desc="Computing logprobs for DP batches",
                total=len(train_batches),
                disable=(self.dist_info.rank != 0),
            ):
                batch_output = self.forward_logprobs(batch)
                if should_collect_outputs:
                    local_outputs.extend(batch_output)
        gpu_usage_info = usage_tracker.usage_info
        assert gpu_usage_info is not None
        return (local_outputs if should_collect_outputs else None), gpu_usage_info

    def _compute_value_outputs_from_local_batches(self, batches: list[SampleTensorDict]) -> tuple[list[TensorDict] | None, GpuUsageInfo]:
        """RayMegatronWorker-only entrypoint for DP-local value batches."""
        train_batches = cast("list[TensorDict]", batches)
        self._assert_unique_real_sample_indices(train_batches, context=str(self.mcore_dist_info))
        logger.info(f"Computing values for {len(batches)} local DP batches on {self.mcore_dist_info}.")
        for model_chunk in self.model:
            model_chunk.eval()
        local_outputs: list[TensorDict] = []
        should_collect_outputs = self.is_output_owner_rank()
        with GpuUsageTracker(f"GPU-{self.mcore_dist_info}") as usage_tracker:
            for batch in tqdm(
                train_batches,
                desc="Computing values for DP batches",
                total=len(train_batches),
                disable=(self.dist_info.rank != 0),
            ):
                batch_output = self.forward_values(batch)
                if should_collect_outputs:
                    local_outputs.extend(batch_output)
        gpu_usage_info = usage_tracker.usage_info
        assert gpu_usage_info is not None
        return (local_outputs if should_collect_outputs else None), gpu_usage_info

    def _compute_logprobs_from_local_batches(self, batches: list[SampleTensorDict]) -> tuple[list[torch.Tensor], GpuUsageInfo]:
        """Run forward-only logprobs over this rank's already-built local DP batches.

        Returned tensors follow the input ``batches`` order and shape. This
        method is used inside actor-local GRPO training, where every rank needs
        its local ``ref_logprobs`` / ``old_logprobs`` tensors before backward.
        """
        train_batches = cast("list[TensorDict]", batches)
        local_outputs, gpu_usage_info = self._compute_logprob_outputs_from_local_batches(batches)
        gather_group = self._get_logprob_output_gather_group()
        outputs = dist_utils.all_gather_list(local_outputs or [], group=gather_group)
        assert outputs is not None
        index_to_pos, logprobs_by_pos = self._cat_logprobs_by_index(outputs)
        batch_logprobs = [self._logprobs_for_batch(batch, index_to_pos, logprobs_by_pos) for batch in train_batches]
        logger.info(
            "Computed gathered logprobs for %d real samples within local model shard group_size=%d.",
            len(index_to_pos),
            dist.get_world_size(group=gather_group),
        )
        return batch_logprobs, gpu_usage_info

    def _get_logprob_output_gather_group(self) -> dist.ProcessGroup:
        if self.mcore_dist_info.cp_size > 1 and self.mcore_dist_info.pp_size > 1:
            return self._get_logprob_tensor_context_pipeline_group()
        if self.mcore_dist_info.cp_size > 1:
            group = mpu.get_tensor_and_context_parallel_group()
        else:
            group = mpu.get_model_parallel_group()
        assert group is not None
        return cast("dist.ProcessGroup", group)

    def _get_logprob_tensor_context_pipeline_group(self) -> dist.ProcessGroup:
        """Gather logprob outputs across all non-DP ranks in this DP shard.

        Megatron's built-in groups cover TP+PP and TP+CP, but not TP+CP+PP.
        Actor-local GRPO logprob recomputation needs that combined shard when
        CP and PP are both enabled: only the output-owner rank produces tensors,
        then every rank in the same DP shard reads them back before training.
        """
        if self._logprob_output_gather_group is not None:
            return self._logprob_output_gather_group

        if self.mcore_dist_info.dp_size == 1:
            self._logprob_output_gather_group = cast("dist.ProcessGroup", dist.group.WORLD)
            return self._logprob_output_gather_group

        rank_dp_pairs = cast(
            "list[tuple[int, int]]",
            dist_utils.all_gather_object((self.dist_info.rank, self.mcore_dist_info.dp_rank)),
        )
        ranks_by_dp: dict[int, list[int]] = {}
        for rank, dp_rank in sorted(rank_dp_pairs):
            ranks_by_dp.setdefault(dp_rank, []).append(rank)

        current_group: dist.ProcessGroup | None = None
        for dp_rank in sorted(ranks_by_dp):
            ranks = ranks_by_dp[dp_rank]
            group = dist.new_group(ranks=ranks)
            if self.dist_info.rank in ranks:
                current_group = cast("dist.ProcessGroup", group)

        assert current_group is not None, (
            f"Could not create logprob gather group for rank={self.dist_info.rank}, "
            f"dp_rank={self.mcore_dist_info.dp_rank}, ranks_by_dp={ranks_by_dp}."
        )
        self._logprob_output_gather_group = current_group
        logger.info(
            "Created TP+CP+PP logprob gather group for dp_rank=%d with ranks=%s.",
            self.mcore_dist_info.dp_rank,
            ranks_by_dp[self.mcore_dist_info.dp_rank],
        )
        return self._logprob_output_gather_group

    @staticmethod
    def _cat_logprobs_by_index(outputs: list[TensorDict]) -> tuple[dict[int, int], torch.Tensor]:
        index_chunks: list[torch.Tensor] = []
        logprob_chunks: list[torch.Tensor] = []
        for output in outputs:
            assert "log_prob" in output
            log_prob = cast("torch.Tensor", output["log_prob"])
            indices = cast("torch.Tensor", output["index"]).flatten().to(device=log_prob.device)
            real_mask = indices >= 0
            if bool(real_mask.any()):
                index_chunks.append(indices[real_mask].long())
                logprob_chunks.append(log_prob[real_mask])
        if not index_chunks:
            assert outputs, "No logprob outputs were produced."
            first_logprob = cast("torch.Tensor", outputs[0]["log_prob"])
            return {}, first_logprob.new_empty((0, *first_logprob.shape[1:]))
        indices = torch.cat(index_chunks, dim=0)
        logprobs = torch.cat(logprob_chunks, dim=0)
        assert int(indices.min().item()) >= 0, f"Logprob sample indices must be non-negative, got {indices.tolist()}."
        assert torch.unique(indices).numel() == indices.numel(), f"Duplicate logprob sample indices: {indices.tolist()}."
        index_to_pos = {int(index): pos for pos, index in enumerate(indices.cpu().tolist())}
        return index_to_pos, logprobs

    def _logprobs_for_batch(
        self,
        batch: TensorDict,
        index_to_pos: dict[int, int],
        logprobs_by_pos: torch.Tensor,
    ) -> torch.Tensor:
        values = torch.zeros_like(cast("torch.Tensor", batch["input_ids"]), dtype=torch.float32)
        indices = cast("torch.Tensor", batch["index"]).flatten()
        real_mask = indices >= 0
        if bool(real_mask.any()):
            real_indices = [int(index) for index in indices[real_mask].cpu().tolist()]
            missing_indices = [index for index in real_indices if index not in index_to_pos]
            assert not missing_indices, (
                f"{self.mcore_dist_info}: sample indices missing from gathered logprobs: {missing_indices}; available_count={len(index_to_pos)}."
            )
            positions = torch.tensor(
                [index_to_pos[index] for index in real_indices],
                device=logprobs_by_pos.device,
                dtype=torch.long,
            )
            selected = logprobs_by_pos[positions].to(device=values.device, dtype=values.dtype)
            assert tuple(selected.shape) == tuple(values[real_mask].shape), (
                f"logprob shape mismatch: {tuple(selected.shape)} != {tuple(values[real_mask].shape)}"
            )
            values[real_mask] = selected
        return values

    def _update_ref_and_old_logprobs_for_batches(self, batches: list[TensorDict]) -> tuple[list[TensorDict], GpuUsageInfo]:
        self.copy_weights_to_cpu("cur_weights")

        self.apply_weights_from_cpu("init_weights")
        ref_logprobs, _ = self._compute_logprobs_from_local_batches(cast("list[SampleTensorDict]", batches))
        for batch, logprobs in zip(batches, ref_logprobs, strict=True):
            batch["ref_logprobs"] = logprobs

        self.apply_weights_from_cpu("cur_weights")
        old_logprobs, old_gpu_usage_info = self._compute_logprobs_from_local_batches(cast("list[SampleTensorDict]", batches))
        for batch, logprobs in zip(batches, old_logprobs, strict=True):
            batch["old_logprobs"] = logprobs
        return batches, old_gpu_usage_info

    def _should_compute_megatron_teacher_logprobs(self) -> bool:
        return isinstance(self.trainer, GrpoTrainer) and self.trainer.config.opd.enabled and self.trainer.config.opd.backend == "megatron"

    def _update_teacher_logprobs_for_batches(self, batches: list[TensorDict]) -> list[TensorDict]:
        if not self._should_compute_megatron_teacher_logprobs():
            return batches

        self.copy_weights_to_cpu("cur_weights")
        try:
            trainer = cast("GrpoTrainer", self.trainer)
            self.apply_weights_from_cpu(trainer.config.opd.teacher_weight_name)
            teacher_logprobs, _ = self._compute_logprobs_from_local_batches(cast("list[SampleTensorDict]", batches))
            for batch, logprobs in zip(batches, teacher_logprobs, strict=True):
                batch["teacher_logprobs"] = logprobs
        finally:
            self.apply_weights_from_cpu("cur_weights")
        return batches

    def _train_from_local_batches(
        self,
        global_step: int,
        batches: list[SampleTensorDict],
        *,
        update_logprobs: bool = False,
    ) -> tuple[int, dict[str, float]]:
        """Train on batches already sliced for this worker's DP rank."""
        if self.config.reset_init_weights_every_k_steps is not None and global_step % self.config.reset_init_weights_every_k_steps == 0:
            self.copy_weights_to_cpu("init_weights")
            if self.dist_info.rank == 0:
                logger.info(f"Reset init weights at step {global_step}")

        train_batches = cast("list[TensorDict]", batches)
        assert train_batches, "local train batches must not be empty."
        if update_logprobs:
            train_batches = self._update_teacher_logprobs_for_batches(train_batches)
            train_batches, _ = self._update_ref_and_old_logprobs_for_batches(train_batches)
        step_metrics: list[dict[str, float]] = []
        with tqdm(
            desc=f"Training from {len(train_batches)} local DP batches",
            total=len(train_batches),
            disable=(self.dist_info.rank != 0),
        ) as progress:
            for batch in train_batches:
                metrics = self.train_step(global_step, batch)
                if metrics is not None:
                    step_metrics.append(metrics)
                    self.log_metrics(global_step, metrics)
                global_step += 1
                progress.update(1)
        self.logger_buffer.sync_and_flush()
        aggregated_metrics = self.logger_buffer.aggregate_step_metrics(step_metrics)
        return global_step, aggregated_metrics

    def _eval_from_local_batches(self, global_step: int, batches: list[SampleTensorDict]) -> dict[str, float]:
        """Evaluate batches already sliced for this worker's DP rank."""
        eval_batches = cast("list[TensorDict]", batches)
        for batch in tqdm(
            eval_batches,
            desc=f"Evaluating {len(eval_batches)} local DP batches",
            total=len(eval_batches),
            disable=(self.dist_info.rank != 0),
        ):
            self.eval_step(global_step=global_step, batch=batch)
        _, merged_metrics = self.logger_buffer.sync_and_flush()
        self.log_metrics(global_step, merged_metrics)
        return merged_metrics

    def log_metrics(self, step: int, metrics: dict[str, float]) -> None:
        if self.dist_info.rank == 0:
            self.metric_logger.log_scalars(name_values=metrics, step=step)

    def magi_prefix_merging_layer_diff(self, samples: SampleTensorDict) -> dict[str, Any]:
        """Three-way per-layer hidden-state diff. See :func:`axrl.utils.megatron.layer_diff.magi_prefix_merging_layer_diff`."""
        from axrl.utils.megatron.layer_diff import magi_prefix_merging_layer_diff

        return magi_prefix_merging_layer_diff(self.model, samples)

    def shutdown(self) -> GpuUsageInfo:  # type: ignore
        if not self.is_on_gpu:
            # move to GPU to correctly release GPU memory
            self.to_gpu()
        with GpuUsageTracker(f"GPU-{self.mcore_dist_info}") as gpu_tracker, Timer(f"Shutting down on {self.mcore_dist_info}", verbose=True):
            del self.model
            if self.optimizer is not None:
                del self.optimizer
            if self.scheduler is not None:
                del self.scheduler
            destroy_num_microbatches_calculator()
            dist_utils.cleanup_distributed()
            gpu_utils.clear_cache()
            super().shutdown()
        assert gpu_tracker.usage_info is not None
        return gpu_tracker.usage_info

    def to_cpu(self) -> None:
        if not self.is_on_gpu:
            logger.info("MegatronWorker is already in CPU.")
            return
        with Timer() as timer:
            torch_memory_saver.pause()
            self.is_on_gpu = False
        gpu_utils.log_gpu_memory_after_move("megatron worker", ["all"], "cpu", timer.elapsed_seconds)

    def to_gpu(self) -> None:
        if self.is_on_gpu:
            logger.info("MegatronWorker is already in GPU.")
            return

        with Timer() as timer:
            torch_memory_saver.resume()
            self.is_on_gpu = True
        gpu_utils.log_gpu_memory_after_move("megatron worker", ["all"], "gpu", timer.elapsed_seconds)
