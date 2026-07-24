from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

import torch.distributed as dist
from megatron.core import mpu
from megatron.core.distributed import DistributedDataParallel as DDP
from megatron.core.models.gpt import GPTModel
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.module import Float16Module

from axrl.utils import dist_utils

if TYPE_CHECKING:
    from axrl.configs import MegatronWorkerConfig
    from axrl.utils.megatron.model_forward import GPTModelForwardFn

logger = logging.getLogger(__name__)


def apply_deterministic_flags() -> None:
    # os.environ.setdefault("NCCL_ALGO", "Tree,Ring")
    # magi_attention.env.general.is_deterministic_mode_enable() reads this.
    os.environ.setdefault("MAGI_ATTENTION_DETERMINISTIC_MODE", "1")
    # torch.use_deterministic_algorithms(mode=True, warn_only=True)
    # torch.backends.cudnn.deterministic = True


def get_model_forward_fn(*, use_magi_merged_forward: bool, use_magi_flat_forward: bool = False) -> GPTModelForwardFn:
    """Pick the `GPTModelForwardFn` implementation for a worker configuration.

    Imports lazily to avoid a circular import (`model_forward.py` imports
    `unwrap_model` from this module) and to keep the baseline path free of
    Magi import cost.

    Precedence (mutually exclusive):
    1. ``use_magi_merged_forward=True`` → Magi merged forward; the worker
       binds ``merge_info`` per microbatch via the trajectory-aware data
       pipeline. The function returned here is a stub that requires the
       caller (the worker's microbatch dispatch) to supply ``merge_info``;
       calling it without one raises.
    2. ``use_magi_flat_forward=True`` → Magi with a flat trie (per-sample
       causal ranges, bit-exact to TE FA3 THD). Diagnostic/test-only.
    3. otherwise → baseline TE FA3 THD path.
    """
    assert not (use_magi_merged_forward and use_magi_flat_forward), "use_magi_merged_forward and use_magi_flat_forward are mutually exclusive"
    if use_magi_merged_forward:
        from axrl.utils.megatron.magi_forward import magi_merged_gptmodel_forward

        return magi_merged_gptmodel_forward
    if use_magi_flat_forward:
        from axrl.utils.megatron.magi_forward import magi_flat_gptmodel_forward

        return magi_flat_gptmodel_forward
    from axrl.utils.megatron.model_forward import gptmodel_forward

    return gptmodel_forward


def _validate_parallel_topology(megatron_config: MegatronWorkerConfig) -> None:
    expected_world_size = megatron_config.world_size()
    actual_world_size = dist.get_world_size()
    if actual_world_size != expected_world_size:
        raise RuntimeError(
            "Megatron world size mismatch: "
            f"expected {expected_world_size} from dp={megatron_config.dp_size}, "
            f"tp={megatron_config.tp_size}, cp={megatron_config.cp_size}, pp={megatron_config.pp_size}, "
            f"but torch.distributed world_size={actual_world_size}."
        )

    actual_tp_size = mpu.get_tensor_model_parallel_world_size()
    if actual_tp_size != megatron_config.tp_size:
        raise RuntimeError(f"Megatron TP size mismatch: expected {megatron_config.tp_size}, got {actual_tp_size}.")

    actual_cp_size = mpu.get_context_parallel_world_size()
    if actual_cp_size != megatron_config.cp_size:
        raise RuntimeError(f"Megatron CP size mismatch: expected {megatron_config.cp_size}, got {actual_cp_size}.")

    actual_pp_size = mpu.get_pipeline_model_parallel_world_size()
    if actual_pp_size != megatron_config.pp_size:
        raise RuntimeError(f"Megatron PP size mismatch: expected {megatron_config.pp_size}, got {actual_pp_size}.")

    actual_dp_size = mpu.get_data_parallel_world_size()
    if actual_dp_size != megatron_config.dp_size:
        raise RuntimeError(f"Megatron DP size mismatch: expected {megatron_config.dp_size}, got {actual_dp_size}.")

    actual_ep_size = mpu.get_expert_model_parallel_world_size()
    if actual_ep_size != megatron_config.ep_size:
        raise RuntimeError(f"Megatron EP size mismatch: expected {megatron_config.ep_size}, got {actual_ep_size}.")

    expected_etp_size = megatron_config.expert_tensor_parallel_size()
    actual_etp_size = mpu.get_expert_tensor_parallel_world_size()
    if actual_etp_size != expected_etp_size:
        raise RuntimeError(f"Megatron ETP size mismatch: expected {expected_etp_size}, got {actual_etp_size}.")


def init_distributed(megatron_config: MegatronWorkerConfig) -> None:
    dist_utils.init_gloabal_process_group(timeout_seconds=megatron_config.distributed_timeout_seconds)

    mpu.initialize_model_parallel(
        tensor_model_parallel_size=megatron_config.tp_size,
        pipeline_model_parallel_size=megatron_config.pp_size,
        virtual_pipeline_model_parallel_size=megatron_config.vpp_size,
        context_parallel_size=megatron_config.cp_size,
        expert_model_parallel_size=megatron_config.ep_size,
        expert_tensor_parallel_size=megatron_config.etp_size,
    )
    _validate_parallel_topology(megatron_config)

    model_parallel_cuda_manual_seed(megatron_config.seed)


def unwrap_model(model: GPTModel | DDP | Float16Module | list[GPTModel | DDP | Float16Module]) -> GPTModel:
    if isinstance(model, list):
        assert len(model) == 1, "Unwrapping a list of models is not supported."
        model = model[0]
    module = model
    while isinstance(module, (DDP, Float16Module)):
        module = module.module
    assert isinstance(module, GPTModel), "Model must be an instance of GPTModel."
    return module
