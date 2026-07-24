# Adapted from https://github.com/volcengine/verl/blob/00a10a8ef389556f957a2f36132b2358fd6a109f/verl/models/mcore/model_forward.py

# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol

import torch
from megatron.core import parallel_state as mpu
from megatron.core.distributed import DistributedDataParallel as DDP
from megatron.core.models.gpt import GPTModel
from megatron.core.transformer.module import Float16Module

from axrl.utils.megatron.pack_utils import postprocess_packed_seqs, preprocess_packed_seqs
from axrl.utils.megatron.router_replay import finish_forward_router_replay, prepare_forward_router_replay
from axrl.utils.megatron.utils import unwrap_model

if TYPE_CHECKING:
    from axrl.utils.megatron.prefix_tree import PrefixMergeInfo


class GPTModelForwardFn(Protocol):
    """Forward-pass callable for a Megatron `GPTModel`.

    Implementations: `gptmodel_forward` (baseline TE+THD causal path) and
    `axrl.utils.megatron.magi_forward.magi_merged_gptmodel_forward` (Magi-Attention
    tree-merged path). The trainer holds one of these and invokes it in
    `forward_step`; the choice is made by the worker from its config.

    Each implementation owns Rollout Routing Replay (R3) setup and
    teardown for its own kernel — the trainer just passes ``routed_experts``
    through. The flat path uses ``preprocess_packed_seqs``-based prep;
    the merged path dispatches routing through ``magi_key`` (different
    CP split). Centralizing this in the forward fn keeps the trainer free
    of forward-fn introspection.

    ``merge_info`` is the per-microbatch prefix-tree metadata. Required
    by the merged path; ignored (and asserted None) by flat-path
    implementations.
    """

    def __call__(
        self,
        model: GPTModel | Float16Module | DDP,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        logits_processor: Callable | None = None,
        logits_processor_args: dict | None = None,
        *,
        routed_experts: torch.Tensor | None = None,
        merge_info: "list[PrefixMergeInfo] | None" = None,
        loss_mask: torch.Tensor | None = None,
    ) -> torch.Tensor | dict[str, torch.Tensor]: ...


def gptmodel_forward(
    model: GPTModel | Float16Module | DDP,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    logits_processor: Callable | None = None,
    logits_processor_args: dict | None = None,
    *,
    routed_experts: torch.Tensor | None = None,
    merge_info: "list[PrefixMergeInfo] | None" = None,
    loss_mask: torch.Tensor | None = None,
) -> torch.Tensor | dict[str, torch.Tensor]:
    """Default forward pass for GPT models with optional sequence packing."""
    assert merge_info is None, "gptmodel_forward (flat path) does not accept merge_info"
    pre_process = unwrap_model(model).pre_process
    post_process = unwrap_model(model).post_process
    batch_size, seq_len = attention_mask.shape[:2]
    input_ids_rmpad, packed_seq_params = preprocess_packed_seqs(input_ids, attention_mask, pre_process=pre_process)
    input_ids_rmpad = input_ids_rmpad.contiguous()
    replay_routers = prepare_forward_router_replay(model, attention_mask, routed_experts, loss_mask=loss_mask)
    try:
        output_orig = model(
            input_ids=input_ids_rmpad,
            attention_mask=None,
            position_ids=None,
            packed_seq_params=packed_seq_params,
        )
    finally:
        finish_forward_router_replay(replay_routers=replay_routers)
    if post_process and logits_processor is not None:
        logits_processor_args = logits_processor_args or {}
        args = {k: preprocess_packed_seqs(v, attention_mask, pre_process=True)[0] for k, v in logits_processor_args.items()}
        output_dict = logits_processor(output_orig, **args)
        outputs = {
            k: postprocess_packed_seqs(v, packed_seq_params, attention_mask, batch_size, seq_len, post_process=post_process)
            for k, v in output_dict.items()
        }
        return outputs

    output = postprocess_packed_seqs(output_orig, packed_seq_params, attention_mask, batch_size, seq_len, post_process=post_process)
    if post_process:
        assert isinstance(output, torch.Tensor), "Model forward should return a Tensor of logits."
        assert output.dim() >= 2, f"Expected logits with at least 2 dims (B, S, ...), got shape {tuple(output.shape)}"
        assert output.size(0) == batch_size, f"Logits batch size mismatch: got {output.size(0)}, expected {batch_size}"
        assert output.size(1) == seq_len, f"Logits seq length mismatch: got {output.size(1)}, expected {seq_len}"
    return output


def gptmodel_forward_qwen2_5_vl(
    model: GPTModel | Float16Module | DDP,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    multi_modal_inputs: dict[str, torch.Tensor],
    logits_processor: Callable | None = None,
    logits_processor_args: dict | None = None,
) -> torch.Tensor | dict[str, torch.Tensor]:
    assert mpu.get_context_parallel_world_size() == 1, "qwen2_5_vl's context parallel is not accurate yet"
    post_process = unwrap_model(model).post_process
    pixel_values = multi_modal_inputs["pixel_values"].to(input_ids.device) if "pixel_values" in multi_modal_inputs else None
    image_grid_thw = multi_modal_inputs["image_grid_thw"].to(input_ids.device) if "image_grid_thw" in multi_modal_inputs else None
    batch_size, seq_len = attention_mask.shape[:2]
    input_ids_rmpad, packed_seq_params = preprocess_packed_seqs(input_ids, attention_mask, pre_process=True)
    input_ids_rmpad = input_ids_rmpad.contiguous()
    output_orig = model(
        input_ids=input_ids_rmpad,
        attention_mask=None,
        position_ids=None,
        packed_seq_params=packed_seq_params,
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
    )

    if post_process and logits_processor is not None:
        logits_processor_args = logits_processor_args or {}
        args = {k: preprocess_packed_seqs(v, attention_mask, pre_process=True)[0] for k, v in logits_processor_args.items()}
        output_dict = logits_processor(output_orig, **args)
        outputs = {
            k: postprocess_packed_seqs(v, packed_seq_params, attention_mask, batch_size, seq_len, post_process=post_process)
            for k, v in output_dict.items()
        }
        return outputs
    output = postprocess_packed_seqs(output_orig, packed_seq_params, attention_mask, batch_size, seq_len, post_process=post_process)
    return output
