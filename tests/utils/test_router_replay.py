"""Tests for router replay tensor shaping and layer selection helpers."""

import torch
from megatron.core.transformer.moe.router_replay import RouterReplay
from megatron.core.transformer.transformer_config import TransformerConfig

from axrl.utils.megatron.router_replay import (
    expand_routed_experts_to_token_positions,
    get_current_rank_layer_info,
    get_micro_batch_router_list,
    select_local_router_replay_tensors,
)


def test_expand_routed_experts_adds_trailing_dummy_token() -> None:
    """Check that replay tensors append the no-prediction placeholder token.

    Router replay stores routing only for predicted tokens, so token-aligned
    tensors must synthesize a final placeholder slot for each sequence.
    """
    attention_mask = torch.tensor(
        [
            [True, True, True, False],
            [True, True, False, False],
        ]
    )
    routed_experts = torch.nested.as_nested_tensor(
        [
            torch.tensor([[[1], [2]], [[3], [4]]], dtype=torch.int16),
            torch.tensor([[[5], [6]]], dtype=torch.int16),
        ],
        layout=torch.jagged,
    )

    expanded = expand_routed_experts_to_token_positions(routed_experts, attention_mask)

    expected = torch.tensor(
        [
            [[[1], [2]], [[3], [4]], [[0], [0]], [[0], [0]]],
            [[[5], [6]], [[0], [0]], [[0], [0]], [[0], [0]]],
        ],
        dtype=torch.int16,
    )
    torch.testing.assert_close(expanded, expected)


def test_expand_routed_experts_uses_distinct_placeholder_experts_for_missing_topk() -> None:
    """Check missing replay slots keep distinct placeholder expert ids per top-k slot."""
    attention_mask = torch.tensor([[True, True, False]])
    routed_experts = torch.nested.as_nested_tensor(
        [
            torch.tensor([[[7, 9], [11, 13]]], dtype=torch.int16),
        ],
        layout=torch.jagged,
    )

    expanded = expand_routed_experts_to_token_positions(routed_experts, attention_mask)

    expected = torch.tensor([[[[7, 9], [11, 13]], [[0, 1], [0, 1]], [[0, 1], [0, 1]]]], dtype=torch.int16)
    torch.testing.assert_close(expanded, expected)


def _make_tf_config(**overrides: object) -> TransformerConfig:
    """Create a minimal Megatron transformer config for PP/VPP routing tests."""
    tf_config = TransformerConfig(
        num_layers=8,
        hidden_size=8,
        num_attention_heads=1,
        pipeline_dtype=torch.float32,
        moe_layer_freq=1,
        pipeline_model_parallel_size=2,
        pipeline_model_parallel_layout=None,
        virtual_pipeline_model_parallel_size=2,
        context_parallel_size=2,
        num_layers_in_first_pipeline_stage=None,
        num_layers_in_last_pipeline_stage=None,
        account_for_embedding_in_pipeline_split=False,
        account_for_loss_in_pipeline_split=False,
    )
    for key, value in overrides.items():
        setattr(tf_config, key, value)
    return tf_config


def test_get_current_rank_layer_info_supports_pp_and_vpp() -> None:
    """Verify layer windows account for both PP and VPP offsets."""
    tf_config = _make_tf_config()

    info = get_current_rank_layer_info(tf_config, pp_rank=0, vp_rank=1)

    assert info == {"start": 4, "end": 6, "count": 2}


def test_get_micro_batch_router_list_offsets_across_vpp_stages() -> None:
    """Verify global RouterReplay instances are sliced to the active VPP stage."""
    tf_config = _make_tf_config()
    RouterReplay.clear_global_router_replay_instances()
    try:
        routers = [RouterReplay() for _ in range(4)]

        local_routers = get_micro_batch_router_list(tf_config, vp_rank=1, pp_rank=0)

        assert local_routers == routers[2:4]
    finally:
        RouterReplay.clear_global_router_replay_instances()


def test_select_local_router_replay_tensors_filters_moe_layers_for_pp_stage() -> None:
    """Verify PP-local replay tensors keep only MoE layers owned by the stage."""
    tf_config = _make_tf_config(
        num_layers=4,
        moe_layer_freq=[1, 0, 1, 0],
        virtual_pipeline_model_parallel_size=None,
        pipeline_model_parallel_size=2,
        context_parallel_size=1,
    )
    layers_topk_idx = torch.tensor(
        [
            [[10], [11], [12]],
            [[20], [21], [22]],
            [[30], [31], [32]],
            [[40], [41], [42]],
        ],
        dtype=torch.int16,
    )

    replay_tensors = select_local_router_replay_tensors(layers_topk_idx, tf_config, pp_rank=1, vp_rank=0)

    assert len(replay_tensors) == 1
    torch.testing.assert_close(replay_tensors[0], torch.tensor([[30], [31], [32]], dtype=torch.int64))
