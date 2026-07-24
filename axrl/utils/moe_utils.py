"""Utilities for Mixture-of-Experts (MoE) routing replay."""

from __future__ import annotations

from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeConfig, Qwen3_5MoeTextConfig


def _qwen35_moe_text_config(hf_config: object) -> Qwen3_5MoeTextConfig | None:
    if isinstance(hf_config, Qwen3_5MoeConfig):
        text_config = hf_config.text_config
        assert isinstance(text_config, Qwen3_5MoeTextConfig)
        return text_config

    text_config = getattr(hf_config, "text_config", None)
    if isinstance(text_config, Qwen3_5MoeTextConfig):
        return text_config
    return None


def get_routing_info_shape(hf_config: object) -> tuple[int, int]:
    """Return ``(num_layers, topk)`` for routing info based on an HF model config.

    ``num_layers`` corresponds to ``hf_config.num_hidden_layers`` (all layers,
    including dense layers which will have dummy routing values).
    ``topk`` corresponds to ``hf_config.num_experts_per_tok``.

    Raises ``ValueError`` if the required attributes are missing.
    """
    num_layers = getattr(hf_config, "num_hidden_layers", None)
    topk = getattr(hf_config, "num_experts_per_tok", None)
    if num_layers is None or topk is None:
        # Qwen3.6 is exposed as a VL wrapper, but routing replay is for the
        # language model whose MoE shape lives under text_config.
        text_config = _qwen35_moe_text_config(hf_config)
        if text_config is not None:
            num_layers = text_config.num_hidden_layers
            topk = text_config.num_experts_per_tok
    if num_layers is None or topk is None:
        raise ValueError(
            "HF config must have 'num_hidden_layers' and 'num_experts_per_tok' "
            f"for MoE routing replay. Got num_hidden_layers={num_layers}, "
            f"num_experts_per_tok={topk}."
        )
    return int(num_layers), int(topk)
