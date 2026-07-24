from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
from megatron.core import tensor_parallel

from axrl.utils.megatron.utils import unwrap_model

if TYPE_CHECKING:
    from megatron.core.models.gpt import GPTModel
    from megatron.core.transformer.module import MegatronModule
    from megatron.core.transformer.transformer_config import TransformerConfig


class LinearForLastLayer(torch.nn.Linear):
    """Scalar value head for the last Megatron pipeline stage.

    Adapted from THUDM/slime commit 680824dd5e01a2e83750bf87fc366ec6fa98766c,
    slime/backends/megatron_utils/model_provider.py#L25-L58:
    https://github.com/THUDM/slime/blob/680824dd5e01a2e83750bf87fc366ec6fa98766c/slime/backends/megatron_utils/model_provider.py#L25-L58
    """

    def __init__(self, input_size: int, output_size: int, *, config: TransformerConfig, bias: bool = True) -> None:
        super().__init__(in_features=input_size, out_features=output_size, bias=bias)
        self.sequence_parallel = config.sequence_parallel
        if self.sequence_parallel:
            self.weight.sequence_parallel = True  # pyright: ignore[reportAttributeAccessIssue]
            if bias:
                assert self.bias is not None
                self.bias.sequence_parallel = True  # pyright: ignore[reportAttributeAccessIssue]

        self.weight.data.normal_(mean=0.0, std=config.init_method_std)
        if bias:
            assert self.bias is not None
            self.bias.data.zero_()

    def forward(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        input_: torch.Tensor,
        weight: torch.Tensor | None = None,
        runtime_gather_output: bool | None = None,  # noqa: FBT001
    ) -> tuple[torch.Tensor, None]:
        del weight, runtime_gather_output
        values = super().forward(input_).float()
        if self.sequence_parallel:
            values = tensor_parallel.gather_from_sequence_parallel_region(values, tensor_parallel_output_grad=False)
        return values, None


def replace_output_layer_with_value_head(model: list[MegatronModule]) -> list[MegatronModule]:
    for model_chunk in model:
        unwrapped_model = unwrap_model(cast("GPTModel", model_chunk))
        if not getattr(unwrapped_model, "post_process", False):
            continue
        config = unwrapped_model.config
        hidden_size = int(config.hidden_size)
        unwrapped_model.output_layer = LinearForLastLayer(  # pyright: ignore[reportAttributeAccessIssue]
            input_size=hidden_size,
            output_size=1,
            config=config,
        )
    return model
