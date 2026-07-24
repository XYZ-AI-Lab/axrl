from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import torch

from axrl.utils.megatron.utils import unwrap_model

if TYPE_CHECKING:
    from megatron.core.models.gpt import GPTModel

logger = logging.getLogger(__name__)


def _axrl_to_fp32(x: Any) -> Any:
    if isinstance(x, torch.Tensor) and x.dtype != torch.float32:
        return x.to(torch.float32)
    return x


class _Fp32WeightCast(torch.autograd.Function):
    """Cast weight to FP32 for forward, sync `grad_added_to_main_grad` in backward.

    Megatron-Core's tensor-parallel linear backward with `gradient_accumulation_fusion=True`
    writes weight gradients directly into `weight.main_grad` and sets
    `weight.grad_added_to_main_grad = True` on the weight tensor passed into forward.

    When we create an FP32 copy of the weight, the backward sets this flag on the FP32 copy,
    not the original BF16/FP16 parameter. The DDP hook later checks this flag on the *original*
    parameter, and if it's False, may incorrectly try to add `.grad` to `.main_grad`.

    This custom autograd function:
    1. Forward: creates FP32 copy with Megatron attributes (including shared `main_grad` buffer)
    2. Backward: syncs `grad_added_to_main_grad` from FP32 copy back to original parameter,
       returns None to prevent any gradient flow through the cast (main_grad already updated)
    """

    @staticmethod
    def forward(ctx: Any, w: torch.Tensor) -> torch.Tensor:
        if w.dtype == torch.float32:
            ctx.original_weight = None
            ctx.w_fp32 = None
            return w

        w_fp32 = w.detach().to(torch.float32)
        w_fp32.requires_grad_(w.requires_grad)

        # Copy Megatron attributes - especially `main_grad` which points to the shared buffer.
        for attr in ("main_grad", "grad_added_to_main_grad", "zero_out_wgrad", "__fsdp_param__"):
            if hasattr(w, attr):
                setattr(w_fp32, attr, getattr(w, attr))

        # Save references for backward
        ctx.original_weight = w
        ctx.w_fp32 = w_fp32

        return w_fp32

    @staticmethod
    def backward(ctx: Any, grad_w_fp32: torch.Tensor | None) -> torch.Tensor | None:  # noqa: ARG004    # type: ignore[override]
        # Sync `grad_added_to_main_grad` from FP32 copy back to original parameter.
        # This ensures the DDP hook sees the correct flag value.
        w_fp32 = ctx.w_fp32
        original = ctx.original_weight

        if w_fp32 is not None and original is not None:
            if hasattr(w_fp32, "grad_added_to_main_grad"):
                original.grad_added_to_main_grad = w_fp32.grad_added_to_main_grad

        # Return None - the fused CUDA kernel already wrote to `main_grad`.
        # We don't want any gradient to flow through the cast operation.
        # Note: grad_w_fp32 is intentionally unused - required by autograd signature.
        return None


def _axrl_fp32_weight(w: torch.Tensor) -> torch.Tensor:
    """Cast `w` to FP32 for matmul while preserving Megatron backward expectations.

    Uses a custom autograd function to properly sync `grad_added_to_main_grad`
    back to the original parameter after backward.
    """
    return _Fp32WeightCast.apply(w)  # type: ignore[arg-type]


def _axrl_cast_lm_head_output_to_fp32(out: Any) -> Any:
    """Cast lm-head outputs to fp32; raise if output structure is unexpected."""
    # Megatron output layers commonly return either:
    # - logits: Tensor
    # - (logits, bias): tuple[Tensor, Tensor|None]
    assert isinstance(out, (torch.Tensor, tuple)), f"Unexpected lm-head output type: {type(out)}"
    if isinstance(out, torch.Tensor):
        return _axrl_to_fp32(out)

    # Tuple path
    assert isinstance(out, tuple)
    assert len(out) == 2, f"Unexpected lm-head output tuple: len={len(out)}"
    logits, bias = out
    assert isinstance(logits, torch.Tensor), f"Unexpected lm-head logits type: {type(logits)}"
    assert bias is None or isinstance(bias, torch.Tensor), f"Unexpected lm-head bias type: {type(bias)}"
    return _axrl_to_fp32(logits), (None if bias is None else _axrl_to_fp32(bias))


def cast_output_layer_to_fp32(
    model: list[GPTModel],
) -> None:
    """Ensure lm-head logits are computed in FP32 without mutating weights.

    In Megatron-Core, logits are produced via `GPTModel._postprocess()` calling
    `self.output_layer(hidden_states, weight=output_weight, ...)` when weights are
    tied, and `self.output_layer(hidden_states, ...)` otherwise.

    We wrap `output_layer.forward` to cast:
    - input activations to FP32
    - the effective lm-head weight tensor to FP32 (passed-in `weight=` or the module's
      own `self.weight` when present)
    - returned logits to FP32 (including tuple outputs like `(logits, bias)`)

    This avoids any in-place dtype changes that can break checkpoint save/load.
    """
    for model_chunk in model:
        unwrapped_model = unwrap_model(model_chunk)
        output_layer = getattr(unwrapped_model, "output_layer", None)
        if output_layer is None:
            continue

        if getattr(output_layer, "_axrl_fp32_lm_head_wrapped", False):
            continue

        logger.info("Enabling FP32 lm-head logits (no in-place casts)")
        orig_forward = output_layer.forward

        def fp32_forward(*args: Any, _orig_forward: Any = orig_forward, _output_layer: Any = output_layer, **kwargs: Any) -> Any:
            # Megatron-Core reference for lm-head forward signature and return type:
            # - megatron/core/tensor_parallel/layers.py
            #   :: ColumnParallelLinear.forward
            #   forward(input_, weight=None, runtime_gather_output=None) -> (output, output_bias)
            args_list = list(args)
            if args_list:
                # Cast input activations to FP32
                args_list[0] = _axrl_to_fp32(args_list[0])

            # Normalize/cast weight from kwargs or positional args.
            weight: torch.Tensor | None = None
            if "weight" in kwargs and isinstance(kwargs["weight"], torch.Tensor):
                weight = kwargs["weight"]
                kwargs["weight"] = _axrl_fp32_weight(weight)
            elif len(args_list) >= 2 and isinstance(args_list[1], torch.Tensor):
                weight = args_list[1]
                args_list[1] = _axrl_fp32_weight(weight)
            else:
                module_weight = getattr(_output_layer, "weight", None)
                if isinstance(module_weight, torch.Tensor):
                    kwargs["weight"] = _axrl_fp32_weight(module_weight)

            out = _orig_forward(*tuple(args_list), **kwargs)
            return _axrl_cast_lm_head_output_to_fp32(out)

        setattr(output_layer, "forward", fp32_forward)  # noqa: B010
        setattr(output_layer, "_axrl_fp32_lm_head_wrapped", True)  # noqa: B010
