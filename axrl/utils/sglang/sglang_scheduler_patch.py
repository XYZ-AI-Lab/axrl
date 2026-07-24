from __future__ import annotations

from typing import Any


def _sanitize_req_logprob_buffers(req: Any) -> None:
    """Make sure logprob-related buffers exist before Scheduler streams outputs."""
    if not getattr(req, "return_logprob", False):
        return

    if getattr(req, "input_token_logprobs_val", None) is None:
        req.input_token_logprobs_val = []
    if getattr(req, "input_token_logprobs_idx", None) is None:
        req.input_token_logprobs_idx = []
    if getattr(req, "input_top_logprobs_val", None) is None:
        req.input_top_logprobs_val = []
    if getattr(req, "input_top_logprobs_idx", None) is None:
        req.input_top_logprobs_idx = []
    if getattr(req, "input_token_ids_logprobs_val", None) is None:
        req.input_token_ids_logprobs_val = []
    if getattr(req, "input_token_ids_logprobs_idx", None) is None:
        req.input_token_ids_logprobs_idx = []


def _sanitize_recv_logprob_buffers(recv_obj: Any, index: int) -> None:
    """Ensure tokenizer receive buffers contain lists, not None."""

    def _fix(container_name: str) -> None:
        container = getattr(recv_obj, container_name, None)
        if container is None or index >= len(container):
            return
        if container[index] is None:
            container[index] = []

    for attr in (
        "input_token_logprobs_val",
        "input_token_logprobs_idx",
        "input_top_logprobs_val",
        "input_top_logprobs_idx",
        "input_token_ids_logprobs_val",
        "input_token_ids_logprobs_idx",
        "output_token_logprobs_val",
        "output_token_logprobs_idx",
        "output_top_logprobs_val",
        "output_top_logprobs_idx",
        "output_token_ids_logprobs_val",
        "output_token_ids_logprobs_idx",
    ):
        _fix(attr)


def _install_scheduler_output_patch() -> None:
    from sglang.srt.managers.scheduler_components import output_streamer  # type: ignore[attr-defined]

    if getattr(output_streamer, "_axrl_stream_output_patched", False):  # pragma: no cover - idempotent guard
        return

    output_cls = output_streamer.SchedulerOutputStreamer
    original_stream_output = output_cls.stream_output  # type: ignore[attr-defined]

    def patched_stream_output(
        self: Any,
        reqs: list[Any],
        return_logprob: bool,  # noqa: FBT001
        skip_req: Any | None = None,
    ) -> None:
        if return_logprob:
            for req in reqs:
                if skip_req is not None and req is skip_req:
                    continue
                _sanitize_req_logprob_buffers(req)
        return original_stream_output(self, reqs, return_logprob, skip_req)

    output_cls.stream_output = patched_stream_output  # type: ignore[assignment]
    output_streamer._axrl_stream_output_patched = True  # type: ignore[attr-defined]


def _install_tokenizer_convert_patch() -> None:
    from sglang.srt.managers import tokenizer_manager  # type: ignore[attr-defined]

    if getattr(tokenizer_manager, "_axrl_convert_logprob_patched", False):  # pragma: no cover - idempotent guard
        return

    original_convert = tokenizer_manager.TokenizerManager.convert_logprob_style

    def patched_convert_logprob_style(
        self: Any,
        meta_info: dict,
        state: Any,
        top_logprobs_num: int,
        token_ids_logprob: Any,
        return_text_in_logprobs: bool,  # noqa: FBT001
        recv_obj: Any,
        recv_obj_index: int,
    ) -> None:
        if getattr(recv_obj, "input_token_logprobs_val", None) is not None:
            _sanitize_recv_logprob_buffers(recv_obj, recv_obj_index)
        return original_convert(
            self,
            meta_info,
            state,
            top_logprobs_num,
            token_ids_logprob,
            return_text_in_logprobs,
            recv_obj,
            recv_obj_index,
        )

    tokenizer_manager.TokenizerManager.convert_logprob_style = patched_convert_logprob_style  # type: ignore[assignment]
    tokenizer_manager._axrl_convert_logprob_patched = True  # type: ignore


def install_scheduler_stream_output_patch() -> None:
    """Install local monkeypatches to keep logprob buffers well-formed."""
    _install_scheduler_output_patch()
    _install_tokenizer_convert_patch()
