from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from axrl.configs import SamplingConfig
from axrl.data import array_utils
from axrl.data.event_timing import EventTiming
from axrl.utils.tensor_store import TensorHandle


@dataclass
class GenerationInput:
    session_id: str
    input_ids: NDArray[np.int32]
    sampling_config: SamplingConfig | None = None
    lora_path: str | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    tool_call_parser: str | None = None
    capture_input_logprobs: bool = False
    # First input token index whose numeric logprob should be returned.
    # Token 0 is not scorable because it has no previous token context.
    input_logprob_start_index: int = 0
    capture_routing: bool = False
    routed_expert_start_index: int = 0
    event_timing: EventTiming = field(default_factory=EventTiming)

    def __post_init__(self) -> None:
        self.input_ids = array_utils.as_i32(self.input_ids)
        assert self.input_logprob_start_index >= 0, "input_logprob_start_index must be non-negative."


@dataclass
class ToolCall:
    """A parsed tool/function call extracted from model output."""

    id: str
    index: int
    name: str
    arguments: str  # JSON string

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "type": "function", "function": {"name": self.name, "arguments": self.arguments}}

    @staticmethod
    def from_dict(data: dict[str, Any], index: int = 0) -> "ToolCall":
        return ToolCall(
            id=data["id"],
            index=index,
            name=data["function"]["name"],
            arguments=data["function"]["arguments"],
        )


@dataclass
class GenerationOutput:
    session_id: str
    output_ids: NDArray[np.int32]
    output_logprobs: NDArray[np.float32]
    output_text: str  # detokenized text without special tokens
    output_text_with_special_tokens: str
    cached_tokens: int
    finish_reason: str
    e2e_elapsed_seconds: float
    stop_reason: int | str | None
    retry: int
    assistant_boundary_token_id: int | None = None
    tool_calls: list[ToolCall] | None = None
    input_logprobs: NDArray[np.float32] | None = None
    input_logprob_token_ids: NDArray[np.int32] | None = None
    input_logprob_start_index: int | None = None
    routing_handle: TensorHandle | None = None
    event_timing: EventTiming = field(default_factory=EventTiming)

    def __post_init__(self) -> None:
        self.output_ids = array_utils.as_i32(self.output_ids)
        self.output_logprobs = array_utils.as_f32(self.output_logprobs)
        if self.input_logprobs is not None:
            self.input_logprobs = array_utils.as_f32(self.input_logprobs)
        if self.input_logprob_token_ids is not None:
            self.input_logprob_token_ids = array_utils.as_i32(self.input_logprob_token_ids)
        if self.input_logprob_start_index is not None:
            assert self.input_logprob_start_index >= 0, "input_logprob_start_index must be non-negative."


@dataclass
class GenerationPair:
    input: GenerationInput
    output: GenerationOutput


"""Examples for ``GenerationInput`` and ``ToolCall``.

Tool calling fields in ``GenerationInput`` follow the OpenAI chat-completions
shape. Example:

    GenerationInput(
        session_id="sess_1",
        input_ids=[1, 2, 3],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Search for relevant information.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                        },
                        "required": ["query"],
                    },
                },
            }
        ],
        tool_choice="auto",
        tool_call_parser="qwen",
    )

``tools`` is the list of available tools in OpenAI function-tool format.
``tool_choice`` may be ``"auto"``, ``"required"``, ``"none"``, or a named-tool
dict such as ``{"type": "function", "function": {"name": "search"}}``.
``tool_call_parser`` names the parser for model-native tool-call output in
``tool_choice="auto"`` mode, for example ``"qwen"``.


Example for ``ToolCall``:
    ToolCall(
        id="call_abc123",
        index=0,
        name="get_current_temperature",
        arguments='{"location": "Paris, France"}',
    )
"""
