from dataclasses import asdict, dataclass, field, fields
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from rich.pretty import pprint

from axrl.configs import SamplingConfig
from axrl.data import array_utils
from axrl.data.generation import ToolCall


@dataclass
class MessagePart:
    type: Literal["text", "image"]
    text: str | None = None
    image: str | None = None

    extra: dict[str, Any] = field(default_factory=dict)
    """extra fields for future extensibility, such as video, audio, etc."""

    def to_dict(self) -> dict:
        known_fields = {f.name for f in fields(self) if f.name != "extra"}
        base = {k: v for k, v in asdict(self).items() if k in known_fields and v is not None}
        return {**base, **self.extra}

    @staticmethod
    def from_dict(data: dict) -> "MessagePart":
        known_fields = {f.name for f in fields(MessagePart) if f.name != "extra"}
        base = {k: data.get(k) for k in known_fields if k in data}
        extra = {k: v for k, v in data.items() if k not in known_fields}
        return MessagePart(**base, extra=extra)  # type: ignore


@dataclass
class Message:
    role: Literal["user", "assistant", "system", "tool"]  # add new role here if needed
    content: str | list[MessagePart]
    tool_calls: list[ToolCall] | None = None  # for assistant messages that invoke tools
    tool_call_id: str | None = None  # for tool-role messages: the id of the tool call being responded to
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        data: dict = {"role": self.role}
        if isinstance(self.content, str):
            data["content"] = self.content
        else:
            data["content"] = [part.to_dict() for part in self.content]
        if self.tool_calls is not None:
            data["tool_calls"] = [tc.to_dict() for tc in self.tool_calls]
        if self.tool_call_id is not None:
            data["tool_call_id"] = self.tool_call_id
        data.update(self.extra)
        return data

    @staticmethod
    def from_dict(data: dict) -> "Message":
        known = {"role", "content", "tool_calls", "tool_call_id"}
        extra = {k: v for k, v in data.items() if k not in known}
        content_raw: str | list[dict] = data["content"]
        content: str | list[MessagePart]
        if isinstance(content_raw, str):
            content = content_raw
        else:
            assert isinstance(content_raw, list), "Content must be a string or a list of dictionaries."
            content = [MessagePart.from_dict(part) for part in content_raw]
        tool_calls_raw = data.get("tool_calls")
        tool_calls: list[ToolCall] | None = None
        if tool_calls_raw is not None:
            tool_calls = [ToolCall.from_dict(tc, index=i) for i, tc in enumerate(tool_calls_raw)]
        message = Message(
            role=data["role"],
            content=content,
            tool_calls=tool_calls,
            tool_call_id=data.get("tool_call_id"),
            extra=extra,
        )
        return message


@dataclass
class GenerationState:
    """Per-conversation inputs for the next ``GenerationInput`` call."""

    session_id: str | None = None
    input_ids: NDArray[np.int32] | None = None
    sampling_config: SamplingConfig | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    tool_call_parser: str | None = None
    capture_routing: bool = False
    captured_routing_rows: int = 0

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {}
        if self.session_id is not None:
            data["session_id"] = self.session_id
        if self.sampling_config is not None:
            data["sampling_config"] = self.sampling_config.model_dump()
        if self.tools is not None:
            data["tools"] = self.tools
        if self.tool_choice is not None:
            data["tool_choice"] = self.tool_choice
        if self.tool_call_parser is not None:
            data["tool_call_parser"] = self.tool_call_parser
        return data

    @staticmethod
    def _parse_sampling_config(value: Any) -> SamplingConfig | None:
        if value is None:
            return None
        if isinstance(value, SamplingConfig):
            return value
        return SamplingConfig.model_validate(value)

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "GenerationState":
        return GenerationState(
            session_id=data.get("session_id"),
            input_ids=array_utils.optional_as_i32(data.get("input_ids")),
            sampling_config=GenerationState._parse_sampling_config(data.get("sampling_config")),
            tools=data.get("tools"),
            tool_choice=data.get("tool_choice"),
            tool_call_parser=data.get("tool_call_parser"),
        )


@dataclass
class Conversation:
    """Represents a multi-turn conversation."""

    messages: list[Message] = field(default_factory=list)
    conversation_id: str = field(default="")
    extra: dict[str, Any] = field(default_factory=dict)
    source: str = field(default="")
    gen_state: GenerationState = field(default_factory=GenerationState)

    def add_message(self, message: Message) -> None:
        self.messages.append(message)

    def deep_copy(self) -> "Conversation":
        import copy

        return copy.deepcopy(self)

    def to_dict(self) -> dict:
        data: dict = {
            "conversation_id": self.conversation_id,
            "messages": [msg.to_dict() for msg in self.messages],
        }
        data.update(self.gen_state.to_dict())
        data.update(self.extra)
        return data

    @staticmethod
    def from_dict(data: dict) -> "Conversation":
        reserved = {f.name for f in fields(Conversation)} | {f.name for f in fields(GenerationState)}
        extra = {k: v for k, v in data.items() if k not in reserved}
        messages_raw: list[dict] = data["messages"]
        messages: list[Message] = [Message.from_dict(msg) for msg in messages_raw]
        conversation_id = data.get("conversation_id", data.get("session_id", ""))
        return Conversation(
            conversation_id=conversation_id,
            messages=messages,
            extra=extra,
            gen_state=GenerationState.from_dict(data),
        )


def _demo_conversation_usage() -> None:
    """Demonstrates basic usage of the Conversation class.

    For more comprehensive tests, refer to `tests/test_conversation.py`.
    """
    input_dict = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, how are you?"},
        ]
    }
    conversation = Conversation.from_dict(input_dict)
    assert len(conversation.messages) == 2
    pprint(conversation.to_dict())


if __name__ == "__main__":
    _demo_conversation_usage()
