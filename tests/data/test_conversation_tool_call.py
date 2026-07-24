from axrl.data.conversation import Conversation, Message
from axrl.data.generation import ToolCall


def test_message_with_tool_calls_round_trip() -> None:
    """Assistant message with tool_calls survives to_dict -> from_dict."""
    tc = ToolCall(id="call_abc", index=0, name="search", arguments='{"query": "python"}')
    msg = Message(role="assistant", content="Let me search.", tool_calls=[tc])

    d = msg.to_dict()
    assert d == {
        "role": "assistant",
        "content": "Let me search.",
        "tool_calls": [
            {"id": "call_abc", "type": "function", "function": {"name": "search", "arguments": '{"query": "python"}'}},
        ],
    }

    restored = Message.from_dict(d)
    assert restored.role == "assistant"
    assert restored.content == "Let me search."
    assert restored.tool_calls is not None
    assert len(restored.tool_calls) == 1
    assert restored.tool_calls[0].id == "call_abc"
    assert restored.tool_calls[0].index == 0
    assert restored.tool_calls[0].name == "search"
    assert restored.tool_calls[0].arguments == '{"query": "python"}'


def test_tool_response_message_round_trip() -> None:
    """Tool-role message with tool_call_id survives to_dict -> from_dict."""
    msg = Message(role="tool", content='{"result": "found 42 results"}', tool_call_id="call_abc")

    d = msg.to_dict()
    assert d == {
        "role": "tool",
        "content": '{"result": "found 42 results"}',
        "tool_call_id": "call_abc",
    }

    restored = Message.from_dict(d)
    assert restored.role == "tool"
    assert restored.tool_call_id == "call_abc"
    assert restored.tool_calls is None


def test_conversation_with_tools_round_trip() -> None:
    """Full conversation with tools/tool_choice/tool_call_parser round-trips."""
    search_tool = {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search the web",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
        },
    }
    raw = {
        "conversation_id": "sess_1",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of France?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "call_1", "type": "function", "function": {"name": "search", "arguments": '{"query": "capital of France"}'}},
                ],
            },
            {"role": "tool", "content": "Paris is the capital of France.", "tool_call_id": "call_1"},
            {"role": "assistant", "content": "The capital of France is Paris."},
        ],
        "tools": [search_tool],
        "tool_choice": "auto",
        "tool_call_parser": "qwen",
    }

    conv = Conversation.from_dict(raw)
    assert conv.gen_state.tools == [search_tool]
    assert conv.gen_state.tool_choice == "auto"
    assert conv.gen_state.tool_call_parser == "qwen"
    assert len(conv.messages) == 5

    # assistant with tool_calls
    assistant_msg = conv.messages[2]
    assert assistant_msg.tool_calls is not None
    assert assistant_msg.tool_calls[0].name == "search"

    # tool response
    tool_msg = conv.messages[3]
    assert tool_msg.role == "tool"
    assert tool_msg.tool_call_id == "call_1"

    # round-trip
    assert conv.to_dict() == raw


def test_multiple_tool_calls_in_one_message() -> None:
    """Assistant can invoke multiple tools in a single message."""
    raw_msg = {
        "role": "assistant",
        "content": "I'll look up both.",
        "tool_calls": [
            {"id": "call_a", "type": "function", "function": {"name": "search", "arguments": '{"query": "weather Paris"}'}},
            {"id": "call_b", "type": "function", "function": {"name": "search", "arguments": '{"query": "weather London"}'}},
        ],
    }

    msg = Message.from_dict(raw_msg)
    assert msg.tool_calls is not None
    assert len(msg.tool_calls) == 2
    assert msg.tool_calls[0].index == 0
    assert msg.tool_calls[1].index == 1
    assert msg.tool_calls[0].id == "call_a"
    assert msg.tool_calls[1].id == "call_b"

    assert msg.to_dict() == raw_msg


def test_parsed_tool_call_from_dict() -> None:
    """ToolCall.from_dict reconstructs correctly."""
    d = {"id": "call_x", "type": "function", "function": {"name": "get_weather", "arguments": '{"city": "Tokyo"}'}}
    tc = ToolCall.from_dict(d, index=3)
    assert tc.id == "call_x"
    assert tc.index == 3
    assert tc.name == "get_weather"
    assert tc.arguments == '{"city": "Tokyo"}'
    assert tc.to_dict() == d
