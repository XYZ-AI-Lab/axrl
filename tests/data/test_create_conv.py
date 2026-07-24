from axrl.data.conversation import Conversation, Message, MessagePart


def test_conversation_from_dict() -> None:
    raw_data = {
        "session_id": "abc123",
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant.",
                "system_tag": "v1",
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's in this image?"},
                    {"type": "image", "image": "/images/example.png", "modality": "input"},
                ],
                "user_tag": "question_1",
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "It appears to be a cat sitting on a sofa."}],
            },
        ],
        "topic": "image_captioning",
    }

    convo = Conversation.from_dict(raw_data)
    as_dict = convo.to_dict()
    expected = raw_data.copy()
    expected["conversation_id"] = expected["session_id"]
    assert as_dict == expected

    assert isinstance(convo, Conversation)
    assert convo.conversation_id == "abc123"
    assert convo.gen_state.session_id == "abc123"
    assert convo.extra["topic"] == "image_captioning"
    assert len(convo.messages) == 3

    sys_msg = convo.messages[0]
    assert sys_msg.role == "system"
    assert isinstance(sys_msg.content, str)
    assert sys_msg.extra["system_tag"] == "v1"

    user_msg = convo.messages[1]
    assert isinstance(user_msg.content, list)
    assert user_msg.content[1].type == "image"
    assert user_msg.content[1].extra["modality"] == "input"
    assert user_msg.extra["user_tag"] == "question_1"


def test_multiple_turn_conversation() -> None:
    convo = Conversation(
        conversation_id="mcp-456",
        messages=[
            Message(role="system", content="You are a helpful assistant."),
            Message(
                role="user",
                content=[
                    MessagePart(type="text", text="Here's the first image:"),
                    MessagePart(type="image", image="/path/to/image1.png"),
                    MessagePart(type="text", text="And the second one:"),
                    MessagePart(type="image", image="/path/to/image2.png"),
                    MessagePart(type="text", text="Please describe these images."),
                ],
            ),
            Message(role="assistant", content="Sure! The first image shows..."),
            Message(role="user", content="What is the difference of these two images?"),
            Message(role="assistant", content="The first image has a cat, while the second has a dog."),
        ],
        extra={"context_type": "multi-image"},
    )

    # Convert to dict and back
    as_dict = convo.to_dict()
    restored = Conversation.from_dict(as_dict)

    # Ensure round-trip integrity
    assert restored == convo
    assert len(restored.messages) == 5
    assert restored.conversation_id == "mcp-456"
    assert restored.gen_state.session_id is None

    # Spot-check turn structure
    assert restored.messages[0].role == "system"
    assert isinstance(restored.messages[1].content, list)
    assert restored.messages[3].role == "user"
    assert isinstance(restored.messages[3].content, str)


if __name__ == "__main__":
    test_conversation_from_dict()
    test_multiple_turn_conversation()
    print("All tests passed.")
