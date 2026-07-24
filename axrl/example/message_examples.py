from typing import Any, Literal

__all__ = [
    "PromptType",
    "prompts",
]

short_vlm_messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "I have two images. The first one is: ",
            },
            {
                "type": "image",
                "image": "/workspaces/axrl/axrl/example/qwen_vl/images/fig1.png",
            },
            {
                "type": "text",
                "text": ". The second one is: ",
            },
            {
                "type": "image",
                "image": "/workspaces/axrl/axrl/example/qwen_vl/images/fig2.png",
            },
            {
                "type": "text",
                "text": ". Please describe the content of these two images, as well as the differences between them.",
            },
        ],
    }
]

long_vlm_messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "I have two images. The first one is: ",
            },
            {
                "type": "image",
                "image": "/workspaces/axrl/axrl/example/qwen_vl/images/fig1.png",
            },
            {
                "type": "text",
                "text": ". The second one is: ",
            },
            {
                "type": "image",
                "image": "/workspaces/axrl/axrl/example/qwen_vl/images/fig2.png",
            },
            {
                "type": "text",
                "text": ". Please describe the content of these two images, as well as the differences between them. "
                "The description should be detailed and include various elements such as colors, shapes, and any notable features. "
                "The response should be at least 8000 words long.",
            },
        ],
    }
]

short_lm_messages = [
    {
        "role": "user",
        "content": "Please write a story about a small bird that sings a special song that makes flowers bloom even in winter.",
    }
]

long_lm_messages = [
    {
        "role": "user",
        "content": "Please write a story about a small bird that sings a special song that makes flowers bloom even in winter. "
        "The story should be detailed and include various elements such as the bird's journey, the challenges it faces, "
        "and the impact of its song on the environment and other creatures. The story should be at least 8000 words long.",
    }
]

PromptType = Literal["short_vlm_messages", "long_vlm_messages", "short_lm_messages", "long_lm_messages"]

prompts: dict[PromptType, Any] = {
    "short_vlm_messages": short_vlm_messages,
    "long_vlm_messages": long_vlm_messages,
    "short_lm_messages": short_lm_messages,
    "long_lm_messages": long_lm_messages,
}
