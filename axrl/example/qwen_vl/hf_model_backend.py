import logging

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from axrl.configs import ModelConfig

logger = logging.getLogger(__name__)

example_messages = [
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


def test_qwen2_5_vl(model_config: ModelConfig) -> None:
    """Test function for Qwen2.5-VL model."""
    # load model
    model_dir = model_config.get_full_path()
    assert model_dir.exists(), f"Model directory {model_dir} does not exist."
    logger.info(f"Loading model from {model_dir}")
    assert torch.cuda.is_available(), "CUDA is not available. Please check your environment."
    device = torch.device("cuda")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        pretrained_model_name_or_path=model_dir,
        local_files_only=True,
        torch_dtype=torch.float32,
        device_map=device,
        # attn_implementation="flash_attention_2",  # is only available for bf16 or fp16 models
    )

    logger.info(f"Model loaded from {model_dir}")

    # default processer
    processor = AutoProcessor.from_pretrained(
        pretrained_model_name_or_path=model_dir,
        trust_remote_code=model_config.trust_remote_code,
    )
    logger.info(f"Processor loaded from {model_dir}")

    messages = example_messages

    # Preparation for inference
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
    logger.info(f"Image inputs: {image_inputs}")
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
        **video_kwargs,  # type: ignore
    )
    inputs = inputs.to(device)
    logger.info(f"Inputs: {inputs}")

    # Inference: Generation of the output
    logger.info("Generating output...")
    generated_ids = model.generate(**inputs, max_new_tokens=2048)  # type: ignore[misc]  # pyright: ignore[reportAttributeAccessIssue]
    generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids, strict=False)]
    output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    logger.info(output_text)


if __name__ == "__main__":
    test_qwen2_5_vl(
        ModelConfig(
            name="Qwen/Qwen2.5-VL-3B-Instruct",
            trust_remote_code=True,
        )
    )
