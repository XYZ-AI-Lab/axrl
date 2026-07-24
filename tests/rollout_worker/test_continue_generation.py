"""Test continue generation behavior for rollout workers."""

import asyncio
from typing import Any

import pytest
from transformers import AutoProcessor

from axrl.configs import EngineType, SamplingConfig
from axrl.data import array_utils
from axrl.ray import ray_utils
from axrl.ray.ray_rollout_worker import RayRolloutWorker
from axrl.worker.rollout_worker import GenerationInput, RolloutWorker
from tests.test_configs import all_engine_types, default_engine_type, make_worker, qwen25_config

messages = [
    {
        "role": "user",
        "content": "Please write a story about a small bird that sings a special song that makes flowers bloom even in winter.",
    }
]


def _build_sampling_config(max_total_tokens: int) -> SamplingConfig:
    """Create deterministic sampling config with a fixed total token budget."""
    return SamplingConfig(
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        max_total_tokens=max_total_tokens,
    )


def log_difference(output1: list[int], output2: list[int], processor: Any) -> None:
    """Log the first difference between two token sequences."""
    min_len = min(len(output1), len(output2))
    for i in range(min_len):
        if output1[i] != output2[i]:
            diff_id1 = output1[max(0, i - 10) : i + 10]
            diff_id2 = output2[max(0, i - 10) : i + 10]
            output_text1 = processor.decode(diff_id1, skip_special_tokens=True)
            output_text2 = processor.decode(diff_id2, skip_special_tokens=True)

            print(f"Output mismatch at index {i}")
            print(f"Output1: ={output_text1}=")
            print(f"Output2: ={output_text2}=")
            print(f"Output1 IDs: ={diff_id1}=")
            print(f"Output2 IDs: ={diff_id2}=")
            break


async def _generate_tokens(worker: RolloutWorker | RayRolloutWorker, input_ids: list[int], sampling_config: SamplingConfig) -> list[int]:
    """Helper that runs the worker and returns output token IDs."""
    generation_input = GenerationInput(
        session_id="continue-generation",
        input_ids=array_utils.as_i32(input_ids),
        sampling_config=sampling_config,
    )
    generation_output = await worker.generate(generation_input)
    return array_utils.to_int_list(generation_output.output_ids)


async def _test_continue_generation(engine_type: EngineType) -> None:
    config = qwen25_config.model_copy(deep=True)
    config.engine_type = engine_type
    worker = make_worker(config, use_ray_worker=True)
    worker.initialize()

    processor: Any = AutoProcessor.from_pretrained(
        pretrained_model_name_or_path=config.model.get_full_path(),
        use_fast=True,
    )
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    input_ids = processor(text=[prompt], return_tensors="pt", add_special_tokens=False)["input_ids"][0].tolist()

    print("Generating baseline output...")
    total_max_tokens = len(input_ids) + 256
    sampling_config = _build_sampling_config(total_max_tokens)
    output_ids = await _generate_tokens(worker, input_ids, sampling_config)
    output_text = processor.decode(output_ids, skip_special_tokens=True)
    print("Generated text, first 500 chars:")
    print("=" * 50)
    print(f"{output_text[:500]}")
    print("=" * 50)

    # Test cache flush consistency
    print("Testing cache flush consistency...")
    await worker.flush_cache()
    new_output_ids = await _generate_tokens(worker, input_ids, sampling_config)
    assert output_ids == new_output_ids
    print("✓ Cache flush produces identical output")

    # Test different input lengths
    print("Testing different input lengths...")
    await worker.flush_cache()
    prefix_tokens = min(10, len(output_ids))
    long_input_ids = input_ids + output_ids[:prefix_tokens]
    target_output_ids = output_ids[prefix_tokens:]
    long_sampling_config = _build_sampling_config(total_max_tokens)
    continued_output_ids = await _generate_tokens(worker, long_input_ids, long_sampling_config)

    if target_output_ids != continued_output_ids:
        log_difference(target_output_ids, continued_output_ids, processor)
        print("✗ Output mismatch with different input length")
    else:
        print("✓ Output consistent with different input length")

    worker.shutdown()
    ray_utils.stop()
    print("Test completed.")


@pytest.mark.parametrize("engine_type", all_engine_types)
def test_continue_generation(engine_type: EngineType) -> None:
    asyncio.run(_test_continue_generation(engine_type))


if __name__ == "__main__":
    test_continue_generation(default_engine_type)
