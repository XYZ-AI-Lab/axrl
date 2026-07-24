import asyncio
import logging
from typing import Any

import pytest
from transformers import AutoProcessor

from axrl.configs import EngineType, SamplingConfig
from axrl.data.generation import GenerationInput, GenerationOutput
from axrl.example.message_examples import PromptType, prompts
from axrl.utils.timer import Timer
from tests.test_configs import all_engine_types, default_engine_type, make_worker
from tests.test_configs import qwen3_config as rollout_config

logger = logging.getLogger(__name__)


async def _test_rollout_worker(engine_type: EngineType, prompt_type: PromptType) -> None:
    """Test rollout worker with full and interrupted rollout tests."""
    print("=" * 100)
    messages = prompts[prompt_type]
    config = rollout_config.model_copy(deep=True)
    config.engine_type = engine_type
    worker = make_worker(config, use_ray_worker=True)
    worker.initialize()

    processor: Any = AutoProcessor.from_pretrained(
        pretrained_model_name_or_path=config.model.get_full_path(),
        use_fast=True,
    )

    prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    input_ids = processor(text=[prompt], return_tensors="pt", add_special_tokens=False)["input_ids"][0].tolist()
    sampling_config = SamplingConfig(
        temperature=0.0,
        max_total_tokens=8192,
    )
    generation_input = GenerationInput(
        session_id="partial-rollout",
        input_ids=input_ids,
        sampling_config=sampling_config,
    )

    async def wait_and_pause() -> float:
        """Wait briefly then pause and resume generation."""
        await asyncio.sleep(1)
        with Timer() as t:
            await worker.pause_generation()
            await worker.resume_generation()
        elapsed_seconds_for_pause = t.elapsed_seconds
        await asyncio.sleep(2)
        return elapsed_seconds_for_pause

    async def test_full_rollout() -> GenerationOutput:
        """Test generation without interruption."""
        print("=" * 50)
        result: GenerationOutput = await worker.generate(generation_input)
        logger.info(f"Full rollout result: {result}")
        await worker.flush_cache()
        return result

    async def test_interrupted_rollout() -> tuple[GenerationOutput, float]:
        """Test generation with pause/resume interruption."""
        print("=" * 50)
        generation_task = asyncio.create_task(worker.generate(generation_input))
        pause_task = asyncio.create_task(wait_and_pause())
        result, elapsed_seconds_for_pause = await asyncio.gather(generation_task, pause_task)
        logger.info(f"Interrupted rollout result: {result}, pause result: {elapsed_seconds_for_pause}")
        await worker.flush_cache()
        return result, elapsed_seconds_for_pause

    result1 = await test_full_rollout()
    result2, _ = await test_interrupted_rollout()

    assert result1.retry == 0, "Full rollout should have zero retries."
    assert result2.retry == 1, "Interrupted rollout should have one retry."
    worker.shutdown()


@pytest.mark.parametrize("engine_type", all_engine_types)
def test_rollout_worker_partial_rollout(engine_type: EngineType) -> None:
    asyncio.run(_test_rollout_worker(engine_type, "long_lm_messages"))


if __name__ == "__main__":
    from axrl.utils.logger import setup_logger

    setup_logger("info")
    test_rollout_worker_partial_rollout(default_engine_type)
