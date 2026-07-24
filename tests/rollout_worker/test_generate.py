from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import pytest
from rich.pretty import pprint
from transformers import AutoProcessor

from axrl.data import GenerationInput
from axrl.data.conversation import Conversation, Message
from tests.test_configs import all_engine_types, make_worker, qwen3_config, qwen25_config

if TYPE_CHECKING:
    from axrl.configs import EngineType, RolloutWorkerConfig


@pytest.mark.parametrize("engine_type", all_engine_types)
@pytest.mark.parametrize(
    "rollout_config",
    [
        pytest.param(qwen25_config, id="qwen2.5"),
        pytest.param(qwen3_config, id="qwen3"),
    ],
)
@pytest.mark.parametrize("use_ray_worker", [True])
def test_generate(engine_type: EngineType, rollout_config: RolloutWorkerConfig, *, use_ray_worker: bool) -> None:
    rollout_config = rollout_config.model_copy(deep=True)
    rollout_config.engine_type = engine_type
    asyncio.run(
        _test_generate(
            rollout_config=rollout_config,
            use_ray_worker=use_ray_worker,
        )
    )


async def _test_generate(rollout_config: RolloutWorkerConfig, *, use_ray_worker: bool) -> None:
    config = rollout_config.model_copy(deep=True)
    worker = make_worker(use_ray_worker=use_ray_worker, config=config)
    worker.initialize()
    prompt = "What's the capital of China?"

    messages = Conversation(messages=[Message(role="user", content=prompt)]).to_dict()["messages"]
    processor: Any = AutoProcessor.from_pretrained(
        pretrained_model_name_or_path=rollout_config.model.get_full_path(),
        use_fast=True,
    )
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    input_ids = processor(text=[prompt], return_tensors="pt", add_special_tokens=False)["input_ids"][0].tolist()
    gen_input = GenerationInput(
        session_id="test_session",
        input_ids=input_ids,
    )
    gen_output = await worker.generate(gen_input)
    pprint(gen_output)
    assert "Beijing" in gen_output.output_text
    assert gen_output.cached_tokens == 0

    await worker.flush_cache()
    gen_output = await worker.generate(gen_input)
    pprint(gen_output)
    assert "Beijing" in gen_output.output_text
    assert gen_output.cached_tokens == 0

    worker.shutdown()


if __name__ == "__main__":
    for cfg in (qwen25_config, qwen3_config):
        asyncio.run(_test_generate(rollout_config=cfg, use_ray_worker=False))
        asyncio.run(_test_generate(rollout_config=cfg, use_ray_worker=True))
