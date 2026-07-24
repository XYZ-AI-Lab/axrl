import asyncio
import logging
from typing import TYPE_CHECKING

import pytest
from transformers import AutoProcessor

from axrl.configs import EngineType
from axrl.data import GenerationInput, GenerationOutput, array_utils
from axrl.data.conversation import Conversation, Message
from axrl.ray import ray_utils
from axrl.ray.ray_rollout_worker import RayRolloutWorker
from axrl.ray.resource_group import Request, ResourceGroup
from tests.test_configs import all_engine_types, qwen3_config

if TYPE_CHECKING:
    from collections.abc import Sequence

    from axrl.configs import RolloutWorkerConfig

logger = logging.getLogger(__name__)


async def _batch_generate(
    worker: RayRolloutWorker,
    input_ids: list[int],
    session_id: str,
    repeat: int,
) -> list[str]:
    requests = [GenerationInput(session_id=session_id, input_ids=array_utils.as_i32(input_ids.copy())) for _ in range(repeat)]
    results: Sequence[GenerationOutput] = await worker.batch_generate(requests)
    return [result.output_text for result in results]


async def _test_ray_rollout_worker_group(engine_type: EngineType, repeat: int = 20) -> None:
    ray_utils.restart()
    resource_group = ResourceGroup(requests=[Request(cpu=1, gpu=2)] * 2)
    config: RolloutWorkerConfig = qwen3_config.model_copy(deep=True)
    config.engine_type = engine_type
    config.name = "ray_test"
    config.num_workers = 2
    config.tp_size = 2
    config.max_imbalance = 2
    worker_group = RayRolloutWorker(RayRolloutWorker.initialize_remote_actor(config, resource_group))

    conversation = Conversation(messages=[Message(role="user", content="What is the capital of China?")])
    processor = AutoProcessor.from_pretrained(
        pretrained_model_name_or_path=config.model.get_full_path(),
        use_fast=True,
    )
    prompt = processor.apply_chat_template(conversation.to_dict()["messages"], add_generation_prompt=True, tokenize=False)
    input_ids = processor(text=[prompt], return_tensors="pt", add_special_tokens=False)["input_ids"][0].tolist()

    logger.info("Testing Ray worker group generation...")
    responses = await _batch_generate(worker_group, input_ids=input_ids, session_id="ray_session", repeat=repeat)
    accuracy = sum("Beijing" in resp for resp in responses) / len(responses)
    logger.info("Generation accuracy: %.2f", accuracy)
    assert accuracy > 0.9, f"Generation accuracy too low: {accuracy}"

    worker_group.shutdown()
    ray_utils.stop()


@pytest.mark.parametrize("engine_type", all_engine_types)
def test_ray_rollout_worker_group(engine_type: EngineType) -> None:
    asyncio.run(_test_ray_rollout_worker_group(engine_type))


if __name__ == "__main__":
    from axrl.utils import setup_logger

    setup_logger("debug")
    test_ray_rollout_worker_group(all_engine_types[0])
