import asyncio
import logging
import uuid

import pytest
import torch
from transformers import AutoProcessor

from axrl.configs import EngineType, MegatronWorkerConfig, ModelConfig, RolloutWorkerConfig, SamplingConfig
from axrl.data import Conversation, GenerationInput, array_utils
from axrl.ray import ray_utils
from axrl.ray.ray_megatron_worker import RayMegatronWorker
from axrl.ray.ray_rollout_worker import RayRolloutWorker
from axrl.ray.resource_group import Request, ResourceGroup
from axrl.utils import gpu_utils, setup_logger
from tests.test_configs import all_engine_types, default_engine_type

logger = logging.getLogger(__name__)


QUESTION_MESSAGES = [
    {
        "role": "user",
        "content": "What is the capital of China?",
    }
]


def _is_correct_capital_answer(text: str) -> bool:
    return "Beijing" in text


def _accuracy(texts: list[str]) -> float:
    if not texts:
        return 0.0
    correct = sum(1 for t in texts if _is_correct_capital_answer(t))
    return correct / len(texts)


def get_rollout_worker_config(engine_type: EngineType, model_config: ModelConfig, num_gpus: int, num_workers: int) -> RolloutWorkerConfig:
    return RolloutWorkerConfig(
        engine_type=engine_type,
        load_dummy_weights=True,
        model=model_config,
        tp_size=num_gpus,
        num_workers=num_workers,
        gpu_memory_utilization=0.4,
        sampling_config=SamplingConfig(
            temperature=0.0,
            max_total_tokens=model_config.seq_length,
        ),
    )


def get_megatron_worker_config(model_config: ModelConfig, num_gpus: int) -> MegatronWorkerConfig:
    return MegatronWorkerConfig(
        model=model_config,
        tp_size=num_gpus,
        inference_only=True,
        dp_size=1,
        pp_size=1,
        vpp_size=None,
    )


def _tokenize_conversation(model_config: ModelConfig, conv: Conversation) -> list[int]:
    processor = AutoProcessor.from_pretrained(
        pretrained_model_name_or_path=model_config.get_full_path(),
        use_fast=True,
    )
    messages = conv.to_dict()["messages"]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    input_ids = processor(text=[prompt], return_tensors="pt", add_special_tokens=False)["input_ids"][0].tolist()
    return input_ids


def _build_question_input_ids(rollout_worker: RayRolloutWorker) -> list[int]:
    conv = Conversation.from_dict({"messages": QUESTION_MESSAGES})
    return _tokenize_conversation(rollout_worker.get_config().model, conv)


def _run_n_generations(rollout_worker: RayRolloutWorker, input_ids: list[int], n: int) -> list[str]:
    async def _run() -> list[str]:
        session_id = str(uuid.uuid4())
        input_ids_array = array_utils.as_i32(input_ids)
        reqs = [GenerationInput(session_id=session_id, input_ids=input_ids_array) for _ in range(n)]
        results = await rollout_worker.batch_generate(reqs)
        return [r.output_text for r in results]

    return asyncio.run(_run())


def _update_and_inference(
    rollout_config: RolloutWorkerConfig,
    megatron_config: MegatronWorkerConfig,
    resource_group_rollout: ResourceGroup,
    resource_group_megatron: ResourceGroup,
    *,
    bucket_size_gb: float = 2.0,
) -> str:
    assert rollout_config.tp_size * rollout_config.num_workers == megatron_config.tp_size

    rollout_worker = RayRolloutWorker(RayRolloutWorker.initialize_remote_actor(rollout_config, resource_group_rollout))
    megatron_worker = RayMegatronWorker(config=megatron_config, resource_group=resource_group_megatron)
    megatron_worker.initialize()

    # Different placement groups => disaggregated weight update path.
    assert resource_group_rollout.pg.id != resource_group_megatron.pg.id

    megatron_worker.build_weight_updater(rollout_worker, bucket_size_gb=bucket_size_gb)
    megatron_worker.connect_rollout_worker()

    input_ids = _build_question_input_ids(rollout_worker)

    before_texts = _run_n_generations(rollout_worker, input_ids, n=10)
    logger.info(f"Before weight update: texts={before_texts}")

    # Pause + flush cache before updating.
    asyncio.run(rollout_worker.pause_generation())
    asyncio.run(rollout_worker.flush_cache())

    # Apply weight updates.
    megatron_worker.update_rollout_model_weights()
    logger.info(f"Weight update done. GPU Memory Usage: {gpu_utils.get_gpu_memory_info()}")

    asyncio.run(rollout_worker.resume_generation())

    after_texts = _run_n_generations(rollout_worker, input_ids, n=10)
    logger.info(f"After weight update: texts={after_texts}")

    rollout_worker.shutdown()
    megatron_worker.shutdown()
    logger.info(f"After shutdown. GPU Memory Usage: {gpu_utils.get_gpu_memory_info()}")
    before_acc = _accuracy(before_texts)
    after_acc = _accuracy(after_texts)
    logger.info(f"Accuracy ({rollout_config.tp_size=}, {rollout_config.num_workers=}): before={before_acc}, after={after_acc}")

    assert before_acc == 0.0
    assert after_acc == 1.0
    return ""


@pytest.mark.parametrize("engine_type", all_engine_types)
@pytest.mark.parametrize(
    ("tp_size", "num_engines"),
    [
        (2, 1),
        (1, 2),
    ],
)
def test_disaggregated_weight_update(engine_type: EngineType, tp_size: int, num_engines: int) -> None:
    setup_logger("info")

    required_gpus = 2 * (tp_size * num_engines)
    if (not torch.cuda.is_available()) or torch.cuda.device_count() < required_gpus:
        pytest.skip(f"Need >= {required_gpus} GPUs for disaggregated test ({tp_size=}, {num_engines=}).")

    ray_utils.restart()

    model_name = "Qwen/Qwen3-0.6B"
    model = ModelConfig(
        name=model_name,
        seq_length=64,
    )

    rollout_config = get_rollout_worker_config(engine_type, model, num_gpus=tp_size, num_workers=num_engines)
    megatron_total_gpus = tp_size * num_engines
    megatron_config = get_megatron_worker_config(model, num_gpus=megatron_total_gpus)

    # Separate placement groups to force disaggregated updater.
    # For rollout: one bundle per engine, with tp_size GPUs per bundle.
    resource_group_rollout = ResourceGroup([Request(gpu=tp_size, cpu=1) for _ in range(num_engines)])
    resource_group_megatron = ResourceGroup([Request(gpu=tp_size, cpu=1) for _ in range(num_engines)])

    _update_and_inference(rollout_config, megatron_config, resource_group_rollout, resource_group_megatron)
    ray_utils.stop()


if __name__ == "__main__":
    test_disaggregated_weight_update(default_engine_type, tp_size=2, num_engines=1)
