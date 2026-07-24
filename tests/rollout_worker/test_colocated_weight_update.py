import asyncio
import logging
import uuid

import pytest
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


def _get_rollout_worker_config(model_config: ModelConfig, tp_size: int, num_workers: int, engine_type: EngineType) -> RolloutWorkerConfig:
    return RolloutWorkerConfig(
        engine_type=engine_type,
        load_dummy_weights=True,
        model=model_config,
        tp_size=tp_size,
        num_workers=num_workers,
        gpu_memory_utilization=0.4,
        sampling_config=SamplingConfig(
            temperature=0.0,
            max_total_tokens=model_config.seq_length,
        ),
    )


def _get_megatron_worker_config(model_config: ModelConfig, total_gpus: int) -> MegatronWorkerConfig:
    return MegatronWorkerConfig(
        model=model_config,
        tp_size=total_gpus,
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
    resource_group: ResourceGroup,
    *,
    bucket_size_gb: float = 2.0,
) -> tuple[list[str], list[str]]:
    assert rollout_config.tp_size * rollout_config.num_workers == megatron_config.tp_size

    rollout_worker = RayRolloutWorker(RayRolloutWorker.initialize_remote_actor(rollout_config, resource_group))
    megatron_worker = RayMegatronWorker(config=megatron_config, resource_group=resource_group)
    megatron_worker.initialize()

    megatron_worker.build_weight_updater(rollout_worker, bucket_size_gb=bucket_size_gb)
    megatron_worker.connect_rollout_worker()

    asyncio.run(rollout_worker.release_gpu_memory(backup_weights_on_cpu=False))
    megatron_worker.to_cpu()
    logger.info("Colocated mode: Released rollout GPU memory and moved Megatron to CPU before weight update.")

    input_ids = _build_question_input_ids(rollout_worker)

    asyncio.run(rollout_worker.resume_gpu_memory())
    logger.info("Colocated mode: Resumed rollout GPU memory after Megatron moved back to GPU.")

    # Run N times with dummy weights.
    before_texts = _run_n_generations(rollout_worker, input_ids, n=10)
    logger.info(f"Before weight update: texts={before_texts}")

    # Pause + flush cache before updating.
    asyncio.run(rollout_worker.pause_generation())
    asyncio.run(rollout_worker.flush_cache())

    asyncio.run(rollout_worker.release_gpu_memory(backup_weights_on_cpu=False))
    megatron_worker.to_gpu()
    logger.info("Colocated mode: Released rollout GPU memory and moved Megatron to GPU before weight update.")

    asyncio.run(rollout_worker.resume_gpu_memory(["weights"]))
    logger.info("Colocated mode: Resumed rollout GPU memory after Megatron moved back to GPU.")

    # Apply weight updates.
    num_updates = 10
    for i in range(num_updates):
        megatron_worker.update_rollout_model_weights()
        logger.info(f"{i + 1}th/{num_updates} Weight update done. GPU Memory Usage: {gpu_utils.get_gpu_memory_info()}")

    megatron_worker.to_cpu()
    asyncio.run(rollout_worker.resume_gpu_memory(["kv_cache"]))
    asyncio.run(rollout_worker.resume_generation())

    # Run N times after weight update.
    after_texts = _run_n_generations(rollout_worker, input_ids, n=10)
    logger.info(f"After weight update: texts={after_texts}")

    rollout_worker.shutdown()
    megatron_worker.shutdown()
    resource_group.shutdown()
    logger.info(f"After shutdown. GPU Memory Usage: {gpu_utils.get_gpu_memory_info()}")

    return before_texts, after_texts


@pytest.mark.parametrize("engine_type", all_engine_types)
@pytest.mark.parametrize(
    ("tp_size", "num_workers"),
    [
        (1, 4),
        (2, 2),
        (4, 1),
    ],
)
def test_colocated_weight_update_accuracy(engine_type: EngineType, tp_size: int, num_workers: int) -> None:
    setup_logger("info")
    ray_utils.restart()

    total_gpus = 4
    assert tp_size * num_workers == total_gpus

    model = ModelConfig(
        name="Qwen/Qwen3-0.6B",
        seq_length=64,
    )

    rollout_config = _get_rollout_worker_config(model, tp_size=tp_size, num_workers=num_workers, engine_type=engine_type)
    megatron_config = _get_megatron_worker_config(model, total_gpus=total_gpus)

    resource_group = ResourceGroup([Request(gpu=tp_size, cpu=1) for _ in range(num_workers)])

    before_texts, after_texts = _update_and_inference(rollout_config, megatron_config, resource_group)
    before_acc = _accuracy(before_texts)
    after_acc = _accuracy(after_texts)
    logger.info(f"Accuracy ({tp_size=}, {num_workers=}): before={before_acc}, after={after_acc}")

    assert before_acc == 0.0
    assert after_acc == 1.0
    logger.info(f"Colocated weight update accuracy test passed for {engine_type} with tp_size={tp_size}, num_workers={num_workers}.")
    ray_utils.stop()


if __name__ == "__main__":
    test_colocated_weight_update_accuracy(engine_type=default_engine_type, tp_size=1, num_workers=4)
