import asyncio
import logging
import uuid

import pytest
import torch
from transformers import AutoProcessor

from axrl.configs import EngineType, MegatronWorkerConfig, ModelConfig, RolloutWorkerConfig, SamplingConfig
from axrl.data import Conversation, GenerationInput, array_utils
from axrl.example.message_examples import short_lm_messages
from axrl.ray import ray_utils
from axrl.ray.ray_megatron_worker import RayMegatronWorker
from axrl.ray.ray_rollout_worker import RayRolloutWorker
from axrl.ray.resource_group import Request, ResourceGroup
from axrl.utils import gpu_utils, setup_logger
from tests.test_configs import all_engine_types, default_engine_type

logger = logging.getLogger(__name__)


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


def _get_expected_output(rollout_config: RolloutWorkerConfig, resource_group: ResourceGroup) -> str:
    rollout_config = rollout_config.model_copy()
    rollout_config.load_dummy_weights = False
    rollout_worker = RayRolloutWorker(RayRolloutWorker.initialize_remote_actor(rollout_config, resource_group))

    conv = Conversation.from_dict({"messages": short_lm_messages})
    input_ids = _tokenize_conversation(rollout_worker.get_config().model, conv)
    req = GenerationInput(session_id="expected_output", input_ids=array_utils.as_i32(input_ids))
    output = asyncio.run(rollout_worker.generate(req))
    logger.info(f"Rollout output with valid weights: {output.output_text}")

    rollout_worker.shutdown()
    gpu_utils.assert_all_gpus_empty(max_used_gb=2)
    return output.output_text


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

    megatron_worker.build_weight_updater(rollout_worker, bucket_size_gb=bucket_size_gb)
    megatron_worker.connect_rollout_worker()

    colocated = resource_group_rollout.pg.id == resource_group_megatron.pg.id
    if colocated:
        asyncio.run(rollout_worker.release_gpu_memory(backup_weights_on_cpu=False))
        megatron_worker.to_cpu()
        logger.info("Colocated mode: Released rollout GPU memory and moved Megatron to CPU before weight update.")

    conv = Conversation.from_dict({"messages": short_lm_messages})
    input_ids = _tokenize_conversation(rollout_worker.get_config().model, conv)
    req = GenerationInput(session_id="update_and_inference", input_ids=array_utils.as_i32(input_ids))

    if colocated:
        asyncio.run(rollout_worker.resume_gpu_memory())
        logger.info("Colocated mode: Resumed rollout GPU memory after Megatron moved back to GPU.")

    # Run once with dummy weights.
    output_before = asyncio.run(rollout_worker.generate(req))
    logger.info(f"Before weight update: rollout output: {output_before.output_text}")

    # Pause + flush cache before updating.
    asyncio.run(rollout_worker.pause_generation())
    asyncio.run(rollout_worker.flush_cache())

    if colocated:
        asyncio.run(rollout_worker.release_gpu_memory(backup_weights_on_cpu=False))
        megatron_worker.to_gpu()
        logger.info("Colocated mode: Released rollout GPU memory and moved Megatron to GPU before weight update.")

    if colocated:
        asyncio.run(rollout_worker.resume_gpu_memory(["weights"]))
        logger.info("Colocated mode: Resumed rollout GPU memory after Megatron moved back to GPU.")

    # Apply weight updates.
    megatron_worker.update_rollout_model_weights()
    logger.info(f"Weight update done. GPU Memory Usage: {gpu_utils.get_gpu_memory_info()}")

    if colocated:
        megatron_worker.to_cpu()
        asyncio.run(rollout_worker.resume_gpu_memory(["kv_cache"]))
        logger.info("Colocated mode: Resumed rollout KV cache GPU memory after weight update.")

    asyncio.run(rollout_worker.resume_generation())

    # Run again after weight update.
    output_after = asyncio.run(rollout_worker.generate(GenerationInput(session_id=str(uuid.uuid4()), input_ids=array_utils.as_i32(input_ids))))
    logger.info(f"After weight update: rollout output: {output_after.output_text}")

    rollout_worker.shutdown()
    megatron_worker.shutdown()
    logger.info(f"After shutdown. GPU Memory Usage: {gpu_utils.get_gpu_memory_info()}")
    return output_after.output_text


@pytest.mark.parametrize("engine_type", all_engine_types)
@pytest.mark.parametrize(
    ("tp_size", "num_engines"),
    [
        (2, 1),
        (1, 2),
    ],
)
def test_weight_update(engine_type: EngineType, tp_size: int, num_engines: int) -> None:
    setup_logger("info")

    # Disaggregated mode needs separate GPU pools for rollout and Megatron.
    required_gpus = 2 * (tp_size * num_engines)
    assert torch.cuda.device_count() >= required_gpus, f"Need >= {required_gpus} GPUs for rollout weight update test ({tp_size=}, {num_engines=})."
    ray_utils.restart()

    model_name = "Qwen/Qwen3-0.6B"
    model = ModelConfig(
        name=model_name,
        seq_length=64,
    )

    rollout_config = get_rollout_worker_config(engine_type, model, num_gpus=tp_size, num_workers=num_engines)
    megatron_config = get_megatron_worker_config(model, num_gpus=tp_size * num_engines)

    # rollout: one bundle per engine, with tp_size GPUs per bundle.
    resource_group_rollout = ResourceGroup([Request(gpu=tp_size, cpu=1) for _ in range(num_engines)])
    resource_group_megatron = ResourceGroup([Request(gpu=tp_size, cpu=1) for _ in range(num_engines)])

    # Expected output from rollout with valid weights (run on the rollout resource group).
    expected_output = _get_expected_output(rollout_config, resource_group_rollout)

    # Colocated weight update: use the same placement group.
    output_colocated = _update_and_inference(
        rollout_config,
        megatron_config,
        resource_group_rollout,
        resource_group_rollout,
    )
    assert expected_output == output_colocated

    # Disaggregated weight update: use separate placement groups.
    output_disaggregated = _update_and_inference(
        rollout_config,
        megatron_config,
        resource_group_rollout,
        resource_group_megatron,
    )
    assert expected_output == output_disaggregated

    logger.info(f"Weight update test passed for engine_type={engine_type}, tp_size={tp_size}, num_engines={num_engines}.")
    ray_utils.stop()


if __name__ == "__main__":
    test_weight_update(default_engine_type, tp_size=1, num_engines=2)
    test_weight_update(default_engine_type, tp_size=2, num_engines=1)
    # python -u tests/rollout_worker/test_weight_update.py 2>&1  | tee tmp/run-weight-update.log
