"""Colocated weight update test for MoE models (Qwen3-30B-A3B).

Tests that weights are correctly synchronized from Megatron to SGLang for
MoE models with expert parallelism (EP) and tensor parallelism (TP).
Verifies:
- Weight update produces correct inference results
- No memory leak across multiple updates
- Works with various Megatron TP/DP/EP configurations
- FP8 rollout sync: Megatron weights can be synced into an FP8 SGLang model
- FP8 MCore compute: Megatron can run with TransformerEngine FP8 enabled during sync
"""

import asyncio
import logging
import uuid
from typing import Literal

import pytest
import torch
from transformers import AutoProcessor

from axrl.configs import MegatronWorkerConfig, ModelConfig, RolloutWorkerConfig, SamplingConfig
from axrl.data import Conversation, GenerationInput, array_utils
from axrl.ray import ray_utils
from axrl.ray.ray_megatron_worker import RayMegatronWorker
from axrl.ray.ray_rollout_worker import RayRolloutWorker
from axrl.ray.resource_group import Request, ResourceGroup
from axrl.utils import setup_logger

logger = logging.getLogger(__name__)

BF16_MODEL_NAME = "Qwen/Qwen3-30B-A3B-Instruct-2507"
FP8_MODEL_NAME = "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"
NUM_GPUS_REQUIRED = 8
R3_MISMATCH_TOPOLOGY_CASES = [
    pytest.param("tp2-cp1-pp1-ep4", 4, 2, 4, 1, 1, id="tp2-cp1-pp1-ep4"),
    pytest.param("tp1-cp1-pp1-ep8", 8, 1, 8, 1, 1, id="tp1-cp1-pp1-ep8"),
    pytest.param("tp2-cp2-pp1-ep4", 2, 2, 4, 2, 1, id="tp2-cp2-pp1-ep4"),
    pytest.param("tp2-cp1-pp2-ep2", 2, 2, 2, 1, 2, id="tp2-cp1-pp2-ep2"),
    pytest.param("tp1-cp2-pp2-ep2", 2, 1, 2, 2, 2, id="tp1-cp2-pp2-ep2"),
]

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


def _tokenize_conversation(model_config: ModelConfig, messages: list[dict]) -> list[int]:
    processor = AutoProcessor.from_pretrained(
        pretrained_model_name_or_path=model_config.get_full_path(),
        trust_remote_code=True,
        use_fast=True,
    )
    conv = Conversation.from_dict({"messages": messages})
    prompt = processor.apply_chat_template(conv.to_dict()["messages"], add_generation_prompt=True, tokenize=False)
    input_ids = processor(text=[prompt], return_tensors="pt", add_special_tokens=False)["input_ids"][0].tolist()
    return input_ids


def _run_n_generations(rollout_worker: RayRolloutWorker, input_ids: list[int], n: int) -> list[str]:
    async def _run() -> list[str]:
        session_id = str(uuid.uuid4())
        input_ids_array = array_utils.as_i32(input_ids)
        reqs = [GenerationInput(session_id=session_id, input_ids=input_ids_array) for _ in range(n)]
        results = await rollout_worker.batch_generate(reqs)
        return [r.output_text for r in results]

    return asyncio.run(_run())


def _test_moe_colocated_weight_update(
    *,
    rollout_model_name: str = BF16_MODEL_NAME,
    megatron_model_name: str = BF16_MODEL_NAME,
    megatron_fp8: Literal["e4m3", "hybrid"] | None = None,
    megatron_use_language_model_only: bool = False,
    rollout_tp_size: int,
    rollout_ep_size: int,
    num_workers: int,
    megatron_tp_size: int,
    megatron_dp_size: int,
    megatron_ep_size: int,
    megatron_cp_size: int = 1,
    megatron_pp_size: int = 1,
    num_updates: int = 3,
    bucket_size_gb: float = 2.0,
) -> None:
    """Run a colocated weight update test for MoE model with given parallelism config.

    There are two independent FP8 knobs in this test:

     1. FP8 rollout model: set rollout_model_name to the FP8 checkpoint. During weight
         sync, exported Megatron weights are quantized to FP8 before sending to SGLang,
         using the rollout model's quantization_config.
     2. FP8 MCore compute: set megatron_fp8 (for example "e4m3") to enable
         TransformerEngine FP8 execution on the Megatron side.

     The BF16 and FP8 knobs are intentionally independent here. The validated weight
     sync path in this repo keeps Megatron source weights in BF16, optionally enables
     FP8 compute in MCore, and quantizes exported weights to FP8 only for the rollout
     side when rollout_model_name points at the FP8 checkpoint.

    Args:
        rollout_model_name: Model for SGLang rollout. FP8 variant triggers FP8 weight sync.
        megatron_model_name: Model path for Megatron checkpoint loading.
        megatron_fp8: Optional TransformerEngine FP8 mode for Megatron, such as "e4m3".
        megatron_use_language_model_only: Load only the text model from Qwen3.5/3.6 VL checkpoints.
        rollout_tp_size: SGLang tensor parallelism size.
        rollout_ep_size: SGLang expert parallelism size.
        num_workers: Number of SGLang worker engines.
        megatron_tp_size: Megatron tensor parallelism size.
        megatron_dp_size: Megatron data parallelism size.
        megatron_ep_size: Megatron expert parallelism size.
        megatron_cp_size: Megatron context parallelism size.
        megatron_pp_size: Megatron pipeline parallelism size.
        num_updates: Number of weight updates to run.
        bucket_size_gb: Size of each weight sync bucket in GB.
    """
    setup_logger("info")
    ray_utils.restart()

    total_gpus = rollout_tp_size * num_workers
    megatron_world_size = megatron_tp_size * megatron_dp_size * megatron_cp_size * megatron_pp_size
    assert total_gpus == megatron_world_size, f"Total GPUs mismatch: rollout={total_gpus}, megatron={megatron_world_size}"

    megatron_model = ModelConfig(
        name=megatron_model_name,
        trust_remote_code=True,
        seq_length=64,
    )

    rollout_model = ModelConfig(
        name=rollout_model_name,
        trust_remote_code=True,
        seq_length=64,
    )

    rollout_config = RolloutWorkerConfig(
        engine_type="sglang",
        load_dummy_weights=True,
        model=rollout_model,
        tp_size=rollout_tp_size,
        ep_size=rollout_ep_size,
        num_workers=num_workers,
        gpu_memory_utilization=0.4,
        dtype="auto",
        sampling_config=SamplingConfig(
            temperature=0.0,
            max_total_tokens=megatron_model.seq_length,
        ),
    )

    megatron_config = MegatronWorkerConfig(
        model=megatron_model,
        tp_size=megatron_tp_size,
        dp_size=megatron_dp_size,
        cp_size=megatron_cp_size,
        ep_size=megatron_ep_size,
        etp_size=1,
        pp_size=megatron_pp_size,
        vpp_size=None,
        inference_only=True,
        fp8=megatron_fp8,
        use_language_model_only=megatron_use_language_model_only,
    )

    resource_group = ResourceGroup([Request(gpu=rollout_tp_size, cpu=1) for _ in range(num_workers)])

    rollout_worker = RayRolloutWorker(RayRolloutWorker.initialize_remote_actor(rollout_config, resource_group))
    megatron_worker = RayMegatronWorker(config=megatron_config, resource_group=resource_group)
    megatron_worker.initialize()

    megatron_worker.build_weight_updater(rollout_worker, bucket_size_gb=bucket_size_gb)
    megatron_worker.connect_rollout_worker()

    asyncio.run(rollout_worker.release_gpu_memory(backup_weights_on_cpu=False))
    megatron_worker.to_cpu()

    # Use megatron model for tokenization (same tokenizer for BF16 and FP8)
    input_ids = _tokenize_conversation(megatron_model, QUESTION_MESSAGES)

    asyncio.run(rollout_worker.resume_gpu_memory())

    # Generate with dummy weights — should produce garbage.
    before_texts = _run_n_generations(rollout_worker, input_ids, n=5)
    logger.info(f"Before weight update: accuracy={_accuracy(before_texts)}, texts={before_texts!r}")

    # Pause + flush + prepare for weight update.
    asyncio.run(rollout_worker.pause_generation())
    asyncio.run(rollout_worker.flush_cache())
    asyncio.run(rollout_worker.release_gpu_memory(backup_weights_on_cpu=False))
    megatron_worker.to_gpu()
    asyncio.run(rollout_worker.resume_gpu_memory(["weights"]))

    # Apply weight updates multiple times to check for memory leaks.
    for i in range(num_updates):
        usage_infos = megatron_worker.update_rollout_model_weights()
        peak_reserved = max(info.peak_mem_reserved_gbs for info in usage_infos)
        logger.info(f"Update {i + 1}/{num_updates} done. Peak reserved: {peak_reserved:.2f} GB")

    megatron_worker.to_cpu()
    asyncio.run(rollout_worker.resume_gpu_memory(["kv_cache"]))
    asyncio.run(rollout_worker.resume_generation())

    # Generate after weight update — should produce correct answers.
    after_texts = _run_n_generations(rollout_worker, input_ids, n=5)
    before_acc = _accuracy(before_texts)
    after_acc = _accuracy(after_texts)
    logger.info(f"Accuracy: before={before_acc}, after={after_acc}, after_texts={after_texts!r}")

    rollout_worker.shutdown()
    megatron_worker.shutdown()
    resource_group.shutdown()

    assert before_acc == 0.0, f"Expected 0% accuracy before weight update, got {before_acc}"
    assert after_acc == 1.0, f"Expected 100% accuracy after weight update, got {after_acc}"
    rollout_is_fp8 = rollout_model_name != BF16_MODEL_NAME
    logger.info(f"MoE colocated weight update test passed! (rollout_fp8={rollout_is_fp8}, mcore_fp8={megatron_fp8})")
    ray_utils.stop()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU is required")
class TestMoeColocatedWeightUpdate:
    """MoE colocated weight update tests with various parallelism configs."""

    @pytest.mark.skipif(
        torch.cuda.is_available() and torch.cuda.device_count() < NUM_GPUS_REQUIRED,
        reason=f"Need at least {NUM_GPUS_REQUIRED} GPUs",
    )
    @pytest.mark.parametrize(
        (
            "topology_name",
            "megatron_dp_size",
            "megatron_tp_size",
            "megatron_ep_size",
            "megatron_cp_size",
            "megatron_pp_size",
        ),
        R3_MISMATCH_TOPOLOGY_CASES,
    )
    def test_r3_mismatch_topology_weight_update(
        self,
        *,
        topology_name: str,
        megatron_dp_size: int,
        megatron_tp_size: int,
        megatron_ep_size: int,
        megatron_cp_size: int,
        megatron_pp_size: int,
    ) -> None:
        """BF16 colocated weight update coverage for axis_recipe/moe/run_r3_mismatch_test.sh."""
        logger.info("Running R3 mismatch topology weight update case: %s", topology_name)
        _test_moe_colocated_weight_update(
            rollout_model_name=BF16_MODEL_NAME,
            megatron_model_name=BF16_MODEL_NAME,
            rollout_tp_size=NUM_GPUS_REQUIRED,
            rollout_ep_size=megatron_ep_size,
            num_workers=1,
            megatron_tp_size=megatron_tp_size,
            megatron_dp_size=megatron_dp_size,
            megatron_ep_size=megatron_ep_size,
            megatron_cp_size=megatron_cp_size,
            megatron_pp_size=megatron_pp_size,
            num_updates=1,
        )

    @pytest.mark.skipif(
        torch.cuda.is_available() and torch.cuda.device_count() < NUM_GPUS_REQUIRED,
        reason=f"Need at least {NUM_GPUS_REQUIRED} GPUs",
    )
    @pytest.mark.parametrize("use_fp8", [False, True], ids=["bf16", "fp8"])
    def test_tp4_ep4_dp4(self, *, use_fp8: bool) -> None:
        """SGLang tp=4 ep=4, Megatron tp=2 dp=4 ep=4."""
        _test_moe_colocated_weight_update(
            rollout_model_name=FP8_MODEL_NAME if use_fp8 else BF16_MODEL_NAME,
            megatron_model_name=BF16_MODEL_NAME,
            rollout_tp_size=4,
            rollout_ep_size=4,
            num_workers=2,
            megatron_tp_size=2,
            megatron_dp_size=4,
            megatron_ep_size=4,
        )

    @pytest.mark.skipif(
        torch.cuda.is_available() and torch.cuda.device_count() < NUM_GPUS_REQUIRED,
        reason=f"Need at least {NUM_GPUS_REQUIRED} GPUs",
    )
    @pytest.mark.parametrize("use_fp8", [False, True], ids=["bf16", "fp8"])
    def test_tp8_ep8(self, *, use_fp8: bool) -> None:
        """SGLang tp=8 ep=8, Megatron tp=2 cp=4 ep=8 — production config."""
        _test_moe_colocated_weight_update(
            rollout_model_name=FP8_MODEL_NAME if use_fp8 else BF16_MODEL_NAME,
            megatron_model_name=BF16_MODEL_NAME,
            rollout_tp_size=8,
            rollout_ep_size=8,
            num_workers=1,
            megatron_tp_size=2,
            megatron_dp_size=1,
            megatron_cp_size=4,
            megatron_ep_size=8,
        )

    @pytest.mark.skipif(
        torch.cuda.is_available() and torch.cuda.device_count() < NUM_GPUS_REQUIRED,
        reason=f"Need at least {NUM_GPUS_REQUIRED} GPUs",
    )
    def test_tp8_ep8_rollout_fp8_mcore_fp8(self) -> None:
        """Production config with FP8 rollout model and FP8 MCore compute."""
        _test_moe_colocated_weight_update(
            rollout_model_name=FP8_MODEL_NAME,
            megatron_model_name=BF16_MODEL_NAME,
            megatron_fp8="e4m3",
            rollout_tp_size=8,
            rollout_ep_size=8,
            num_workers=1,
            megatron_tp_size=2,
            megatron_dp_size=1,
            megatron_cp_size=4,
            megatron_ep_size=8,
        )


if __name__ == "__main__":
    # Regression for colocated MoE weight sync when rollout EP is smaller than
    # rollout TP (SGLang tp=8 ep=4). This covers a non-ep==tp topology with
    # MCore tp=2 dp=4 ep=4, where expert weight layout bugs can be masked by
    # the production tp=8 ep=8 path.
    _test_moe_colocated_weight_update(
        rollout_model_name=BF16_MODEL_NAME,
        megatron_model_name=BF16_MODEL_NAME,
        rollout_tp_size=8,
        rollout_ep_size=4,
        num_workers=1,
        megatron_tp_size=2,
        megatron_dp_size=4,
        megatron_ep_size=4,
        num_updates=1,
    )
