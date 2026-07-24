"""Test model loading, saving, and snapshotting."""

import logging
from pathlib import Path

import psutil
import torch
from rich.pretty import pprint
from torch import Tensor

from axrl.configs import ModelConfig
from axrl.data import Sample, SampleTensorDict
from axrl.example.config_examples import CONV_EXAMPLE_PATH, get_megatron_trainer_config
from axrl.ray import ray_utils
from axrl.ray.ray_megatron_worker import RayMegatronWorker
from axrl.ray.resource_group import Request, ResourceGroup
from axrl.utils import gpu_utils, setup_logger
from axrl.utils.gpu_utils import GpuUsageInfo
from axrl.utils.kl_utils import compare_logprobs
from axrl.worker.hf_worker import HFWorker
from tests.mcore.test_training_curve_consistency import get_train_eval_samples

logger = logging.getLogger(__name__)


def compute_hf_logprobs(model_path: Path, batch_size: int, samples: list[Sample]) -> tuple[Tensor, list[GpuUsageInfo]]:
    gpu_utils.assert_all_gpus_empty()
    worker = HFWorker(model_path=model_path)
    worker.initialize()
    hf_logprobs, hf_gpu_usage = worker.compute_logprobs(SampleTensorDict.from_samples(samples), batch_size=batch_size)
    worker.shutdown()
    logger.info(f"After HF Inference GPU usage: {gpu_utils.get_gpu_memory_info()}")
    return hf_logprobs, hf_gpu_usage


def print_memory_info(tag: str) -> None:
    logger.info(f"{tag} Memory Info:")
    mem = psutil.virtual_memory()
    logger.info(f"Total: {mem.total / 1024**3:.2f} GB")
    logger.info(f"Available: {mem.available / 1024**3:.2f} GB")
    logger.info(f"Used: {mem.used / 1024**3:.2f} GB")
    logger.info(f"Percent: {mem.percent}%")
    logger.info(f"GPU memory: {gpu_utils.get_gpu_memory_info()}")


def test_model_io_and_snapshots() -> None:
    ray_utils.restart()
    # prepare two model configs:
    model1 = ModelConfig(name="Qwen/Qwen3-0.6B")
    model2 = ModelConfig(name="Qwen/Qwen3-0.6B-Base")
    config = get_megatron_trainer_config(tp_size=2, dp_size=2)
    config.model = model1

    data_path = CONV_EXAMPLE_PATH

    _, eval_samples = get_train_eval_samples(
        conversation_path=data_path,
        model_config=config.model,
    )

    hf_logprobs_1, _ = compute_hf_logprobs(
        model_path=model1.get_full_path(),
        batch_size=config.eval_micro_batch_size,
        samples=eval_samples,
    )

    hf_logprobs_2, _ = compute_hf_logprobs(
        model_path=model2.get_full_path(),
        batch_size=config.eval_micro_batch_size,
        samples=eval_samples,
    )

    loss_masks = torch.tensor([sample.loss_mask for sample in eval_samples])

    logprobs_diff_result = compare_logprobs(loss_masks, hf_logprobs_1, hf_logprobs_2)
    logger.info("HF Model 1 vs Model 2 LogprobsDiffResult:")
    pprint(logprobs_diff_result)
    assert logprobs_diff_result.cosine_similarity < 0.9

    resource_group = ResourceGroup(requests=[Request(cpu=1, gpu=4)] * 1)
    megatron_worker = RayMegatronWorker(config=config, resource_group=resource_group)
    megatron_worker.initialize()
    mcore_logprobs, _ = megatron_worker.compute_logprobs(
        samples=SampleTensorDict.from_samples(eval_samples),
        batch_size=config.global_batch_size,
    )
    diff_hf_mcore = compare_logprobs(loss_masks, hf_logprobs_1, mcore_logprobs)
    logger.info("HF Model 1 vs Megatron Model 1 LogprobsDiffResult:")
    pprint(diff_hf_mcore)
    assert diff_hf_mcore.cosine_similarity > 0.999

    print_memory_info("Before snapshot")
    megatron_worker.copy_weights_to_cpu(name="model1")
    print_memory_info("After snapshot")

    megatron_worker.load_hf_weights(hf_model_dir=model2.get_full_path())
    mcore_logprobs_2, _ = megatron_worker.compute_logprobs(
        samples=SampleTensorDict.from_samples(eval_samples),
        batch_size=config.global_batch_size,
    )
    diff_hf_mcore_2 = compare_logprobs(loss_masks, hf_logprobs_2, mcore_logprobs_2)
    logger.info("HF Model 2 vs Megatron Model 2 LogprobsDiffResult:")
    pprint(diff_hf_mcore_2)
    assert diff_hf_mcore_2.cosine_similarity > 0.999

    megatron_worker.apply_weights_from_cpu(name="model1")
    mcore_logprobs_1_again, _ = megatron_worker.compute_logprobs(
        samples=SampleTensorDict.from_samples(eval_samples),
        batch_size=config.global_batch_size,
    )
    diff_hf_mcore_1_again = compare_logprobs(loss_masks, hf_logprobs_1, mcore_logprobs_1_again)
    logger.info("HF Model 1 vs Megatron Model 1 Again LogprobsDiffResult:")
    pprint(diff_hf_mcore_1_again)
    assert diff_hf_mcore_1_again.cosine_similarity > 0.99

    megatron_worker.shutdown()
    ray_utils.stop()


if __name__ == "__main__":
    from axrl.utils import setup_logger

    setup_logger("info")
    test_model_io_and_snapshots()
