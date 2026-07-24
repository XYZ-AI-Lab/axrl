import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from rich.pretty import pprint
from torch import Tensor
from tqdm import tqdm

from axrl.configs import MegatronWorkerConfig, ModelConfig
from axrl.data import Sample, SampleTensorDict
from axrl.example.config_examples import CONV_EXAMPLE_PATH
from axrl.ray import ray_utils
from axrl.ray.ray_megatron_worker import RayMegatronWorker
from axrl.ray.resource_group import Request, ResourceGroup
from axrl.utils import gpu_utils, setup_logger
from axrl.utils.gpu_utils import GpuUsageInfo
from axrl.utils.kl_utils import LogprobsDiffResult, compare_logprobs
from axrl.worker.hf_worker import HFWorker
from tests.mcore.test_training_curve_consistency import get_train_eval_samples
from tests.test_configs import get_consistency_checking_configs

logger = logging.getLogger(__name__)


@dataclass
class InferenceConsistencyResult(LogprobsDiffResult):
    base_inference_seconds: float
    base_max_gpu_memory_reserved_gbs: float
    base_max_gpu_memory_allocated_gbs: float

    teset_inference_seconds: float
    test_max_gpu_memory_reserved_gbs: float
    test_max_gpu_memory_allocated_gbs: float


def filter_logprobs_with_loss_mask(hf_seq: list[float], mcore_seq: list[float], loss_mask: list[bool]) -> tuple[list[float], list[float]]:
    seq_len = len(hf_seq)
    assert seq_len == len(mcore_seq)
    assert sum(loss_mask[seq_len:]) == 0
    loss_mask = loss_mask[:seq_len]
    assert sum(loss_mask) > 0
    hf_seq_masked = [lp for lp, lm in zip(hf_seq, loss_mask, strict=True) if lm]
    mcore_seq_masked = [lp for lp, lm in zip(mcore_seq, loss_mask, strict=True) if lm]
    assert len(hf_seq_masked) == len(mcore_seq_masked)
    return hf_seq_masked, mcore_seq_masked


def create_inference_consistency_result(
    samples: list[Sample],
    hf_logprobs: Tensor,
    hf_gpu_usages: list[GpuUsageInfo],
    mcore_logprobs: Tensor,
    mcore_gpu_usages: list[GpuUsageInfo],
) -> InferenceConsistencyResult:
    """Create InferenceConsistencyResult from HF and Megatron inference outputs."""
    loss_masks = torch.tensor([sample.loss_mask for sample in samples])
    logprobs_diff_result = compare_logprobs(loss_masks, hf_logprobs, mcore_logprobs)

    hf_inference_seconds = max(usage.cpu_time_s for usage in hf_gpu_usages)
    hf_max_gpu_memory_reserved_gbs = max(usage.peak_mem_reserved_gbs for usage in hf_gpu_usages)
    hf_max_gpu_memory_allocated_gbs = max(usage.peak_mem_gbs for usage in hf_gpu_usages)

    mcore_inference_seconds = max(usage.cpu_time_s for usage in mcore_gpu_usages)
    mcore_max_gpu_memory_reserved_gbs = max(usage.peak_mem_reserved_gbs for usage in mcore_gpu_usages)
    mcore_max_gpu_memory_allocated_gbs = max(usage.peak_mem_gbs for usage in mcore_gpu_usages)

    return InferenceConsistencyResult(
        base_inference_seconds=hf_inference_seconds,
        base_max_gpu_memory_reserved_gbs=hf_max_gpu_memory_reserved_gbs,
        base_max_gpu_memory_allocated_gbs=hf_max_gpu_memory_allocated_gbs,
        teset_inference_seconds=mcore_inference_seconds,
        test_max_gpu_memory_reserved_gbs=mcore_max_gpu_memory_reserved_gbs,
        test_max_gpu_memory_allocated_gbs=mcore_max_gpu_memory_allocated_gbs,
        **logprobs_diff_result.__dict__,
    )


def compute_hf_logprobs(model_path: Path, batch_size: int, samples: list[Sample]) -> tuple[Tensor, list[GpuUsageInfo]]:
    gpu_utils.assert_all_gpus_empty()
    worker = HFWorker(model_path=model_path)
    worker.initialize()
    hf_logprobs, hf_gpu_usage = worker.compute_logprobs(SampleTensorDict.from_samples(samples), batch_size=batch_size)
    worker.shutdown()
    logger.info(f"After HF Inference GPU usage: {gpu_utils.get_gpu_memory_info()}")
    return hf_logprobs, hf_gpu_usage


def compute_mcore_logprobs(config: MegatronWorkerConfig, resource_group: ResourceGroup, samples: list[Sample]) -> tuple[Tensor, list[GpuUsageInfo]]:
    gpu_utils.assert_all_gpus_empty()
    megatron_worker = RayMegatronWorker(config=config, resource_group=resource_group)
    megatron_worker.initialize()
    mcore_logprobs, mcore_gpu_usages = megatron_worker.compute_logprobs(
        samples=SampleTensorDict.from_samples(samples),
        batch_size=config.global_batch_size,  # inside of mcore, it forward in a micro-batch manner
    )
    megatron_worker.shutdown()
    logger.info(f"After Megatron Inference GPU usage: {gpu_utils.get_gpu_memory_info()}")
    return mcore_logprobs, mcore_gpu_usages


def compare_configs(check_results: list[tuple[MegatronWorkerConfig, InferenceConsistencyResult]], save_csv_path: Path) -> None:
    def _get_config_name(config: MegatronWorkerConfig) -> str:
        return f"t{config.tp_size}_p{config.pp_size}_vp{config.vpp_size or 1}_c{config.cp_size}_d{config.dp_size}"

    name_results = [(_get_config_name(config), result) for config, result in check_results]
    names = [name for name, _ in name_results]
    results = [result for _, result in name_results]
    data = pd.DataFrame([r.__dict__ for r in results], index=names)
    for col in data.columns:
        print(f"\n====== {col} ======")
        nunique = data[col].nunique()
        if nunique == 1:
            print(f"All values are the same: {data[col].iloc[0]}")
        else:
            sorted_values = data[col].sort_values(ascending=False)
            print(sorted_values.to_string())
    data.to_csv(save_csv_path)


def test_inference_consistency() -> None:
    setup_logger("info")
    ray_utils.restart()
    model_config = ModelConfig(
        name="Qwen/Qwen3-0.6B",
        seq_length=1024 * 2,
    )
    result_path = Path("/workspaces/axrl/tests/mcore/consistency_checking_result.csv")
    _test_inference_consistency(model_config, result_path=result_path)


def _test_inference_consistency(model_config: ModelConfig, result_path: Path) -> None:
    data_path = CONV_EXAMPLE_PATH
    test_configs = get_consistency_checking_configs(model_config=model_config)
    _, eval_samples = get_train_eval_samples(
        conversation_path=data_path,
        model_config=test_configs[0].model,
    )

    hf_logprobs, hf_gpu_usages = compute_hf_logprobs(
        model_path=test_configs[0].model.get_full_path(),
        batch_size=test_configs[0].eval_micro_batch_size,
        samples=eval_samples,
    )
    hf_logprobs = hf_logprobs.cpu()
    gpu_utils.clear_cache(verbose=True)

    check_results: list[tuple[MegatronWorkerConfig, InferenceConsistencyResult]] = []
    for config in tqdm(test_configs, desc="Compute lobprobs with Megatron and Check Consistency", total=len(test_configs)):
        ray_utils.restart()
        resource_group = ResourceGroup(requests=[Request(cpu=1, gpu=config.world_size())])
        mcore_logprobs, mcore_gpu_usages = compute_mcore_logprobs(
            config=config,
            resource_group=resource_group,
            samples=eval_samples,
        )

        check_result = create_inference_consistency_result(
            samples=eval_samples,
            hf_logprobs=hf_logprobs,
            hf_gpu_usages=hf_gpu_usages,
            mcore_logprobs=mcore_logprobs,
            mcore_gpu_usages=mcore_gpu_usages,
        )
        logger.info("Inference Consistency Check Result:")
        pprint(check_result)
        check_results.append((config, check_result))
        assert check_result.cosine_similarity > 0.999, f"Cosine similarity too low: {check_result.cosine_similarity}"
        assert check_result.k2 < 0.005, f"KL Divergence k2 too high: {check_result.k2}"
        ray_utils.stop()
    if __name__ == "__main__":
        compare_configs(check_results, save_csv_path=result_path)


if __name__ == "__main__":
    test_inference_consistency()

# python -u tests/mcore/test_mcore_hf_infer_consistency.py 2>&1  | tee tmp/test-mcore-consistency.log
