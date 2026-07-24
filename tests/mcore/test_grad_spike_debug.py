"""Test gradient spike detection, snapshotting, and reproduction.

Strategy:
  1. Train for a few epochs on normal SFT data (warmup).
  2. Inject a "poison" batch where labels are random tokens — this produces
     very high cross-entropy loss → large gradients → spike.
  3. Verify spike is detected, snapshot is saved to disk.
  4. Call reproduce_spike() on the saved snapshot and verify it reproduces
     the same grad norm.

Run:
    python -u tests/mcore/test_grad_spike_debug.py 2>&1 | tee tmp/test-grad-spike-debug.log
"""

import json
import logging
import random
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch
from transformers import AutoTokenizer

from axrl.configs import MegatronWorkerConfig, ModelConfig
from axrl.data import Sample, SampleTensorDict
from axrl.data.sft_sample_converter import SftSampleConverter
from axrl.example.config_examples import CONV_EXAMPLE_PATH, get_megatron_trainer_config
from axrl.ray import ray_utils
from axrl.ray.ray_megatron_worker import RayMegatronWorker
from axrl.ray.resource_group import Request, ResourceGroup
from axrl.utils import zst_utils

logger = logging.getLogger(__name__)


def get_config() -> MegatronWorkerConfig:
    config = get_megatron_trainer_config(tp_size=1, dp_size=1, pp_size=1, cp_size=1).model_copy()
    config.global_batch_size = 32
    config.train_micro_batch_size = 4
    config.eval_micro_batch_size = 8
    config.log_every_k_steps = 1
    config.use_dynamic_batch_size = True
    config.num_epochs = 1  # we control epochs manually

    # Enable spike detection with low thresholds for testing
    config.spike_debug.enabled = True
    config.spike_debug.warmup_steps = 3  # small warmup so we detect spikes quickly
    config.spike_debug.spike_ratio = 3.0  # 3x median should catch the poison batch
    config.spike_debug.history_window = 50
    config.spike_debug.max_snapshots = 3
    config.spike_debug.save_per_param_grads = True
    return config


def get_train_samples(config: MegatronWorkerConfig) -> list[Sample]:
    """Load normal SFT training samples."""
    conversations = zst_utils.load_zst(CONV_EXAMPLE_PATH)
    builder = SftSampleConverter(config.model)
    samples = [builder.process(conv) for conv in conversations[:128]]
    assert len(samples) > 0
    return samples


def create_poison_samples(normal_samples: list[Sample], model_config: ModelConfig, num_poison: int = 32) -> list[Sample]:
    """Create poison samples with random labels to trigger a gradient spike.

    Takes normal samples and replaces their labels with random token IDs.
    The model's predictions will be far from these random targets, producing
    very high cross-entropy loss and large gradients.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_config.get_full_path(), trust_remote_code=True)
    vocab_size = tokenizer.vocab_size
    rng = random.Random(42)
    poison_samples: list[Sample] = []
    for i in range(num_poison):
        base = normal_samples[i % len(normal_samples)]
        seq_len = len(base.input_ids)

        # Replace labels with random token IDs where loss_mask is True.
        # Keep IGNORE_INDEX (-100) where loss_mask is False.
        poison_labels = base.labels.copy()
        for j in range(seq_len):
            if base.loss_mask[j]:
                poison_labels[j] = rng.randint(0, vocab_size - 1)

        poison_samples.append(
            Sample(
                input_ids=base.input_ids,
                labels=poison_labels,
                loss_mask=base.loss_mask,
                attention_mask=base.attention_mask,
                position_ids=base.position_ids,
                reward=0.0,
                reward_baseline=0.0,
                advantage=np.zeros(len(base.input_ids), dtype=np.float32),
            )
        )
    return poison_samples


def find_spike_snapshot_dir(config: MegatronWorkerConfig) -> Path | None:
    """Find the most recent spike snapshot directory."""
    spike_dir = config.get_checkpoint_dir().parent / "spike_snapshots"
    if not spike_dir.exists():
        return None
    iter_dirs = sorted(spike_dir.glob("iter_*"))
    if not iter_dirs:
        return None
    return iter_dirs[-1]


def run_spike_test() -> None:
    config = get_config()
    world_size = config.world_size()
    num_gpus = torch.cuda.device_count()
    assert num_gpus >= world_size, f"Need {world_size} GPUs, found {num_gpus}"

    # Prepare data
    normal_samples = get_train_samples(config)
    poison_samples = create_poison_samples(normal_samples, config.model, num_poison=32)
    # Assign a unique trajectory_id per sample so the trajectory-grouped iterator
    # treats each as its own trajectory (gradient-update count = num_samples / global_batch_size).
    for trajectory_id, sample in enumerate(normal_samples):
        sample.trajectory_id = trajectory_id
    for trajectory_id, sample in enumerate(poison_samples):
        sample.trajectory_id = trajectory_id
    normal_tensor_dict = SampleTensorDict.from_samples(normal_samples)
    poison_tensor_dict = SampleTensorDict.from_samples(poison_samples)

    # Clean up any previous spike snapshots
    spike_dir = config.get_checkpoint_dir().parent / "spike_snapshots"
    if spike_dir.exists():
        shutil.rmtree(spike_dir)

    ray_utils.restart()
    resource_group = ResourceGroup([Request(cpu=1, gpu=world_size)])
    worker = RayMegatronWorker(config=config, resource_group=resource_group)
    worker.initialize()

    # === Phase 1: Warmup with normal data ===
    logger.info("=== Phase 1: Warmup with normal data ===")
    global_step = 0
    num_warmup_epochs = 3  # enough to get past warmup_steps=3
    for epoch in range(num_warmup_epochs):
        global_step, train_metrics = worker.train(
            samples=normal_tensor_dict,
            global_step=global_step,
            data_shuffle_seed=epoch,
            compute_logprobs=False,
        )
        logger.info(
            f"Warmup epoch {epoch}: step={global_step}, "
            f"loss={train_metrics.get('actor_train/loss', 'N/A')}, "
            f"grad_norm={train_metrics.get('actor_train/grad_norm', 'N/A')}"
        )

    # Verify no spikes during warmup
    warmup_snapshot = find_spike_snapshot_dir(config)
    assert warmup_snapshot is None, f"Unexpected spike during warmup: {warmup_snapshot}"
    logger.info(f"Warmup complete. No spikes detected. global_step={global_step}")

    # === Phase 2: Inject poison data to trigger spike ===
    logger.info("=== Phase 2: Inject poison data ===")
    global_step, poison_metrics = worker.train(
        samples=poison_tensor_dict,
        global_step=global_step,
        data_shuffle_seed=999,
        compute_logprobs=False,
    )
    logger.info(
        "Poison epoch: step=%s, loss=%s, grad_norm=%s",
        global_step,
        poison_metrics.get("actor_train/loss", "N/A"),
        poison_metrics.get("actor_train/grad_norm", "N/A"),
    )

    # === Phase 3: Verify spike was detected and snapshot saved ===
    logger.info("=== Phase 3: Verify spike snapshot ===")
    snapshot_dir = find_spike_snapshot_dir(config)
    assert snapshot_dir is not None, "No spike snapshot found — spike was not detected!"
    logger.info(f"Spike snapshot found at: {snapshot_dir}")

    # Verify snapshot contents
    assert (snapshot_dir / "metadata.json").exists(), "Megatron checkpoint metadata.json missing"
    assert (snapshot_dir / "axrl_metadata.json").exists(), "AXRL spike axrl_metadata.json missing"
    assert any(snapshot_dir.glob("batch_rank*.pt")), "batch files missing"
    if config.spike_debug.save_per_param_grads:
        assert any(snapshot_dir.glob("grad_info_rank*.pt")), "grad_info files missing"

    # Read metadata
    with (snapshot_dir / "axrl_metadata.json").open() as f:
        metadata = json.load(f)
    logger.info(
        f"Spike metadata: global_step={metadata['global_step']}, "
        f"grad_norm={metadata['grad_norm']:.4f}, "
        f"median={metadata['grad_norm_median']:.4f}, "
        f"ratio={metadata['spike_ratio_actual']:.2f}x"
    )
    assert metadata["spike_ratio_actual"] >= config.spike_debug.spike_ratio, (
        f"Spike ratio {metadata['spike_ratio_actual']:.2f} < threshold {config.spike_debug.spike_ratio}"
    )

    # === Phase 4: Reproduce the spike ===
    logger.info("=== Phase 4: Reproduce spike ===")
    result = worker.reproduce_spike(snapshot_dir)
    logger.info(
        f"Reproduction result: reproduced={result['reproduced']}, "
        f"original_grad_norm={result['original_grad_norm']:.6f}, "
        f"replayed_grad_norm={result['replayed_grad_norm']:.6f}, "
        f"relative_diff={result['relative_diff']:.6f}"
    )

    # Without full deterministic mode (CUBLAS_WORKSPACE_CONFIG, etc.), CUDA
    # operations have minor non-determinism. Since we detect spikes at 5x+
    # the median, 20% tolerance is sufficient to confirm the same spike.
    assert result["relative_diff"] < 0.20, (
        f"Failed to reproduce spike within 20% tolerance! "
        f"original={result['original_grad_norm']:.6f}, "
        f"replayed={result['replayed_grad_norm']:.6f}, "
        f"relative_diff={result['relative_diff']:.6f}"
    )
    logger.info(f"Spike reproduced within tolerance: relative_diff={result['relative_diff']:.4f}")

    worker.shutdown()
    ray_utils.stop()
    logger.info("=== All spike debug tests PASSED ===")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU is required")
def test_grad_spike_debug() -> None:
    run_spike_test()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    run_spike_test()
    # python -u tests/mcore/test_grad_spike_debug.py 2>&1 | tee tmp/test-grad-spike-debug.log
