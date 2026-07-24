import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pytest
import torch

from axrl.configs import MegatronWorkerConfig, ModelConfig
from axrl.data import Conversation, Sample, SampleTensorDict
from axrl.data.sft_sample_converter import SftSampleConverter
from axrl.example.config_examples import CONV_EXAMPLE_PATH, get_megatron_trainer_config
from axrl.ray import ray_utils
from axrl.ray.ray_megatron_worker import RayMegatronWorker
from axrl.ray.resource_group import Request, ResourceGroup
from axrl.utils import zst_utils

logger = logging.getLogger(__name__)


target_curve_path = Path("tests/mcore/target-curve.csv")


def get_train_eval_samples(conversation_path: Path, model_config: ModelConfig) -> tuple[list[Sample], list[Sample]]:
    conversations: list[Conversation] = zst_utils.load_zst(conversation_path)
    builder = SftSampleConverter(model_config)
    train_samples = [builder.process(conv) for conv in conversations[:128]]
    eval_samples = [builder.process(conv) for conv in conversations[-128:]]
    assert len(train_samples) > 0
    assert len(eval_samples) > 0
    return train_samples, eval_samples


@dataclass(frozen=True)
class Case:
    name: str
    tp: int = 1
    dp: int = 1
    pp: int = 1
    cp: int = 1
    train_micro_batch_size: int = 4
    use_dynamic_batch_size: bool = True

    def world_size(self) -> int:
        return self.dp * self.tp * self.pp * self.cp

    @staticmethod
    def get_base_config() -> MegatronWorkerConfig:
        config = get_megatron_trainer_config(
            tp_size=1,
            dp_size=1,
            pp_size=1,
            cp_size=1,
        ).model_copy()

        config.global_batch_size = 32
        config.train_micro_batch_size = 4
        config.eval_micro_batch_size = 8
        config.log_every_k_steps = 1
        config.use_dynamic_batch_size = True
        config.num_epochs = 6
        return config

    def update_config(self, base: MegatronWorkerConfig) -> MegatronWorkerConfig:
        config = base.model_copy()
        config.tp_size = self.tp
        config.dp_size = self.dp
        config.pp_size = self.pp
        config.cp_size = self.cp
        config.train_micro_batch_size = self.train_micro_batch_size
        config.use_dynamic_batch_size = self.use_dynamic_batch_size
        return config


def get_test_cases() -> list[Case]:
    # One combined parallel config exercises tp/cp/pp at once instead of
    # running a full matrix; tp * cp * pp = 8 ranks. Two micro-batch variants
    # cover dynamic vs non-dynamic batching on top of that single topology.
    return [
        Case(name="tp2_cp2_pp2_micro1", tp=2, cp=2, pp=2, train_micro_batch_size=1),
        Case(name="tp2_cp2_pp2_micro1_no_dynamic_batching", tp=2, cp=2, pp=2, train_micro_batch_size=1, use_dynamic_batch_size=False),
    ]


def get_dataset(config: MegatronWorkerConfig) -> tuple[list[Sample], list[Sample]]:
    data_path = CONV_EXAMPLE_PATH
    assert data_path.exists(), f"Missing example dataset at {data_path}"
    train_samples, eval_samples = get_train_eval_samples(
        conversation_path=data_path,
        model_config=config.model,
    )
    return train_samples, eval_samples


def _required_gpus(config: MegatronWorkerConfig) -> int:
    return config.world_size()


def _run_train_curve(config: MegatronWorkerConfig) -> pd.DataFrame:
    world_size = _required_gpus(config)
    if world_size > torch.cuda.device_count():
        pytest.skip(f"Requires {world_size} GPUs, found {torch.cuda.device_count()}")
    train_samples, eval_samples = get_dataset(config)
    ray_utils.restart()

    resource_group = ResourceGroup([Request(cpu=1, gpu=world_size)])
    worker = RayMegatronWorker(config=config, resource_group=resource_group)

    curve: list[dict] = []
    global_step = 0
    worker.initialize()
    # One trajectory per training sample so global_batch_size is interpreted as trajectory count.
    for trajectory_id, sample in enumerate(train_samples):
        sample.trajectory_id = trajectory_id
    for trajectory_id, sample in enumerate(eval_samples):
        sample.trajectory_id = trajectory_id
    train_tensor_dict = SampleTensorDict.from_samples(train_samples)
    eval_dataset = SampleTensorDict.from_samples(eval_samples)
    for _ in range(config.num_epochs):
        global_step, train_metrics = worker.train(
            samples=train_tensor_dict,
            global_step=global_step,
            data_shuffle_seed=global_step,
            compute_logprobs=False,
        )
        eval_metrics = worker.eval(samples=eval_dataset, global_step=global_step)
        train_loss = train_metrics["actor_train/loss"]
        eval_loss = eval_metrics["eval/loss"]
        train_grad_norm = train_metrics["actor_train/grad_norm"]
        curve.append(
            {
                "global_step": global_step,
                "train_loss": train_loss,
                "eval_loss": eval_loss,
                "train_grad_norm": train_grad_norm,
            }
        )
    worker.shutdown()
    ray_utils.stop()
    return pd.DataFrame(curve)


def save_baseline_curve() -> None:
    config = Case.get_base_config()
    curve_df = _run_train_curve(config=config)
    curve_df.to_csv(target_curve_path, index=False)
    last_eval_loss = curve_df["eval_loss"].iloc[-1]
    last_train_loss = curve_df["train_loss"].iloc[-1]
    assert last_eval_loss > 1.0, f"Unexpected last eval loss: {last_eval_loss}"
    assert last_train_loss < 0.2, f"Unexpected last train loss: {last_train_loss}"
    logger.info(f"Saved baseline curve to {target_curve_path}")


def save_all_curves() -> None:
    base_config = Case.get_base_config()
    output_dir = Path("tmp/test-mcores")
    output_dir.mkdir(parents=True, exist_ok=True)

    for case in get_test_cases():
        test_config = case.update_config(base=base_config)
        curve_path = Path(output_dir / f"curve-{case.name}.csv")
        try:
            curve_df = _run_train_curve(config=test_config)
            curve_df.to_csv(curve_path, index=False)
            logger.info(f"Saved curve for case {case.name} to {curve_path}")
        except Exception as e:
            curve_path.write_text(f"Error: {e}")


@pytest.mark.parametrize("case", get_test_cases(), ids=lambda c: c.name)
def test_megatron_curve_consistency(case: Case) -> None:
    target_curve = pd.read_csv(target_curve_path)
    base_config = Case.get_base_config()
    test_config = case.update_config(base=base_config)
    logger.info(f"Running test case: {case}")
    test_curve = _run_train_curve(config=test_config)
    pd.testing.assert_frame_equal(test_curve, target_curve, rtol=0.1)


if __name__ == "__main__":
    save_baseline_curve()
    save_all_curves()
    # python -u tests/mcore/test_training_curve_consistency.py 2>&1 | tee tmp/test-curve-consistency.log
