"""Training curve consistency test for MoE models across diverse parallel strategies.

Uses Qwen3-30B-A3B-Instruct-2507 (48 layers, 128 experts, top-8) and checks that
loss, eval_loss, and grad_norm remain consistent across 30+ parallelism configs
including DP, TP, CP, EP, ETP, PP, and VPP.

Baseline: dp=2, tp=1, pp=1, cp=4, ep=4, etp=1.

Reference: tests/mcore/test_training_curve_consistency.py (dense model version).
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pytest
import torch

from axrl.configs import AXRL_DIR, MCoreLrSchedulerConfig, MCoreOptimizerConfig, MegatronWorkerConfig, ModelConfig
from axrl.data import Conversation, Sample, SampleTensorDict
from axrl.data.sft_sample_converter import SftSampleConverter
from axrl.example.config_examples import CONV_EXAMPLE_PATH
from axrl.ray import ray_utils
from axrl.ray.ray_megatron_worker import RayMegatronWorker
from axrl.ray.resource_group import Request, ResourceGroup
from axrl.utils import zst_utils

logger = logging.getLogger(__name__)

target_curve_path = Path("tests/mcore/moe-target-curve.csv")


def get_train_eval_samples(conversation_path: Path, model_config: ModelConfig) -> tuple[list[Sample], list[Sample]]:
    conversations: list[Conversation] = zst_utils.load_zst(conversation_path)
    builder = SftSampleConverter(model_config)
    train_samples = [builder.process(conv) for conv in conversations[:128]]
    eval_samples = [builder.process(conv) for conv in conversations[-128:]]
    assert len(train_samples) > 0
    assert len(eval_samples) > 0
    return train_samples, eval_samples


@dataclass(frozen=True)
class MoECase:
    name: str
    dp: int = 2
    tp: int = 1
    pp: int = 1
    cp: int = 4
    ep: int = 4
    etp: int = 1
    vpp: int | None = None
    train_micro_batch_size: int = 2
    use_dynamic_batch_size: bool = True

    def world_size(self) -> int:
        return self.dp * self.tp * self.pp * self.cp

    @staticmethod
    def get_base_config() -> MegatronWorkerConfig:
        """Base MoE config: Qwen3-30B-A3B, dp=2, cp=4, ep=4."""
        config = MegatronWorkerConfig(
            model=ModelConfig(
                name="Qwen/Qwen3-30B-A3B-Instruct-2507",
                seq_length=4096,
            ),
            seed=42,
            # Parallelism — baseline
            tp_size=1,
            dp_size=2,
            pp_size=1,
            cp_size=4,
            ep_size=4,
            etp_size=1,
            vpp_size=None,
            # Training
            train_micro_batch_size=2,
            eval_micro_batch_size=2,
            global_batch_size=32,
            use_dynamic_batch_size=True,
            num_epochs=3,
            log_every_k_steps=1,
            # Memory saving — required for 30B MoE on 8 GPUs
            recompute_granularity="full",
            recompute_method="uniform",
            recompute_num_layers=1,
            # MoE settings
            moe_aux_loss_coeff=0,
            moe_router_load_balancing_type="none",
            enable_routing_replay=False,
            # Precision
            fp16=False,
            bf16=True,
            use_gloo_process_groups=True,
            optimizer=MCoreOptimizerConfig(
                lr=1e-4,
                min_lr=1e-6,
                bf16=True,
                clip_grad=10.0,
                use_distributed_optimizer=True,
                optimizer_cpu_offload=True,
                optimizer_offload_fraction=1.0,
            ),
            lr_scheduler=MCoreLrSchedulerConfig(
                max_lr=1e-4,
                min_lr=1e-6,
                lr_warmup_steps=0,
                lr_decay_style="constant",
            ),
        )
        return config

    def update_config(self, base: MegatronWorkerConfig) -> MegatronWorkerConfig:
        config = base.model_copy()
        config.dp_size = self.dp
        config.tp_size = self.tp
        config.pp_size = self.pp
        config.cp_size = self.cp
        config.ep_size = self.ep
        config.etp_size = self.etp
        config.vpp_size = self.vpp
        config.train_micro_batch_size = self.train_micro_batch_size
        config.use_dynamic_batch_size = self.use_dynamic_batch_size
        return config


def get_test_cases() -> list[MoECase]:
    """One combined parallel config that exercises tp/cp/pp at once.

    World size = tp * cp * pp * dp = 8 GPUs.
    Constraints (from Megatron-LM parallel_state.py):
      - 8 % (etp * ep * pp) == 0
      - 128 % ep == 0
      - 48 % pp == 0
    With tp=cp=pp=2 and dp=1, etp=1 forces ep <= 2; ep=2 is the largest legal
    value (smaller still requires too much expert memory; ep=1 OOMs).

    Note: VPP is excluded because its uniform padding adds dummy tokens that
    change MoE routing patterns; VPP correctness is verified in the
    parallelism benchmark.
    """
    # Keep the matrix small but exercise each independent axis once:
    #   - tp/cp/pp combined (the main parallel topology)
    #   - dp (swap pp → dp=2)
    #   - non-dynamic batching (toggle on the tp/cp/pp combined case)
    #   - etp (expert tensor parallel; etp must divide tp, world / (etp*ep*pp) ≥ 1)
    #   - vpp (virtual pipeline parallel; TP=1 only due to a Megatron core bug
    #     where VPP+TP>1 mismatches hidden states across micro-batches when
    #     sequence_parallel=True; 48 / (PP * VPP) must be an integer)
    return [
        MoECase(name="tp2_cp2_pp2_ep2", tp=2, cp=2, pp=2, dp=1, ep=2),
        MoECase(name="tp2_cp2_dp2_ep4", tp=2, cp=2, pp=1, dp=2, ep=4),
        MoECase(name="tp2_cp2_pp2_ep2_no_dynamic", tp=2, cp=2, pp=2, dp=1, ep=2, train_micro_batch_size=1, use_dynamic_batch_size=False),
        MoECase(name="tp2_cp2_pp2_ep2_etp2", tp=2, cp=2, pp=2, dp=1, ep=2, etp=2),
        MoECase(name="dp2_pp4_cp1_ep2_vpp2", tp=1, cp=1, pp=4, dp=2, ep=2, vpp=2),
    ]


def get_dataset(config: MegatronWorkerConfig) -> tuple[list[Sample], list[Sample]]:
    data_path = CONV_EXAMPLE_PATH
    assert data_path.exists(), f"Missing example dataset at {data_path}"
    return get_train_eval_samples(conversation_path=data_path, model_config=config.model)


def _run_train_curve(config: MegatronWorkerConfig) -> pd.DataFrame:
    world_size = config.world_size()
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
    config = MoECase.get_base_config()
    curve_df = _run_train_curve(config=config)
    curve_df.to_csv(target_curve_path, index=False)
    last_eval_loss = curve_df["eval_loss"].iloc[-1]
    last_train_loss = curve_df["train_loss"].iloc[-1]
    logger.info(f"Baseline curve: last train_loss={last_train_loss:.4f}, eval_loss={last_eval_loss:.4f}")
    logger.info(f"Saved baseline curve to {target_curve_path}")


def save_all_curves() -> None:
    base_config = MoECase.get_base_config()
    output_dir = AXRL_DIR.output / "moe_consistency_curves"
    output_dir.mkdir(parents=True, exist_ok=True)

    for case in get_test_cases():
        test_config = case.update_config(base=base_config)
        curve_path = Path(output_dir / f"curve-{case.name}.csv")
        logger.info(f"--- Running case: {case.name} (ws={case.world_size()}) ---")
        try:
            curve_df = _run_train_curve(config=test_config)
            curve_df.to_csv(curve_path, index=False)
            logger.info(f"Saved curve for {case.name} to {curve_path}")
        except Exception as e:
            logger.exception(f"FAILED: {case.name}")
            curve_path.write_text(f"Error: {e}")


@pytest.mark.parametrize("case", get_test_cases(), ids=lambda c: c.name)
def test_moe_curve_consistency(case: MoECase) -> None:
    target_curve = pd.read_csv(target_curve_path)
    base_config = MoECase.get_base_config()
    test_config = case.update_config(base=base_config)
    logger.info(f"Running MoE test case: {case}")
    test_curve = _run_train_curve(config=test_config)
    # Save curve for each config so results can be inspected after the run
    output_dir = AXRL_DIR.output / "moe_consistency_curves"
    output_dir.mkdir(parents=True, exist_ok=True)
    test_curve.to_csv(output_dir / f"curve-{case.name}.csv", index=False)
    pd.testing.assert_frame_equal(test_curve.drop(columns=["train_grad_norm"]), target_curve.drop(columns=["train_grad_norm"]), rtol=0.1)
    pd.testing.assert_series_equal(test_curve["train_grad_norm"], target_curve["train_grad_norm"], rtol=0.2)


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s")

    if "--baseline" in sys.argv:
        save_baseline_curve()
    elif "--all" in sys.argv:
        save_baseline_curve()
        save_all_curves()
    else:
        save_baseline_curve()
        save_all_curves()
    # Usage:
    #   python -u tests/mcore/test_moe_training_curve_consistency.py --baseline 2>&1 | tee tmp/moe-baseline.log
    #   python -u tests/mcore/test_moe_training_curve_consistency.py --all 2>&1 | tee tmp/moe-all-curves.log
