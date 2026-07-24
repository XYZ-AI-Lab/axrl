from pathlib import Path

from axrl.configs import (
    DatasetConfig,
    GrpoTrainerConfig,
    MCoreLrSchedulerConfig,
    MCoreOptimizerConfig,
    MegatronWorkerConfig,
    MetricLoggerConfig,
    ModelConfig,
    OnlineRLTrainConfig,
    PPOValueConfig,
    RolloutWorkerConfig,
    SamplingConfig,
)
from axrl.pipeline.config import ControllerConfig, PipelineExperimentConfig
from axrl.utils import config_utils


def _prepare_default_configs(
    model_name: str,
    max_length: int,
    *,
    colocated: bool = True,
) -> PipelineExperimentConfig:
    model_config = ModelConfig(name=model_name, seq_length=max_length, trust_remote_code=True)
    optimizer_config = MCoreOptimizerConfig(
        optimizer="adam",
        lr=1e-6,
        min_lr=1e-6,
        use_distributed_optimizer=True,
        clip_grad=1.0,
        bf16=True,
    )
    lr_scheduler_config = MCoreLrSchedulerConfig(
        lr_decay_style="constant",
        init_lr=1e-6,
        max_lr=1e-6,
        lr_warmup_steps=0,
    )
    megatron_worker_config = MegatronWorkerConfig(
        model=model_config,
        model_role="actor",
        optimizer=optimizer_config,
        lr_scheduler=lr_scheduler_config,
        tp_size=2,
        cp_size=1,
        pp_size=2,
        vpp_size=None,
        log_every_k_steps=1,
        checkpoint_dir="ppo_gsm8k/checkpoints/megatron_actor",
        global_batch_size=512,
        train_micro_batch_size=2,
        eval_micro_batch_size=2,
        log_gpu_usaegs=True,
        bf16=True,
        fp16=False,
    )
    value_worker_config = megatron_worker_config.model_copy(deep=True)
    value_worker_config.model_role = "value"
    value_worker_config.checkpoint_dir = "ppo_gsm8k/checkpoints/megatron_value"

    sampling_config = SamplingConfig(temperature=1.0, max_total_tokens=max_length, top_p=1.0)
    rollout_worker_config = RolloutWorkerConfig(
        model=ModelConfig(name=model_name, seq_length=max_length, trust_remote_code=True),
        sampling_config=sampling_config,
        dp_size=1,
        num_workers=2,
        tp_size=2,
        gpu_memory_utilization=0.7,
        max_running_requests=128,
    )

    logger_config = MetricLoggerConfig(
        logger_type="tensorboard",
        name="main-process",
        group_name="ppo_gsm8k",
        project_name="PPO-GSM8K-2026-07-20",
    )
    eval_sampling_config = SamplingConfig(
        temperature=0.0,
        max_total_tokens=max_length,
    )

    grpo = GrpoTrainerConfig(
        loss_type="ppo",
        clip_ratio_high=0.2,
        clip_ratio_low=0.2,
        dual_clip_neg_adv_factor=None,
        dual_soft_clip=None,
        normalize_advantages_over_valid_tokens_in_batch=False,
        ppo_value=PPOValueConfig(
            gamma=1.0,
            gae_lambda=1.0,
            value_clip=0.2,
            value_loss_coef=1.0,
            num_value_only_updates=0,
        ),
        is_base_logprobs="old_logprobs",
        kl_base_logprobs="old_logprobs",
        kl_control_alpha=0.0,
    )
    grpo.entropy_control.compute_entropy = True
    online_rl_train = OnlineRLTrainConfig(
        batch_rollout_for_n_global_updates=4,
        model_sync_every_n_global_updates=4,
        num_rollouts_per_conversation=8,
        eval_every_n_global_updates=16,
        max_global_updates=128,
        checkpoint_every_n_global_updates=128,
    )

    return PipelineExperimentConfig(
        controller=ControllerConfig(
            run_mode="online_rl_train",
            output_dir_name="ppo_gsm8k",
            colocated=colocated,
            num_rollout_actors=8,
            num_cpus_per_actor=2,
            max_running_requests=rollout_worker_config.max_running_requests,
        ),
        megatron_worker=megatron_worker_config,
        value_worker=value_worker_config,
        rollout_worker=rollout_worker_config,
        train_datasets=[DatasetConfig(name="openai/gsm8k/train")],
        test_datasets=[DatasetConfig(name="openai/gsm8k/test", eval_num_rollouts_per_prompt=1)],
        grpo=grpo,
        logger=logger_config,
        eval_sampling_config=eval_sampling_config,
        train_sampling_config=sampling_config,
        online_rl_train=online_rl_train,
    )


def _create_default_config() -> None:
    config_path = Path("axis_recipe/ppo_gsm8k/pipeline_config.yaml")
    configs = _prepare_default_configs(
        model_name="Qwen/Qwen2.5-1.5B-Instruct",
        max_length=1024 * 2,
    )
    config_utils.save_to_yaml(configs, config_path)
    print(f"Default pipeline config created at: {config_path}")
    loaded_configs = config_utils.load_and_validate_config(
        PipelineExperimentConfig,
        str(config_path),
        load_env_config=False,
    )
    assert loaded_configs == configs, "Loaded config does not match the saved config."


if __name__ == "__main__":
    _create_default_config()
