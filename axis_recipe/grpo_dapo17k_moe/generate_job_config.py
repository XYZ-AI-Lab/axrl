from pathlib import Path

from axrl.configs import (
    DatasetConfig,
    GradSpikeDebugConfig,
    GrpoTrainerConfig,
    MCoreLrSchedulerConfig,
    MCoreOptimizerConfig,
    MegatronWorkerConfig,
    MetricLoggerConfig,
    MismatchTestConfig,
    ModelConfig,
    OnlineRLTrainConfig,
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
    model_config = ModelConfig(
        name=model_name,
        seq_length=max_length,
        trust_remote_code=True,
    )
    optimizer_config = MCoreOptimizerConfig(
        optimizer="adam",
        lr=1e-6,
        min_lr=1e-7,
        use_distributed_optimizer=True,
        clip_grad=1.0,
        bf16=True,
        weight_decay=0.1,
        optimizer_cpu_offload=True,
        optimizer_offload_fraction=1.0,
    )
    lr_scheduler_config = MCoreLrSchedulerConfig(
        lr_decay_style="constant",
        init_lr=1e-7,
        max_lr=1e-6,
        lr_warmup_steps=40,
        start_wd=0.1,
        end_wd=0.1,
    )
    megatron_worker_config = MegatronWorkerConfig(
        model=model_config,
        optimizer=optimizer_config,
        lr_scheduler=lr_scheduler_config,
        tp_size=8,
        cp_size=1,
        pp_size=1,
        ep_size=8,
        vpp_size=None,
        log_every_k_steps=1,
        global_batch_size=512,
        train_micro_batch_size=2,
        eval_micro_batch_size=4,
        log_gpu_usaegs=True,
        bf16=True,
        fp16=False,
        apply_rope_fusion=True,
        enable_fp32_lm_head=False,
        shuffle_train_data=False,
        # memory saving
        recompute_granularity="full",
        recompute_method="uniform",
        recompute_num_layers=1,
        spike_debug=GradSpikeDebugConfig(spike_ratio=5.0),
    )

    sampling_config = SamplingConfig(temperature=1.0, max_total_tokens=max_length, top_p=1.0)
    rollout_worker_config = RolloutWorkerConfig(
        model=ModelConfig(name=model_name, seq_length=max_length, trust_remote_code=True),
        sampling_config=sampling_config,
        dp_size=1,
        num_workers=2,
        tp_size=4,
        ep_size=4,
        pp_size=1,
        gpu_memory_utilization=0.7,
        # Note: increase `max_running_requests` will increase efficiency and off-policy ratio.
        max_running_requests=1024 * 2,
        max_num_batched_tokens=2 * max_length,
        attention_backend=None,
        dtype="auto",
        enable_fp32_lm_head=True,  # to try: False
    )

    logger_config = MetricLoggerConfig(
        logger_type="tensorboard",
        name="main-process",
        group_name="GRPO-DAPO17k-Experiment",
        project_name="dapo17K-moe",
    )

    mismatch_test_config = MismatchTestConfig(
        enabled=False,
    )

    grpo = GrpoTrainerConfig(
        clip_ratio_high=0.28,
        clip_ratio_low=0.2,
        loss_type="grpo",  # to try: grpo2, is_type: rollout_logprob, seq-level clipping.
        loss_agg_type="token-mean",
        micro_batch_denominator_type="sequence",
        dual_clip_neg_adv_factor=10,  # to try: 3
        dual_soft_clip=3,
        is_base_logprobs="old_logprobs",
        kl_control_alpha=0,
        kl_base_logprobs="old_logprobs",
    )
    online_rl_train = OnlineRLTrainConfig(
        eval_on_start=True,
        num_rollouts_per_conversation=8,
        model_sync_every_n_global_updates=4,
        batch_rollout_for_n_global_updates=4,
        eval_every_n_global_updates=64,
        checkpoint_every_n_global_updates=1024,
        max_global_updates=30000,
        filter_zero_std=False,
        reward_mean_type="group",
        reward_std_type="group",
        sample_type="uniform",
        rollout_save_filename="dapo17k-moe",
    )

    eval_sampling_config = SamplingConfig(
        temperature=0.7,
        top_p=0.9,
        max_total_tokens=max_length,
    )

    configs = PipelineExperimentConfig(
        controller=ControllerConfig(
            run_mode="online_rl_train",
            output_dir_name="grpo-dapo17k-moe",
            colocated=colocated,
            num_rollout_actors=16,
            num_cpus_per_actor=4,
            max_running_requests=rollout_worker_config.max_running_requests,
        ),
        megatron_worker=megatron_worker_config,
        rollout_worker=rollout_worker_config,
        train_datasets=[DatasetConfig(name="BytedTsinghua-SIA/DAPO-Math-17k")],
        test_datasets=[DatasetConfig(name="BytedTsinghua-SIA/AIME-2024", eval_num_rollouts_per_prompt=64)],
        logger=logger_config,
        grpo=grpo,
        online_rl_train=online_rl_train,
        eval_sampling_config=eval_sampling_config,
        train_sampling_config=sampling_config,
        mismatch_test=mismatch_test_config,
    )

    return configs


def _create_default_config() -> None:
    config_path = Path("axis_recipe/grpo_dapo17k_moe/pipeline_config.yaml")
    configs = _prepare_default_configs(
        # model_name="Qwen/Qwen3-4B-Instruct-2507",
        # model_name="Qwen/Qwen3-0.6B-Base",
        # model_name="Qwen/Qwen2.5-1.5B",
        model_name="Qwen/Qwen3-30B-A3B-Base",
        max_length=1024 * 16,
    )
    config_utils.save_to_yaml(configs, config_path)
    print(f"Default config created at: {config_path}")
    loaded_configs = config_utils.load_and_validate_config(
        PipelineExperimentConfig,
        str(config_path),
        load_env_config=False,
    )
    assert loaded_configs == configs, "Loaded config does not match the saved config."


if __name__ == "__main__":
    _create_default_config()
