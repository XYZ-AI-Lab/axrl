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
    RolloutWorkerConfig,
    SamplingConfig,
    StrictBaseModel,
)
from axrl.pipeline.config import ControllerConfig, PipelineExperimentConfig
from axrl.utils import config_utils


class SearchR1VerifierConfig(StrictBaseModel):
    structure_format_score: float = 0.1
    retrieval_score: float = 0.2


class SearchClientConfig(StrictBaseModel):
    request_timeout: float = 30.0
    max_connections: int = 4096
    max_keepalive_connections: int = 1024
    max_retries: int = 30
    retry_backoff_seconds: float = 1.0


class SearchR1Config(PipelineExperimentConfig):
    verifier: SearchR1VerifierConfig = SearchR1VerifierConfig()
    search_client: SearchClientConfig = SearchClientConfig()


def _prepare_default_configs(
    model_name: str,
    max_length: int,
    *,
    colocated: bool = True,
) -> SearchR1Config:
    model_config = ModelConfig(name=model_name, seq_length=max_length, trust_remote_code=True)
    optimizer_config = MCoreOptimizerConfig(
        optimizer="adam",
        lr=1e-6,
        min_lr=1e-7,
        weight_decay=0.1,
        use_distributed_optimizer=True,
        clip_grad=1.0,
        bf16=True,
    )
    lr_scheduler_config = MCoreLrSchedulerConfig(
        lr_decay_style="constant",
        init_lr=1e-7,
        max_lr=1e-6,
        lr_warmup_steps=50,
    )
    megatron_worker_config = MegatronWorkerConfig(
        model=model_config,
        optimizer=optimizer_config,
        lr_scheduler=lr_scheduler_config,
        tp_size=2,
        cp_size=2,
        pp_size=1,
        vpp_size=None,
        log_every_k_steps=1,
        global_batch_size=512,
        train_micro_batch_size=4,
        eval_micro_batch_size=8,
        log_gpu_usaegs=True,
        bf16=True,
        fp16=False,
    )

    sampling_config = SamplingConfig(
        temperature=1.0,
        max_total_tokens=max_length,
        top_p=1.0,
    )
    rollout_worker_config = RolloutWorkerConfig(
        model=ModelConfig(name=model_name, seq_length=max_length, trust_remote_code=True),
        sampling_config=sampling_config,
        dp_size=1,
        num_workers=4,
        tp_size=2,
        gpu_memory_utilization=0.7,
        max_running_requests=1024,
    )

    logger_config = MetricLoggerConfig(
        logger_type="tensorboard",
        name="main-process",
        group_name="GRPO-local-test",
        project_name="SearchR1",
    )

    eval_sampling_config = SamplingConfig(
        temperature=1.0,
        max_total_tokens=max_length,
        top_p=1.0,
    )

    grpo = GrpoTrainerConfig(
        clip_ratio_high=0.28,
        clip_ratio_low=0.2,
        loss_type="grpo2",
    )
    online_rl_train = OnlineRLTrainConfig(
        model_sync_every_n_global_updates=4,
        batch_rollout_for_n_global_updates=4,
        filter_zero_std=False,
        num_rollouts_per_conversation=16,
        reward_std_type="group",
        sample_type="uniform-no-easy",
    )

    verifier = SearchR1VerifierConfig(
        structure_format_score=0.01,
        retrieval_score=0.2,
    )

    configs = SearchR1Config(
        controller=ControllerConfig(
            run_mode="online_rl_train",
            output_dir_name="search-r1",
            colocated=colocated,
            num_rollout_actors=16,
            num_cpus_per_actor=4,
            max_running_requests=rollout_worker_config.max_running_requests,
        ),
        megatron_worker=megatron_worker_config,
        rollout_worker=rollout_worker_config,
        grpo=grpo,
        online_rl_train=online_rl_train,
        train_datasets=[DatasetConfig(name="RUC-NLPIR/FlashRAG_datasets/nq/train")],
        test_datasets=[DatasetConfig(name="RUC-NLPIR/FlashRAG_datasets/nq/test", eval_num_rollouts_per_prompt=1)],
        logger=logger_config,
        eval_sampling_config=eval_sampling_config,
        train_sampling_config=sampling_config,
        verifier=verifier,
    )

    return configs


def _create_default_config() -> None:
    config_path = Path("axis_recipe/search_r1/search-r1-config.yaml")
    configs = _prepare_default_configs(
        # model_name="Qwen/Qwen3-4B-Instruct-2507",
        # model_name="Qwen/Qwen3-0.6B-Base",
        # model_name="Qwen/Qwen2.5-1.5B",
        model_name="Qwen/Qwen2.5-3B-Instruct",
        max_length=1024 * 8,
    )
    config_utils.save_to_yaml(configs, config_path)
    print(f"Default config created at: {config_path}")
    loaded_configs = config_utils.load_and_validate_config(
        SearchR1Config,
        str(config_path),
        load_env_config=False,
    )
    assert loaded_configs == configs, "Loaded config does not match the saved config."


if __name__ == "__main__":
    _create_default_config()
