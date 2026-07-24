from __future__ import annotations

from pathlib import Path

from pydantic import Field

from axrl.configs import (
    DatasetConfig,
    GrpoTrainerConfig,
    MCoreLrSchedulerConfig,
    MCoreOptimizerConfig,
    MegatronWorkerConfig,
    MetricLoggerConfig,
    ModelConfig,
    OnlineRLTrainConfig,
    OPDConfig,
    RolloutWorkerConfig,
    SamplingConfig,
    StrictBaseModel,
)
from axrl.pipeline.config import ControllerConfig, EvalOnlyConfig, PipelineExperimentConfig
from axrl.runner.e2b_runner import E2BRunnerConfig
from axrl.utils import config_utils
from axrl.utils.tunnel import TunnelConfig

OpenAIProxyTunnelConfig = TunnelConfig


class OpenAIProxyExposureConfig(StrictBaseModel):
    exposed_base_url: str | None = None
    allow_out: list[str] = Field(default_factory=list)
    tunnel: OpenAIProxyTunnelConfig | None = Field(default_factory=OpenAIProxyTunnelConfig)


class OpenAIProxyConfig(StrictBaseModel):
    # Valid parser names are SGLang registry keys:
    # - tool_call_parser: FunctionCallParser.ToolCallParserEnum
    #   https://github.com/sgl-project/sglang/blob/127b9e3283f7c2a43234b852ff5c9f1796d53624/python/sglang/srt/function_call/function_call_parser.py#L56-L84
    # - reasoning_parser: ReasoningParser.DetectorMap
    #   https://github.com/sgl-project/sglang/blob/127b9e3283f7c2a43234b852ff5c9f1796d53624/python/sglang/srt/parser/reasoning_parser.py#L610-L631
    host: str = "0.0.0.0"  # noqa: S104 - this is a bind address; public_host controls the advertised URL.
    public_host: str | None = None
    port: int = 0
    served_model_name: str | None = None
    request_timeout_seconds: float = 1200.0
    adapter_num_processors: int = 1
    adapter_timeout_seconds: float = 300.0
    tool_call_parser: str | None = "qwen"  # None, "auto", or SGLang tool parser, e.g. "qwen", "glm45".
    reasoning_parser: str | None = "qwen3"  # None, "auto", or SGLang reasoning parser, e.g. "qwen3", "deepseek-r1".
    exposure: OpenAIProxyExposureConfig = OpenAIProxyExposureConfig()


class OpenHandsLauncherConfig(StrictBaseModel):
    command: list[str] = Field(default_factory=lambda: ["openhands"])
    api_key: str = "EMPTY"
    llm_timeout_seconds: float | None = 1200.0
    suppress_banner: bool = True
    load_public_skills: bool = False
    extra_env: dict[str, str] = Field(default_factory=dict)
    e2b: E2BRunnerConfig = E2BRunnerConfig()


class OpenHandsEnvConfig(StrictBaseModel):
    test_file_dir: str = "tmp/openhands-case-study"
    initial_request_timeout_seconds: float = 360.0
    request_timeout_seconds: float = 60.0
    max_model_calls: int = 16
    collect_file_on_finish: bool = True


class BlackBoxRLConfig(PipelineExperimentConfig):
    openai_proxy: OpenAIProxyConfig = OpenAIProxyConfig()
    openhands: OpenHandsLauncherConfig = OpenHandsLauncherConfig()
    openhands_env: OpenHandsEnvConfig = OpenHandsEnvConfig()
    verifier_e2b: E2BRunnerConfig = E2BRunnerConfig(timeout_seconds=120, request_timeout_seconds=60.0)
    verifier_num_processors: int = 1
    verifier_timeout_seconds: float = 30.0
    verifier_memory_limit_gib: int = 16


def _prepare_default_configs(
    model_name: str,
    max_length: int,
    *,
    colocated: bool = True,
) -> BlackBoxRLConfig:
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
        tp_size=4,
        dp_size=1,
        cp_size=2,
        pp_size=1,
        vpp_size=None,
        recompute_granularity="full",
        recompute_method="uniform",
        recompute_num_layers=1,
        log_every_k_steps=1,
        global_batch_size=512,
        train_micro_batch_size=1,
        eval_micro_batch_size=1,
        log_gpu_usaegs=True,
        bf16=True,
        fp16=False,
        use_magi_merged_forward=True,
    )
    train_sampling_config = SamplingConfig(
        temperature=1,
        top_p=1,
        max_total_tokens=max_length,
    )
    eval_sampling_config = SamplingConfig(
        temperature=0.7,
        top_p=0.9,
        max_total_tokens=max_length,
    )
    rollout_worker_config = RolloutWorkerConfig(
        model=model_config,
        sampling_config=train_sampling_config,
        dp_size=1,
        num_workers=2,
        tp_size=4,
        gpu_memory_utilization=0.7,
        max_running_requests=32,
        max_running_requests_eval=None,
    )
    logger_config = MetricLoggerConfig(
        logger_type="tensorboard",
        name="main-process",
        group_name="blackbox-rl-local-test",
        project_name="BlackBoxRL",
    )
    grpo = GrpoTrainerConfig(
        clip_ratio_high=0.28,
        clip_ratio_low=0.2,
        loss_type="grpo2",
        opd=OPDConfig(sglang_worker=RolloutWorkerConfig(max_running_requests=32)),
    )
    online_rl_train = OnlineRLTrainConfig(
        model_sync_every_n_global_updates=4,
        batch_rollout_for_n_global_updates=4,
        filter_zero_std=False,
        num_rollouts_per_conversation=8,
        sample_type="uniform",
        eval_every_n_global_updates=16,
        rollout_save_every_n_global_updates=16,
        checkpoint_every_n_global_updates=None,
    )
    return BlackBoxRLConfig(
        controller=ControllerConfig(
            run_mode="eval_only",
            output_dir_name="blackbox-rl",
            colocated=colocated,
            num_rollout_actors=32,
            num_cpus_per_actor=1,
            max_running_requests=32,
        ),
        eval_only=EvalOnlyConfig(),
        megatron_worker=megatron_worker_config,
        rollout_worker=rollout_worker_config,
        grpo=grpo,
        online_rl_train=online_rl_train,
        train_datasets=[DatasetConfig(name="newfacade/LeetCodeDataset/train")],
        test_datasets=[DatasetConfig(name="newfacade/LeetCodeDataset/test", eval_num_rollouts_per_prompt=8)],
        logger=logger_config,
        eval_sampling_config=eval_sampling_config,
        train_sampling_config=train_sampling_config,
    )


def _create_default_config() -> None:
    config_path = Path("axis_recipe/blackbox_rl/blackbox-rl-config.yaml")
    configs = _prepare_default_configs(
        model_name="Qwen/Qwen3-4B-Instruct-2507",
        max_length=65536,
    )
    config_utils.save_to_yaml(configs, config_path)
    print(f"Default config created at: {config_path}")
    loaded_configs = config_utils.load_and_validate_config(
        BlackBoxRLConfig,
        str(config_path),
        load_env_config=False,
    )
    print(loaded_configs)


if __name__ == "__main__":
    _create_default_config()
