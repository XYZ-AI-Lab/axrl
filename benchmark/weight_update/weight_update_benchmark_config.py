from pathlib import Path

from axrl.configs import LogLevel, MegatronWorkerConfig, MetricLoggerConfig, ModelConfig, RolloutWorkerConfig, SamplingConfig, StrictBaseModel
from axrl.utils import config_utils


class WeightUpdateBenchmarkConfig(StrictBaseModel):
    """Configs for the colocated weight update benchmark."""

    rollout_worker: RolloutWorkerConfig = RolloutWorkerConfig(load_dummy_weights=False)
    megatron_worker: MegatronWorkerConfig = MegatronWorkerConfig(
        inference_only=True,
        dp_size=1,
        pp_size=1,
        vpp_size=None,
    )
    logger: MetricLoggerConfig = MetricLoggerConfig(
        logger_type="tensorboard",
        project_name="weight-update-bench",
        group_name="colocated-weight-update",
        name="main-process",
    )
    bucket_size_gb: float = 2.0
    warmup_updates: int = 1
    measured_updates: int = 3
    max_runtime_seconds: float = 300.0
    restart_ray: bool = True
    log_level: LogLevel = "info"


def prepare_default_config() -> WeightUpdateBenchmarkConfig:
    model_config = ModelConfig(
        name="Qwen/Qwen3-32B",
        seq_length=64,
        trust_remote_code=True,
    )
    rollout_worker_config = RolloutWorkerConfig(
        engine_type="sglang",
        model=model_config,
        sampling_config=SamplingConfig(
            temperature=0.0,
            max_total_tokens=64,
        ),
        tp_size=4,
        num_workers=1,
        gpu_memory_utilization=0.4,
        load_dummy_weights=False,
    )
    megatron_worker_config = MegatronWorkerConfig(
        model=model_config,
        tp_size=4,
        dp_size=1,
        pp_size=1,
        vpp_size=None,
        inference_only=True,
    )
    logger_config = MetricLoggerConfig(
        logger_type="tensorboard",
        project_name="weight-update-bench",
        group_name="colocated-qwen3-32b",
        name="tp4-nw1-bucket2g",
    )
    return WeightUpdateBenchmarkConfig(
        rollout_worker=rollout_worker_config,
        megatron_worker=megatron_worker_config,
        logger=logger_config,
        bucket_size_gb=2.0,
        warmup_updates=1,
        measured_updates=3,
        max_runtime_seconds=300.0,
        restart_ray=True,
        log_level="info",
    )


def create_default_config() -> None:
    config_path = Path(__file__).with_name("weight_update_benchmark.yaml")
    config = prepare_default_config()
    config_utils.save_to_yaml(config, config_path)
    print(f"Default config created at: {config_path}")
    loaded_config = config_utils.load_and_validate_config(
        WeightUpdateBenchmarkConfig,
        str(config_path),
        load_env_config=False,
    )
    assert loaded_config == config, "Loaded config does not match the saved config."


if __name__ == "__main__":
    create_default_config()
