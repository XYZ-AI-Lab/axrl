from axrl.configs import (
    DatasetConfig,
    GrpoConfig,
    MegatronWorkerConfig,
    MetricLoggerConfig,
    MismatchTestConfig,
    RolloutWorkerConfig,
    SamplingConfig,
    StrictBaseModel,
)


class GrpoExperimentConfig(StrictBaseModel):
    megatron_worker: MegatronWorkerConfig = MegatronWorkerConfig()
    rollout_worker: RolloutWorkerConfig = RolloutWorkerConfig()
    grpo: GrpoConfig = GrpoConfig()
    eval_sampling_config: SamplingConfig = SamplingConfig()
    train_datasets: list[DatasetConfig] | None = None
    test_datasets: list[DatasetConfig] | None = None
    colocated: bool = True
    mismatch_test: MismatchTestConfig = MismatchTestConfig()
    logger: MetricLoggerConfig = MetricLoggerConfig()
    eval_on_start: bool = True
    eval_only: bool = False
    debug_train: bool = False
