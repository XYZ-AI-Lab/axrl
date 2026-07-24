from __future__ import annotations

from typing import Literal

from axrl.configs import (
    DatasetConfig,
    GrpoTrainerConfig,
    MegatronWorkerConfig,
    MetricLoggerConfig,
    MismatchTestConfig,
    ModelConfig,
    OnlineRLTrainConfig,
    RolloutWorkerConfig,
    SamplingConfig,
    SftTrainerConfig,
    StrictBaseModel,
)

PipelineRunMode = Literal[
    "online_rl_train",  # roll out new data, then train RL
    "replay_rl_train",  # train RL from saved rollout data, no new rollouts
    "eval_only",  # run eval rollouts, no training
    "sft_train",  # supervised train from dataset, no rollouts
    "mismatch_test",
]


class ControllerConfig(StrictBaseModel):
    """Control-loop settings for pipeline execution."""

    run_mode: PipelineRunMode = "online_rl_train"
    output_dir_name: str = "grpo"
    colocated: bool = True
    num_rollout_actors: int = 16
    # CPU resources reserved for each rollout actor.
    # Include room for actor-local helper pools such as tokenizers or metrics.
    num_cpus_per_actor: int = 4
    max_running_requests: int = 256
    smoke_eval_rollouts: int | None = None
    save_eval_rollouts: bool = True
    allow_prefix_merging: bool = True


class EvalOnlyConfig(StrictBaseModel):
    """Settings used only by ``controller.run_mode='eval_only'``."""

    model: ModelConfig | None = None


class ReplayRLTrainConfig(StrictBaseModel):
    """Inputs for RL training without generating new rollouts."""

    sample_dict_path: str | None = None
    rollout_groups_path: str | None = None


class PipelineExperimentConfig(StrictBaseModel):
    controller: ControllerConfig = ControllerConfig()
    eval_only: EvalOnlyConfig = EvalOnlyConfig()

    # worker configs
    megatron_worker: MegatronWorkerConfig = MegatronWorkerConfig()
    value_worker: MegatronWorkerConfig | None = None
    rollout_worker: RolloutWorkerConfig = RolloutWorkerConfig()

    # dataset configs
    train_datasets: list[DatasetConfig] | None = None
    test_datasets: list[DatasetConfig] | None = None

    # sampling configs
    eval_sampling_config: SamplingConfig = SamplingConfig()
    train_sampling_config: SamplingConfig = SamplingConfig()

    # trainer configs
    grpo: GrpoTrainerConfig = GrpoTrainerConfig()
    sft: SftTrainerConfig = SftTrainerConfig()
    online_rl_train: OnlineRLTrainConfig = OnlineRLTrainConfig()
    replay_rl_train: ReplayRLTrainConfig = ReplayRLTrainConfig()
    mismatch_test: MismatchTestConfig = MismatchTestConfig()

    # logger config
    logger: MetricLoggerConfig = MetricLoggerConfig()
