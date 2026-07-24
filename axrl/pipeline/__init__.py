from typing import TYPE_CHECKING, Any

from axrl.configs import GrpoTrainerConfig, MismatchTestConfig, OnlineRLTrainConfig, SftTrainerConfig
from axrl.pipeline.config import ControllerConfig, EvalOnlyConfig, PipelineExperimentConfig, PipelineRunMode, ReplayRLTrainConfig

if TYPE_CHECKING:
    from axrl.pipeline.controller import PipelineController
    from axrl.pipeline.rollout_actor import RolloutActor
    from axrl.pipeline.rollout_data import RolloutGroup, RolloutRuntime, TrainGroupBatch
    from axrl.pipeline.utils import PipelineWorkerPlacement, rollout_from_env, shutdown_pipeline_workers, start_ray_infer_worker

__all__ = [
    "ControllerConfig",
    "EvalOnlyConfig",
    "GrpoTrainerConfig",
    "MismatchTestConfig",
    "OnlineRLTrainConfig",
    "PipelineController",
    "PipelineExperimentConfig",
    "PipelineRunMode",
    "PipelineWorkerPlacement",
    "ReplayRLTrainConfig",
    "RolloutActor",
    "RolloutGroup",
    "RolloutRuntime",
    "SftTrainerConfig",
    "TrainGroupBatch",
    "rollout_from_env",
    "shutdown_pipeline_workers",
    "start_ray_infer_worker",
]


def __getattr__(name: str) -> Any:
    if name == "PipelineController":
        from axrl.pipeline.controller import PipelineController

        return PipelineController
    if name == "RolloutActor":
        from axrl.pipeline.rollout_actor import RolloutActor

        return RolloutActor
    if name in {"RolloutGroup", "RolloutRuntime", "TrainGroupBatch"}:
        from axrl.pipeline import rollout_data

        return getattr(rollout_data, name)
    if name in {"PipelineWorkerPlacement", "rollout_from_env", "shutdown_pipeline_workers", "start_ray_infer_worker"}:
        from axrl.pipeline import utils

        return getattr(utils, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
