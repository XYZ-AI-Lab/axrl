import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from megatron.bridge.training.config import OptimizerConfig

from pydantic import BaseModel, ConfigDict


@dataclass
class DirInfo:
    data: Path
    model: Path
    output: Path
    log: Path


def _prepare_global_dirs() -> DirInfo:
    """Load AXRL directories from environment variables or local defaults.

    Keep this import-time helper side-effect free. Some tools import config
    classes only to generate or validate YAML; touching remote filesystems here
    can block those lightweight commands before they do any useful work.
    Directory creation belongs at the concrete read/write call site.
    """
    axrl_root = Path.home() / "axrl-data"
    output_dir_name = os.environ.get("AXRL_OUTPUT_DIR_NAME", "default")
    output_dir = Path(os.environ.get("AXRL_OUTPUT_DIR", str(axrl_root / "outputs" / output_dir_name)))

    dir_info = DirInfo(
        data=Path(os.environ.get("AXRL_DATA_DIR", str(axrl_root / "datasets"))),
        model=Path(os.environ.get("AXRL_MODEL_DIR", str(axrl_root / "models"))),
        output=output_dir,
        log=Path(os.environ.get("AXRL_LOG_DIR", str(output_dir / "logs"))),
    )

    return dir_info


AXRL_DIR = _prepare_global_dirs()
STABLE_MODEL_DIR = Path(os.environ.get("AXRL_STABLE_MODEL_DIR", str(AXRL_DIR.model)))
SHARED_MODEL_DIR = Path(os.environ.get("AXRL_SHARED_MODEL_DIR", str(AXRL_DIR.model / "shared_models")))
RAY_TASK_QUEUE_MAX_CONCURRENCY = 8192


def resolve_model_name(model_name: str) -> str:
    """Return an absolute shared-model path when the model is absent from stable storage."""
    model_path = Path(model_name)
    if model_path.is_absolute():
        return str(model_path)
    if (STABLE_MODEL_DIR / model_path).exists():
        return model_name

    shared_model_path = SHARED_MODEL_DIR / model_path
    if shared_model_path.exists():
        return str(shared_model_path.absolute())
    return model_name


class StrictBaseModel(BaseModel):
    """Base model that raises an error if extra fields are present."""

    model_config = ConfigDict(extra="forbid")


LogLevel = Literal["debug", "info", "warning", "error", "critical"]

ValueAggType = Literal["token-mean", "seq-mean-token-mean", "seq-mean-token-std", "token-max", "token-min", "token-std", "token-p05", "token-p95"]

SampleType = Literal[
    "uniform",  # uniform sampling
    "uniform-no-easy",  # uniform sampling excluding easy samples with 90%+ success rate
    "low-success-rate",  # samples with lower success rate are prioritized
    "intermediate-success-rate",  # samples with intermediate success rate are prioritized
]


class RolloutSamplerConfig(StrictBaseModel):
    """Configs for the rollout sampler."""

    sample_type: SampleType = "uniform"
    epsilon: float = 0.05


class LoggerConfig(StrictBaseModel):
    """Configs for a logger."""

    name: str = "axrl"
    level: LogLevel = "info"


MetricLoggerType = Literal["tensorboard", "wandb", "console"]


class MetricLoggerConfig(StrictBaseModel):
    """Configs for a metric recorder."""

    logger_type: MetricLoggerType = "console"
    name: str = "main-process"
    group_name: str = "default_group"
    project_name: str = "axrl"
    run_id: str | None = None

    def get_log_dir(self) -> Path:
        """Get the full path to the log directory."""
        path = AXRL_DIR.log
        return path.absolute()


class DatasetConfig(StrictBaseModel):
    name: str
    data_path: str | None = None
    # Eval-only fields. Must be set on entries in test_datasets; ignored on train_datasets entries.
    eval_num_rollouts_per_prompt: int | None = None
    subset_key: str | None = None
    """Optional key in conv.extra used to split eval metrics into subsets (e.g. 'data_source')."""


class HfDataConfig(StrictBaseModel):
    """Configs for a Hugging Face dataset."""

    repo_id: str = "open-r1/DAPO-Math-17k-Processed"
    filename: str = "all/train-00000-of-00001.parquet"

    def get_full_path(self) -> Path:
        """Get the full path to the dataset file."""
        path = AXRL_DIR.data / self.repo_id / self.filename
        return path.absolute()


class ModelConfig(StrictBaseModel):
    """Configs for a model."""

    name: str = "Qwen/Qwen3-0.6B-Base"  # Other sizes: 7B, 32B, 72B
    trust_remote_code: bool = True
    seq_length: int = 2048

    def get_full_path(self) -> Path:
        """Get the full name of the model."""
        resolved_name = Path(resolve_model_name(self.name))
        if resolved_name.is_absolute():
            return resolved_name
        path = AXRL_DIR.model / self.name
        return path.absolute()


Role = Literal["user", "assistant", "system", "tool"]
IGNORE_INDEX: int = -100


PlacementGroupStrategy = Literal["PACK", "SPREAD", "STRICT_PACK", "STRICT_SPREAD"]


class WorkerConfig(StrictBaseModel):
    """Configs for the worker."""

    name: str = "default-worker"


class InferWorkerConfig(WorkerConfig):
    """Configs for the inference worker."""


class SamplingConfig(StrictBaseModel):
    """Configs for sampling."""

    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    min_p: float = 0.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    repetition_penalty: float = 1.0
    max_total_tokens: int = 4096
    max_new_tokens: int | None = None
    stop: list[str] | str | None = None


class OAIClientConfig(InferWorkerConfig):
    """Configs for a simple OpenAI-compatible generation client.

    The first implementation targets SGLang's native ``/generate`` endpoint so
    we can send token ids and receive token-id-aligned logprobs. The name
    reflects the service boundary rather than use of the official OpenAI SDK.
    """

    base_url: str
    sampling_config: SamplingConfig = SamplingConfig()
    request_timeout_seconds: float | None = None
    max_connections: int | None = 512
    max_keepalive_connections: int | None = 128
    retry_initial_sleep_seconds: float = 1.0
    retry_max_sleep_seconds: float = 30.0


EngineType = Literal["sglang"]


class RolloutWorkerConfig(InferWorkerConfig):
    """Configs for the rollout worker."""

    engine_type: EngineType = "sglang"
    model: ModelConfig = ModelConfig()
    # Legacy rollout-worker default; pipeline recipes should prefer
    # PipelineExperimentConfig.train_sampling_config.
    sampling_config: SamplingConfig = SamplingConfig()
    gpu_memory_utilization: float = 0.6
    dp_size: int = 1
    tp_size: int = 2
    pp_size: int = 1
    ep_size: int = 1
    moe_a2a_backend: Literal["none", "deepep", "mooncake", "mori", "ascend_fuseep", "flashinfer"] = "none"
    moe_runner_backend: str | None = None
    enable_routing_replay: bool = False
    num_workers: int = 1
    clear_partial_outputs_after_abort: bool = False
    continue_generation_after_abort: bool = True
    log_level: LogLevel = "info"
    max_running_requests: int = 128
    max_running_requests_eval: int | None = None
    max_num_batched_tokens: int | None = None
    load_dummy_weights: bool = False
    attention_backend: str | None = None
    kv_cache_dtype: Literal["auto", "fp8_e4m3", "fp8_e5m2"] = "auto"
    prefill_max_requests: int | None = None
    enable_fp32_lm_head: bool = True
    enable_metrics: bool = True
    dtype: Literal["auto", "float16", "bfloat16"] = "bfloat16"
    master_addr: str = "127.0.0.1"
    master_port: int = 20250
    nnodes: int = 1
    node_rank: int = 0
    max_imbalance: int = 16

    def gpus_per_worker(self) -> int:
        """Get the number of GPUs per worker."""
        return self.tp_size * self.pp_size


TeacherBackend = Literal["sglang", "megatron"]
OPDStudentLogprobSource = Literal["rollout_logprobs", "old_logprobs", "cur_logprobs"]


class OPDConfig(StrictBaseModel):
    """On-policy distillation settings shared by rollout annotation and training loss."""

    enabled: bool = False
    backend: TeacherBackend = "sglang"
    teacher_model: ModelConfig | None = None

    # SGLang teacher runtime. Keep this rollout-worker-shaped so model
    # parallelism, memory, dtype, and request limits remain explicit.
    sglang_worker: RolloutWorkerConfig = RolloutWorkerConfig()
    # Optional before service startup. The controller fills this with the
    # actual SGLang endpoint selected by Ray placement before rollout actors start.
    sglang_host: str | None = None
    sglang_port: int | None = None

    opd_alpha: float = 0.1
    reverse_kl_clip: float | None = 10.0
    student_logprob_source: OPDStudentLogprobSource = "cur_logprobs"
    normalize_student_logprob_scale: bool = False
    teacher_weight_name: str = "opd_teacher_weights"

    def model_post_init(self, __context: object, /) -> None:
        if not self.enabled:
            return
        assert 0.0 <= self.opd_alpha <= 1.0, "OPD opd_alpha must be between 0 and 1."
        assert self.teacher_model is not None, "OPD requires teacher_model when enabled."
        if self.backend == "sglang":
            assert self.sglang_worker.model == self.teacher_model, "OPD SGLang teacher model must match sglang_worker.model."
            assert self.sglang_port is not None, "OPD SGLang backend requires sglang_port."
        else:
            assert self.backend == "megatron", f"Unsupported OPD backend: {self.backend!r}."
            assert self.teacher_weight_name, "OPD Megatron backend requires a non-empty teacher_weight_name."


class DataloaderConfig(StrictBaseModel):
    """Configs for the dataloader."""

    num_workers: int = 4


class GradSpikeDebugConfig(StrictBaseModel):
    """Configuration for gradient spike debug snapshotting.

    When enabled, monitors gradient norms and saves a full debug snapshot
    (model + optimizer checkpoint, input batch, per-param grad norms, metadata)
    when a spike is detected. A spike is defined as grad_norm exceeding
    ``spike_ratio`` times the median of the last ``history_window`` steps.

    Detection is skipped during the first ``warmup_steps`` to avoid false
    positives while training is warming up.  Snapshots are saved to a
    dedicated ``spike_snapshots/`` directory under the output dir, and
    Megatron's built-in ``most_recent_k`` rotation keeps only the last
    ``max_snapshots`` checkpoints.
    """

    enabled: bool = False
    warmup_steps: int = 10
    spike_ratio: float = 2.0
    history_window: int = 100
    max_snapshots: int = 3
    save_per_param_grads: bool = True


class MCoreOptimizerConfig(StrictBaseModel):
    """Optimizer settings independent of Megatron import time."""

    optimizer: str = "adam"
    lr: float | None = None
    min_lr: float | None = None
    weight_decay: float = 0.01
    use_distributed_optimizer: bool = True
    clip_grad: float = 1.0
    fp16: bool = False
    bf16: bool = False
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    optimizer_cpu_offload: bool = False
    optimizer_offload_fraction: float = 0.0
    use_stateless_adam: bool = False

    def to_megatron_config(self) -> "OptimizerConfig":
        from megatron.bridge.training.config import OptimizerConfig

        payload = self.model_dump()
        use_stateless_adam = payload.pop("use_stateless_adam")
        assert not use_stateless_adam, "use_stateless_adam is reserved but not implemented yet."
        config = OptimizerConfig(**payload)
        config.finalize()
        return config


class MCoreLrSchedulerConfig(StrictBaseModel):
    """Configs for the learning rate scheduler."""

    init_lr: float = 1e-5
    max_lr: float = 1e-5
    min_lr: float = 1e-6
    lr_warmup_steps: int = 100
    lr_decay_steps: int = 1000000
    lr_decay_style: Literal["constant", "cosine", "linear"] = "constant"
    start_wd: float = 0.01
    end_wd: float = 0.01
    wd_incr_steps: int = 1000000
    wd_incr_style: Literal["constant", "cosine", "linear"] = "constant"
    use_checkpoint_opt_param_scheduler: bool | None = True
    override_opt_param_scheduler: bool | None = False
    wsd_decay_steps: int | None = None
    lr_wsd_decay_style: str | None = None


TrainerType = Literal["sft"]
MegatronModelRole = Literal["actor", "value"]


class SftTrainerConfig(StrictBaseModel):
    """Configs for the SFT trainer."""

    compute_accuracy: bool = False
    compute_entropy: bool = False


SFTConfig = SftTrainerConfig


class MegatronWorkerConfig(StrictBaseModel):
    """Configs for the Megatron worker."""

    model: ModelConfig = ModelConfig()
    model_role: MegatronModelRole = "actor"

    seed: int = 42
    # Parallelism Configs
    tp_size: int = 1
    pp_size: int = 1
    vpp_size: int | None = None
    dp_size: int = 1
    cp_size: int = 1  # context parallel size
    ep_size: int = 1
    etp_size: int | None = 1  # expert tensor parallel size
    distributed_timeout_seconds: int = 1800
    enable_routing_replay: bool = False
    replay_routing_for_loss_tokens_only: bool = False

    # FP8 mixed precision (MCore forward/backward via TransformerEngine)
    fp8: Literal["e4m3", "hybrid"] | None = None
    fp8_recipe: Literal["tensorwise", "delayed", "mxfp8", "blockwise", "custom"] = "blockwise"

    # Dataloader
    data_loader: DataloaderConfig = DataloaderConfig()
    padding_sample_length: int = 4

    # Training Hyperparameters
    train_micro_batch_size: int = 4  # micro-batch size per GPU
    eval_micro_batch_size: int = 8
    global_batch_size: int = 32  # accumulate gradients
    use_dynamic_batch_size: bool = True
    num_epochs: int = 100

    # MoE
    moe_aux_loss_coeff: float = 0
    moe_router_load_balancing_type: Literal["none", "aux_loss", "seq_aux_loss", "global_aux_loss", "sinkhorn"] = "none"

    # Memory Saving
    recompute_granularity: Literal["full", "selective"] | None = None
    recompute_method: Literal["uniform", "block"] | None = None
    recompute_num_layers: int | None = None
    recompute_modules: list[str] | None = None  # choices: "core_attn", "moe_act", "layernorm", "mla_up_proj", "mlp", "moe", "shared_experts"
    cpu_offloading: bool = False
    cpu_offloading_num_layers: int = 0  # Number of layers to offload (0 = all)
    cpu_offloading_double_buffering: bool = False

    # Checkpointing and Logging

    log_every_k_steps: int = 10
    checkpoint_dir: str = "checkpoints/megatron"
    checkpoint_step: int | None = None
    most_recent_checkpoint_k: int = 2
    save_hf_checkpoint: bool = False  # save HF-format checkpoint alongside megatron checkpoint
    reset_init_weights_every_k_steps: int | None = None
    log_microbatches: bool = False
    log_gpu_usaegs: bool = False
    metric_logger_config: MetricLoggerConfig = MetricLoggerConfig()
    log_level: LogLevel = "info"

    fp16: bool = False
    bf16: bool = True
    use_gloo_process_groups: bool = True
    apply_rope_fusion: bool | None = None

    attention_backend: str | None = "flash"  # None means use default (auto)

    deterministic_mode: bool = True
    batch_invariant_mode: bool = False
    use_language_model_only: bool = False

    spike_debug: GradSpikeDebugConfig = GradSpikeDebugConfig()

    optimizer: MCoreOptimizerConfig = MCoreOptimizerConfig(
        lr=1e-5,
        min_lr=1e-6,
        bf16=True,
        use_distributed_optimizer=True,
    )
    lr_scheduler: MCoreLrSchedulerConfig = MCoreLrSchedulerConfig()

    # data loader
    num_dataloader_workers: int = 0
    inference_only: bool = False  # when True, does not initialize optimizer and scheduler
    shuffle_train_data: bool = False

    enable_fp32_lm_head: bool = False

    # Trajectory-aware merged forward. When True, the worker hands the trainer
    # a Magi-Attention-based ``GPTModelForwardFn`` that packs each trajectory's
    # turn-samples into a prefix-tree-merged layout (one ``calc_attn`` call per
    # microbatch with offset-shifted q/k ranges). When False, the baseline
    # TE+THD causal forward is used. Applies to every trainer the worker runs,
    # not SFT specifically.
    use_magi_merged_forward: bool = False

    # Diagnostic / test-only: route every forward through Magi `calc_attn`
    # with a FLAT trie (per-sample causal ranges; same kernel as the
    # merged path). Verified bit-exact against the TE FA3 THD baseline,
    # so useful for isolating the prefix-merging delta in tests. Mutually
    # exclusive with `use_magi_merged_forward`.
    use_magi_flat_forward: bool = False

    def world_size(self) -> int:
        """Return Megatron's regular world size for this configuration.

        Related Megatron-LM code (commit-pinned permalink):
        https://github.com/NVIDIA/Megatron-LM/blob/3bec9aa97dda898d16ff5a89bac0ed2b6682b172/megatron/core/parallel_state.py#L736-L743
        """
        return self.dp_size * self.tp_size * self.pp_size * self.cp_size

    def expert_tensor_parallel_size(self) -> int:
        """Return the expert tensor parallel size Megatron will use.

        When etp_size is None, Megatron defaults it to tp_size.
        Related Megatron-LM code (commit-pinned permalink):
        https://github.com/NVIDIA/Megatron-LM/blob/3bec9aa97dda898d16ff5a89bac0ed2b6682b172/megatron/core/parallel_state.py#L786-L787
        """
        return self.etp_size if self.etp_size is not None else self.tp_size

    def expert_parallel_group_size(self) -> int:
        """Return the expert TP x EP x PP size used for expert-group divisibility checks."""
        return self.expert_tensor_parallel_size() * self.ep_size * self.pp_size

    def get_checkpoint_dir(self) -> Path:
        """Get the full path to the checkpoint directory."""
        path = AXRL_DIR.output / self.checkpoint_dir
        return path.absolute()


class EntropyControlConfig(StrictBaseModel):
    """Configs for entropy control."""

    target_entropy: float = 0.1  # the target entropy value to achieve
    alpha: float = 0.0  # strength of entropy control, 0 means no control
    # Keep this disabled when alpha is 0 in normal GRPO runs: the entropy pass
    # does a vocab-wide reduction and is expensive. Enable it only when you need
    # raw entropy metrics/debugging while entropy control itself is off.
    compute_entropy: bool = False
    top_k: int = -1  # top k for logits masking when calculating entropy, -1 means no masking
    top_quantile: float = 1.0  # filter tokens with top quantiles entropy when calculating entropy, 1.0 means no filtering
    easy_score_cutoff: float = 0.4  # only consider samples with average score below this cutoff for entropy control, 1.0 means no filtering
    epsilon: float = 1e-6


RewardMeanType = Literal["group"]
RewardStdType = Literal["group"]
LossType = Literal["grpo", "ppo", "tis", "gspo", "grpo2", "topr", "kimi2_5"]
IsBaseLogprobsType = Literal["old_logprobs", "rollout_logprobs"]
KlBaseLogprobsType = Literal["old_logprobs", "ref_logprobs"]
MicroBatchDenominatorType = Literal["token", "sequence", "seq_turn"]


class PPOValueConfig(StrictBaseModel):
    """Configs for PPO value targets, value loss, and value-worker warmup."""

    gamma: float = 1.0
    gae_lambda: float = 1.0
    value_clip: float | None = 0.2
    value_loss_coef: float = 1.0
    num_value_only_updates: int = 0
    use_stateless_value_model: bool = False


class GrpoTrainerConfig(StrictBaseModel):
    """Configs consumed by the GRPO trainer."""

    loss_type: LossType = "grpo2"
    clip_ratio_high: float = 0.28
    clip_ratio_low: float = 0.2
    dual_clip_neg_adv_factor: float | None = 3.0  # dual clip factor for negative advantages
    dual_soft_clip: float | None = 3
    log_sample: bool = True
    normalize_advantage_by_batch_std: bool = False
    loss_agg_type: ValueAggType = "token-mean"
    micro_batch_denominator_type: MicroBatchDenominatorType = "token"
    entropy_control: EntropyControlConfig = EntropyControlConfig()
    kl_control_alpha: float = 0.0  # strength of KL control, 0 means no KL control
    is_base_logprobs: IsBaseLogprobsType = "old_logprobs"  # importance sampling based on which logprobs
    kl_base_logprobs: KlBaseLogprobsType = "old_logprobs"
    # Truncated Rollout/Trainer IS: https://fengyao.notion.site/off-policy-rl
    mismatch_token_clip_max: float | None = None  # 2
    # Geometric sequence masking: https://richardli.xyz/rl-collapse
    mismatch_seq_masking_low: float | None = None  # 0.998
    mismatch_seq_masking_high: float | None = None  # 1.002
    mismatch_token_veto_threshold: float | None = None  # 1e-4
    # IcePop: https://hijkzzz.notion.site/online-ice-pop
    icepop_masking_low: float | None = None  # 0.5
    icepop_masking_high: float | None = None  # 5
    turn_reward_alpha: float = 0.0  # advantage = final_adv + alpha * normalized_turn_reward; 0 disables
    seq_turn_alpha: float | None = None  # seq_turn denominator: total_sequences + alpha * total_turns; None uses turn_reward_alpha
    opd: OPDConfig = OPDConfig()

    # PPO actor-side advantage normalization. Inactive unless loss_type == "ppo".
    normalize_advantages_over_valid_tokens_in_batch: bool = False
    ppo_value: PPOValueConfig | None = None


class OnlineRLTrainConfig(StrictBaseModel):
    """Configs for the online rollout -> train control loop."""

    eval_on_start: bool = True
    num_rollouts_per_conversation: int = 8  # total_rollouts_per_epoch = conversation_per_epoch * num_rollouts_per_conversation
    model_sync_every_n_global_updates: int = 4  # num of global batch updates to sync model weights from megatron worker to rollout worker
    checkpoint_every_n_global_updates: int | None = 1024
    eval_every_n_global_updates: int = 16
    batch_rollout_for_n_global_updates: int = 4
    max_global_updates: int = 4000
    strict_on_policy: bool = False
    reward_mean_type: RewardMeanType = "group"
    reward_std_type: RewardStdType = "group"
    reward_history_size: int = 64
    sample_type: SampleType = "uniform"
    filter_zero_std: bool = False
    max_prompt_length: int | None = None
    sort_sampled_prompts_by_response_length: bool = True  # sort prompts by historical response length descendingly after sampling.
    rollout_save_filename: str = "valid_rollouts"
    rollout_save_every_n_global_updates: int | None = None  # if not None, save valid rollouts to disk every n global updates for analysis/debugging
    save_all_rollouts: bool = False  # if True, snapshot all rollouts (incl. filter_zero_std-dropped groups); else only training-valid


class GrpoConfig(GrpoTrainerConfig):
    """Legacy combined GRPO config kept for the old controller."""

    eval_on_start: bool = True
    num_rollouts_per_conversation: int = 16
    model_sync_every_n_global_updates: int = 4
    checkpoint_every_n_global_updates: int = 1024
    eval_every_n_global_updates: int = 64
    batch_rollout_for_n_global_updates: int = 8
    max_global_updates: int = 2000
    strict_on_policy: bool = False
    reward_mean_type: RewardMeanType = "group"
    reward_std_type: RewardStdType = "group"
    reward_history_size: int = 64
    sample_type: SampleType = "uniform"
    filter_zero_std: bool = False
    max_prompt_length: int | None = None
    sort_sampled_prompts_by_response_length: bool = True
    rollout_save_filename: str = "valid_rollouts"
    rollout_save_every_n_global_updates: int | None = None
    save_all_rollouts: bool = False


class MismatchTestConfig(StrictBaseModel):
    """Configs for mismatch test."""

    # Legacy GrpoController dispatch flag. PipelineController ignores this
    # field and uses controller.run_mode="mismatch_test" instead.
    enabled: bool = False
    name: str = "cur-exp"
    output_dir: str = "tmp/mismatch-test"
    override_rollouts_if_exists: bool = False
    baseline_samples_path: str | None = None
    baseline_name: str | None = None

    @staticmethod
    def _sanitize_name(name: str) -> str:
        return name.replace("/", "_")

    def get_output_dir(self) -> Path:
        return Path(self.output_dir)

    def get_result_dir(self) -> Path:
        return self.get_output_dir() / "results"

    def get_data_root(self) -> Path:
        return self.get_result_dir() / "data"

    def get_log_dir(self) -> Path:
        return self.get_result_dir() / "log"

    def get_fig_dir(self) -> Path:
        return self.get_result_dir() / "figs"

    def get_exp_data_dir(self, name: str) -> Path:
        return self.get_data_root() / self._sanitize_name(name)

    def get_valid_rollouts_path(self, name: str) -> Path:
        filename = self._sanitize_name(name)
        return self.get_exp_data_dir(name) / f"valid_rollouts-{filename}.zst"

    def get_training_samples_path(self, name: str) -> Path:
        filename = self._sanitize_name(name)
        return self.get_exp_data_dir(name) / f"training_samples-{filename}.zst"

    def get_result_path(self, name: str) -> Path:
        filename = self._sanitize_name(name)
        return self.get_log_dir() / f"{filename}.zst"
