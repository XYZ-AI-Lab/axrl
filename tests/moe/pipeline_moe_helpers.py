from __future__ import annotations

from typing import TYPE_CHECKING, Any

from axrl.agent.rollout_agent import RolloutAgent
from axrl.configs import (
    DatasetConfig,
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
from axrl.envs.math_env import MathEnv
from axrl.metrics.response_metric import ResponseMetricCalculator
from axrl.pipeline import ControllerConfig
from axrl.pipeline.config import PipelineExperimentConfig
from axrl.pipeline.utils import rollout_from_env
from axrl.processor.conversation_tokenizer import ConversationTokenizer
from axrl.processor.processor_pool import ProcessorPool
from axrl.recipe.base_recipe import BaseRecipe
from axrl.verifier.dapo_verifier import DapoVerifier

if TYPE_CHECKING:
    from axrl.data import Conversation, RolloutResult
    from axrl.pipeline import RolloutRuntime
    from axrl.worker.infer_worker import InferWorker


class MoeMathRecipe(BaseRecipe):
    def initialize_local_processors(self, worker_id: str) -> dict[str, InferWorker[Any, Any]]:
        assert worker_id, "worker_id must be non-empty when initializing MoE math workers."
        return {
            "verifier": ProcessorPool(DapoVerifier, config=None, num_processors=1),
            "metric": ProcessorPool(ResponseMetricCalculator, config=None, num_processors=1),
            "conv_tokenizer": ProcessorPool(ConversationTokenizer, config=self.config.rollout_worker.model, num_processors=1),
        }

    async def run_rollout(self, conversation: Conversation, runtime: RolloutRuntime) -> RolloutResult:
        sampling_config = conversation.gen_state.sampling_config
        assert sampling_config is not None, f"Rollout conversation {conversation.conversation_id!r} is missing sampling config."
        assert "answer" in conversation.extra, f"Rollout conversation {conversation.conversation_id!r} is missing extra['answer']."
        conv_tokenizer = runtime.get_local_worker("conv_tokenizer")
        gen_input = await conv_tokenizer.generate(conversation)
        conversation.gen_state.input_ids = gen_input.input_ids
        max_length = sampling_config.max_total_tokens
        if max_length <= 0:
            max_length = self.config.rollout_worker.model.seq_length

        env = MathEnv(
            score_provider=runtime.get_local_worker("verifier"),
            metric_calculator=runtime.get_local_worker("metric"),
            conv_tokenizer=conv_tokenizer,
            conv=conversation,
            label=conversation.extra["answer"],
            max_length=max_length,
            return_sample=True,
        )
        agent = RolloutAgent(runtime.rollout_worker)
        return await rollout_from_env(env, agent, sampling_config)


def make_moe_pipeline_config(
    *,
    model_name: str,
    max_length: int,
    project_name: str,
    run_name: str,
    output_dir: str,
    baseline_name: str | None,
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
        lr_warmup_steps=10,
        start_wd=0.1,
        end_wd=0.1,
    )
    megatron_worker_config = MegatronWorkerConfig(
        model=model_config,
        optimizer=optimizer_config,
        lr_scheduler=lr_scheduler_config,
        tp_size=1,
        cp_size=4,
        pp_size=1,
        ep_size=4,
        etp_size=1,
        dp_size=2,
        vpp_size=None,
        log_every_k_steps=1,
        global_batch_size=128,
        train_micro_batch_size=1,
        eval_micro_batch_size=1,
        log_gpu_usaegs=True,
        bf16=True,
        fp16=False,
        apply_rope_fusion=True,
        enable_fp32_lm_head=False,
        shuffle_train_data=True,
        recompute_granularity="full",
        recompute_method="uniform",
        recompute_num_layers=1,
    )

    sampling_config = SamplingConfig(temperature=1.0, max_total_tokens=max_length, top_p=1.0)
    rollout_worker_config = RolloutWorkerConfig(
        model=ModelConfig(name=model_name, seq_length=max_length, trust_remote_code=True),
        sampling_config=sampling_config,
        dp_size=1,
        num_workers=1,
        tp_size=8,
        pp_size=1,
        ep_size=8,
        gpu_memory_utilization=0.7,
        max_running_requests=2000,
        max_num_batched_tokens=2 * max_length,
        attention_backend=None,
        dtype="auto",
        enable_fp32_lm_head=True,
    )

    grpo = GrpoTrainerConfig(
        clip_ratio_high=0.28,
        clip_ratio_low=0.2,
        loss_type="grpo2",
        loss_agg_type="token-mean",
        dual_clip_neg_adv_factor=10,
        dual_soft_clip=3,
        is_base_logprobs="rollout_logprobs",
        kl_control_alpha=0.1,
        kl_base_logprobs="old_logprobs",
    )

    controller = ControllerConfig(
        run_mode="mismatch_test",
        colocated=colocated,
        num_rollout_actors=16,
        max_running_requests=rollout_worker_config.max_running_requests,
    )
    online_rl_train = OnlineRLTrainConfig(
        eval_on_start=True,
        num_rollouts_per_conversation=8,
        model_sync_every_n_global_updates=8,
        checkpoint_every_n_global_updates=1024,
        eval_every_n_global_updates=128,
        batch_rollout_for_n_global_updates=8,
        max_global_updates=30000,
        strict_on_policy=False,
        reward_mean_type="group",
        reward_std_type="group",
        sample_type="uniform",
        filter_zero_std=False,
        sort_sampled_prompts_by_response_length=True,
        rollout_save_filename="valid_rollouts",
        rollout_save_every_n_global_updates=None,
        save_all_rollouts=False,
    )

    mismatch_test = MismatchTestConfig(
        name=run_name,
        output_dir=output_dir,
        override_rollouts_if_exists=True,
        baseline_name=baseline_name,
    )
    logger_config = MetricLoggerConfig(
        logger_type="tensorboard",
        name="main-process",
        group_name=run_name,
        project_name=project_name,
    )
    return PipelineExperimentConfig(
        controller=controller,
        megatron_worker=megatron_worker_config,
        rollout_worker=rollout_worker_config,
        train_datasets=[DatasetConfig(name="BytedTsinghua-SIA/DAPO-Math-17k")],
        test_datasets=[DatasetConfig(name="BytedTsinghua-SIA/AIME-2024", eval_num_rollouts_per_prompt=128)],
        logger=logger_config,
        grpo=grpo,
        online_rl_train=online_rl_train,
        eval_sampling_config=SamplingConfig(temperature=0.7, top_p=0.9, max_total_tokens=max_length * 2),
        train_sampling_config=sampling_config,
        mismatch_test=mismatch_test,
    )
