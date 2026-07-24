from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, override

from axrl.agent.rollout_agent import RolloutAgent
from axrl.envs.math_env import MathEnv
from axrl.metrics.response_metric import ResponseMetricCalculator
from axrl.pipeline.config import PipelineExperimentConfig
from axrl.pipeline.controller import PipelineController
from axrl.pipeline.utils import rollout_from_env
from axrl.processor.conversation_tokenizer import ConversationTokenizer
from axrl.processor.processor_pool import ProcessorPool
from axrl.recipe.base_recipe import BaseRecipe
from axrl.utils import setup_logger
from axrl.utils.config_utils import load_and_validate_config
from axrl.verifier.gsm8k import GSM8KVerifier

if TYPE_CHECKING:
    from axrl.data import Conversation, GenerationInput, RolloutResult
    from axrl.pipeline.rollout_data import RolloutRuntime
    from axrl.worker.infer_worker import InferWorker


class GSM8KPipelineRecipe(BaseRecipe):
    @override
    def initialize_local_processors(self, worker_id: str) -> dict[str, InferWorker[Any, Any]]:
        assert worker_id, "worker_id must be non-empty when initializing GSM8K rollout workers."
        return {
            "verifier": ProcessorPool(GSM8KVerifier, config=None, num_processors=1),
            "metric": ProcessorPool(ResponseMetricCalculator, config=None, num_processors=1),
            "conv_tokenizer": ProcessorPool(ConversationTokenizer, config=self.config.rollout_worker.model, num_processors=1),
        }

    @override
    async def run_rollout(self, conversation: Conversation, runtime: RolloutRuntime) -> RolloutResult:
        session_id = conversation.gen_state.session_id or conversation.conversation_id
        sampling_config = conversation.gen_state.sampling_config
        assert sampling_config is not None, f"Rollout conversation {session_id!r} is missing gen_state.sampling_config."
        assert "answer" in conversation.extra, f"Rollout conversation {session_id!r} is missing extra['answer']."

        conv_tokenizer = runtime.get_local_worker("conv_tokenizer")
        gen_input: GenerationInput = await conv_tokenizer.generate(conversation)
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


async def run_controller() -> None:
    setup_logger("info")
    config = load_and_validate_config(
        PipelineExperimentConfig,
        config_path="axis_recipe/grpo_gsm8k/pipeline_config.yaml",
        print_configs=True,
    )
    controller = PipelineController(config, GSM8KPipelineRecipe(config))
    await controller.start()


if __name__ == "__main__":
    asyncio.run(run_controller())
