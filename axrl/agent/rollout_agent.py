from __future__ import annotations

import logging
from typing import TYPE_CHECKING, override

from axrl.agent.base_agent import BaseAgent
from axrl.data import Conversation, GenerationInput, GenerationOutput, array_utils
from axrl.utils.timer import SessionTimer

if TYPE_CHECKING:
    from axrl.configs import SamplingConfig
    from axrl.ray.ray_rollout_worker import RayRolloutWorker

logger = logging.getLogger(__name__)


class RolloutAgent(BaseAgent):
    def __init__(
        self,
        rollout_worker: RayRolloutWorker,
    ) -> None:
        super().__init__()
        self.rollout_worker = rollout_worker

    @override
    async def act(self, conversation: Conversation, config: SamplingConfig) -> GenerationOutput:
        output: GenerationOutput | None = None
        gen_state = conversation.gen_state
        assert gen_state.input_ids is not None
        assert gen_state.session_id, "Conversation gen_state.session_id must be assigned before rollout generation."
        sampling_config = gen_state.sampling_config or config
        req = GenerationInput(
            session_id=gen_state.session_id,
            input_ids=array_utils.as_i32(gen_state.input_ids),
            sampling_config=sampling_config,
            tools=gen_state.tools,
            tool_choice=gen_state.tool_choice,
            tool_call_parser=gen_state.tool_call_parser,
            capture_routing=gen_state.capture_routing,
            routed_expert_start_index=gen_state.captured_routing_rows,
        )
        with SessionTimer(gen_state.session_id, "async", "Rollout generation"):
            output = await self.rollout_worker.generate(req)
        assert output is not None and len(output.output_ids) > 0, "Rollout generation failed to produce output_ids."
        return output
