import copy
from typing import Any

from axrl.data import Conversation
from axrl.data.rollout_trace import RolloutTrace


class BaseEnv:
    def __init__(self, conv: Conversation) -> None:
        self.conv = copy.deepcopy(conv)

    async def step(self, action: Any) -> tuple[Conversation, float, bool, RolloutTrace | None, Any]:
        """Returns: observation, reward, done, info."""
        raise NotImplementedError
