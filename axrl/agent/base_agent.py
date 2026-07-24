import logging
from typing import Any

from axrl.data import Conversation

logger = logging.getLogger(__name__)


class BaseAgent:
    def __init__(self) -> None:
        pass

    async def act(self, conversation: Conversation, config: Any) -> Any:
        """Action method to be implemented by subclasses."""
        raise NotImplementedError
