import json
import logging
import time
from dataclasses import dataclass
from typing import Literal, Self

logger = logging.getLogger(__name__)


class Timer:
    def __init__(self, name: str = "Unnamed Timer", *, verbose: bool = False) -> None:
        self.name = name
        self._start_time: float
        self.elapsed_seconds: float
        self.verbose = verbose

    def start(self) -> None:
        self._start_time = time.perf_counter()
        logger.debug(f"Starting: [{self.name}]")
        if self.verbose:
            logger.info(f"Starting: [{self.name}]")

    def stop(self) -> None:
        end_time: float = time.perf_counter()
        self.elapsed_seconds = end_time - self._start_time
        logger.debug(f"Finished: [{self.name}] in {self.elapsed_seconds:.4f} seconds.")
        if self.verbose:
            logger.info(f"Finished: [{self.name}] in {self.elapsed_seconds:.4f} seconds.")

    def __repr__(self) -> str:
        return f"Timer(name={self.name}, elapsed_seconds={self.elapsed_seconds:.4f})"

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        self.stop()


ExecutionKind = Literal["sync", "async"]


@dataclass(frozen=True)
class SessionTimerData:
    session_id: str
    execution_kind: ExecutionKind
    function_name: str

    PREFIX = "SessionTimer: "

    def to_log_name(self) -> str:
        payload = {
            "session_id": self.session_id,
            "execution_kind": self.execution_kind,
            "function_name": self.function_name,
        }
        return self.PREFIX + json.dumps(payload, sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_log_name(cls, name: str) -> Self | None:
        if not name.startswith(cls.PREFIX):
            return None
        try:
            payload = json.loads(name[len(cls.PREFIX) :])
        except json.JSONDecodeError:
            logger.warning("Failed to parse SessionTimer payload: %s", name)
            return None

        session_id = payload.get("session_id")
        execution_kind = payload.get("execution_kind")
        function_name = payload.get("function_name")
        if execution_kind not in {"sync", "async"} or not isinstance(session_id, str) or not isinstance(function_name, str):
            logger.warning("Invalid SessionTimer payload: %s", payload)
            return None
        return cls(
            session_id=session_id,
            execution_kind=execution_kind,
            function_name=function_name,
        )


class SessionTimer(Timer):
    def __init__(
        self,
        session_id: str,
        execution_kind: ExecutionKind,
        function_name: str,
        *,
        verbose: bool = False,
    ) -> None:
        self.data = SessionTimerData(
            session_id=session_id,
            execution_kind=execution_kind,
            function_name=function_name,
        )
        super().__init__(self.data.to_log_name(), verbose=verbose)

    @staticmethod
    def parse_log_name(name: str) -> SessionTimerData | None:
        return SessionTimerData.from_log_name(name)
