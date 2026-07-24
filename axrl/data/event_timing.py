from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass
class EventTiming:
    """Small timestamp carrier for async request/response handoffs."""

    scheduled_at: float | None = None
    worker_received_at: float | None = None
    worker_returned_at: float | None = None
    driver_received_at: float | None = None

    @staticmethod
    def now() -> float:
        return time.time()

    def mark_scheduled(self, timestamp: float | None = None) -> None:
        self.scheduled_at = self.now() if timestamp is None else timestamp

    def mark_worker_received(self, timestamp: float | None = None) -> None:
        self.worker_received_at = self.now() if timestamp is None else timestamp

    def mark_worker_returned(self, timestamp: float | None = None) -> None:
        self.worker_returned_at = self.now() if timestamp is None else timestamp

    def mark_driver_received(self, timestamp: float | None = None) -> None:
        self.driver_received_at = self.now() if timestamp is None else timestamp

    @staticmethod
    def _elapsed(start: float | None, end: float | None) -> float | None:
        if start is None or end is None:
            return None
        return max(0.0, end - start)

    @property
    def schedule_to_worker_seconds(self) -> float | None:
        return self._elapsed(self.scheduled_at, self.worker_received_at)

    @property
    def worker_runtime_seconds(self) -> float | None:
        return self._elapsed(self.worker_received_at, self.worker_returned_at)

    @property
    def worker_return_to_driver_seconds(self) -> float | None:
        return self._elapsed(self.worker_returned_at, self.driver_received_at)

    @property
    def schedule_to_driver_seconds(self) -> float | None:
        return self._elapsed(self.scheduled_at, self.driver_received_at)

    @property
    def driver_worker_overhead_seconds(self) -> float | None:
        """Elapsed time outside worker runtime, robust to cross-node clock skew.

        This is the aggregate driver->worker plus worker->driver handoff time:
        (driver_received - scheduled) - (worker_returned - worker_received).
        It does not try to split the two directions because that requires
        synchronized clocks or an extra symmetry assumption.
        """
        schedule_to_driver_seconds = self.schedule_to_driver_seconds
        worker_runtime_seconds = self.worker_runtime_seconds
        if schedule_to_driver_seconds is None or worker_runtime_seconds is None:
            return None
        return max(0.0, schedule_to_driver_seconds - worker_runtime_seconds)
