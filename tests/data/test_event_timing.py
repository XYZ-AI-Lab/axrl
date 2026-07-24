import logging

import pytest

from axrl.data import EventTiming, GenerationOutput, array_utils
from axrl.metrics.response_metric import ResponseMetricCalculator
from axrl.worker.infer_router import _warn_if_event_timing_slow


def _generation_output(event_timing: EventTiming) -> GenerationOutput:
    return GenerationOutput(
        session_id="session-1",
        output_ids=array_utils.as_i32([1, 2, 3]),
        output_logprobs=array_utils.as_f32([0.0, 0.0, 0.0]),
        output_text="ok",
        output_text_with_special_tokens="ok",
        cached_tokens=0,
        finish_reason="stop",
        e2e_elapsed_seconds=0.2,
        stop_reason=None,
        retry=0,
        event_timing=event_timing,
    )


def test_event_timing_tracks_standard_intervals() -> None:
    timing = EventTiming()
    timing.mark_scheduled(10.0)
    timing.mark_worker_received(10.5)
    timing.mark_worker_returned(12.0)
    timing.mark_driver_received(13.25)

    assert timing.schedule_to_worker_seconds == pytest.approx(0.5)
    assert timing.worker_runtime_seconds == pytest.approx(1.5)
    assert timing.worker_return_to_driver_seconds == pytest.approx(1.25)
    assert timing.schedule_to_driver_seconds == pytest.approx(3.25)
    assert timing.driver_worker_overhead_seconds == pytest.approx(1.75)


def test_response_metric_includes_driver_worker_overhead_and_router_warns(caplog: pytest.LogCaptureFixture) -> None:
    timing = EventTiming(scheduled_at=1.0, worker_received_at=1.1, worker_returned_at=2.0, driver_received_at=6.5)

    metric = ResponseMetricCalculator().process(_generation_output(timing))
    with caplog.at_level(logging.WARNING):
        _warn_if_event_timing_slow("session-1", timing)

    assert metric.driver_worker_overhead_seconds == pytest.approx(4.6)
    assert "Generation routing overhead is 4.600s" in caplog.text


def test_router_warning_ignores_cross_node_clock_skew(caplog: pytest.LogCaptureFixture) -> None:
    timing = EventTiming(
        scheduled_at=100.0,
        worker_received_at=65.1,
        worker_returned_at=66.0,
        driver_received_at=102.0,
    )

    with caplog.at_level(logging.WARNING):
        _warn_if_event_timing_slow("session-1", timing)

    assert timing.worker_return_to_driver_seconds == pytest.approx(36.0)
    assert timing.driver_worker_overhead_seconds == pytest.approx(1.1)
    assert "Generation routing overhead" not in caplog.text
