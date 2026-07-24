from __future__ import annotations

import signal
import threading
from contextlib import contextmanager
from typing import Any


@contextmanager
def temporary_signal_patch() -> Any:
    """Temporarily bypass signal handler registration when not running in main thread.

    sglang registers custom signal handlers in Engine.__init__ which raises a ValueError
    if executed from a non-main thread (can happen inside some Ray actor startup paths).
    We monkeypatch signal.signal during engine construction to silently skip registrations
    outside the main thread, then restore the original function.

    Related issue on Github: https://github.com/sgl-project/sglang/issues/4319
    Related patch in verl: https://github.com/volcengine/verl/blob/4ed7811813fb7a42c71bc7b20eb2c6d58c4754a4/verl/workers/rollout/sglang_rollout/sglang_rollout.py#L88
    """
    original_signal = signal.signal

    def safe_signal(sig: int, handler: Any) -> Any:
        if threading.current_thread() is not threading.main_thread():
            return handler  # Skip registration; return handler for compatibility
        return original_signal(sig, handler)

    signal.signal = safe_signal  # type: ignore
    try:
        yield
    finally:
        signal.signal = original_signal  # type: ignore
