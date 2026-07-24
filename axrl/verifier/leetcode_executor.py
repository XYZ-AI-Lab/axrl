from __future__ import annotations

import builtins
import contextlib
import io
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from collections.abc import Iterator
    from types import FrameType

SUBPROCESS_RESULT_PREFIX = "__AXRL_LEETCODE_RESULT__ "


def run_unsafe_execute_subprocess() -> None:
    original_stdout = sys.__stdout__ if sys.__stdout__ is not None else sys.stdout
    try:
        payload = json.loads(sys.stdin.read())
        problem = payload["problem"]
        result = unsafe_execute(
            prompt=str(problem["prompt"]),
            completion=str(payload["completion"]),
            test=str(problem["test"]),
            entry_point=str(problem["entry_point"]),
            timeout=float(payload["timeout"]),
            memory_limit_bytes=None if payload["memory_limit_bytes"] is None else int(payload["memory_limit_bytes"]),
        )
    except BaseException as exc:
        result = f"failed: subprocess wrapper: {exc}"

    encoded = json.dumps(result, ensure_ascii=False)
    original_stdout.write(f"{SUBPROCESS_RESULT_PREFIX}{encoded}\n")
    original_stdout.flush()


def unsafe_execute(
    *,
    prompt: str,
    completion: str,
    test: str,
    entry_point: str,
    timeout: float,
    memory_limit_bytes: int | None,
) -> str:
    # Adapted from the upstream LeetCodeDataset execution harness:
    # https://github.com/newfacade/LeetCodeDataset/blob/main/eval_lcd/execution.py#L12-L51
    # The same pattern descends from OpenAI HumanEval's executor:
    # https://github.com/openai/human-eval/blob/master/human_eval/execution.py#L14-L58
    with tempfile.TemporaryDirectory() as temp_dir:
        cwd = Path.cwd()
        rmtree = shutil.rmtree
        rmdir = os.rmdir
        chdir = os.chdir
        try:
            os.chdir(temp_dir)
            reliability_guard(maximum_memory_bytes=memory_limit_bytes)
            check_program = "\n".join(
                [
                    prompt,
                    completion,
                    test,
                    f"check({entry_point})",
                ]
            )
            exec_globals: dict[str, Any] = {}
            with swallow_io(), time_limit(timeout):
                exec(check_program, exec_globals)  # noqa: S102
            return "passed"
        except TimeoutError:
            return "timed out"
        except BaseException as exc:
            message = str(exc).strip() or type(exc).__name__
            return f"failed: {message}"
        finally:
            shutil.rmtree = rmtree
            os.rmdir = rmdir
            os.chdir = chdir
            os.chdir(cwd)


@contextlib.contextmanager
def time_limit(seconds: float) -> Iterator[None]:
    # Reference: https://github.com/newfacade/LeetCodeDataset/blob/main/eval_lcd/execution.py#L85-L96
    def _signal_handler(_signum: int, _frame: FrameType | None) -> None:
        raise TimeoutError("Timed out")

    old_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _signal_handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)


@contextlib.contextmanager
def swallow_io() -> Iterator[None]:
    # Reference: https://github.com/newfacade/LeetCodeDataset/blob/main/eval_lcd/execution.py#L98-L132
    stream = _WriteOnlyStringIO()
    with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream), redirect_stdin(stream):
        yield


@contextlib.contextmanager
def redirect_stdin(stream: io.StringIO) -> Iterator[None]:
    old_stdin = sys.stdin
    sys.stdin = stream
    try:
        yield
    finally:
        sys.stdin = old_stdin


class _WriteOnlyStringIO(io.StringIO):
    def read(self, _size: int | None = -1, /) -> str:
        raise OSError("Read disabled")

    def readline(self, _size: int | None = -1, /) -> str:  # type: ignore[override]
        raise OSError("Read disabled")

    def readlines(self, _hint: int = -1, /) -> list[str]:  # type: ignore[override]
        raise OSError("Read disabled")

    def readable(self) -> bool:
        return False


def reliability_guard(maximum_memory_bytes: int | None = None) -> None:
    # Reference: https://github.com/newfacade/LeetCodeDataset/blob/main/eval_lcd/execution.py#L154-L240
    if maximum_memory_bytes is not None:
        import resource

        set_resource_limit(resource.RLIMIT_AS, maximum_memory_bytes)
        set_resource_limit(resource.RLIMIT_DATA, maximum_memory_bytes)
        if sys.platform != "darwin":
            set_resource_limit(resource.RLIMIT_STACK, maximum_memory_bytes)

    for name in ("exit", "quit", "help"):
        setattr(builtins, name, None)
    os.environ["OMP_NUM_THREADS"] = "1"

    for name in (
        "kill",
        "system",
        "putenv",
        "remove",
        "removedirs",
        "rmdir",
        "fchdir",
        "setuid",
        "fork",
        "forkpty",
        "killpg",
        "rename",
        "renames",
        "truncate",
        "replace",
        "unlink",
        "fchmod",
        "fchown",
        "chmod",
        "chown",
        "chroot",
        "lchflags",
        "lchmod",
        "lchown",
        "getcwd",
        "chdir",
    ):
        if hasattr(os, name):
            setattr(os, name, None)

    for name in ("rmtree", "move", "chown"):
        setattr(shutil, name, None)
    for name in ("Popen",):
        setattr(subprocess, name, None)

    for name in ("ipdb", "joblib", "psutil", "tkinter"):
        sys.modules[name] = cast("Any", None)


def set_resource_limit(resource_id: int, maximum_bytes: int) -> None:
    import resource

    _soft, hard = resource.getrlimit(resource_id)
    limit = maximum_bytes if hard == resource.RLIM_INFINITY else min(maximum_bytes, hard)
    resource.setrlimit(resource_id, (limit, limit))
