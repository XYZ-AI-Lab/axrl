from __future__ import annotations

import ast
import asyncio
import json
import logging
import re
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any, override

from e2b import AsyncSandbox, CommandExitException

from axrl.runner.e2b_runner import E2BRunnerConfig, resolve_e2b_api_key
from axrl.verifier import leetcode_executor
from axrl.verifier.base_verifier import BaseVerifier, VerifierInput, VerifierOutput

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_SECONDS = 3.0
DEFAULT_MEMORY_LIMIT_BYTES = 16 * 1024 * 1024 * 1024
_SUBPROCESS_TIMEOUT_GRACE_SECONDS = 1.0
_SUBPROCESS_ENTRYPOINT = "from axrl.verifier.leetcode_executor import run_unsafe_execute_subprocess; run_unsafe_execute_subprocess()"
_EXECUTOR_SOURCE_PATH = Path(leetcode_executor.__file__).resolve()
_E2B_SUBPROCESS_PYTHON = "python"
_E2B_WORKDIR = "/tmp/axrl-leetcode"  # noqa: S108 - this path is inside the remote E2B sandbox.
_E2B_PAYLOAD_PATH = f"{_E2B_WORKDIR}/payload.json"
_E2B_REPO_ROOT = "/tmp/axrl-verifier-src"  # noqa: S108 - this path is inside the remote E2B sandbox.
_E2B_EXECUTOR_PATH = f"{_E2B_REPO_ROOT}/axrl/verifier/leetcode_executor.py"
_PYTHON_FENCE_RE = re.compile(r"```(?:python|py)\s*\n(.*?)```", re.IGNORECASE | re.DOTALL)
_ANY_FENCE_RE = re.compile(r"```\s*\n(.*?)```", re.DOTALL)
_CODE_STARTERS = ("import ", "from ", "class ", "def ")

# Code extraction follows newfacade/LeetCodeDataset's two-step shape:
# 1. prefer fenced python blocks: https://github.com/newfacade/LeetCodeDataset/blob/main/eval_lcd/data.py#L58-L69
# 2. fall back to syntax-valid spans: https://github.com/newfacade/LeetCodeDataset/blob/main/eval_lcd/data.py#L72-L94


@dataclass(frozen=True)
class LeetCodeProblem:
    task_id: str
    prompt: str
    test: str
    entry_point: str


@dataclass(frozen=True)
class LeetCodeExecutionResult:
    passed: bool
    result: str
    completion: str


def extract_leetcode_code(output_text: str) -> str:
    blocks = _PYTHON_FENCE_RE.findall(output_text) or _ANY_FENCE_RE.findall(output_text)
    if blocks:
        return max(blocks, key=_nonempty_line_count).strip()

    syntax_valid_span = _longest_syntax_valid_span(output_text)
    if syntax_valid_span is not None:
        return syntax_valid_span.strip()

    return output_text.strip()


class LeetCodeVerifier(BaseVerifier):
    def __init__(self, config: Any | None = None) -> None:
        super().__init__(config=config)
        _validate_e2b_only_runner(config)
        self.timeout = _get_timeout(config)
        self.memory_limit_bytes = _get_memory_limit_bytes(config)
        self.e2b_config = _get_e2b_config(config)

    @override
    def verify(self, label: str | list[str], output_text: str, *, verbose: bool = False) -> float:
        result = self.check(label=label, output_text=output_text)
        if verbose:
            logger.info(f"LeetCodeVerifier: score={1.0 if result.passed else 0.0}, result={result.result!r}, completion={result.completion!r}")
        return 1.0 if result.passed else 0.0

    @override
    def process(self, item: VerifierInput) -> VerifierOutput:
        result = self.check(label=item.label, output_text=item.output_text)
        return VerifierOutput(
            score=1.0 if result.passed else 0.0,
            infos={
                "result": result.result,
                "completion": result.completion,
            },
        )

    def check(self, label: str | list[str], output_text: str) -> LeetCodeExecutionResult:
        assert isinstance(label, str)
        completion = extract_leetcode_code(output_text)
        try:
            problem = _load_problem(label)
        except Exception as exc:
            logger.warning("Failed to parse LeetCode verifier label: %s", exc)
            return LeetCodeExecutionResult(passed=False, result=f"failed: invalid label: {exc}", completion=completion)

        try:
            result = check_leetcode_correctness(
                problem=problem,
                completion=completion,
                timeout=self.timeout,
                memory_limit_bytes=self.memory_limit_bytes,
                e2b_config=self.e2b_config,
            )
        except Exception as exc:
            result = _format_verifier_infra_failure(exc)
            logger.warning("LeetCode verifier infrastructure failed for task %s: %s", problem.task_id, result)
        return LeetCodeExecutionResult(passed=result == "passed", result=result, completion=completion)


def check_leetcode_correctness(
    *,
    problem: LeetCodeProblem,
    completion: str,
    timeout: float,
    memory_limit_bytes: int | None = DEFAULT_MEMORY_LIMIT_BYTES,
    e2b_config: E2BRunnerConfig | None = None,
) -> str:
    # Mirrors the upstream subprocess timeout wrapper:
    # https://github.com/newfacade/LeetCodeDataset/blob/main/eval_lcd/execution.py#L54-L82
    payload = json.dumps(
        {
            "problem": {
                "task_id": problem.task_id,
                "prompt": problem.prompt,
                "test": problem.test,
                "entry_point": problem.entry_point,
            },
            "completion": completion,
            "timeout": timeout,
            "memory_limit_bytes": memory_limit_bytes,
        },
        ensure_ascii=False,
    )
    return _run_subprocess_in_e2b(payload=payload, timeout=timeout, e2b_config=e2b_config)


def _get_timeout(config: Any | None) -> float:
    if config is None:
        return DEFAULT_TIMEOUT_SECONDS
    if isinstance(config, dict):
        timeout = config.get("max_timeout", config.get("timeout", DEFAULT_TIMEOUT_SECONDS))
    else:
        timeout = getattr(config, "max_timeout", getattr(config, "timeout", DEFAULT_TIMEOUT_SECONDS))
    return max(float(timeout), 0.001)


def _get_memory_limit_bytes(config: Any | None) -> int | None:
    if config is None:
        return DEFAULT_MEMORY_LIMIT_BYTES
    if isinstance(config, dict):
        value = config.get("memory_limit_bytes", DEFAULT_MEMORY_LIMIT_BYTES)
    else:
        value = getattr(config, "memory_limit_bytes", DEFAULT_MEMORY_LIMIT_BYTES)
    if value is None:
        return None
    value = int(value)
    if value <= 0:
        raise ValueError("LeetCode verifier memory limit must be greater than zero.")
    return value


def _validate_e2b_only_runner(config: Any | None) -> None:
    if config is None:
        return
    if isinstance(config, dict):
        runner = config.get("runner", "e2b")
    else:
        runner = getattr(config, "runner", "e2b")
    if runner != "e2b":
        raise ValueError(f"The blackbox LeetCode verifier only supports runner='e2b', got {runner!r}.")


def _get_e2b_config(config: Any | None) -> E2BRunnerConfig:
    if config is None:
        return E2BRunnerConfig()
    if isinstance(config, dict):
        value = config.get("e2b")
    else:
        value = getattr(config, "e2b", None)
    if value is None:
        return E2BRunnerConfig()
    if isinstance(value, E2BRunnerConfig):
        return value
    if isinstance(value, dict):
        return E2BRunnerConfig(**value)
    raise TypeError(f"LeetCode verifier e2b config must be a dict or E2BRunnerConfig, got {type(value).__name__}.")


def _load_problem(label: str) -> LeetCodeProblem:
    data = json.loads(label)
    return LeetCodeProblem(
        task_id=str(data["task_id"]),
        prompt=str(data["prompt"]),
        test=str(data["test"]),
        entry_point=str(data["entry_point"]),
    )


def _nonempty_line_count(code: str) -> int:
    return sum(1 for line in code.splitlines() if line.strip())


def _is_syntax_valid(code: str) -> bool:
    try:
        ast.parse(code)
    except (SyntaxError, MemoryError):
        return False
    return True


def _longest_syntax_valid_span(text: str) -> str | None:
    lines = text.splitlines()
    best: str | None = None
    best_length = 0
    for start, line in enumerate(lines):
        if not line.lstrip().startswith(_CODE_STARTERS):
            continue
        for end in range(len(lines), start, -1):
            code = "\n".join(lines[start:end])
            line_count = _nonempty_line_count(code)
            if line_count <= best_length:
                continue
            if _is_syntax_valid(code):
                best = code
                best_length = line_count
                break
    return best


def _run_subprocess_in_e2b(*, payload: str, timeout: float, e2b_config: E2BRunnerConfig | None) -> str:
    return asyncio.run(_run_subprocess_in_e2b_async(payload=payload, timeout_seconds=timeout, e2b_config=e2b_config or E2BRunnerConfig()))


def _format_verifier_infra_failure(exc: BaseException) -> str:
    message = str(exc).replace("\n", " ").strip()
    if len(message) > 1000:
        message = f"{message[:1000]}..."
    return f"failed: verifier infrastructure error: {type(exc).__name__}: {message}"


async def _run_subprocess_in_e2b_async(*, payload: str, timeout_seconds: float, e2b_config: E2BRunnerConfig) -> str:
    sandbox = await AsyncSandbox.create(
        template=e2b_config.template,
        timeout=e2b_config.timeout_seconds,
        metadata={"axrl_runner": "leetcode-verifier", **e2b_config.metadata},
        envs=e2b_config.envs,
        secure=e2b_config.secure,
        allow_internet_access=False,
        api_key=resolve_e2b_api_key(e2b_config),
        request_timeout=e2b_config.request_timeout_seconds,
    )
    try:
        await _write_e2b_verifier_files(sandbox=sandbox, payload=payload, e2b_config=e2b_config)
        return await _run_e2b_verifier_command(sandbox=sandbox, timeout_seconds=timeout_seconds, e2b_config=e2b_config)
    finally:
        await sandbox.kill(request_timeout=e2b_config.request_timeout_seconds)


async def _write_e2b_verifier_files(*, sandbox: AsyncSandbox, payload: str, e2b_config: E2BRunnerConfig) -> None:
    request_timeout = e2b_config.request_timeout_seconds
    user = e2b_config.user
    await sandbox.files.make_dir(_E2B_WORKDIR, user=user, request_timeout=request_timeout)
    await sandbox.files.make_dir(f"{_E2B_REPO_ROOT}/axrl", user=user, request_timeout=request_timeout)
    await sandbox.files.make_dir(f"{_E2B_REPO_ROOT}/axrl/verifier", user=user, request_timeout=request_timeout)
    await sandbox.files.write(_E2B_PAYLOAD_PATH, payload, user=user, request_timeout=request_timeout)
    await sandbox.files.write(f"{_E2B_REPO_ROOT}/axrl/__init__.py", "", user=user, request_timeout=request_timeout)
    await sandbox.files.write(f"{_E2B_REPO_ROOT}/axrl/verifier/__init__.py", "", user=user, request_timeout=request_timeout)
    await sandbox.files.write(
        _E2B_EXECUTOR_PATH,
        _EXECUTOR_SOURCE_PATH.read_text(encoding="utf-8"),
        user=user,
        request_timeout=request_timeout,
    )


async def _run_e2b_verifier_command(*, sandbox: AsyncSandbox, timeout_seconds: float, e2b_config: E2BRunnerConfig) -> str:
    command = _e2b_subprocess_command(_E2B_PAYLOAD_PATH)
    try:
        result = await sandbox.commands.run(
            command,
            cwd=_E2B_WORKDIR,
            user=e2b_config.user,
            timeout=timeout_seconds + _SUBPROCESS_TIMEOUT_GRACE_SECONDS,
            request_timeout=e2b_config.request_timeout_seconds,
        )
        returncode = int(result.exit_code)
        stdout = str(result.stdout)
    except TimeoutError:
        return "timed out"
    except CommandExitException as exc:
        returncode = int(exc.exit_code)
        stdout = str(exc.stdout)
        if exc.stderr:
            stdout = f"{stdout}\n{exc.stderr}"
    return _parse_subprocess_output(returncode=returncode, stdout=stdout)


def _e2b_subprocess_command(payload_path: str) -> str:
    command = [
        f"PYTHONPATH={shlex.quote(_E2B_REPO_ROOT)}",
        "exec",
        shlex.quote(_E2B_SUBPROCESS_PYTHON),
        "-c",
        shlex.quote(_SUBPROCESS_ENTRYPOINT),
        "<",
        shlex.quote(payload_path),
    ]
    return " ".join(command)


def _parse_subprocess_output(*, returncode: int, stdout: str) -> str:
    for line in reversed(stdout.splitlines()):
        if not line.startswith(leetcode_executor.SUBPROCESS_RESULT_PREFIX):
            continue
        try:
            return str(json.loads(line[len(leetcode_executor.SUBPROCESS_RESULT_PREFIX) :]))
        except json.JSONDecodeError as exc:
            return f"failed: invalid subprocess result: {exc}"

    if returncode != 0:
        detail = f": {stdout.strip()[-1000:]}" if stdout.strip() else ""
        return f"failed: subprocess exited with code {returncode}{detail}"
    return "failed: subprocess produced no result"
