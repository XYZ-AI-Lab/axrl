from __future__ import annotations

import asyncio
import codecs
import json
import logging
import shlex
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from axis_recipe.blackbox_rl.config import OpenHandsLauncherConfig

logger = logging.getLogger(__name__)

_STDOUT_READ_CHUNK_BYTES = 64 * 1024
_MAX_PENDING_STDOUT_LINE_CHARS = 5_000_000
_MAX_STORED_STDOUT_LINE_CHARS = 200_000


def get_openhand_launch_script(
    config: OpenHandsLauncherConfig,
    *,
    task: str,
    llm_base_url: str,
    llm_model: str,
) -> str:
    argv = [
        *config.command,
        "--headless",
        "--json",
        "--always-approve",
        "--override-with-envs",
        "-t",
        task,
    ]
    env = _build_env(config, llm_base_url=llm_base_url, llm_model=llm_model)
    command = f"env {_env_args(env)} {shlex.join(argv)}"
    setup_script = _sandbox_openhands_setup_script(home=env.get("HOME"))
    if setup_script:
        return f"{setup_script}\nexec {command}"
    return f"exec {command}"


async def read_stdout(
    stdout: Any,
    *,
    session_id: str,
    stdout_lines: list[str],
    json_events: list[dict[str, Any]],
) -> None:
    decoder = codecs.getincrementaldecoder("utf-8")("replace")
    line_buffer = ""
    json_buffer: list[str] | None = None
    while True:
        chunk = await _read_stdout_chunk(stdout, _STDOUT_READ_CHUNK_BYTES)
        if not chunk:
            break
        line_buffer += decoder.decode(chunk)
        while "\n" in line_buffer:
            line, line_buffer = line_buffer.split("\n", 1)
            json_buffer = _handle_stdout_line(
                session_id=session_id,
                line=line.rstrip("\r"),
                json_buffer=json_buffer,
                stdout_lines=stdout_lines,
                json_events=json_events,
            )
        if len(line_buffer) <= _MAX_PENDING_STDOUT_LINE_CHARS:
            continue
        logger.warning(
            "OpenHands[%s] stdout line exceeded %d chars without newline; truncating buffered line.",
            session_id,
            _MAX_PENDING_STDOUT_LINE_CHARS,
        )
        json_buffer = _handle_stdout_line(
            session_id=session_id,
            line=_truncate_stdout_line(line_buffer),
            json_buffer=None,
            stdout_lines=stdout_lines,
            json_events=json_events,
        )
        line_buffer = ""
    line_buffer += decoder.decode(b"", final=True)
    if line_buffer:
        _handle_stdout_line(
            session_id=session_id,
            line=line_buffer.rstrip("\r"),
            json_buffer=json_buffer,
            stdout_lines=stdout_lines,
            json_events=json_events,
        )


def _build_env(config: OpenHandsLauncherConfig, *, llm_base_url: str, llm_model: str) -> dict[str, str]:
    env = dict(config.extra_env)
    if not config.load_public_skills:
        env["HOME"] = str(PurePosixPath(config.e2b.workdir) / ".openhands-home")
    env["LLM_MODEL"] = f"openai/{llm_model}" if not llm_model.startswith("openai/") else llm_model
    env["LLM_BASE_URL"] = llm_base_url
    env["LLM_API_KEY"] = config.api_key
    if config.llm_timeout_seconds is not None:
        env["LLM_TIMEOUT"] = f"{config.llm_timeout_seconds:g}"
    if config.suppress_banner:
        env["OPENHANDS_SUPPRESS_BANNER"] = "1"
    return env


def _sandbox_openhands_setup_script(*, home: str | None) -> str:
    if not home:
        return ""
    marketplace_json = json.dumps(
        {
            "name": "axrl-empty",
            "owner": {"name": "AXRL"},
            "plugins": [],
            "skills": [],
        },
        separators=(",", ":"),
    )
    return "\n".join(
        [
            "set -euo pipefail",
            f"export HOME={shlex.quote(home)}",
            'cache_dir="${HOME}/.openhands/cache/skills"',
            'repo_path="${cache_dir}/public-skills"',
            'marketplace_path="${repo_path}/marketplaces/default.json"',
            'if [ ! -d "${repo_path}/.git" ] || [ ! -f "${marketplace_path}" ]; then',
            '  rm -rf "${repo_path}"',
            '  mkdir -p "${repo_path}/marketplaces" "${repo_path}/skills"',
            f"  printf '%s\\n' {shlex.quote(marketplace_json)} > \"${{marketplace_path}}\"",
            '  : > "${repo_path}/skills/.gitkeep"',
            '  git -C "${repo_path}" init -b main >/dev/null',
            '  git -C "${repo_path}" config user.email axrl@example.invalid',
            '  git -C "${repo_path}" config user.name AXRL',
            '  git -C "${repo_path}" add marketplaces/default.json skills/.gitkeep',
            '  git -C "${repo_path}" commit -m "Initialize empty public skills cache" >/dev/null',
            "fi",
        ]
    )


async def _read_stdout_chunk(stdout: Any, size: int) -> bytes:
    if isinstance(stdout, asyncio.StreamReader):
        return await stdout.read(size)
    return await asyncio.to_thread(stdout.read, size)


def _handle_stdout_line(
    *,
    session_id: str,
    line: str,
    json_buffer: list[str] | None,
    stdout_lines: list[str],
    json_events: list[dict[str, Any]],
) -> list[str] | None:
    stdout_lines.append(_truncate_stdout_line(line))
    if line == "--JSON Event--":
        return []
    if json_buffer is not None:
        json_buffer.append(line)
        try:
            event = json.loads("\n".join(json_buffer))
        except json.JSONDecodeError:
            if sum(len(part) for part in json_buffer) > _MAX_PENDING_STDOUT_LINE_CHARS:
                logger.warning("OpenHands[%s] JSON event block exceeded parse buffer; dropping block", session_id)
                return None
            return json_buffer
        if isinstance(event, dict):
            json_events.append(event)
        return None
    try:
        event = json.loads(line)
    except json.JSONDecodeError:
        logger.debug("OpenHands[%s]: %s", session_id, _truncate_stdout_line(line))
        return None
    if isinstance(event, dict):
        json_events.append(event)
    return None


def _truncate_stdout_line(line: str) -> str:
    if len(line) <= _MAX_STORED_STDOUT_LINE_CHARS:
        return line
    return f"{line[:_MAX_STORED_STDOUT_LINE_CHARS]}... [truncated stdout line; original_chars={len(line)}]"


def _env_args(env: dict[str, str]) -> str:
    return " ".join(shlex.quote(f"{key}={value}") for key, value in env.items())
