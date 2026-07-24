from __future__ import annotations

import dataclasses
import html
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from axrl.data import RolloutResult


def write_openhands_case_report(
    result: RolloutResult,
    path: Path,
    *,
    token_decoder: Callable[[Sequence[int]], str] | None = None,
) -> None:
    metric = result.metric
    conv = result.conversation
    messages = [message.to_dict() for message in conv.messages]
    stdout_lines = [str(line) for line in conv.extra.get("openhands_stdout_lines", [])]
    openhands_events = conv.extra.get("openhands_json_events", [])
    model_events = conv.extra.get("openai_io_events") or conv.extra.get("sglang_io_events", [])
    summary = {
        "conversation_id": conv.conversation_id,
        "score": getattr(metric, "score", None),
        "num_model_calls": getattr(metric, "num_model_calls", None),
        "openhands_exit_code": getattr(metric, "openhands_exit_code", None),
        "num_turn_samples": len(result.trace.turn_samples) if result.trace is not None else 0,
        "openhands_events": len(openhands_events),
        "model_io_events": len(model_events),
    }
    timeline = _build_case_timeline(openhands_events=openhands_events, model_events=model_events)
    prefix_elider = _PrefixElider()
    generated_at = datetime.now().astimezone().isoformat(timespec="seconds")
    stdout_text = "\n".join(stdout_lines)
    metrics_html = "\n".join(
        f"<div class='metric'><span>{html.escape(str(key))}</span><strong>{html.escape(str(value))}</strong></div>" for key, value in summary.items()
    )
    timeline_html = "\n".join(_render_timeline_item(item, prefix_elider=prefix_elider) for item in timeline)
    turn_samples = result.trace.turn_samples if result.trace is not None else []
    decoded_samples_html = _render_decoded_turn_samples(turn_samples, token_decoder=token_decoder, prefix_elider=prefix_elider)
    missing_capture_html = ""
    if not model_events:
        missing_capture_html = """
<section class="card warn-card">
  <h3>Model I/O Capture Missing</h3>
  <p class="muted">
    This report was generated from a run that did not persist model-boundary input/output records.
    Re-run the rollout with the current capture code to populate model input/output cards in the timeline.
  </p>
</section>
"""
    body = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>OpenHands Black-Box RL I/O Case</title>
  <style>
    :root {{
      --bg: #f6f7f9;
      --panel: #ffffff;
      --ink: #18202b;
      --muted: #64748b;
      --line: #d8dee4;
      --openhands: #2563eb;
      --sglang-in: #7c3aed;
      --sglang-out: #059669;
      --warn: #b45309;
      --bad: #b91c1c;
      --code-bg: #f8fafc;
      --stripe: #eef2f7;
      --omitted-bg: #fff7ed;
      --omitted-border: #fb923c;
      --token-bg: #ecfeff;
      --token-border: #06b6d4;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      background: var(--bg);
      color: var(--ink);
      font-family: system-ui, sans-serif;
      line-height: 1.45;
      margin: 0;
    }}
    a {{ color: #0f766e; text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    .layout {{
      display: grid;
      grid-template-columns: 260px minmax(0, 1fr);
      min-height: 100vh;
    }}
    .sidebar {{
      align-self: start;
      background: #eef2f7;
      border-right: 1px solid var(--line);
      height: 100vh;
      overflow: auto;
      padding: 24px 18px;
      position: sticky;
      top: 0;
    }}
    .sidebar h1 {{ font-size: 18px; margin: 0 0 14px; }}
    .sidebar a {{ display: block; font-size: 14px; padding: 6px 0; }}
    .content {{
      max-width: 1360px;
      padding: 28px 34px 60px;
    }}
    .hero {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      margin-bottom: 18px;
      padding: 22px;
    }}
    h1, h2, h3 {{ line-height: 1.2; }}
    h1 {{ font-size: 28px; margin: 0 0 8px; }}
    h2 {{ font-size: 20px; margin: 28px 0 12px; }}
    h3 {{ font-size: 16px; margin: 0 0 10px; }}
    .muted {{ color: var(--muted); }}
    .grid {{ display: grid; gap: 12px; }}
    .cols {{ grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
    }}
    .metric {{
      border-bottom: 1px solid #edf0f5;
      display: flex;
      gap: 12px;
      justify-content: space-between;
      padding: 8px 0;
    }}
    .metric:last-child {{ border-bottom: 0; }}
    .badge {{
      background: #e2e8f0;
      border-radius: 999px;
      color: #334155;
      display: inline-block;
      font-size: 12px;
      font-weight: 800;
      padding: 2px 8px;
    }}
    .badge.openhands {{ background: #dbeafe; color: #1d4ed8; }}
    .badge.model-input,
    .badge.sglang-input {{ background: #ede9fe; color: #6d28d9; }}
    .badge.model-output,
    .badge.sglang-output {{ background: #dcfce7; color: #047857; }}
    .badge.warn {{ background: #fef3c7; color: var(--warn); }}
    .timeline {{ display: grid; gap: 12px; }}
    .sample-grid {{ display: grid; gap: 12px; }}
    .timeline-item {{
      border-left: 5px solid #64748b;
    }}
    .timeline-item.openhands {{ border-left-color: var(--openhands); }}
    .timeline-item.model-input,
    .timeline-item.sglang-input {{ border-left-color: var(--sglang-in); }}
    .timeline-item.model-output,
    .timeline-item.sglang-output {{ border-left-color: var(--sglang-out); }}
    .timeline-item.warn-event {{ border-left-color: var(--warn); }}
    .badge.warn-event {{ background: #fef3c7; color: var(--warn); }}
    .sample-turn {{ border-left: 5px solid #0f766e; }}
    .timeline-title {{
      align-items: center;
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-bottom: 8px;
    }}
    .timeline-title h3 {{ margin: 0; }}
    .kv-list {{
      display: grid;
      gap: 4px;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      list-style: none;
      margin: 10px 0 0;
      padding: 0;
    }}
    .kv-list li {{
      color: var(--muted);
      font-size: 13px;
      overflow-wrap: anywhere;
    }}
    .kv-list strong {{ color: var(--ink); }}
    .warn-card {{ border-left: 5px solid var(--warn); }}
    details {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      margin: 10px 0;
    }}
    details.warn-detail {{
      border-color: #f59e0b;
      border-left: 5px solid var(--warn);
    }}
    summary {{
      cursor: pointer;
      font-weight: 700;
      padding: 12px 14px;
    }}
    details.warn-detail summary {{
      background: #fffbeb;
      color: var(--warn);
    }}
    details > .details-body {{ padding: 0 14px 14px; }}
    pre {{
      background: var(--code-bg);
      border: 1px solid var(--line);
      border-radius: 6px;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
      font-size: 12px;
      line-height: 1.55;
      max-height: 620px;
      overflow: auto;
      overflow-wrap: anywhere;
      padding: 12px;
      white-space: pre-wrap;
      word-break: break-word;
    }}
    pre.striped-lines {{ padding: 0; }}
    .pre-line {{
      display: block;
      min-height: 1.55em;
      overflow-wrap: anywhere;
      padding: 0 12px;
      white-space: pre-wrap;
      word-break: break-word;
    }}
    .pre-line:first-child {{ padding-top: 12px; }}
    .pre-line:last-child {{ padding-bottom: 12px; }}
    .pre-line:nth-child(odd) {{ background: var(--code-bg); }}
    .pre-line:nth-child(even) {{ background: var(--stripe); }}
    .pre-line.omitted-content {{
      background: var(--omitted-bg) !important;
      border-left: 4px solid var(--omitted-border);
      color: #7c2d12;
      padding-left: 8px;
    }}
    .pre-line.token-summary {{
      background: var(--token-bg) !important;
      border-left: 4px solid var(--token-border);
      padding-left: 8px;
    }}
    .chat-boundary {{
      background: #fef3c7;
      border: 1px solid #f59e0b;
      border-radius: 4px;
      color: #78350f;
      display: inline-block;
      font-weight: 800;
      line-height: 1.25;
      margin: 0 2px;
      padding: 0 4px;
    }}
    .chat-boundary.role-system {{ background: #dbeafe; border-color: #60a5fa; color: #1e3a8a; }}
    .chat-boundary.role-user {{ background: #dcfce7; border-color: #4ade80; color: #14532d; }}
    .chat-boundary.role-assistant {{ background: #ede9fe; border-color: #a78bfa; color: #4c1d95; }}
    .chat-boundary.role-tool {{ background: #fee2e2; border-color: #f87171; color: #7f1d1d; }}
    @media (max-width: 900px) {{
      .layout {{ display: block; }}
      .sidebar {{ height: auto; position: static; }}
      .content {{ padding: 20px; }}
    }}
  </style>
  <script>
    document.addEventListener("DOMContentLoaded", () => {{
      stripePreBlocks();
      highlightChatBoundaries();
    }});
    function stripePreBlocks() {{
      for (const pre of document.querySelectorAll("pre")) {{
        if (pre.dataset.striped === "true") {{
          continue;
        }}
        const lines = pre.textContent.split("\\n");
        pre.textContent = "";
        pre.classList.add("striped-lines");
        pre.dataset.striped = "true";
        for (const line of lines) {{
          const span = document.createElement("span");
          span.className = "pre-line";
          span.textContent = line.length > 0 ? line : "\\u200b";
          markCompactLine(span);
          pre.appendChild(span);
        }}
      }}
    }}
    function markCompactLine(line) {{
      const text = line.textContent;
      if (
        /\\[\\d+\\s+shared prefix tokens with/.test(text) ||
        /\\[(?:content elided: about|totally)\\s+\\d+\\s+(?:text\\s+)?tokens?\\b.*(?:shared with previous|same as previous)/.test(text)
      ) {{
        line.classList.add("omitted-content");
      }}
      if (/\\(total\\s+\\d+\\s+(?:tokens?|logprobs?)\\.\\)/.test(text)) {{
        line.classList.add("token-summary");
      }}
    }}
    function escapeHtml(value) {{
      const entities = {{
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        "\\"": "&quot;",
        "'": "&#x27;",
      }};
      return value.replace(/[&<>"']/g, (char) => entities[char]);
    }}
    function roleClass(role) {{
      const normalized = (role || "").toLowerCase().replace(/[^a-z0-9_-]/g, "-");
      return normalized ? ` role-${{normalized}}` : "";
    }}
    function highlightChatBoundaries() {{
      const markerPattern = /(&lt;\\|im_(?:start|end)\\|&gt;)([A-Za-z0-9_-]+)?/g;
      for (const line of document.querySelectorAll(".pre-line")) {{
        const text = line.textContent;
        if (!text.includes("<|im_")) {{
          continue;
        }}
        const escaped = escapeHtml(text);
        const highlighted = escaped.replace(markerPattern, (_match, marker, role) => {{
          const label = `${{marker}}${{role || ""}}`;
          return `<span class="chat-boundary${{roleClass(role)}}" title="ChatML boundary">${{label}}</span>`;
        }});
        line.innerHTML = highlighted;
      }}
    }}
  </script>
</head>
<body>
  <div class="layout">
    <aside class="sidebar">
      <h1>OpenHands I/O Case</h1>
      <a href="#summary">Summary</a>
      <a href="#timeline">I/O Timeline</a>
      <a href="#decoded-samples">Decoded Samples</a>
      <a href="#final-conversation">Final Conversation JSON</a>
      <a href="#raw-stdout">Raw Stdout</a>
    </aside>
    <main class="content">
      <section id="summary" class="hero">
        <h1>OpenHands Black-Box RL I/O Case</h1>
        <p class="muted">Generated report: {html.escape(generated_at)}</p>
        <div class="grid cols">
          <section class="card">{metrics_html}</section>
        </div>
      </section>

      <h2 id="timeline">I/O Timeline</h2>
      <p class="muted">
        Chronological OpenHands events and OpenAI-compatible model-boundary input/output records.
        Model cards show OpenAI-compatible request/response payloads for readability.
        Repeated token prefixes are collapsed with placeholders such as
        <code>[100 shared prefix tokens with previous trainable sample (...prefix ending preview)]</code>.
      </p>
      {missing_capture_html}
      <div class="timeline">
        {timeline_html}
      </div>

      <h2 id="decoded-samples">Decoded Trainable Samples</h2>
      <p class="muted">
        Each card is one trainable turn sample from the rollout trace, decoded from sample.input_ids.
        Placeholders compare token-level prefixes of the whole trainable sample, not semantic equality of prior assistant messages.
      </p>
      {decoded_samples_html}

      <h2 id="final-conversation">Final Conversation JSON</h2>
      {_render_details("Raw final conversation JSON", messages, prefix_elider=prefix_elider, stream_key="final-conversation")}

      <h2 id="raw-stdout">Raw Stdout</h2>
      {_render_details("Raw OpenHands stdout", stdout_text, prefix_elider=prefix_elider, stream_key="openhands-stdout")}
    </main>
  </div>
</body>
</html>
"""
    path.write_text(body, encoding="utf-8")


def write_openhands_eval_report(
    results: Sequence[RolloutResult],
    path: Path,
    *,
    source_rollout_path: Path | None = None,
    case_report_dir: Path | None = None,
    case_report_limit: int = 8,
    token_decoder: Callable[[Sequence[int]], str] | None = None,
) -> list[Path]:
    """Write a summary report for saved OpenHands evaluation rollouts.

    The top-level report stays compact and links to representative case reports
    generated with ``write_openhands_case_report``. Eval rollouts usually omit
    trainable samples/routing, but their conversations still carry OpenAI proxy
    events, OpenHands events/stdout, solution text, and metrics.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [_eval_rollout_row(index, result) for index, result in enumerate(results)]
    case_links = _write_representative_eval_cases(
        results,
        rows,
        report_path=path,
        case_report_dir=case_report_dir,
        case_report_limit=case_report_limit,
        token_decoder=token_decoder,
    )
    summary = _eval_summary(rows)
    calls_relation = _group_rows_by_metric(rows, "num_model_calls")
    status_counts = Counter(str(row["status"]) for row in rows)
    generated_at = datetime.now().astimezone().isoformat(timespec="seconds")
    source_html = html.escape(str(source_rollout_path)) if source_rollout_path is not None else "not supplied"
    summary_cards = _render_eval_summary_cards(summary)
    status_rows = _render_status_table(status_counts, total=len(rows))
    calls_rows = _render_relation_table(calls_relation)
    case_rows = _render_eval_case_links(case_links, rows, report_path=path)
    rollout_rows = _render_eval_rollout_rows(rows, case_links=case_links, report_path=path)
    body = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>OpenHands Eval Rollout Report</title>
  <style>
    :root {{
      --bg: #f6f7f9;
      --panel: #ffffff;
      --ink: #18202b;
      --muted: #64748b;
      --line: #d8dee4;
      --pass: #047857;
      --fail: #b91c1c;
      --warn: #b45309;
      --info: #2563eb;
      --stripe: #f8fafc;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      background: var(--bg);
      color: var(--ink);
      font-family: system-ui, sans-serif;
      line-height: 1.45;
      margin: 0;
    }}
    a {{ color: #0f766e; text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    .layout {{
      display: grid;
      grid-template-columns: 260px minmax(0, 1fr);
      min-height: 100vh;
    }}
    .sidebar {{
      align-self: start;
      background: #eef2f7;
      border-right: 1px solid var(--line);
      height: 100vh;
      overflow: auto;
      padding: 24px 18px;
      position: sticky;
      top: 0;
    }}
    .sidebar h1 {{ font-size: 18px; margin: 0 0 14px; }}
    .sidebar a {{ display: block; font-size: 14px; padding: 6px 0; }}
    .content {{
      max-width: 1360px;
      padding: 28px 34px 60px;
    }}
    .hero, .card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
    }}
    .hero {{
      margin-bottom: 18px;
      padding: 22px;
    }}
    .card {{ padding: 16px; }}
    h1, h2, h3 {{ line-height: 1.2; }}
    h1 {{ font-size: 28px; margin: 0 0 8px; }}
    h2 {{ font-size: 20px; margin: 28px 0 12px; }}
    h3 {{ font-size: 16px; margin: 0 0 10px; }}
    .muted {{ color: var(--muted); }}
    .grid {{ display: grid; gap: 12px; }}
    .cols {{ grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); }}
    .metric {{
      border-bottom: 1px solid #edf0f5;
      display: flex;
      gap: 12px;
      justify-content: space-between;
      padding: 8px 0;
    }}
    .metric:last-child {{ border-bottom: 0; }}
    .metric span {{ color: var(--muted); }}
    .metric strong {{ overflow-wrap: anywhere; text-align: right; }}
    .badge {{
      background: #e2e8f0;
      border-radius: 999px;
      color: #334155;
      display: inline-block;
      font-size: 12px;
      font-weight: 800;
      padding: 2px 8px;
      white-space: nowrap;
    }}
    .badge.pass {{ background: #dcfce7; color: var(--pass); }}
    .badge.fail {{ background: #fee2e2; color: var(--fail); }}
    .badge.warn {{ background: #fef3c7; color: var(--warn); }}
    .badge.info {{ background: #dbeafe; color: var(--info); }}
    table {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-collapse: collapse;
      border-radius: 8px;
      display: block;
      max-width: 100%;
      overflow: auto;
      width: max-content;
    }}
    th, td {{
      border-bottom: 1px solid #edf0f5;
      padding: 8px 10px;
      text-align: left;
      vertical-align: top;
      white-space: nowrap;
    }}
    th {{
      background: #eef2f7;
      color: #334155;
      font-size: 12px;
      position: sticky;
      top: 0;
      z-index: 1;
    }}
    tr:nth-child(even) td {{ background: var(--stripe); }}
    tr.pass-row td:first-child {{ border-left: 5px solid var(--pass); }}
    tr.fail-row td:first-child {{ border-left: 5px solid var(--fail); }}
    tr.timeout-row td:first-child {{ border-left: 5px solid var(--warn); }}
    td.wrap {{
      max-width: 520px;
      overflow-wrap: anywhere;
      white-space: normal;
    }}
    .section-note {{
      margin: -4px 0 12px;
    }}
    @media (max-width: 900px) {{
      .layout {{ display: block; }}
      .sidebar {{ height: auto; position: static; }}
      .content {{ padding: 20px; }}
    }}
  </style>
</head>
<body>
  <div class="layout">
    <aside class="sidebar">
      <h1>Eval Rollouts</h1>
      <a href="#summary">Summary</a>
      <a href="#status">Status Buckets</a>
      <a href="#turn-score">Turns vs Score</a>
      <a href="#cases">Case Links</a>
      <a href="#rollouts">All Rollouts</a>
    </aside>
    <main class="content">
      <section id="summary" class="hero">
        <h1>OpenHands Eval Rollout Report</h1>
        <p class="muted">Generated report: {html.escape(generated_at)}</p>
        <p class="muted">Source rollouts: {source_html}</p>
        <div class="grid cols">{summary_cards}</div>
      </section>

      <h2 id="status">Status Buckets</h2>
      <p class="muted section-note">
        High-level buckets derived from score, OpenHands timeout, solution collection, JSON events, and invalid tool-call counters.
      </p>
      {status_rows}

      <h2 id="turn-score">Turns vs Score</h2>
      <p class="muted section-note">Grouped by model-call count. This helps spot whether failures are concentrated in long trajectories.</p>
      {calls_rows}

      <h2 id="cases">Representative Case Studies</h2>
      <p class="muted section-note">These pages use the same tuned OpenHands case-study renderer as the rollout smoke reports.</p>
      {case_rows}

      <h2 id="rollouts">All Rollouts</h2>
      {rollout_rows}
    </main>
  </div>
</body>
</html>
"""
    path.write_text(body, encoding="utf-8")
    return list(case_links.values())


def _eval_rollout_row(index: int, result: RolloutResult) -> dict[str, Any]:
    metric = result.metric
    conv = result.conversation
    extra = conv.extra
    openhands_events = _listish(extra.get("openhands_json_events"))
    model_events = _listish(extra.get("openai_io_events") or extra.get("sglang_io_events"))
    stdout_lines = _listish(extra.get("openhands_stdout_lines"))
    solution = str(extra.get("openhands_solution") or "")
    score = _metric_number(metric, "score")
    timeout = int(_metric_number(metric, "openhands_process_timeout") or 0)
    invalid_tool_calls = int(_metric_number(metric, "num_invalid_tool_calls") or 0)
    solution_chars = int(_metric_number(metric, "collected_solution_chars") or len(solution))
    row = {
        "index": index,
        "conversation_id": conv.conversation_id,
        "task_id": extra.get("task_id") or conv.conversation_id,
        "difficulty": extra.get("difficulty"),
        "score": score,
        "normal_finish": int(_metric_number(metric, "normal_finish") or 0),
        "num_model_calls": int(_metric_number(metric, "num_model_calls") or 0),
        "token_count": int(_metric_number(metric, "token_count") or 0),
        "timeout": timeout,
        "initial_request_timeout": int(_metric_number(metric, "initial_request_timeout") or 0),
        "request_timeout": int(_metric_number(metric, "request_timeout") or 0),
        "verifier_timeout": int(_metric_number(metric, "verifier_timeout") or 0),
        "invalid_tool_calls": invalid_tool_calls,
        "solution_chars": solution_chars,
        "stdout_lines": int(_metric_number(metric, "openhands_stdout_lines") or len(stdout_lines)),
        "openhands_events": len(openhands_events),
        "model_io_events": len(model_events),
        "exit_code": _metric_number(metric, "openhands_exit_code"),
        "blackbox_total_seconds": _metric_number(metric, "blackbox_total_seconds"),
        "blackbox_llm_total_seconds": _metric_number(metric, "blackbox_llm_total_seconds"),
        "wait_request_total_seconds": _metric_number(metric, "blackbox_wait_request_total_seconds"),
        "llm_turn_mean_latency": _metric_number(metric, "llm_turn_mean_latency"),
        "llm_turn_mean_output_tokens": _metric_number(metric, "llm_turn_mean_output_tokens"),
        "forced_score_reason": str(extra.get("blackbox_forced_score_reason") or ""),
    }
    row["status"] = _eval_status(row)
    return row


def _metric_number(metric: object, name: str) -> int | float | None:
    value = getattr(metric, name, None)
    if isinstance(value, np.generic):
        return cast("int | float", value.item())
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value
    return None


def _eval_status(row: dict[str, Any]) -> str:
    score = row.get("score")
    if isinstance(score, (int, float)) and score > 0:
        return "pass"
    if row["timeout"]:
        if row["solution_chars"] > 0:
            return "timeout_after_solution"
        if row["openhands_events"] > 0:
            return "timeout_after_events"
        return "timeout_no_events"
    if row["request_timeout"]:
        return "request_timeout"
    if row["verifier_timeout"]:
        return "verifier_timeout"
    if row["invalid_tool_calls"]:
        return "fail_invalid_tool_call"
    if row["solution_chars"] > 0:
        return "fail_verifier"
    return "fail_no_solution"


def _eval_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    passed = sum(1 for row in rows if row["status"] == "pass")
    timeouts = sum(int(row["timeout"]) for row in rows)
    initial_request_timeouts = sum(int(row["initial_request_timeout"]) for row in rows)
    request_timeouts = sum(int(row["request_timeout"]) for row in rows)
    verifier_timeouts = sum(int(row["verifier_timeout"]) for row in rows)
    invalid_rollouts = sum(1 for row in rows if row["invalid_tool_calls"])
    normal_finishes = sum(int(row["normal_finish"]) for row in rows)
    max_calls = max((int(row["num_model_calls"]) for row in rows), default=0)
    return {
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "pass_rate": _ratio(passed, total),
        "normal_finishes": normal_finishes,
        "normal_finish_rate": _ratio(normal_finishes, total),
        "timeouts": timeouts,
        "timeout_rate": _ratio(timeouts, total),
        "initial_request_timeouts": initial_request_timeouts,
        "initial_request_timeout_rate": _ratio(initial_request_timeouts, total),
        "request_timeouts": request_timeouts,
        "request_timeout_rate": _ratio(request_timeouts, total),
        "verifier_timeouts": verifier_timeouts,
        "verifier_timeout_rate": _ratio(verifier_timeouts, total),
        "timeout_no_events": sum(1 for row in rows if row["status"] == "timeout_no_events"),
        "timeout_after_work": sum(1 for row in rows if str(row["status"]).startswith("timeout_after")),
        "invalid_tool_rollouts": invalid_rollouts,
        "invalid_tool_calls": sum(int(row["invalid_tool_calls"]) for row in rows),
        "with_solution": sum(1 for row in rows if row["solution_chars"] > 0),
        "without_solution": sum(1 for row in rows if row["solution_chars"] <= 0),
        "max_model_calls_observed": max_calls,
        "rollouts_at_max_calls": sum(1 for row in rows if row["num_model_calls"] == max_calls),
        "avg_model_calls": _mean(row["num_model_calls"] for row in rows),
        "p50_model_calls": _percentile([row["num_model_calls"] for row in rows], 0.5),
        "p95_model_calls": _percentile([row["num_model_calls"] for row in rows], 0.95),
        "avg_token_count": _mean(row["token_count"] for row in rows),
        "p95_token_count": _percentile([row["token_count"] for row in rows], 0.95),
        "avg_total_seconds": _mean(row["blackbox_total_seconds"] for row in rows if row["blackbox_total_seconds"] is not None),
        "avg_llm_seconds": _mean(row["blackbox_llm_total_seconds"] for row in rows if row["blackbox_llm_total_seconds"] is not None),
        "avg_wait_request_seconds": _mean(row["wait_request_total_seconds"] for row in rows if row["wait_request_total_seconds"] is not None),
    }


def _ratio(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _mean(values: Any) -> float:
    cleaned = [float(value) for value in values if value is not None]
    return float(sum(cleaned) / len(cleaned)) if cleaned else 0.0


def _percentile(values: list[Any], percentile: float) -> float:
    cleaned = sorted(float(value) for value in values if value is not None)
    if not cleaned:
        return 0.0
    index = min(len(cleaned) - 1, max(0, round((len(cleaned) - 1) * percentile)))
    return cleaned[index]


def _group_rows_by_metric(rows: list[dict[str, Any]], metric_name: str) -> list[dict[str, Any]]:
    groups: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row[metric_name]].append(row)
    relation_rows: list[dict[str, Any]] = []
    for value, group in sorted(groups.items(), key=lambda item: item[0]):
        passed = sum(1 for row in group if row["status"] == "pass")
        timeouts = sum(int(row["timeout"]) for row in group)
        request_timeouts = sum(int(row["request_timeout"]) for row in group)
        relation_rows.append(
            {
                metric_name: value,
                "rollouts": len(group),
                "passed": passed,
                "pass_rate": _ratio(passed, len(group)),
                "timeouts": timeouts,
                "timeout_rate": _ratio(timeouts, len(group)),
                "request_timeouts": request_timeouts,
                "request_timeout_rate": _ratio(request_timeouts, len(group)),
                "avg_token_count": _mean(row["token_count"] for row in group),
                "avg_turn_output_tokens": _mean(
                    row["llm_turn_mean_output_tokens"] for row in group if row["llm_turn_mean_output_tokens"] is not None
                ),
                "avg_total_seconds": _mean(row["blackbox_total_seconds"] for row in group if row["blackbox_total_seconds"] is not None),
            }
        )
    return relation_rows


def _write_representative_eval_cases(
    results: Sequence[RolloutResult],
    rows: list[dict[str, Any]],
    *,
    report_path: Path,
    case_report_dir: Path | None,
    case_report_limit: int,
    token_decoder: Callable[[Sequence[int]], str] | None,
) -> dict[int, Path]:
    if case_report_limit <= 0:
        return {}
    output_dir = case_report_dir or report_path.with_name(f"{report_path.stem}-cases")
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = _select_representative_eval_rows(rows, limit=case_report_limit)
    case_links: dict[int, Path] = {}
    for label, index in selected:
        result = results[index]
        filename = f"{index:03d}-{_safe_report_name(label)}-{_safe_report_name(result.conversation.conversation_id)}.html"
        case_path = output_dir / filename
        write_openhands_case_report(result, case_path, token_decoder=token_decoder)
        case_links[index] = case_path
    return case_links


def _select_representative_eval_rows(rows: list[dict[str, Any]], *, limit: int) -> list[tuple[str, int]]:
    selected: list[tuple[str, int]] = []
    used: set[int] = set()

    def add(label: str, candidates: list[dict[str, Any]], *, reverse: bool = False, key: str = "token_count") -> None:
        if len(selected) >= limit or not candidates:
            return
        ordered = sorted(candidates, key=lambda row: row[key], reverse=reverse)
        for row in ordered:
            index = int(row["index"])
            if index not in used:
                selected.append((label, index))
                used.add(index)
                return

    pass_rows = [row for row in rows if row["status"] == "pass"]
    fail_verifier_rows = [row for row in rows if row["status"] == "fail_verifier"]
    timeout_no_event_rows = [row for row in rows if row["status"] == "timeout_no_events"]
    timeout_after_work_rows = [row for row in rows if str(row["status"]).startswith("timeout_after")]
    request_timeout_rows = [row for row in rows if row["status"] == "request_timeout"]
    invalid_rows = [row for row in rows if row["invalid_tool_calls"]]
    max_calls = max((int(row["num_model_calls"]) for row in rows), default=0)
    add("pass-short", pass_rows, key="num_model_calls")
    add("pass-long", pass_rows, reverse=True)
    add("fail-verifier", fail_verifier_rows, reverse=True)
    add("timeout-no-events", timeout_no_event_rows, reverse=True)
    add("timeout-after-work", timeout_after_work_rows, reverse=True)
    add("request-timeout", request_timeout_rows, reverse=True)
    add("invalid-tool-call", invalid_rows, reverse=True)
    add("max-calls", [row for row in rows if row["num_model_calls"] == max_calls], reverse=True)
    add("longest-rollout", rows, reverse=True)
    for row in rows:
        if len(selected) >= limit:
            break
        index = int(row["index"])
        if index not in used:
            selected.append(("additional", index))
            used.add(index)
    return selected


def _render_eval_summary_cards(summary: dict[str, Any]) -> str:
    card_groups = [
        {
            "total": summary["total"],
            "passed": summary["passed"],
            "failed": summary["failed"],
            "pass_rate": _format_percent(summary["pass_rate"]),
        },
        {
            "process_timeouts": summary["timeouts"],
            "process_timeout_rate": _format_percent(summary["timeout_rate"]),
            "normal_finishes": summary["normal_finishes"],
            "normal_finish_rate": _format_percent(summary["normal_finish_rate"]),
            "initial_request_timeouts": summary["initial_request_timeouts"],
            "initial_request_timeout_rate": _format_percent(summary["initial_request_timeout_rate"]),
            "request_timeouts": summary["request_timeouts"],
            "request_timeout_rate": _format_percent(summary["request_timeout_rate"]),
            "verifier_timeouts": summary["verifier_timeouts"],
            "verifier_timeout_rate": _format_percent(summary["verifier_timeout_rate"]),
        },
        {
            "timeout_no_events": summary["timeout_no_events"],
            "timeout_after_work": summary["timeout_after_work"],
            "invalid_tool_rollouts": summary["invalid_tool_rollouts"],
            "invalid_tool_calls": summary["invalid_tool_calls"],
            "with_solution": summary["with_solution"],
            "without_solution": summary["without_solution"],
        },
        {
            "avg_model_calls": _format_number(float(summary["avg_model_calls"])),
            "p50_model_calls": _format_number(float(summary["p50_model_calls"])),
            "p95_model_calls": _format_number(float(summary["p95_model_calls"])),
            "rollouts_at_max_calls": summary["rollouts_at_max_calls"],
        },
        {
            "avg_token_count": _format_number(float(summary["avg_token_count"])),
            "p95_token_count": _format_number(float(summary["p95_token_count"])),
            "avg_total_seconds": _format_number(float(summary["avg_total_seconds"])),
            "avg_llm_seconds": _format_number(float(summary["avg_llm_seconds"])),
        },
        {
            "avg_wait_request_seconds": _format_number(float(summary["avg_wait_request_seconds"])),
            "max_model_calls_observed": summary["max_model_calls_observed"],
        },
    ]
    return "\n".join(_render_metric_card(group) for group in card_groups)


def _render_metric_card(metrics: dict[str, Any]) -> str:
    metric_html = "\n".join(
        f"<div class='metric'><span>{html.escape(str(key))}</span><strong>{html.escape(str(value))}</strong></div>" for key, value in metrics.items()
    )
    return f"<section class='card'>{metric_html}</section>"


def _render_status_table(status_counts: Counter[str], *, total: int) -> str:
    rows = "\n".join(
        f"<tr><td>{html.escape(status)}</td><td>{count}</td><td>{html.escape(_format_percent(_ratio(count, total)))}</td></tr>"
        for status, count in sorted(status_counts.items())
    )
    return f"""
<table>
  <thead><tr><th>Status</th><th>Rollouts</th><th>Share</th></tr></thead>
  <tbody>{rows}</tbody>
</table>
"""


def _render_relation_table(rows: list[dict[str, Any]]) -> str:
    body = "\n".join(
        "<tr>"
        f"<td>{row['num_model_calls']}</td>"
        f"<td>{row['rollouts']}</td>"
        f"<td>{row['passed']}</td>"
        f"<td>{html.escape(_format_percent(row['pass_rate']))}</td>"
        f"<td>{row['timeouts']}</td>"
        f"<td>{html.escape(_format_percent(row['timeout_rate']))}</td>"
        f"<td>{row['request_timeouts']}</td>"
        f"<td>{html.escape(_format_percent(row['request_timeout_rate']))}</td>"
        f"<td>{html.escape(_format_number(row['avg_token_count']))}</td>"
        f"<td>{html.escape(_format_number(row['avg_turn_output_tokens']))}</td>"
        f"<td>{html.escape(_format_number(row['avg_total_seconds']))}</td>"
        "</tr>"
        for row in rows
    )
    return f"""
<table>
  <thead>
    <tr>
      <th>Model Calls</th><th>Rollouts</th><th>Passed</th><th>Pass Rate</th>
      <th>Process Timeouts</th><th>Process Timeout Rate</th>
      <th>Request Timeouts</th><th>Request Timeout Rate</th><th>Avg Trajectory Output Tokens</th>
      <th>Avg Turn Output Tokens</th><th>Avg Total Seconds</th>
    </tr>
  </thead>
  <tbody>{body}</tbody>
</table>
"""


def _render_eval_case_links(case_links: dict[int, Path], rows: list[dict[str, Any]], *, report_path: Path) -> str:
    if not case_links:
        return "<section class='card'><p class='muted'>No representative cases were generated.</p></section>"
    row_by_index = {int(row["index"]): row for row in rows}
    body = "\n".join(
        "<tr>"
        f"<td>{_status_badge(row_by_index[index])}</td>"
        f"<td>{index}</td>"
        f"<td class='wrap'>{html.escape(str(row_by_index[index]['conversation_id']))}</td>"
        f"<td>{html.escape(str(row_by_index[index]['score']))}</td>"
        f"<td>{row_by_index[index]['num_model_calls']}</td>"
        f"<td><a href='{html.escape(_relative_href(path, report_path.parent))}'>case report</a></td>"
        "</tr>"
        for index, path in sorted(case_links.items())
    )
    return f"""
<table>
  <thead><tr><th>Status</th><th>Index</th><th>Conversation</th><th>Score</th><th>Calls</th><th>Link</th></tr></thead>
  <tbody>{body}</tbody>
</table>
"""


def _render_eval_rollout_rows(rows: list[dict[str, Any]], *, case_links: dict[int, Path], report_path: Path) -> str:
    body = "\n".join(_render_eval_rollout_row(row, case_links=case_links, report_path=report_path) for row in rows)
    return f"""
<table>
  <thead>
    <tr>
      <th>Status</th><th>Index</th><th>Conversation</th><th>Difficulty</th><th>Score</th>
      <th>Calls</th><th>Trajectory Output Tokens</th><th>Normal Finish</th>
      <th>Process Timeout</th><th>Initial Request Timeout</th><th>Request Timeout</th><th>Verifier Timeout</th><th>Invalid Tools</th>
      <th>Solution Chars</th><th>OH Events</th><th>Model I/O</th><th>Stdout</th>
      <th>Total Seconds</th><th>LLM Seconds</th><th>Mean Turn Tokens</th><th>Reason</th><th>Case</th>
    </tr>
  </thead>
  <tbody>{body}</tbody>
</table>
"""


def _render_eval_rollout_row(row: dict[str, Any], *, case_links: dict[int, Path], report_path: Path) -> str:
    index = int(row["index"])
    if row["timeout"] or row["request_timeout"] or row["verifier_timeout"]:
        row_class = "timeout-row"
    elif row["status"] == "pass":
        row_class = "pass-row"
    else:
        row_class = "fail-row"
    case_link = ""
    if index in case_links:
        case_link = f"<a href='{html.escape(_relative_href(case_links[index], report_path.parent))}'>case</a>"
    return (
        f"<tr class='{row_class}'>"
        f"<td>{_status_badge(row)}</td>"
        f"<td>{index}</td>"
        f"<td class='wrap'>{html.escape(str(row['conversation_id']))}</td>"
        f"<td>{html.escape(str(row.get('difficulty') or ''))}</td>"
        f"<td>{html.escape(str(row['score']))}</td>"
        f"<td>{row['num_model_calls']}</td>"
        f"<td>{row['token_count']}</td>"
        f"<td>{row['normal_finish']}</td>"
        f"<td>{row['timeout']}</td>"
        f"<td>{row['initial_request_timeout']}</td>"
        f"<td>{row['request_timeout']}</td>"
        f"<td>{row['verifier_timeout']}</td>"
        f"<td>{row['invalid_tool_calls']}</td>"
        f"<td>{row['solution_chars']}</td>"
        f"<td>{row['openhands_events']}</td>"
        f"<td>{row['model_io_events']}</td>"
        f"<td>{row['stdout_lines']}</td>"
        f"<td>{html.escape(_format_optional_number(row['blackbox_total_seconds']))}</td>"
        f"<td>{html.escape(_format_optional_number(row['blackbox_llm_total_seconds']))}</td>"
        f"<td>{html.escape(_format_optional_number(row['llm_turn_mean_output_tokens']))}</td>"
        f"<td class='wrap'>{html.escape(_shorten(row['forced_score_reason'], limit=180))}</td>"
        f"<td>{case_link}</td>"
        "</tr>"
    )


def _status_badge(row: dict[str, Any]) -> str:
    status = str(row["status"])
    if status == "pass":
        css_class = "pass"
    elif "timeout" in status:
        css_class = "warn"
    else:
        css_class = "fail"
    return f"<span class='badge {css_class}'>{html.escape(status)}</span>"


def _relative_href(path: Path, base_dir: Path) -> str:
    try:
        return str(path.resolve().relative_to(base_dir.resolve()))
    except ValueError:
        return str(path)


def _safe_report_name(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "-" for ch in value)
    return safe[:96] or "case"


def _format_percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def _format_optional_number(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return _format_number(float(value))
    return str(value)


def _build_case_timeline(openhands_events: Any, model_events: Any) -> list[dict[str, Any]]:
    timeline: list[dict[str, Any]] = []
    ordinal = 0
    for event in _listish(openhands_events):
        if not isinstance(event, dict):
            continue
        timestamp = _event_timestamp(event)
        timeline.append(
            {
                "kind": "openhands",
                "css_kind": "openhands",
                "badge": "OpenHands",
                "title": _openhands_event_title(event),
                "timestamp": timestamp,
                "summary": _openhands_event_summary(event),
                "payload": event,
                "sort_key": _event_sort_key(timestamp, ordinal),
                "ordinal": ordinal,
            }
        )
        ordinal += 1
    for event in _listish(model_events):
        if not isinstance(event, dict):
            continue
        kind = str(event.get("kind") or "model_event")
        css_kind = kind.replace("_", "-")
        title = _model_event_title(kind)
        timestamp = _event_timestamp(event)
        timeline.append(
            {
                "kind": kind,
                "css_kind": css_kind,
                "badge": title,
                "title": f"{title} #{event.get('call_index', '?')}",
                "timestamp": timestamp,
                "summary": _model_event_summary(event),
                "details": _model_event_details(event),
                "sort_key": _event_sort_key(timestamp, ordinal),
                "ordinal": ordinal,
            }
        )
        ordinal += 1
    return sorted(timeline, key=lambda item: (item["sort_key"], item["ordinal"]))


def _model_event_title(kind: str) -> str:
    if kind in {"model_input", "sglang_input"}:
        return "Model Input"
    if kind in {"model_output", "sglang_output"}:
        return "Model Output"
    return "Model Event"


def _listish(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _event_timestamp(event: dict[str, Any]) -> str:
    timestamp = event.get("timestamp") or event.get("created_at") or event.get("time")
    return str(timestamp) if timestamp is not None else ""


def _event_sort_key(timestamp: str, ordinal: int) -> tuple[int, float | int]:
    if not timestamp:
        return (1, ordinal)
    try:
        parsed = datetime.fromisoformat(timestamp)
    except ValueError:
        return (1, ordinal)
    return (0, parsed.timestamp())


def _openhands_event_title(event: dict[str, Any]) -> str:
    kind = str(event.get("kind") or event.get("type") or "OpenHands Event")
    source = event.get("source")
    tool_name = event.get("tool_name")
    if tool_name:
        return f"{kind}: {tool_name}"
    if source:
        return f"{kind}: {source}"
    return kind


def _openhands_event_summary(event: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for key in ("source", "tool_name", "security_risk", "summary"):
        if event.get(key) is not None:
            summary[key] = _shorten(event[key])
    action = event.get("action")
    if isinstance(action, dict):
        summary["action_kind"] = action.get("kind")
        for key in ("command", "path", "message"):
            if action.get(key) is not None:
                summary[f"action_{key}"] = _shorten(action[key])
                break
    observation = event.get("observation")
    if isinstance(observation, dict):
        summary["observation_kind"] = observation.get("kind")
        if observation.get("is_error") is not None:
            summary["is_error"] = observation.get("is_error")
        if observation.get("exit_code") is not None:
            summary["exit_code"] = observation.get("exit_code")
        content = _content_preview(observation.get("content"))
        if content:
            summary["observation_content"] = content
    thought = _content_preview(event.get("thought"))
    if thought:
        summary["thought"] = thought
    if not summary:
        for key in ("id", "kind", "type"):
            if event.get(key) is not None:
                summary[key] = _shorten(event[key])
    return {key: value for key, value in summary.items() if value not in (None, "")}


def _model_event_summary(event: dict[str, Any]) -> dict[str, Any]:
    summary_raw = event.get("summary")
    summary: dict[str, Any] = {str(key): value for key, value in summary_raw.items()} if isinstance(summary_raw, dict) else {}
    payload_raw = event.get("payload")
    payload: dict[str, Any] = {str(key): value for key, value in payload_raw.items()} if isinstance(payload_raw, dict) else {}
    kind = event.get("kind")
    if kind in {"model_input", "sglang_input"}:
        summary.update(
            _generation_input_summary(
                generation_input=payload.get("generation_input"),
                preparation=payload.get("generation_input_preparation"),
                response_context=payload.get("response_context"),
            )
        )
    elif kind in {"model_output", "sglang_output"}:
        summary.update(_generation_output_summary(payload.get("generation_output")))
        invalid_tool_call = payload.get("invalid_tool_call")
        if invalid_tool_call is not None:
            invalid_json = _jsonable_for_report(invalid_tool_call)
            summary["invalid_tool_call"] = True
            if isinstance(invalid_json, dict):
                summary["invalid_tool_call_message"] = invalid_json.get("message")
                summary["invalid_tool_name"] = invalid_json.get("tool_name")
        request_timeout = payload.get("request_timeout")
        if request_timeout is not None:
            timeout_json = _jsonable_for_report(request_timeout)
            summary["request_timeout"] = True
            if isinstance(timeout_json, dict):
                summary["request_timeout_message"] = timeout_json.get("message")
                summary["request_timeout_seconds"] = timeout_json.get("timeout_seconds")
    for key in ("call_index", "request_id", "session_id"):
        if event.get(key) is not None:
            summary[key] = event[key]
    return {key: _shorten(value) for key, value in summary.items()}


def _generation_input_summary(
    *,
    generation_input: object | None,
    preparation: object | None,
    response_context: object | None,
) -> dict[str, object]:
    request_json = _openai_request_from_context(response_context)
    input_ids = getattr(generation_input, "input_ids", None)
    prompt_tokens = getattr(response_context, "prompt_tokens", None)
    messages = request_json.get("messages") if isinstance(request_json, dict) else None
    summary: dict[str, object | None] = {
        "model": request_json.get("model") if isinstance(request_json, dict) else None,
        "messages": len(messages) if isinstance(messages, list) else None,
        "input_tokens": len(input_ids) if input_ids is not None else prompt_tokens,
        "routed_expert_start_index": getattr(generation_input, "routed_expert_start_index", None),
        "shared_prefix_tokens": getattr(preparation, "shared_prefix_tokens", None),
        "previous_routing_rows": getattr(preparation, "previous_routing_rows", None),
        "preserved_routing_rows": getattr(preparation, "preserved_routing_rows", None),
        "dropped_routing_rows": getattr(preparation, "dropped_routing_rows", None),
        "preserved_routing_handles": getattr(preparation, "preserved_routing_handles", None),
    }
    return {key: value for key, value in summary.items() if value is not None}


def _generation_output_summary(generation_output: object | None) -> dict[str, object]:
    if generation_output is None:
        return {}
    output_ids = getattr(generation_output, "output_ids", None)
    tool_calls = getattr(generation_output, "tool_calls", None)
    summary: dict[str, object | None] = {
        "finish_reason": getattr(generation_output, "finish_reason", None),
        "output_tokens": len(output_ids) if output_ids is not None else None,
        "cached_tokens": getattr(generation_output, "cached_tokens", None),
        "retry": getattr(generation_output, "retry", None),
        "e2e_elapsed_seconds": getattr(generation_output, "e2e_elapsed_seconds", None),
        "tool_calls": len(tool_calls or []),
    }
    return {key: value for key, value in summary.items() if value is not None}


def _model_event_details(event: dict[str, Any]) -> list[tuple[str, Any]]:
    payload_raw = event.get("payload")
    payload: dict[str, Any] = {str(key): value for key, value in payload_raw.items()} if isinstance(payload_raw, dict) else {}
    kind = event.get("kind")
    if kind in {"model_input", "sglang_input"}:
        details = [("OpenAI-compatible request", _openai_request_from_payload(payload))]
        if "generation_input" in payload:
            details.append(("GenerationInput", payload["generation_input"]))
        if "response_context" in payload:
            details.append(("OpenAI response context", _response_context_for_display(payload["response_context"])))
        return details
    if kind in {"model_output", "sglang_output"}:
        output_details: list[tuple[str, Any]] = []
        if "openai_response" in payload:
            output_details.append(("OpenAI-compatible response", payload["openai_response"]))
        if "invalid_tool_call" in payload:
            output_details.append(("Invalid tool call (not sent to OpenHands)", payload["invalid_tool_call"]))
        if "request_timeout" in payload:
            output_details.append(("Request timeout after response (OpenHands terminated)", payload["request_timeout"]))
        if "generation_output" in payload:
            output_details.append(("GenerationOutput", payload["generation_output"]))
        if not output_details:
            output_details.append(("Model output payload", payload))
        return output_details
    return [("OpenAI-compatible payload", payload or event)]


def _openai_request_from_payload(payload: dict[str, Any]) -> object:
    if "openai_request" in payload:
        return payload["openai_request"]
    return _openai_request_from_context(payload.get("response_context")) or payload


def _openai_request_from_context(response_context: object | None) -> object | None:
    if response_context is None:
        return None
    return getattr(response_context, "request_json", None)


def _response_context_for_display(response_context: object) -> object:
    context = _jsonable_for_report(response_context)
    if isinstance(context, dict) and "request_json" in context:
        context = dict(context)
        context["request_json"] = "[rendered above as OpenAI-compatible request]"
    return context


def _content_preview(value: Any) -> str:
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, dict) and item.get("text") is not None:
                parts.append(str(item["text"]))
            elif item is not None:
                parts.append(str(item))
        return _shorten("\n".join(parts))
    if isinstance(value, str):
        return _shorten(value)
    return ""


def _shorten(value: Any, *, limit: int = 500) -> str:
    text = value if isinstance(value, str) else json.dumps(_jsonable_for_report(value), ensure_ascii=False, sort_keys=True)
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "..."


def _render_timeline_item(item: dict[str, Any], *, prefix_elider: _PrefixElider) -> str:
    css_kind = html.escape(str(item["css_kind"]))
    timestamp = html.escape(str(item.get("timestamp") or "no timestamp"))
    title = html.escape(str(item["title"]))
    badge = html.escape(str(item["badge"]))
    summary_raw = item.get("summary")
    summary: dict[str, Any] = summary_raw if isinstance(summary_raw, dict) else {}
    has_warning = bool(summary.get("invalid_tool_call") or summary.get("request_timeout"))
    section_class = f"{css_kind} warn-event" if has_warning else css_kind
    badge_class = f"{css_kind} warn-event" if has_warning else css_kind
    summary_html = "\n".join(f"<li><strong>{html.escape(str(key))}</strong>: {html.escape(str(value))}</li>" for key, value in summary.items())
    if summary_html:
        summary_html = f"<ul class='kv-list'>{summary_html}</ul>"
    else:
        summary_html = "<p class='muted'>No compact summary fields available.</p>"
    details = item.get("details")
    if isinstance(details, list):
        details_html = "\n".join(
            _render_details(str(detail_summary), payload, prefix_elider=prefix_elider, stream_key=f"{item.get('kind')}.{detail_summary}")
            for detail_summary, payload in details
        )
    else:
        details_html = _render_details("Raw payload", item.get("payload"), prefix_elider=prefix_elider, stream_key=f"{item.get('kind')}.raw")
    return f"""
<section class="card timeline-item {section_class}">
  <div class="timeline-title">
    <span class="badge {badge_class}">{badge}</span>
    <h3>{title}</h3>
    <span class="muted">{timestamp}</span>
  </div>
  {summary_html}
  {details_html}
</section>
"""


def _render_decoded_turn_samples(
    turn_samples: Sequence[Any],
    *,
    token_decoder: Callable[[Sequence[int]], str] | None,
    prefix_elider: _PrefixElider,
) -> str:
    if not turn_samples:
        return """
<section class="card">
  <p class="muted">No trainable turn samples were attached to this rollout result.</p>
</section>
"""

    cards: list[str] = []
    for index, sample in enumerate(turn_samples, start=1):
        input_ids = _active_sample_input_ids(sample)
        decoded_text = prefix_elider.elide_decoded_tokens(
            key="decoded-turn-sample",
            token_ids=input_ids,
            token_decoder=token_decoder,
            label="previous trainable sample",
        )
        summary = {
            "active_input_tokens": len(input_ids),
            "trainable_tokens": _sample_trainable_token_count(sample),
            "reward": getattr(sample, "reward", None),
            "reward_baseline": getattr(sample, "reward_baseline", None),
        }
        summary_html = "\n".join(
            f"<li><strong>{html.escape(str(key))}</strong>: {html.escape(str(value))}</li>" for key, value in summary.items() if value is not None
        )
        cards.append(
            f"""
<section class="card sample-turn">
  <div class="timeline-title">
    <span class="badge">Sample</span>
    <h3>Turn {index}</h3>
  </div>
  <ul class="kv-list">{summary_html}</ul>
  <pre>{html.escape(decoded_text)}</pre>
</section>
"""
        )
    return '<div class="sample-grid">\n' + "".join(cards) + "\n</div>"


def _active_sample_input_ids(sample: Any) -> list[int]:
    input_ids = [int(token_id) for token_id in getattr(sample, "input_ids", [])]
    attention_mask = getattr(sample, "attention_mask", None)
    if attention_mask is None:
        return input_ids
    mask = [bool(value) for value in attention_mask]
    if len(mask) != len(input_ids):
        return input_ids
    return [token_id for token_id, active in zip(input_ids, mask, strict=True) if active]


def _sample_trainable_token_count(sample: Any) -> int:
    loss_mask = getattr(sample, "loss_mask", None)
    if loss_mask is None:
        return 0
    if isinstance(loss_mask, np.ndarray):
        return int(np.count_nonzero(loss_mask))
    return sum(1 for value in loss_mask if bool(value))


def _decode_sample_input_ids(
    input_ids: Sequence[int],
    token_decoder: Callable[[Sequence[int]], str] | None,
) -> str:
    if token_decoder is None:
        return "Token decoder was not supplied when this report was generated."
    try:
        return token_decoder(input_ids)
    except Exception as exc:  # pragma: no cover - defensive report path
        return f"Token decode failed: {type(exc).__name__}: {exc}"


def _render_details(
    summary: str,
    payload: Any,
    *,
    prefix_elider: _PrefixElider,
    stream_key: str,
) -> str:
    detail_class = ' class="warn-detail"' if "invalid tool call" in summary.lower() or "timeout" in summary.lower() else ""
    return f"""
<details{detail_class}>
  <summary>{html.escape(summary)}</summary>
  <div class="details-body">
    <pre>{_payload_html(payload, prefix_elider=prefix_elider, stream_key=stream_key)}</pre>
  </div>
</details>
"""


def _payload_html(payload: Any, *, prefix_elider: _PrefixElider, stream_key: str) -> str:
    if isinstance(payload, str):
        return html.escape(prefix_elider.elide_text(stream_key, payload, label="previous text"))
    jsonable = _jsonable_for_report(payload)
    compacted = _elide_json_prefixes(jsonable, prefix_elider=prefix_elider, stream_key=stream_key)
    return html.escape(json.dumps(compacted, ensure_ascii=False, indent=2, sort_keys=True))


class _PrefixElider:
    def __init__(self, *, min_text_prefix_chars: int = 240, min_token_prefix: int = 32, prefix_tail_preview_chars: int = 100) -> None:
        self.min_text_prefix_chars = min_text_prefix_chars
        self.min_token_prefix = min_token_prefix
        self.prefix_tail_preview_chars = prefix_tail_preview_chars
        self._previous_text: dict[str, str] = {}
        self._previous_tokens: dict[str, list[int]] = {}
        self._previous_messages: dict[str, list[Any]] = {}

    def elide_text(self, key: str, text: str, *, label: str) -> str:
        previous = self._previous_text.get(key)
        self._previous_text[key] = text
        if previous is None:
            return text
        prefix_len = _common_text_prefix_len(previous, text)
        if prefix_len < self.min_text_prefix_chars:
            return text
        suffix = text[prefix_len:]
        estimated_tokens = _estimated_text_tokens(text[:prefix_len])
        placeholder = f"[{estimated_tokens} text tokens / {prefix_len} chars shared prefix with {label}]"
        if not suffix:
            return placeholder
        return placeholder + suffix

    def elide_token_list(self, key: str, token_ids: list[int], *, label: str) -> list[Any]:
        previous = self._previous_tokens.get(key)
        self._previous_tokens[key] = list(token_ids)
        if previous is None:
            return token_ids
        prefix_len = _common_token_prefix_len(previous, token_ids)
        if prefix_len < self.min_token_prefix:
            return token_ids
        suffix = token_ids[prefix_len:]
        placeholder = f"[{prefix_len} shared prefix tokens with {label}]"
        return [placeholder, *suffix] if suffix else [placeholder]

    def elide_decoded_tokens(
        self,
        *,
        key: str,
        token_ids: list[int],
        token_decoder: Callable[[Sequence[int]], str] | None,
        label: str,
    ) -> str:
        previous = self._previous_tokens.get(key)
        self._previous_tokens[key] = list(token_ids)
        if previous is None:
            return _decode_sample_input_ids(token_ids, token_decoder)
        prefix_len = _common_token_prefix_len(previous, token_ids)
        if prefix_len < self.min_token_prefix:
            return _decode_sample_input_ids(token_ids, token_decoder)
        suffix_ids = token_ids[prefix_len:]
        placeholder = self._token_prefix_placeholder(
            prefix_len=prefix_len,
            prefix_token_ids=token_ids[:prefix_len],
            token_decoder=token_decoder,
            label=label,
        )
        if not suffix_ids:
            return placeholder
        return placeholder + "\n" + _decode_sample_input_ids(suffix_ids, token_decoder)

    def _token_prefix_placeholder(
        self,
        *,
        prefix_len: int,
        prefix_token_ids: Sequence[int],
        token_decoder: Callable[[Sequence[int]], str] | None,
        label: str,
    ) -> str:
        preview = _decoded_prefix_tail_preview(
            prefix_token_ids,
            token_decoder=token_decoder,
            char_limit=self.prefix_tail_preview_chars,
        )
        if preview:
            return f"[{prefix_len} shared prefix tokens with {label} ({preview})]"
        return f"[{prefix_len} shared prefix tokens with {label}]"

    def elide_messages(self, key: str, messages: list[Any]) -> list[Any]:
        previous = self._previous_messages.get(key)
        self._previous_messages[key] = messages
        if previous is None:
            return messages
        prefix_count = _common_json_prefix_len(previous, messages)
        if prefix_count == 0:
            return messages
        elided_prefix = [_message_with_elided_content(message, preview_chars=self.prefix_tail_preview_chars) for message in messages[:prefix_count]]
        return [*elided_prefix, *messages[prefix_count:]]


def _elide_json_prefixes(value: Any, *, prefix_elider: _PrefixElider, stream_key: str) -> Any:
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, item in value.items():
            item_key = f"{stream_key}.{key}"
            if key == "messages" and isinstance(item, list):
                result[key] = prefix_elider.elide_messages("messages", item)
            elif key in {"input_ids", "output_ids"} and _is_int_list(item):
                result[key] = _token_list_preview([int(token_id) for token_id in item])
            elif key == "output_logprobs" and _is_number_list(item):
                result[key] = _number_list_preview([float(value) for value in item], value_name="logprob", plural_name="logprobs")
            elif key in {"content", "text", "output_text", "output_text_with_special_tokens", "thought", "command", "stdout"} and isinstance(
                item, str
            ):
                result[key] = prefix_elider.elide_text(item_key, item, label=f"previous {key}")
            else:
                result[key] = _elide_json_prefixes(item, prefix_elider=prefix_elider, stream_key=item_key)
        return result
    if isinstance(value, list):
        if _is_int_list(value):
            return _token_list_preview([int(token_id) for token_id in value])
        return [_elide_json_prefixes(item, prefix_elider=prefix_elider, stream_key=f"{stream_key}[]") for item in value]
    return value


def _is_int_list(value: Any) -> bool:
    return isinstance(value, list) and bool(value) and all(isinstance(item, int) for item in value)


def _is_number_list(value: Any) -> bool:
    return isinstance(value, list) and bool(value) and all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in value)


def _token_list_preview(token_ids: Sequence[int], *, head: int = 4, tail: int = 4) -> str:
    token_count = len(token_ids)
    if token_count == 0:
        return "[] (total 0 tokens.)"
    if token_count <= head + tail + 1:
        body = ", ".join(str(token_id) for token_id in token_ids)
    else:
        head_text = ", ".join(str(token_id) for token_id in token_ids[:head])
        tail_text = ", ".join(str(token_id) for token_id in token_ids[-tail:])
        body = f"{head_text}, ..., {tail_text}"
    token_word = "token" if token_count == 1 else "tokens"
    return f"[{body} (total {token_count} {token_word}.)]"


def _number_list_preview(values: Sequence[float], *, value_name: str, plural_name: str, head: int = 4, tail: int = 4) -> str:
    value_count = len(values)
    if value_count == 0:
        return f"[] (total 0 {plural_name}.)"
    if value_count <= head + tail + 1:
        body = ", ".join(_format_number(value) for value in values)
    else:
        head_text = ", ".join(_format_number(value) for value in values[:head])
        tail_text = ", ".join(_format_number(value) for value in values[-tail:])
        body = f"{head_text}, ..., {tail_text}"
    unit = value_name if value_count == 1 else plural_name
    return f"[{body} (total {value_count} {unit}.)]"


def _format_number(value: float) -> str:
    return f"{value:.6g}"


def _common_token_prefix_len(left: Sequence[int], right: Sequence[int]) -> int:
    limit = min(len(left), len(right))
    for index in range(limit):
        if int(left[index]) != int(right[index]):
            return index
    return limit


def _common_json_prefix_len(left: Sequence[Any], right: Sequence[Any]) -> int:
    limit = min(len(left), len(right))
    for index in range(limit):
        if left[index] != right[index]:
            return index
    return limit


def _common_text_prefix_len(left: str, right: str) -> int:
    limit = min(len(left), len(right))
    for index in range(limit):
        if left[index] != right[index]:
            return _trim_text_prefix_to_boundary(left, index)
    return _trim_text_prefix_to_boundary(left, limit)


def _trim_text_prefix_to_boundary(text: str, prefix_len: int) -> int:
    while prefix_len > 0 and prefix_len < len(text) and not text[prefix_len - 1].isspace():
        prefix_len -= 1
    return prefix_len


def _estimated_text_tokens(text: str) -> int:
    return len(text.split())


def _message_with_elided_content(message: Any, *, preview_chars: int) -> Any:
    if not isinstance(message, dict):
        return message
    elided_message = dict(message)
    content_text = _message_content_text(message.get("content"))
    token_count = _estimated_text_tokens(content_text)
    preview = _text_tail_preview(content_text, char_limit=preview_chars)
    if preview:
        elided_message["content"] = f"[content elided: about {token_count} tokens shared with previous input ({preview})]"
    else:
        elided_message["content"] = f"[content elided: about {token_count} tokens shared with previous input]"
    return elided_message


def _message_content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict):
                text = part.get("text")
                if text is not None:
                    parts.append(str(text))
                else:
                    parts.append(json.dumps(part, ensure_ascii=False, sort_keys=True))
            elif part is not None:
                parts.append(str(part))
        return "\n".join(parts)
    if content is None:
        return ""
    return str(content)


def _text_tail_preview(text: str, *, char_limit: int) -> str:
    if char_limit <= 0:
        return ""
    preview = " ".join(text.split())
    if not preview:
        return ""
    if len(preview) <= char_limit:
        return preview
    return "..." + preview[-(char_limit - 3) :]


def _decoded_prefix_tail_preview(
    token_ids: Sequence[int],
    *,
    token_decoder: Callable[[Sequence[int]], str] | None,
    char_limit: int,
    token_window: int = 256,
) -> str:
    if token_decoder is None or char_limit <= 0 or not token_ids:
        return ""
    tail_ids = token_ids[-token_window:]
    try:
        decoded = token_decoder(tail_ids)
    except Exception:  # pragma: no cover - defensive report path
        return ""
    return _text_tail_preview(decoded, char_limit=char_limit)


def _jsonable_for_report(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _jsonable_for_report(dataclasses.asdict(cast("Any", value)))
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable_for_report(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable_for_report(item) for item in value]
    if isinstance(value, set):
        return sorted(_jsonable_for_report(item) for item in value)
    if isinstance(value, Path):
        return str(value)
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return _jsonable_for_report(model_dump(mode="json"))
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    try:
        json.dumps(value)
    except TypeError:
        return repr(value)
    return value
