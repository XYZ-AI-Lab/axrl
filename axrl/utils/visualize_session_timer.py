"""Visualize structured SessionTimer logs as an interactive Plotly timeline."""

from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path
from typing import cast

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

DEFAULT_LOG_PATH = Path("tmp/2026-05-18-12-search-agent-browsecomp-r3-last-tool_2nodes_397b7117/browsecompr3lasttool.log")
DEFAULT_OUTPUT_DIR = Path("tmp/session-timer-visualization")
SESSION_TIMER_PREFIX = "SessionTimer: "
EVENT_COLUMNS = ["line_no", "timestamp", "event", "session_id", "execution_kind", "function_name", "label", "duration_s"]
SEGMENT_COLUMNS = ["session_id", "function_name", "execution_kind", "label", "start_time", "end_time", "duration_s", "line_no_start"]
COMPLETED_COLUMNS = [*SEGMENT_COLUMNS, "line_no_end"]
ACTIVE_COLUMNS = ["window_start", "window_end", "label", "execution_kind", "active_sessions"]
DURATION_COLUMNS = ["window_start", "window_end", "label", "execution_kind", "mean_duration_s", "completed_count"]

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
TIMESTAMP_RE = re.compile(r"(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
START_RE = re.compile(r"Starting: \[(?P<name>SessionTimer: .*?)\]\s*$")
FINISH_RE = re.compile(r"Finished: \[(?P<name>SessionTimer: .*?)\] in (?P<duration>[0-9.]+) seconds\.?")
ActiveStart = tuple[pd.Timestamp, int, str, str, str]  # timestamp, line_no, execution_kind, function_name, label


def label_for(function_name: str, execution_kind: str) -> str:
    return f"[{execution_kind}] {function_name}"


def parse_session_timer_name(name: str) -> tuple[str, str, str] | None:
    if not name.startswith(SESSION_TIMER_PREFIX):
        return None
    try:
        payload = json.loads(name[len(SESSION_TIMER_PREFIX) :])
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    session_id = payload.get("session_id")
    execution_kind = payload.get("execution_kind")
    function_name = payload.get("function_name")
    if execution_kind not in {"sync", "async"} or not isinstance(session_id, str) or not isinstance(function_name, str):
        return None
    return session_id, execution_kind, function_name


def parse_log(log_path: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    with log_path.open("r", encoding="utf-8", errors="replace") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = ANSI_RE.sub("", raw_line.rstrip())
            timestamp_match = TIMESTAMP_RE.search(line)
            if timestamp_match is None:
                continue
            match = START_RE.search(line)
            event, duration_s = "start", None
            if match is None:
                match = FINISH_RE.search(line)
                if match is None:
                    continue
                event, duration_s = "finish", float(match.group("duration"))
            parsed_name = parse_session_timer_name(match.group("name"))
            if parsed_name is None:
                continue
            session_id, execution_kind, function_name = parsed_name
            rows.append(
                {
                    "line_no": line_no,
                    "timestamp": pd.Timestamp(timestamp_match.group("timestamp")),
                    "event": event,
                    "session_id": session_id,
                    "execution_kind": execution_kind,
                    "function_name": function_name,
                    "label": label_for(function_name, execution_kind),
                    "duration_s": duration_s,
                }
            )
    return pd.DataFrame(rows, columns=EVENT_COLUMNS).sort_values("line_no").reset_index(drop=True)


def append_segment(
    rows: list[dict[str, object]], session_id: str, stack: list[ActiveStart], start_time: pd.Timestamp, end_time: pd.Timestamp
) -> None:
    if not stack or end_time <= start_time:
        return
    top = stack[-1]
    rows.append(
        {
            "session_id": session_id,
            "function_name": top[3],
            "execution_kind": top[2],
            "label": top[4],
            "start_time": start_time,
            "end_time": end_time,
            "duration_s": (end_time - start_time).total_seconds(),
            "line_no_start": top[1],
        }
    )


def find_match(stack: list[ActiveStart], event: pd.Series) -> int | None:
    for idx in range(len(stack) - 1, -1, -1):
        item = stack[idx]
        if item[2] == event["execution_kind"] and item[3] == event["function_name"]:
            return idx
    return None


def build_segments_and_completed(events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if events.empty:
        return pd.DataFrame(columns=SEGMENT_COLUMNS), pd.DataFrame(columns=COMPLETED_COLUMNS)
    stacks: dict[str, list[ActiveStart]] = {}
    last_time: dict[str, pd.Timestamp] = {}
    segments: list[dict[str, object]] = []
    completed: list[dict[str, object]] = []
    analysis_end_time = events["timestamp"].max() + pd.Timedelta(seconds=1)

    for _, event in events.iterrows():
        session_id = event["session_id"]
        timestamp = event["timestamp"]
        stack = stacks.setdefault(session_id, [])
        append_segment(segments, session_id, stack, last_time.setdefault(session_id, timestamp), timestamp)
        last_time[session_id] = timestamp

        if event["event"] == "start":
            stack.append((timestamp, int(event["line_no"]), event["execution_kind"], event["function_name"], event["label"]))
            continue

        match_idx = find_match(stack, event)
        if match_idx is None:
            start_time = timestamp - pd.Timedelta(seconds=float(event["duration_s"])) if pd.notna(event["duration_s"]) else pd.NaT
            line_no_start = None
        else:
            start = stack.pop(match_idx)
            start_time, line_no_start = start[0], start[1]
        duration_s = float(event["duration_s"]) if pd.notna(event["duration_s"]) else (timestamp - start_time).total_seconds()
        completed.append(
            {
                "session_id": session_id,
                "function_name": event["function_name"],
                "execution_kind": event["execution_kind"],
                "label": event["label"],
                "start_time": start_time,
                "end_time": timestamp,
                "duration_s": duration_s,
                "line_no_start": line_no_start,
                "line_no_end": int(event["line_no"]),
            }
        )

    for session_id, stack in stacks.items():
        append_segment(segments, session_id, stack, last_time[session_id], analysis_end_time)
    return pd.DataFrame(segments, columns=SEGMENT_COLUMNS), pd.DataFrame(completed, columns=COMPLETED_COLUMNS)


def build_active_timeline(segments: pd.DataFrame, window_seconds: int) -> pd.DataFrame:
    if segments.empty:
        return pd.DataFrame(columns=ACTIVE_COLUMNS)
    rows: list[dict[str, object]] = []
    window_delta = pd.Timedelta(seconds=window_seconds)
    for segment in segments.itertuples(index=False):
        start_time = cast("pd.Timestamp", segment.start_time)
        end_time = cast("pd.Timestamp", segment.end_time)
        window_start = start_time.floor(f"{window_seconds}s")
        while window_start < end_time:
            window_end = window_start + window_delta
            overlap_s = (min(end_time, window_end) - max(start_time, window_start)).total_seconds()
            if overlap_s > 0:
                rows.append(
                    {
                        "window_start": window_start,
                        "window_end": window_end,
                        "label": segment.label,
                        "execution_kind": segment.execution_kind,
                        "active_sessions": overlap_s / window_seconds,
                    }
                )
            window_start = window_end
    timeline = pd.DataFrame(rows, columns=ACTIVE_COLUMNS)
    if timeline.empty:
        return timeline
    timeline = timeline.groupby(["window_start", "window_end", "label", "execution_kind"], dropna=False, as_index=False).agg(
        active_sessions=("active_sessions", "sum")
    )
    total = timeline.groupby(["window_start", "window_end"], dropna=False, as_index=False).agg(active_sessions=("active_sessions", "sum"))
    total["label"], total["execution_kind"] = "Total active sessions", "total"
    return pd.concat([timeline, total], ignore_index=True).sort_values(["window_start", "label"]).reset_index(drop=True)


def build_duration_timeline(completed: pd.DataFrame, window_seconds: int) -> pd.DataFrame:
    if completed.empty:
        return pd.DataFrame(columns=DURATION_COLUMNS)
    completed = completed.copy()
    completed["window_start"] = completed["end_time"].dt.floor(f"{window_seconds}s")
    completed["window_end"] = completed["window_start"] + pd.Timedelta(seconds=window_seconds)
    return completed.groupby(["window_start", "window_end", "label", "execution_kind"], dropna=False, as_index=False).agg(
        mean_duration_s=("duration_s", "mean"), completed_count=("duration_s", "size")
    )


def ordered_labels(data: pd.DataFrame, value_col: str) -> list[str]:
    return [] if data.empty else data.groupby("label")[value_col].sum().sort_values(ascending=False).index.tolist()


def style_maps(labels: list[str]) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD"]
    dashes = ["solid", "dash", "dot", "dashdot", "longdash"]
    symbols = ["circle", "square", "diamond", "cross", "x", "triangle-up", "triangle-down", "star"]
    return (
        {label: ("#111111" if label == "Total active sessions" else colors[idx % len(colors)]) for idx, label in enumerate(labels)},
        {label: ("solid" if label == "Total active sessions" else dashes[idx % len(dashes)]) for idx, label in enumerate(labels)},
        {label: ("circle" if label == "Total active sessions" else symbols[idx % len(symbols)]) for idx, label in enumerate(labels)},
    )


def build_sorted_hover_data(data: pd.DataFrame, *, x_col: str, y_col: str, title: str) -> pd.DataFrame:
    if data.empty:
        return pd.DataFrame(columns=[x_col, y_col, "hover_html"])
    rows: list[dict[str, object]] = []
    for x_value, group in data.groupby(x_col, sort=True):
        sorted_group = group[group[y_col] > 0].sort_values(y_col, ascending=False)
        if sorted_group.empty:
            continue
        hover_lines = [f"<b>{html.escape(title)}</b>", f"time={x_value}"]
        for _, item in sorted_group.iterrows():
            hover_lines.append(f"{html.escape(str(item['label']))}: <b>{float(item[y_col]):.3f}</b>")
        rows.append({x_col: x_value, y_col: float(sorted_group[y_col].max()), "hover_html": "<br>".join(hover_lines)})
    return pd.DataFrame(rows)


def add_sorted_hover_trace(fig: go.Figure, data: pd.DataFrame, *, x_col: str, y_col: str, title: str, row: int) -> None:
    hover_data = build_sorted_hover_data(data, x_col=x_col, y_col=y_col, title=title)
    if hover_data.empty:
        return
    fig.add_trace(
        go.Scatter(
            x=hover_data[x_col],
            y=hover_data[y_col],
            mode="markers",
            marker={"color": "rgba(0,0,0,0)", "size": 24},
            name=title,
            customdata=hover_data["hover_html"],
            hovertemplate="%{customdata}<extra></extra>",
            showlegend=False,
        ),
        row=row,
        col=1,
    )


def add_traces(
    fig: go.Figure,
    data: pd.DataFrame,
    labels: list[str],
    y_col: str,
    row: int,
    maps: tuple[dict[str, str], dict[str, str], dict[str, str]],
    shown: set[str],
) -> None:
    color_map, dash_map, symbol_map = maps
    for label in labels:
        trace_data = data[data["label"] == label].sort_values("window_start")
        if trace_data.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=trace_data["window_start"],
                y=trace_data[y_col],
                mode="lines+markers",
                name=label,
                legendgroup=label,
                showlegend=label not in shown,
                line={"color": color_map[label], "dash": dash_map[label], "width": 3.2 if label == "Total active sessions" else 1.8},
                marker={"symbol": symbol_map[label], "size": 8 if label == "Total active sessions" else 6},
                hoverinfo="skip",
            ),
            row=row,
            col=1,
        )
        shown.add(label)


def write_plot(active_timeline: pd.DataFrame, duration_timeline: pd.DataFrame, output_html: Path) -> None:
    active_labels = ordered_labels(active_timeline, "active_sessions")
    duration_labels = ordered_labels(duration_timeline, "completed_count")
    maps = style_maps(list(dict.fromkeys(active_labels + duration_labels)))
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.12,
        subplot_titles=("Fig 1: Active sessions by top timer", "Fig 2: Completed timer duration"),
    )
    shown: set[str] = set()
    add_traces(fig, active_timeline, active_labels, "active_sessions", 1, maps, shown)
    add_sorted_hover_trace(fig, active_timeline, x_col="window_start", y_col="active_sessions", title="Fig 1 active sessions", row=1)
    add_traces(fig, duration_timeline, duration_labels, "mean_duration_s", 2, maps, shown)
    add_sorted_hover_trace(fig, duration_timeline, x_col="window_start", y_col="mean_duration_s", title="Fig 2 average duration", row=2)
    fig.update_yaxes(title_text="Average active sessions", row=1, col=1)
    fig.update_yaxes(title_text="Mean duration (s)", row=2, col=1)
    fig.update_xaxes(title_text="Time", row=2, col=1)
    fig.update_layout(
        height=900,
        width=1500,
        template="plotly_white",
        hovermode="x unified",
        hoverlabel={"align": "left", "namelength": -1, "font": {"size": 12}},
        legend={"title": {"text": "timer type"}, "x": 1.02, "y": 1.0},
        margin={"l": 80, "r": 360, "t": 80, "b": 60},
    )
    output_html.parent.mkdir(parents=True, exist_ok=True)
    pio.write_html(fig, output_html, include_plotlyjs=True, full_html=True)
    print(f"Plot written to {output_html}")


def visualize_session_timer(log_path: Path, window_seconds: int, output_dir: Path, output_html: Path | None = None) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    events = parse_log(log_path)
    segments, completed = build_segments_and_completed(events)
    active_timeline = build_active_timeline(segments, window_seconds)
    duration_timeline = build_duration_timeline(completed, window_seconds)
    output_html = output_html or output_dir / "session_timer_timeline.html"
    for name, data in {
        "session_timer_events.csv": events,
        "session_timer_active_segments.csv": segments,
        "session_timer_completed.csv": completed,
        "fig1_active_sessions.csv": active_timeline,
        "fig2_completed_durations.csv": duration_timeline,
    }.items():
        data.to_csv(output_dir / name, index=False)
    write_plot(active_timeline, duration_timeline, output_html)
    print(f"events={len(events)} segments={len(segments)} completed={len(completed)} html={output_html} csv_dir={output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize structured AXRL SessionTimer logs.")
    parser.add_argument("--log-path", type=Path, default=DEFAULT_LOG_PATH)
    parser.add_argument("-k", "--window-seconds", type=int, default=30)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-html", type=Path, default=None)
    args = parser.parse_args()
    if args.window_seconds <= 0:
        raise ValueError("--window-seconds must be positive")
    visualize_session_timer(args.log_path, args.window_seconds, args.output_dir, args.output_html)


if __name__ == "__main__":
    main()

# Run with python -m axrl.utils.visualize_session_timer
