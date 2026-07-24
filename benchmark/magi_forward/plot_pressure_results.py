"""Plot the pressure-test results from `pressure_test.py`.

Figures:
    fig0: max merged-tree path length vs turns (inference matrix sweep)
    fig1: per-trajectory packed tokens vs turns
    fig2: inference time vs turns
    fig3: inference peak memory vs turns
    fig4: training time vs turns
    fig5: training peak memory vs turns
    fig6: training time vs optimizer-offload fraction (bar)
    fig7: training peak memory vs optimizer-offload fraction (bar)
    fig8: training time vs recompute strategy (bar)
    fig9: training peak memory vs recompute strategy (bar)

Usage:
    python -m benchmark.magi_forward.plot_pressure_results \
        --input tmp/magi-bench-pressure/results.csv \
        --train-input tmp/magi-bench-pressure/train_results.csv \
        --offload-input tmp/magi-bench-pressure/train_offload_sweep.csv \
        --recompute-input tmp/magi-bench-pressure/train_recompute_sweep.csv \
        --output-dir tmp/magi-bench-pressure
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter

if TYPE_CHECKING:
    from matplotlib.patches import Rectangle


def _human_compact(value: float, _pos: int = 0) -> str:
    """Format axis ticks as ``50k`` / ``2.5M`` / ``1.2B`` instead of ``50000``."""
    abs_v = abs(value)
    if abs_v >= 1e9:
        return f"{value / 1e9:g}B"
    if abs_v >= 1e6:
        return f"{value / 1e6:g}M"
    if abs_v >= 1e3:
        return f"{value / 1e3:g}k"
    return f"{value:g}"


logger = logging.getLogger(__name__)

TRAJECTORIES = 4
TOKENS_PER_TURN = 1024


def _load_results(csv_path: str) -> pd.DataFrame:
    """Tolerant CSV loader (error column may contain commas; not quoted)."""
    rows: list[dict] = []
    path = Path(csv_path)
    if not path.exists():
        return pd.DataFrame()
    with path.open() as f:
        next(f, None)
        for line in f:
            parts = line.rstrip("\n").split(",", 9)
            if len(parts) != 10:
                continue
            cfg, turns, time_s, mem_gb, sum_packed, max_packed, max_path, seq_length, notes, error = parts
            rows.append(
                {
                    "parallel_config": cfg,
                    "turns": int(turns),
                    "median_time_s": float(time_s) if time_s else None,
                    "peak_mem_reserved_gb": float(mem_gb) if mem_gb else None,
                    "sum_total_padded": int(sum_packed) if sum_packed else None,
                    "max_total_padded": int(max_packed) if max_packed else None,
                    "max_path_length": int(max_path) if max_path else None,
                    "seq_length": int(seq_length) if seq_length else None,
                    "notes": notes,
                    "error": error,
                },
            )
    return pd.DataFrame(rows)


def _plot_vs_turns(
    df: pd.DataFrame,
    *,
    metric_column: str,
    y_label: str,
    title: str,
    out_path: Path,
) -> None:
    if df.empty:
        logger.warning("no rows; skipping %s", out_path)
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.lineplot(
        data=df,
        x="turns",
        y=metric_column,
        hue="parallel_config",
        marker="o",
        ax=ax,
    )
    ax.set_xscale("log", base=2)
    ax.set_xlabel("turns per conversation")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(visible=True, alpha=0.3)
    ax.yaxis.set_major_formatter(FuncFormatter(_human_compact))
    ax.xaxis.set_major_formatter(FuncFormatter(_human_compact))
    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    logger.info("wrote %s", out_path)


def _plot_bar(
    df: pd.DataFrame,
    *,
    label_column: str,
    metric_column: str,
    y_label: str,
    title: str,
    out_path: Path,
) -> None:
    if df.empty:
        logger.warning("no rows; skipping %s", out_path)
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.barplot(data=df, x=label_column, y=metric_column, ax=ax, color="steelblue")
    ax.set_xlabel(label_column)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(visible=True, axis="y", alpha=0.3)
    ax.yaxis.set_major_formatter(FuncFormatter(_human_compact))
    for label in ax.get_xticklabels():
        label.set_rotation(20)
        label.set_horizontalalignment("right")
    # Annotate each bar with its value. ``ax.patches`` here are
    # ``matplotlib.patches.Rectangle`` objects (seaborn's barplot output);
    # cast for the typechecker, which only sees the base ``Patch`` class.

    for patch, value in zip(ax.patches, df[metric_column].tolist(), strict=True):
        if value is None or pd.isna(value):
            continue
        rect = cast("Rectangle", patch)
        ax.annotate(
            _human_compact(float(value)),
            (rect.get_x() + rect.get_width() / 2.0, rect.get_height()),
            ha="center",
            va="bottom",
            fontsize=9,
        )
    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    logger.info("wrote %s", out_path)


def _parse_notes_field(notes: str, key: str) -> str | None:
    """Extract ``key=value`` from a ``;``-separated ``notes`` cell."""
    for raw in notes.split(";"):
        chunk = raw.strip()
        if chunk.startswith(f"{key}="):
            return chunk.split("=", 1)[1]
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="tmp/magi-bench-pressure/results.csv")
    parser.add_argument("--train-input", default="tmp/magi-bench-pressure/train_results.csv")
    parser.add_argument("--offload-input", default="tmp/magi-bench-pressure/train_offload_sweep.csv")
    parser.add_argument("--recompute-input", default="tmp/magi-bench-pressure/train_recompute_sweep.csv")
    parser.add_argument("--output-dir", default="tmp/magi-bench-pressure")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    inf_df = _load_results(args.input)
    if not inf_df.empty:
        inf_df = inf_df[inf_df["error"] == ""].sort_values("turns")
        _plot_vs_turns(
            inf_df,
            metric_column="max_path_length",
            y_label="max path length (tokens)",
            title="Fig 0: max merged-tree path length vs turns",
            out_path=out_dir / "fig0.png",
        )
        _plot_vs_turns(
            inf_df,
            metric_column="max_total_padded",
            y_label="packed tokens / trajectory",
            title="Fig 1: packed tokens per trajectory vs turns",
            out_path=out_dir / "fig1.png",
        )
        _plot_vs_turns(
            inf_df,
            metric_column="median_time_s",
            y_label="time (s)",
            title="Fig 2: inference time vs turns",
            out_path=out_dir / "fig2.png",
        )
        _plot_vs_turns(
            inf_df,
            metric_column="peak_mem_reserved_gb",
            y_label="peak mem (GB)",
            title="Fig 3: inference peak memory vs turns",
            out_path=out_dir / "fig3.png",
        )

    train_df = _load_results(args.train_input)
    if not train_df.empty:
        train_df = train_df[train_df["error"] == ""].sort_values("turns")
        _plot_vs_turns(
            train_df,
            metric_column="median_time_s",
            y_label="time (s)",
            title="Fig 4: training time vs turns",
            out_path=out_dir / "fig4.png",
        )
        _plot_vs_turns(
            train_df,
            metric_column="peak_mem_reserved_gb",
            y_label="peak mem (GB)",
            title="Fig 5: training peak memory vs turns",
            out_path=out_dir / "fig5.png",
        )

    offload_df = _load_results(args.offload_input)
    if not offload_df.empty:
        offload_df = offload_df.copy()
        offload_df["offload_fraction"] = offload_df["notes"].apply(
            lambda s: _parse_notes_field(s, "offload"),
        )
        offload_df = offload_df[offload_df["offload_fraction"].notna()].sort_values("offload_fraction")
        # OOM rows have empty median_time_s; keep them in the bar plot as
        # zero-height entries so the absent bar tells you it failed.
        offload_df["median_time_s"] = offload_df["median_time_s"].fillna(0.0)
        offload_df["peak_mem_reserved_gb"] = offload_df["peak_mem_reserved_gb"].fillna(0.0)
        _plot_bar(
            offload_df,
            label_column="offload_fraction",
            metric_column="median_time_s",
            y_label="time (s)",
            title="Fig 6: training time vs optimizer offload fraction",
            out_path=out_dir / "fig6.png",
        )
        _plot_bar(
            offload_df,
            label_column="offload_fraction",
            metric_column="peak_mem_reserved_gb",
            y_label="peak mem (GB)",
            title="Fig 7: training peak memory vs optimizer offload fraction",
            out_path=out_dir / "fig7.png",
        )

    recompute_df = _load_results(args.recompute_input)
    if not recompute_df.empty:
        recompute_df = recompute_df.copy()
        recompute_df["strategy"] = recompute_df["notes"].apply(
            lambda s: _parse_notes_field(s, "recompute"),
        )
        recompute_df = recompute_df[recompute_df["strategy"].notna()]
        recompute_df["median_time_s"] = recompute_df["median_time_s"].fillna(0.0)
        recompute_df["peak_mem_reserved_gb"] = recompute_df["peak_mem_reserved_gb"].fillna(0.0)
        _plot_bar(
            recompute_df,
            label_column="strategy",
            metric_column="median_time_s",
            y_label="time (s)",
            title="Fig 8: training time vs recompute strategy",
            out_path=out_dir / "fig8.png",
        )
        _plot_bar(
            recompute_df,
            label_column="strategy",
            metric_column="peak_mem_reserved_gb",
            y_label="peak mem (GB)",
            title="Fig 9: training peak memory vs recompute strategy",
            out_path=out_dir / "fig9.png",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
