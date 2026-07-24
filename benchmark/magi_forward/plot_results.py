"""Plot benchmark results saved by ``run_magi_forward_benchmark.py``.

Reads ``--input`` CSV, emits Fig 1-4 as PNG to ``--output-dir``. Style is
intentionally minimal (similar to ``axrl/metrics/report_mismatch.py``):
seaborn lineplot with markers, hue = method.

Each figure title states the global batch size, micro-batch size, and cp
so the plot is self-contained.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

logger = logging.getLogger(__name__)


GLOBAL_BATCH_SIZE = 128  # trajectories per global step (both sweeps)
MICRO_BATCH_SIZE = 2
CONTEXT_PARALLEL = 8
METHOD_HUE_ORDER = ["magi_merged", "gptmodel_forward"]


def _plot_metric(
    df: pd.DataFrame,
    *,
    sweep: str,
    metric_column: str,
    y_label: str,
    title: str,
    out_path: Path,
    x_label: str,
) -> None:
    rows = df[df["sweep"] == sweep].copy()
    if rows.empty:
        logger.warning("no rows for sweep=%s; skipping %s", sweep, out_path)
        return
    rows = rows.sort_values("size")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.lineplot(
        data=rows,
        x="size",
        y=metric_column,
        hue="method",
        hue_order=[m for m in METHOD_HUE_ORDER if m in rows["method"].unique()],
        marker="o",
        ax=ax,
    )
    ax.set_xscale("log", base=2)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(visible=True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    logger.info("wrote %s", out_path)


def _load_results(csv_path: str) -> pd.DataFrame:
    """Tolerant CSV loader: error column can contain commas (CUDA OOM messages).

    The writer doesn't quote, so we split with maxsplit=5 to match the
    writer's column count.
    """
    rows: list[dict] = []
    with Path(csv_path).open() as f:
        next(f, None)  # skip header
        for line in f:
            parts = line.rstrip("\n").split(",", 5)
            if len(parts) != 6:
                continue
            sweep, size_str, method, time_str, mem_str, error = parts
            rows.append(
                {
                    "sweep": sweep,
                    "size": int(size_str),
                    "method": method,
                    "median_time_s": float(time_str) if time_str else None,
                    "peak_mem_reserved_gb": float(mem_str) if mem_str else None,
                    "error": error,
                },
            )
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="tmp/magi-bench/results.csv")
    parser.add_argument("--output-dir", default="tmp/magi-bench")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    df = _load_results(args.input)
    df = df[df["error"] == ""]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    flat_suffix = f" (gbs={GLOBAL_BATCH_SIZE} trajectories, mbs={MICRO_BATCH_SIZE}, cp={CONTEXT_PARALLEL})"
    merged_suffix = f" (gbs={GLOBAL_BATCH_SIZE} trajectories, mbs={MICRO_BATCH_SIZE}, cp={CONTEXT_PARALLEL}, ~1K tok/turn)"

    _plot_metric(
        df,
        sweep="flat",
        metric_column="median_time_s",
        y_label="median compute_logprobs time (s)",
        title="Fig 1: flat samples - time vs seq length" + flat_suffix,
        out_path=out_dir / "fig1.png",
        x_label="seq length per sample (tokens)",
    )
    _plot_metric(
        df,
        sweep="flat",
        metric_column="peak_mem_reserved_gb",
        y_label="peak GPU memory reserved (GB)",
        title="Fig 2: flat samples - peak memory vs seq length" + flat_suffix,
        out_path=out_dir / "fig2.png",
        x_label="seq length per sample (tokens)",
    )
    _plot_metric(
        df,
        sweep="merged",
        metric_column="median_time_s",
        y_label="median compute_logprobs time (s)",
        title="Fig 3: merged samples - time vs turns" + merged_suffix,
        out_path=out_dir / "fig3.png",
        x_label="turns per conversation",
    )
    _plot_metric(
        df,
        sweep="merged",
        metric_column="peak_mem_reserved_gb",
        y_label="peak GPU memory reserved (GB)",
        title="Fig 4: merged samples - peak memory vs turns" + merged_suffix,
        out_path=out_dir / "fig4.png",
        x_label="turns per conversation",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
