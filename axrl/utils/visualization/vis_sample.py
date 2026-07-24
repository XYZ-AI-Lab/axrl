from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import plotly.express as px
import streamlit as st
from transformers import AutoProcessor

from axrl.configs import AXRL_DIR
from axrl.trainer.grpo_exp_config import GrpoExperimentConfig
from axrl.utils import setup_logger, zst_utils
from axrl.utils.config_utils import load_and_validate_config

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from axrl.data import Conversation, Sample
    from axrl.metrics.response_metric import ResponseMetric

logger = logging.getLogger(__name__)

GroupFilterType = Literal["pass", "empty_response", "zero_std_all_fail", "zero_std_all_success"]


@dataclass
class SampleData:
    input_ids: list[int]
    chars: list[str]
    label_probs: list[float]
    label_loss_mask: list[bool]


def process_sample(sample: Sample, decoder: Any) -> SampleData:
    attention_mask = sample.attention_mask
    seq_len = sum(1 for mask in attention_mask if mask)
    input_ids = sample.input_ids[:seq_len]
    logprobs = sample.rollout_logprobs
    assert logprobs is not None
    assert sample.loss_mask is not None
    logprobs = logprobs[:seq_len]
    label_probs = [float(np.exp(logprob)) for logprob in logprobs]
    label_loss_mask = sample.loss_mask[:seq_len]
    chars: list[str] = [decoder.decode([int(token_id)], skip_special_tokens=False) for token_id in input_ids]

    input_ids_list = input_ids.tolist()
    label_loss_mask_list = label_loss_mask.tolist()

    return SampleData(
        input_ids=input_ids_list,
        chars=chars,
        label_probs=label_probs,
        label_loss_mask=label_loss_mask_list,
    )


def _visualize_single_sample(st: Any, idx: int, sample: Sample, metric: ResponseMetric, sample_data: SampleData) -> None:
    import html

    st.header(f"Sample {idx + 1}")
    st.write(f"**Reward:** {sample.reward:.4f}")
    if hasattr(sample, "reward_baseline"):
        st.write(f"**Reward Baseline:** {sample.reward_baseline:.4f}")

    st.subheader("Response Metrics")
    st.json(asdict(metric))

    chars = sample_data.chars
    label_probs = sample_data.label_probs
    label_loss_mask = sample_data.label_loss_mask
    input_ids = sample_data.input_ids

    html_content = (
        '<div style="font-family: monospace; line-height: 1.5; white-space: pre-wrap; '
        'word-break: break-all; background-color: #f0f0f0; padding: 10px; border-radius: 5px;">'
    )

    for i in range(len(chars)):
        char = chars[i]
        prob = 1 if i == 0 else label_probs[i - 1]
        loss_mask = False if i == 0 else label_loss_mask[i - 1]

        # Escape HTML and Markdown special characters to prevent rendering issues
        safe_char = html.escape(char)
        safe_char = safe_char.replace("\n", "&#10;")
        safe_char = safe_char.replace("$", "&#36;")
        safe_char = safe_char.replace("`", "&#96;")
        safe_char = safe_char.replace("[", "&#91;")
        safe_char = safe_char.replace("]", "&#93;")

        if not loss_mask:
            bg_color = "#d3d3d3"
        else:
            p = max(0.0, min(1.0, prob))
            if p < 0.5:
                r = 255
                g = int(255 * (p * 2))
                b = 0
            else:
                r = int(255 * (1 - p) * 2)
                g = 255
                b = 0
            bg_color = f"rgba({r}, {g}, {b}, 0.6)"

        # Ensure token_id is a standard python int
        try:
            token_id = int(input_ids[i])
        except Exception:
            token_id = input_ids[i]

        tooltip = f"Index: {i}&#10;Token ID: {token_id}&#10;Prob: {prob:.4f}"
        html_content += f'<span class="token-hover" style="background-color: {bg_color};" title="{tooltip}">{safe_char}</span>'

    html_content += "</div>"
    st.markdown(html_content, unsafe_allow_html=True)
    st.divider()


def visualize_rollouts(all_samples: list[tuple[Sample, ResponseMetric]], tokenizer_path: Path, num_samples: int = 5) -> None:
    st.title("Rollout Visualization")

    # --- Filters ---
    st.sidebar.header("Filters")
    score_filter = st.sidebar.radio("Score Filter", ["All", "Score 0", "Score 1"])
    quality_filter = st.sidebar.radio("Low Quality Filter", ["All", "Low Quality", "High Quality"])

    filtered_samples = []
    for sample, metric in all_samples:
        # Score filter
        if score_filter == "Score 0":
            if metric.score != 0:
                continue
        elif score_filter == "Score 1":
            if metric.score != 1:
                continue

        # Quality filter
        if quality_filter == "Low Quality":
            if not metric.is_low_quality:
                continue
        elif quality_filter == "High Quality":
            if metric.is_low_quality:
                continue

        filtered_samples.append((sample, metric))

    st.write(f"Filtered samples: {len(filtered_samples)} / {len(all_samples)}")

    if not filtered_samples:
        st.warning("No samples match the filters.")
        return

    # --- Overall Statistics ---
    st.header("Overall Statistics (Filtered)")

    # Token Count Histogram
    token_counts = [m.token_count for _, m in filtered_samples]
    fig_tokens = px.histogram(token_counts, nbins=50, title="Token Count Distribution", labels={"value": "Token Count"})
    st.plotly_chart(fig_tokens)

    # Probability Histogram (Sampled for performance)
    all_probs = []
    # Sample up to 100 items to estimate the probability distribution
    stats_sample_indices = np.random.choice(len(filtered_samples), size=min(100, len(filtered_samples)), replace=False)
    for i in stats_sample_indices:
        s, _ = filtered_samples[i]
        if s.rollout_logprobs is not None:
            seq_len = sum(1 for mask in s.attention_mask if mask)
            logprobs = s.rollout_logprobs[:seq_len]
            probs = np.exp(logprobs)
            all_probs.extend(probs)

    if all_probs:
        fig_probs = px.histogram(all_probs, nbins=50, title="Token Probability Distribution (Sampled)", labels={"value": "Probability"})
        st.plotly_chart(fig_probs)

    # --- Visualization ---
    num_to_visualize = min(num_samples, len(filtered_samples))
    indices_to_visualize = np.random.choice(len(filtered_samples), size=num_to_visualize, replace=False)
    samples_to_visualize = [filtered_samples[i] for i in indices_to_visualize]

    st.info(f"Visualizing {len(samples_to_visualize)} samples.")

    decoder = AutoProcessor.from_pretrained(
        pretrained_model_name_or_path=tokenizer_path,
        use_fast=True,
    )

    st.markdown(
        """
        <style>
        .token-hover:hover {
            text-decoration: underline;
            text-decoration-style: solid;
            text-decoration-thickness: 2px;
            color: blue !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    for idx, (sample, metric) in enumerate(samples_to_visualize):
        sample_data = process_sample(sample, decoder)
        _visualize_single_sample(st, idx, sample, metric, sample_data)


def _visualize_rollouts() -> None:
    config = load_and_validate_config(
        GrpoExperimentConfig,
        config_path="axis_recipe/grpo/grpo_config.yaml",
        print_configs=True,
    )
    # await controller.run()
    data_path = AXRL_DIR.output / "grpo" / f"{config.grpo.rollout_save_filename}-latest.zst"
    tokenizer_path = config.rollout_worker.model.get_full_path()
    valid_rollouts: list[Sequence[tuple[Conversation, Sample, ResponseMetric]]] = zst_utils.load_zst(data_path)
    all_samples: list[tuple[Sample, ResponseMetric]] = [(sample, metric) for episode in valid_rollouts for _, sample, metric in episode]
    visualize_rollouts(all_samples, tokenizer_path, num_samples=5)
    # await controller.train_from_snapshot_rollouts()


if __name__ == "__main__":
    setup_logger(level="info")
    _visualize_rollouts()
    # streamlit run axrl/utils/visualization/vis_sample.py
