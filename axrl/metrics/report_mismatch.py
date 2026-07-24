import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from tqdm import tqdm
from transformers import AutoProcessor

from axrl.configs import GrpoTrainerConfig, ModelConfig, StrictBaseModel
from axrl.data import SampleTensorDict
from axrl.utils import setup_logger, zst_utils
from axrl.utils.config_utils import load_and_validate_config
from axrl.utils.gpu_utils import GpuUsageInfo
from axrl.utils.logger.metric_logger import MetricLogger

if TYPE_CHECKING:
    import torch


logger = logging.getLogger(__name__)

SCATTER_PLOT_ALPHA = 0.5


def _make_legend_transparent(ax: Axes) -> None:
    legend = ax.get_legend()
    if legend is None:
        return
    frame = legend.get_frame()
    frame.set_facecolor("none")
    frame.set_alpha(0)
    frame.set_edgecolor("none")


class MismatchReportTask(StrictBaseModel):
    exp_sample_data_path: str = "tmp/mismatch-test/results/data/baseline-64/training_samples-baseline-64.zst"
    exp_name: str = "exp_name"
    baseline_sample_data_path: str | None = None
    baseline_name: str = "baseline"
    output_fig_path: str = "tmp/mismatch-test/results/figs/kl_mismatch_report.png"
    output_txt_path: str = "tmp/mismatch-test/results/log/top_mismatch.log"
    model: ModelConfig = ModelConfig(
        name="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        seq_length=1024 * 16,
        trust_remote_code=True,
    )


@dataclass
class MCoreRunMetrics:
    end_to_end_time_sec: float | None = None
    old_logprobs_gpu_usage: list[GpuUsageInfo] = field(default_factory=list)


@dataclass
class MismatchRunResult:
    success: bool
    grpo_config: GrpoTrainerConfig | None = None
    metadata: dict[str, Any] | None = None
    response_metrics: dict[str, float] | None = None
    rollout_throughput: float | None = None
    mcore: MCoreRunMetrics | None = None
    mismatch_metrics: dict[str, float] | None = None


@dataclass
class TokenInfo:
    sequence_id: int
    pos: int
    token_id: int
    rollout_logprob: float
    model_logprob: float
    loss_mask: bool
    score: float
    word: str | None = None


def get_token_infos(
    samples: SampleTensorDict,
) -> pd.DataFrame:
    token_infos: list[TokenInfo] = []
    for seq_id, sample in tqdm(enumerate(samples), total=len(samples), desc="Extracting token infos"):
        attention_mask: torch.Tensor = sample["attention_mask"]
        seq_len: int = attention_mask.sum().item()  # type: ignore
        input_ids = sample["input_ids"].tolist()[:seq_len]
        rollout_logprobs = sample["rollout_logprobs"].tolist()[:seq_len]
        model_logprobs = sample["ref_logprobs"].tolist()[:seq_len]
        loss_mask = sample["loss_mask"].tolist()[:seq_len]
        score = sample["reward"].item()
        for pos in range(seq_len):
            token_info = TokenInfo(
                sequence_id=seq_id,
                pos=pos,
                token_id=input_ids[pos],
                rollout_logprob=0 if pos == 0 else rollout_logprobs[pos - 1],
                model_logprob=0 if pos == 0 else model_logprobs[pos - 1],
                loss_mask=False if pos == 0 else bool(loss_mask[pos - 1]),
                score=score,
            )
            token_infos.append(token_info)
    logger.info(f"Extracted {len(token_infos)} token infos from {len(samples)} samples.")
    data = pd.DataFrame([asdict(ti) for ti in token_infos])
    logger.info(f"Created DataFrame with shape {data.shape}, columns: {data.columns.tolist()}.")
    return data


def _report_logprob_mismatch(data: pd.DataFrame, ax: Axes, hue_order: list[str]) -> None:
    sns.histplot(
        data=data,
        x=data["rollout_logprob"] - data["model_logprob"],
        bins=100,
        ax=ax,
        stat="count",
        element="step",
        hue="Source",
        hue_order=hue_order,
        alpha=0.8,
        fill=False,
    )
    ax.set_title("Rollout_Logprob - Model_Logprob")
    ax.set_xlabel("Logprob Mismatch")
    ax.set_ylabel("Count")
    ax.set_yscale("log")
    ax.axvline(0, color="red", linestyle="--", linewidth=1)  # vertical line at 0
    _make_legend_transparent(ax)


def _report_pos_to_kl_mismatch(
    data: pd.DataFrame, agg_type: Literal["mean", "std", "max"], ax: Axes, hue_order: list[str], num_bins: int = 100
) -> None:
    data["pos_quantile_bin"] = pd.qcut(data["pos"], q=num_bins, duplicates="drop")
    kl_by_pos = data.groupby(["pos_quantile_bin", "Source"]).agg(pos_mean=("pos", "mean"), KL1=("KL1", agg_type)).reset_index()
    sns.lineplot(
        data=kl_by_pos,
        x="pos_mean",
        y="KL1",
        hue="Source",
        hue_order=hue_order,
        ax=ax,
        alpha=0.6,
    )
    ax.set_title(f"Token Position vs KL1 ({agg_type})")
    ax.set_xlabel("Token Position")
    ax.set_ylabel(f"KL Mismatch ({agg_type})")
    if agg_type == "max":
        ax.set_yscale("log")
    _make_legend_transparent(ax)


def _report_rollout_prob_to_kl_mismatch(
    data: pd.DataFrame, agg_type: Literal["mean", "std", "max"], ax: Axes, hue_order: list[str], num_bins: int = 100
) -> None:
    data["rollout_prob_quantile_bin"] = pd.qcut(data["rollout_prob"], q=num_bins, duplicates="drop")
    kl_by_rollout_prob = (
        data.groupby(["rollout_prob_quantile_bin", "Source"]).agg(rollout_prob_mean=("rollout_prob", "mean"), KL1=("KL1", agg_type)).reset_index()
    )
    sns.lineplot(
        data=kl_by_rollout_prob,
        x="rollout_prob_mean",
        y="KL1",
        hue="Source",
        hue_order=hue_order,
        ax=ax,
        alpha=0.6,
    )
    ax.set_title(f"Rollout Probability vs KL1 ({agg_type})")
    ax.set_xlabel("Rollout Probability")
    ax.set_ylabel(f"KL Mismatch ({agg_type})")
    _make_legend_transparent(ax)


def _report_top_mismatch_logprob_to_prob(top_data: pd.DataFrame, top_size: int, ax: Axes, hue_order: list[str]) -> None:
    sns.scatterplot(
        data=top_data,
        x=top_data["rollout_logprob"],
        y=top_data["model_logprob"],
        hue="Source",
        hue_order=hue_order,
        ax=ax,
        alpha=SCATTER_PLOT_ALPHA,
        s=15,
        linewidth=0,
        marker="o",
    )
    ax.set_title(f"Top {top_size} Mismatch: Logprob vs Logprob")
    ax.set_xlabel("Rollout Logprob")
    ax.set_ylabel("Model Logprob")
    ax.grid(visible=True, alpha=0.2)
    min_logprob = min(top_data["rollout_logprob"].min(), top_data["model_logprob"].min())
    max_logprob = max(top_data["rollout_logprob"].max(), top_data["model_logprob"].max())
    ax.plot([min_logprob, max_logprob], [min_logprob, max_logprob], color="red", linestyle="--", linewidth=1)
    _make_legend_transparent(ax)


def _report_top_mismatch_pos_to_kl_mismatch(top_data: pd.DataFrame, top_size: int, ax: Axes, hue_order: list[str]) -> None:
    sns.scatterplot(
        data=top_data,
        x=top_data["pos"],
        y=top_data["KL1"],
        hue="Source",
        hue_order=hue_order,
        ax=ax,
        alpha=SCATTER_PLOT_ALPHA,
        s=15,
        linewidth=0,
        marker="o",
    )
    ax.set_title(f"Top {top_size} Mismatch: Position vs KL1")
    ax.set_xlabel("Token Position")
    ax.set_ylabel("KL Mismatch")
    ax.set_yscale("log")
    ax.grid(visible=True, alpha=0.2)
    _make_legend_transparent(ax)


def _report_top_mismatch_rollout_prob_to_ratio(top_data: pd.DataFrame, top_size: int, ax: Axes, hue_order: list[str]) -> None:
    sns.scatterplot(
        data=top_data,
        x=top_data["rollout_prob"],
        y=top_data["ratio"] - 1,
        hue="Source",
        hue_order=hue_order,
        ax=ax,
        alpha=SCATTER_PLOT_ALPHA,
        s=15,
        linewidth=0,
        marker="o",
    )
    ax.set_title(f"Top {top_size} Mismatch: Rollout_Prob vs (Ratio-1)")
    ax.set_xlabel("Rollout Probability")
    ax.set_ylabel("Importance Sampling Ratio")
    ax.set_yscale("symlog", linthresh=2)
    ax.grid(visible=True, alpha=0.2)
    # add a horizontal line at y=0
    ax.axhline(0, color="red", linestyle="--", linewidth=1)
    _make_legend_transparent(ax)


def prepare_data(name_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    datas: list[pd.DataFrame] = []
    # merge dataset
    for name, data in name_data.items():
        logger.info(f"Data '{name}': shape={data.shape}.")
        data["Source"] = name
        datas.append(data)
    data = pd.concat(datas, ignore_index=True)
    logger.info(f"Merged data shape: {data.shape}.")

    data["rollout_prob"] = np.exp(data["rollout_logprob"])
    data["model_prob"] = np.exp(data["model_logprob"])

    all_data = data
    data = all_data[all_data["loss_mask"]]
    data["KL1"] = (data["rollout_logprob"] - data["model_logprob"]).abs()
    data["ratio"] = np.exp(data["rollout_logprob"] - data["model_logprob"])
    return data


def get_top_kl_data(data: pd.DataFrame, top_size: int) -> pd.DataFrame:
    top_dfs = []
    # get top mismatch by kl for each group
    for _, group in data.groupby("Source"):
        top_mismatch = group.nlargest(top_size, "KL1")
        top_dfs.append(top_mismatch)

    top_data = pd.concat(top_dfs, ignore_index=True)
    return top_data


def get_hue_order(config: MismatchReportTask) -> list[str]:
    hue_order = [config.exp_name]
    if config.baseline_sample_data_path is not None:
        assert config.baseline_name is not None
        hue_order.insert(0, config.baseline_name)
    return hue_order


def report_kl_mismatch(
    config: MismatchReportTask,
    name_data: dict[str, pd.DataFrame],
    output_path: Path,
    *,
    metric_logger: MetricLogger | None = None,
    step: int = 0,
) -> dict[str, float]:
    logger.info(f"Generating KL mismatch report to {output_path}...")
    data = prepare_data(name_data)

    mean_kl = {str(k): float(v) for k, v in data.groupby("Source")["KL1"].mean().to_dict().items()}
    std_kl = {str(k): float(v) for k, v in data.groupby("Source")["KL1"].std().to_dict().items()}
    max_kl = {str(k): float(v) for k, v in data.groupby("Source")["KL1"].max().to_dict().items()}
    fig, axes = plt.subplots(3, 3, figsize=(12, 8))
    title = "KL1 (mean/std/max): " + ", ".join(f"{k}: ({mean_kl[k]:.4f}/{std_kl[k]:.4f}/{max_kl[k]:.4f})" for k in sorted(mean_kl))
    fig.suptitle(title)
    top_size: int = 200
    num_bins: int = 50
    top_data = get_top_kl_data(data, top_size=top_size)
    hue_order = get_hue_order(config)

    _report_logprob_mismatch(data, ax=axes[0, 0], hue_order=hue_order)
    _report_top_mismatch_logprob_to_prob(top_data, top_size=top_size, ax=axes[0, 1], hue_order=hue_order)
    _report_top_mismatch_pos_to_kl_mismatch(top_data, top_size=top_size, ax=axes[0, 2], hue_order=hue_order)

    _report_pos_to_kl_mismatch(data, agg_type="mean", ax=axes[1, 0], hue_order=hue_order, num_bins=num_bins)
    _report_pos_to_kl_mismatch(data, agg_type="std", ax=axes[1, 1], hue_order=hue_order, num_bins=num_bins)
    _report_pos_to_kl_mismatch(data, agg_type="max", ax=axes[1, 2], hue_order=hue_order, num_bins=num_bins)

    _report_rollout_prob_to_kl_mismatch(data, agg_type="mean", ax=axes[2, 0], hue_order=hue_order, num_bins=num_bins)
    _report_rollout_prob_to_kl_mismatch(data, agg_type="std", ax=axes[2, 1], hue_order=hue_order, num_bins=num_bins)
    _report_rollout_prob_to_kl_mismatch(data, agg_type="max", ax=axes[2, 2], hue_order=hue_order, num_bins=num_bins)

    for ax in axes.flat:
        _make_legend_transparent(ax)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=300)
    logger.info(f"Saved KL mismatch report to {output_path}.")

    metrics: dict[str, float] = {}
    metrics.update({f"KL-Mismatch/mean_KL1_{k}": v for k, v in mean_kl.items()})
    metrics.update({f"KL-Mismatch/std_KL1_{k}": v for k, v in std_kl.items()})
    metrics.update({f"KL-Mismatch/max_KL1_{k}": v for k, v in max_kl.items()})
    logger.info(f"KL mismatch metrics: {metrics}.")
    if metric_logger is not None:
        metric_logger.log_image(output_path.stem, fig, step=step)
    plt.close(fig)
    return metrics


def print_top_mismatch(
    name_data: dict[str, pd.DataFrame],
    model_config: ModelConfig,
    txt_path: Path,
    top_k: int = 20,
    context_window_one_side: int = 20,
) -> None:
    processor: Any = AutoProcessor.from_pretrained(
        pretrained_model_name_or_path=model_config.get_full_path(),
        use_fast=True,
    )

    outputs: list[str] = []
    for name, all_data in name_data.items():
        all_ids = all_data["token_id"].unique().tolist()
        id_to_word: dict[int, str] = {}
        # decode all unique token ids
        for token_id in tqdm(all_ids, desc=f"Decoding tokens for {name}"):
            word = processor.decode(token_ids=[token_id], skip_special_tokens=False)
            id_to_word[token_id] = word
        all_data["word"] = all_data["token_id"].map(id_to_word)
        all_data["rollout_prob"] = np.exp(all_data["rollout_logprob"])
        all_data["model_prob"] = np.exp(all_data["model_logprob"])
        data = all_data[all_data["loss_mask"]]
        data["KL1"] = (data["rollout_logprob"] - data["model_logprob"]).abs()
        outputs.append(f"########## Report for {name} ##########")
        outputs.append(f"Total tokens: {len(all_data)}, Tokens with loss_mask: {all_data['loss_mask'].sum()}")
        outputs.append(f"Mean KL1 Mismatch: {data['KL1'].mean():.4f}")
        outputs.append(f"Max KL1 Mismatch: {data['KL1'].max():.4f}")
        outputs.append(f"=== Top {top_k} Mismatch Tokens ===")

        top_mismatch = data.nlargest(top_k, "KL1")

        for _, row in top_mismatch.iterrows():
            sequence_id = int(row["sequence_id"])
            pos = int(row["pos"])
            begin_pos = max(0, pos - context_window_one_side)
            end_pos = pos + context_window_one_side + 1
            case_data = all_data[(all_data["sequence_id"] == sequence_id) & (all_data["pos"] >= begin_pos) & (all_data["pos"] < end_pos)]
            case_data = case_data.sort_values("pos")
            outputs.append(f"-- Sequence ID: {sequence_id}, Token Pos: {pos}, KL Mismatch: {row['KL1']:.4f} --")
            full_text = "".join(case_data["word"].tolist())
            outputs.append(f"Full Text Context:\n```{full_text}```\n")
            # outputs.append(f"{'Pos':>5} {'Token_ID':>10} {'Rollout_Logprob':>20} {'Model_Logprob':>20} {'Loss_Mask':>10} {'Score':>5}")
            outputs.append(
                f"{'Pos':>10} {'Token_ID(Word)':>20} {'Rollout_Logprob':>20} {'Model_Logprob':>20} {'KL1':>15} {'Rollout_Prob':>15} {'Model_Prob':>15} {'Loss_Mask':>10} {'Score':>5}"  # noqa: E501
            )
            for _, case_row in case_data.iterrows():
                token_id = int(case_row["token_id"])
                rollout_logprob = case_row["rollout_logprob"]
                model_logprob = case_row["model_logprob"]
                rollout_prob = case_row["rollout_prob"]
                model_prob = case_row["model_prob"]
                loss_mask = case_row["loss_mask"]
                score = case_row["score"]
                word = case_row["word"]
                kl1 = abs(rollout_logprob - model_logprob)
                token_str = f"{token_id}"
                token_str += f"('{word}')"
                highlight = ">>" if int(case_row["pos"]) == pos else "  "
                outputs.append(
                    f"{highlight} {case_row['pos']:>10} {token_str:>20} {rollout_logprob:>20.4f} {model_logprob:>20.4f} {kl1:>15.4f} {rollout_prob:>15.6f} {model_prob:>15.6f} {loss_mask:>10} {score:>5} "  # noqa: E501
                )
            outputs.append(f"-- End of Sequence ID: {sequence_id} --\n\n")
        outputs.append(f"########## End of Report for {name} ##########\n\n")

    txt_path.parent.mkdir(parents=True, exist_ok=True)
    txt_path.write_text("\n".join(outputs))
    logger.info(f"Saved top mismatch report to {txt_path}.")


def report_mismatch(
    config: MismatchReportTask,
    *,
    metric_logger: MetricLogger | None = None,
    step: int = 0,
) -> dict[str, float]:
    name_path: dict[str, Path] = {
        config.exp_name: Path(config.exp_sample_data_path),
    }
    if config.baseline_sample_data_path is not None:
        assert Path(config.baseline_sample_data_path).exists(), f"Baseline path {config.baseline_sample_data_path} does not exist."
        name_path[config.baseline_name] = Path(config.baseline_sample_data_path)
    fig_path = Path(config.output_fig_path)
    name_data: dict[str, pd.DataFrame] = {}
    for name, path in name_path.items():
        samples: SampleTensorDict = zst_utils.load_zst(path)
        logger.info(f"Loaded dataset from {path} with {len(samples)} samples.")
        data = get_token_infos(samples)
        logger.info(f"Get {len(data)} token infos for '{name}'.")
        name_data[name] = data

    print_top_mismatch(
        name_data,
        model_config=config.model,
        txt_path=Path(config.output_txt_path),
        top_k=20,
        context_window_one_side=20,
    )

    metrics = report_kl_mismatch(config, name_data, fig_path, metric_logger=metric_logger, step=step)
    return metrics


if __name__ == "__main__":
    setup_logger("info")
    config: MismatchReportTask = load_and_validate_config(MismatchReportTask)
    logger.info(f"Loaded config: {config}")
    report_mismatch(config)
