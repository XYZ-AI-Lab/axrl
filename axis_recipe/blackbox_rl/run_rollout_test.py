from __future__ import annotations

import argparse
import asyncio
import json
import multiprocessing as mp
import sys
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from axis_recipe.blackbox_rl.config import BlackBoxRLConfig
    from axrl.configs import ModelConfig
    from axrl.data import RolloutResult


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a real OpenHands black-box RL rollout smoke test.")
    parser.add_argument("--num_cases", type=int, default=2, help="Number of dataset cases to roll out.")
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    return args


async def _run() -> None:
    args = _parse_args()

    from axis_recipe.blackbox_rl.config import BlackBoxRLConfig
    from axrl.configs import AXRL_DIR
    from axrl.utils import setup_logger
    from axrl.utils.config_utils import load_and_validate_config

    setup_logger("info")
    config = load_and_validate_config(
        BlackBoxRLConfig,
        config_path="axis_recipe/blackbox_rl/blackbox-rl-config.yaml",
        print_configs=False,
    )
    report_dir = _case_report_dir(_controller_output_dir(config, output_root=AXRL_DIR.output))
    case_report_started_at = time.time_ns()
    results = await _run_blackbox_rollout_test(config, num_cases=args.num_cases)

    summary: list[dict[str, Any]] = []
    for idx, result in enumerate(results):
        metric = result.metric
        summary.append(
            {
                "idx": idx,
                "conversation_id": result.conversation.conversation_id,
                "score": getattr(metric, "score", None),
                "num_model_calls": getattr(metric, "num_model_calls", None),
                "openhands_exit_code": getattr(metric, "openhands_exit_code", None),
                "test_file": result.conversation.extra.get("openhands_test_file"),
                "solution_chars": len(result.conversation.extra.get("openhands_solution", "")),
                "turn_samples": len(result.trace.turn_samples) if result.trace is not None else 0,
            }
        )
    case_paths = sorted(str(path) for path in report_dir.glob("case-*.html") if path.stat().st_mtime_ns >= case_report_started_at)
    print("ROLLOUT_RESULTS_JSON=" + json.dumps(summary, ensure_ascii=False, sort_keys=True), flush=True)
    print("CASE_REPORTS_JSON=" + json.dumps(case_paths, ensure_ascii=False, sort_keys=True), flush=True)
    bad_results = [item for item in summary if not item["num_model_calls"] or not item["turn_samples"]]
    if bad_results:
        raise RuntimeError(f"OpenHands rollout smoke produced no model calls for {len(bad_results)} case(s): {bad_results}")


async def _run_blackbox_rollout_test(config: BlackBoxRLConfig, *, num_cases: int) -> list[RolloutResult]:
    from axis_recipe.blackbox_rl.openhands_case_report import write_openhands_case_report
    from axis_recipe.blackbox_rl.train_blackbox_rl import BlackBoxRLRecipe
    from axrl.configs import AXRL_DIR
    from axrl.pipeline.controller import PipelineController

    controller = PipelineController(config, BlackBoxRLRecipe(config))
    try:
        await controller.initialize()
        if controller.megatron_worker is not None:
            await controller.prepare_for_weight_updates()
            controller.megatron_worker.update_rollout_model_weights()
            await controller.switch_to_rollout()
        if controller.test_datasets:
            dataset = controller.test_datasets[0]
            dataset_config = controller.eval_dataset_configs[0]
            conversations = controller._build_eval_rollout_conversations(dataset, dataset_config, max_rollouts=num_cases)
        else:
            train_dataset = controller.train_dataset
            assert train_dataset is not None, "Blackbox rollout test requires a train or test dataset."
            indices = train_dataset.sample(num_samples=num_cases)
            conversations = []
            for case_idx, dataset_idx in enumerate(indices):
                conv = train_dataset.get_conv(dataset_idx).deep_copy()
                conv.extra["answer"] = train_dataset.get_label(dataset_idx)
                conv.gen_state.sampling_config = controller.config.eval_sampling_config
                conv.gen_state.session_id = f"{conv.conversation_id}:manual-test:{case_idx}"
                conversations.append(conv)
        assert conversations, "Blackbox rollout test requires at least one rollout conversation."
        await controller.enqueue_rollout_conversations(conversations)
        _, result_queue = controller._check_rollout_ready()
        results = await controller._collect_expected_rollout_results(result_queue, expected_result_count=len(conversations))
        report_dir = _case_report_dir(controller.output_dir or _controller_output_dir(config, output_root=AXRL_DIR.output))
        report_dir.mkdir(parents=True, exist_ok=True)
        tokenizer = _load_tokenizer(config.rollout_worker.model)
        for case_idx, result in enumerate(results):
            report_path = report_dir / f"case-{case_idx}-{_safe_name(result.conversation.conversation_id)}.html"
            write_openhands_case_report(result, report_path, token_decoder=lambda token_ids: _decode_token_ids(tokenizer, token_ids))
            print(f"Wrote OpenHands case report: {report_path}", flush=True)
        return results
    finally:
        controller.shutdown()
        await controller.shutdown_recipe()


def _load_tokenizer(model_config: ModelConfig) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        model_config.get_full_path(),
        trust_remote_code=model_config.trust_remote_code,
    )


def _decode_token_ids(tokenizer: Any, token_ids: Sequence[int]) -> str:
    return str(
        tokenizer.decode(
            [int(token_id) for token_id in token_ids],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
    )


def _safe_name(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "-" for ch in value)
    return safe[:96] or "case"


def _controller_output_dir(config: BlackBoxRLConfig, *, output_root: Path) -> Path:
    return output_root / config.controller.output_dir_name


def _case_report_dir(output_dir: Path) -> Path:
    return output_dir / "openhands_cases"


if __name__ == "__main__":
    mp.freeze_support()
    asyncio.run(_run())
