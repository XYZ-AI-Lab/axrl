---
name: axrl-run-regression-tests
description: Run axrl validation end-to-end by using skills/axrl-fix-linting/SKILL.md first, then skills/axrl-run-gpu-task/SKILL.md to execute scripts/run-tests.sh on a remote GPU node, iterating on failures and benchmark table updates.
argument-hint: Optional context such as ssh='-p 18802 root@10.100.20.14', progress=agent-task/progress-run-tests.md, or specific failing tests/tables
user-invocable: true
---

# axrl-run-regression-tests

Use this skill when the user asks to pass lint and then run the full axrl
test suite on a GPU node, especially when benchmark or mismatch tables may
need updating.

## Required Sub-Skills

Load and follow these repo skills in order:

1. `skills/axrl-fix-linting/SKILL.md`
   - Run local lint/type validation first.
   - Keep fixes minimal and non-semantic.
   - Stop before changing logits, sampling, rewards, or training behavior
     unless the user explicitly approves.
2. `skills/axrl-run-gpu-task/SKILL.md`
   - Use its SSH, environment setup, progress-log, and remote execution
     workflow.
   - Run `bash ltp/install.sh` on each GPU node, then
     `source ~/.axrl_env.sh`, before GPU commands.

This skill only orchestrates those workflows and adds test-suite-specific
retry, logging, and benchmark-table policy.

## Defaults

- Progress file: use an explicit `progress=...` if provided; otherwise use
  `agent-task/progress-run-tests.md`.
- Full test command on the GPU node. `scripts/run-tests.sh` runs pytest with
  stdout passthrough (`-s`), streams pytest INFO logs into the main harness log
  with `--log-level=INFO` and `--log-cli-level=INFO`, and also writes a
  separate pytest INFO log, so expensive regression-table values are available
  without a focused rerun:

```bash
bash scripts/run-tests.sh
```

- Optional override for the pytest INFO log path:

```bash
AXRL_PYTEST_LOG_FILE="${AXRL_OUTPUT_DIR}/pytest-info-custom.log" bash scripts/run-tests.sh
```

- If the user gives an SSH target, use it. Otherwise resolve it through
  `skills/axrl-run-gpu-task/SKILL.md`.
- Do not stage, unstage, commit, reset, or otherwise modify Git index or
  history.

## Workflow

1. Read both required sub-skills.
2. Initialize or update the progress file with the resolved SSH target,
   validation command, current status, and a timestamp.
3. Run `bash scripts/run-precommit-check.sh` locally by following
   `axrl-fix-linting`.
4. Fix lint/type failures minimally, then rerun the lint command until it
   passes or a semantic blocker is clearly recorded.
5. Follow `axrl-run-gpu-task` to prepare the GPU node and verify:
   - `nvidia-smi` works
   - `~/.axrl_env.sh` is sourced
   - `pip show axrl` points at this shared working tree
6. Run the full GPU suite with `bash scripts/run-tests.sh`.
7. While any long GPU run is active, update the progress file at least every
   5 minutes with:
   - running command and remote PID or log path
   - latest pass/fail/error summary
   - current test file or phase, if visible
   - retries or failures since the last update
   - any suspicious numeric results
8. For failures, record the failure in the progress file, rerun the narrowest
   failing test with enough logging, fix the root cause, rerun the focused
   test, then rerun the full suite.

## Benchmark And Mismatch Tables

When tests print replacement numbers for tables such as R3 mismatch,
benchmark, FP8 mismatch, or MoE benchmark snapshots:

- Always check these table-bearing regression files when their tests run or
  fail:
  - `tests/magi-forward/test_magi_forward_r3_mismatch.py`
  - `tests/moe/test_r3_benchmark_results.py`
  - `tests/moe/test_fp8_mismatch_results.py`
- First, try to extract the values from the full-run log files:
  `${AXRL_OUTPUT_DIR}/run-tests-*.log` and
  `${AXRL_OUTPUT_DIR}/pytest-info-*.log`.
- Rerun the focused test only if the full-run logs or persisted artifacts do
  not contain the needed table values. Use logging enabled, for example:

```bash
python -u -X faulthandler -m pytest <test-path-or-nodeid> -vv -s --log-cli-level=INFO --tb=long
```

- Update checked-in table values only from clearly logged test output for the
  same test/configuration.
- Keep table edits mechanical: numbers and surrounding expected text only.
- After a table update, rerun the focused test if the first full run did not
  already prove that exact table output. Then include it in the next full
  regression pass.

## Potential Wrong Results To Report

Call out these even if the test process exits successfully:

- NaN or infinite metrics.
- KL, mismatch, off-policy, or routing values outside the intended range.
- Benchmark scores or throughput that drift sharply from the checked-in table.
- Missing train/eval metric names expected by existing GRPO users.
- Changes that suggest different behavior for the same GRPO config.
- Suspicious sample packing, prefix merging, MoE routing, data shuffling, or
  logging behavior observed in test output.

## Done When

- Local `bash scripts/run-precommit-check.sh` passes.
- Remote `bash scripts/run-tests.sh` passes on the GPU node.
- Any benchmark table updates are supported by focused logged output.
- The progress file records the final commands, evidence, failures/retries,
  potential wrong-result notes, and final status.
