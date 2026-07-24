---
name: axrl-spike-replay
description: Reproduce and analyze AXRL gradient spike snapshots, run mitigation experiments on a GPU node, and document proven spike-mitigation methods.
argument-hint: Provide one or more job names or spike snapshot directories, plus an optional SSH target such as `ssh -p 26461 root@10.100.4.23`.
user-invocable: true
---

# axrl-spike-replay

Use this skill when asked to reproduce, analyze, or mitigate AXRL gradient
spikes from saved `checkpoints/spike_snapshots/` directories.

## Inputs

- Job names, for example `axrl-v002-052915-math-moe-r3`.
- Direct spike snapshot roots, for example
  `/mnt/3fs1/data/<user>/outputs/<run>/checkpoints/spike_snapshots/`.
- Optional GPU SSH target. If supplied, follow `skills/axrl-run-gpu-task/SKILL.md`.

## Workflow

1. Resolve each job to role output directories.
   - Use `skills/axrl-copy-log-to-tmp-and-analyze/SKILL.md` to copy and inspect
     the job logs.
   - Prefer output directories matching `*~<job_name>__<role>`.
2. Find spike snapshots.
   - Look under `<output_dir>/checkpoints/spike_snapshots/iter_*`.
   - Immediately preserve any retained snapshots under a non-rotating sibling
     directory such as `<output_dir>/checkpoints/spike_snapshots-active/`.
     Prefer hardlinked copies on the same filesystem for very large snapshots;
     use the preserved directory for replay and analysis, then delete that
     preserved hardlink directory after all mitigation experiments and final
     documentation are finished.
   - Read `metadata.json`, `run_config.yaml`, `batch_rank*.pt`,
     `grad_info_rank*.pt`, and `routing_payload_rank*.pt`.
   - Record world size, TP/DP/PP/EP/ETP, grad norm, median norm, spike ratio,
     and routing payload counts.
3. Maintain an ongoing task file under `agent-task/spike-replay/`.
   - Track active snapshot, replay command, experiment table, observations,
     failed attempts, and next steps.
4. Reproduce the spike on a GPU node.
   - Prefer the saved `MegatronWorker.reproduce_spike(snapshot_dir)` path.
   - If the saved snapshot world size is larger than the available node, first
     attempt a reduced-rank replay only as a diagnostic and clearly mark it as
     not full-fidelity.
   - Spike replay only needs model weights, saved batch tensors, and routing
     payloads to recompute gradients. Optimizer and RNG state may be skipped
     for replay diagnostics, especially when testing a reduced DP layout.
   - Megatron distributed checkpoints reserve `metadata.json`. AXRL replay
     metadata should live in `axrl_metadata.json`; for old snapshots that wrote
     AXRL metadata to `metadata.json`, create a temporary replay directory that
     restores Megatron's checkpoint metadata and copies the AXRL metadata to
     `axrl_metadata.json`.
   - Keep command output in `tmp/spike-replay/<job>/<timestamp>/`.
5. Analyze the batch before mitigation.
   - Inspect per-sample and per-token signals: advantages, old/ref/rollout
     logprobs, rollout/old importance-sampling ratios, loss mask lengths,
     token IDs, turn indexes, rewards, and routing metadata.
   - Inspect per-rank `grad_info_rank*.pt` for largest parameter grad norms.
6. Try mitigation methods.
   - Prefer patch-based replay experiments that do not permanently alter the
     main training path.
   - Log every attempt in the task file, including failures.
   - Useful families: token/sequence masking by ratio outliers, advantage
     clipping, log-ratio clipping, denominator/aggregation changes,
     `is_base_logprobs` variants, GRPO setting changes, aux/router settings,
     initial-model replay, fixed-routing on/off comparisons, Magi merged-forward
     on/off comparisons when compatible with the saved batch layout, and
     combinations of proven smaller changes.
7. Add proven methods only.
   - Put clean, successful, non-duplicate methods in
     `spike-mitigation-methods/` as one markdown file per method.
   - Include job/role, snapshot path, key signals, patch, replay result,
     residual spike status, and rationale.

## Recorded Methods

- `enable-token-mismatch-tis-max2.md`: best current full-R3 stabilizer for
  `axrl-v002-052915-math-moe-r3` step 85; keeps R3 enabled and sets
  `mismatch_token_clip_max=2`.
- `use-current-routing-for-first-moe-layer.md`: layer-selective R3 mitigation;
  uses current routing for the first global MoE layer and keeps R3 for later
  layers.
- `replay-r3-on-loss-tokens-only.md`: token-selective R3 mitigation; replays
  routing only on `loss_mask=True` output tokens and uses current routing on
  non-loss tokens. Target-passing on step 85, partial on Step1. Its evidence
  includes the Step85 SGLang/MCore prompt-route diagnostics and the normalized
  loss-vs-non-loss logprob mismatch comparison.
- `balance-negative-advantages.md`: partial sign-balance mitigation for batches
  where negative-advantage magnitude dominates.
- `disable-routing-replay.md`: diagnostic mitigation only; useful for causal
  testing, not a clean production fix by itself.

## Output

Report:
- The resolved output directory and selected snapshot.
- Whether full-fidelity replay was possible on the provided GPU node.
- Baseline replay grad norm and relative difference when available.
- Mitigation attempts run, with result metrics.
- Any proven method added to `spike-mitigation-methods/`.
