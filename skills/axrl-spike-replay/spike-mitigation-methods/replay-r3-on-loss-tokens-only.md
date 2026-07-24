# Replay R3 On Loss Tokens Only

## Description

Replay rollout routing only where `loss_mask=True`, and use current-model routing on the non-loss/prefix tokens. This keeps R3 on the tokens that directly contribute policy-gradient loss, while removing fixed rollout routing from prompt and shared-prefix context tokens.

The current step85 evidence says the prompt/non-loss issue is not a simple one-request SGLang prefill-vs-decode shift; a focused SGLang-only run showed prompt logprobs and prompt routed experts match full-prefill scoring. The sharper issue is that default SGLang prompt prefill is not batch invariant: identical prompt prefill logprobs and routed experts can vary when the scheduler splits requests into different prefill subbatches. Enabling `ServerArgs.enable_deterministic_inference=True` activates SGLang batch-invariant ops and removes that duplicated-prompt gap in the diagnostic. Loss-token-only R3 avoids replaying scheduler-subbatch-specific prompt routes inside MCore.

## Status

Validated as a strong diagnostic mitigation, with one target-passing replay and one partial replay:

- On `axrl-v002-052915-math-moe-r3` step 85, loss-token-only R3 reduced the no-sequence-mask reduced-DP replay norm from `0.4421454` to `0.1346995`, and the active-mask replay norm to `0.1284255`, below the `normal * 1.5 = 0.1445443` target.
- On the Step1 on-policy spike, loss-token-only R3 reduced the active-path reduced-DP replay norm from `1.8579500` to `0.1941783` (`89.55%` reduction), but did not reach the `normal * 1.5 = 0.1471065` target.

Do not treat this as a universal production fix yet. It is useful when debugging whether non-loss/prefix routing replay is responsible for a spike, and it may be useful in combination with layer-selective routing or other stabilizers.

## Step85 Spike

- Job: `axrl-v002-052915-math-moe-r3`
- Role: `moer3r1`
- Snapshot: `checkpoints/spike_snapshots-active/iter_0000085`
- Saved spike norm: `0.4033715`
- Median recent norm: `0.0963629`
- Target: `0.1445443`
- Replay mode: reduced diagnostic replay mapping original source ranks `24-31` onto 8 GPUs (`dp_size=1`)

## Step1 Spike

- Job: `axrl-v002-052915-math-moe-r3_step1`
- Role: `moer3r1`
- Snapshot: `checkpoints/spike_snapshots-active/iter_0000055`
- Saved spike norm: `0.7231703`
- Median recent norm: `0.0980710`
- Target: `0.1471065`
- Replay mode: reduced diagnostic replay mapping original source ranks `0-7` onto 8 GPUs (`dp_size=1`)

## Evidence

Step85 source ranks `24-31` show a target-passing result when replayed routes are kept only for supervised loss positions:

- full R3 no-sequence-mask replay: `0.4421454`
- R3 only on `loss_mask=True`, no-sequence-mask replay: `0.1346995`
- R3 only on `loss_mask=True`, active-mask replay: `0.1284255`
- R3 only on non-loss/prompt positions, no-sequence-mask replay: `0.1974342`
- boundary-only R3, no-sequence-mask replay: `0.1475855`

The node2 implementation builds the mask as `batch["loss_mask"].bool() & attention_mask`, then packs/dispatches that mask through the same flat or Magi path as the input tokens before selecting replayed vs current router top-k. This confirms the result is supervised-loss-token R3, not shifted-token or boundary-only R3.

The R3-on/off diagnostic should be interpreted carefully:

- R3 on means saved SGLang routes replayed inside MCore.
- R3 off means fresh MCore routes.
- A focused SGLang-only diagnostic showed prompt input logprobs match full-prefill input logprobs with max abs `1.9e-6`, prompt routes have `0` mismatched tokens in every layer, assistant decode logprobs match full-prefill scoring exactly at shift `0`, and assistant routes have `0` mismatched tokens in every layer.
- A second SGLang-only diagnostic with duplicated identical prompts showed prompt prefill batch-shape sensitivity: within a batch14 call split into prefill subbatches of `4` and `10`, prompt logprob mean abs diff was `0.072290` and prompt route ordered top-k match was only `65.6005%` despite `98.2789%` slot-in-set overlap.
- Re-running that duplicated-prompt diagnostic with SGLang batch-invariant mode enabled reduced the same within-batch14 prompt logprob mean/p95/max diff to `0`, and route ordered top-k match became `100%`.
- In end-to-end MCore alignment, normalized mean-absolute gaps show that the raw non-loss gap is mostly scale-driven by larger prompt NLL. With batch-invariant disabled on both sides, `R3-on vs R3-off` normalized ratio was `0.0784` on loss tokens and `0.0561` on non-loss tokens. With batch-invariant enabled on both sides, it was `0.0701` on loss tokens and `0.0892` on non-loss tokens.
- Batch-invariant mode makes `R3-on vs rollout` much closer for non-loss tokens (`0.0471 -> 0.0165` normalized ratio), but fresh MCore without R3 remains materially different from the SGLang rollout reference.

The prompt/non-loss raw logprob gap is much larger than the supervised loss-token raw gap, but this is mostly explained by prompt tokens having larger NLL. The best current explanation is that SGLang default prefill has scheduler-subbatch-sensitive prompt routes, and even batch-invariant SGLang prompt routes remain different from fresh MCore prompt routing. Loss-only R3 avoids replaying non-loss/prompt routes whose source data can depend on SGLang prefill scheduler subbatch shape.

The Step1 source-rank group `0-7` is also strongly R3-sensitive, but loss-token-only R3 is not sufficient there:

- full R3 active-path replay: `1.8579500`
- R3 off / current routing diagnostic: `0.2182103`
- current routing for first four MoE layers, R3 for layers `4+`: `0.1486031`
- R3 only on `loss_mask=True` tokens: `0.1941783`

The Step1 loss-token-only run logged replay-token-mask rates around `0.76-0.79` on local packed shards, so the token mask was active. The result shows that replaying rollout routes on prompt/non-loss tokens is not required for most of the gradient amplification. However, since the Step1 loss-token-only norm remains above target, the main high-gain path still includes replayed routing on trainable output tokens, especially in early MoE layers.

This should be interpreted together with the early-layer evidence:

- Current routing for first four MoE layers is stronger than loss-token-only R3.
- Loss-token-only R3 is stronger than leaving full R3 unchanged.
- Therefore, the spike is best explained by early-layer replay on the trainable/output path, not by a pure prefix-routing bug.

## Implementation

In replay/debug code, build a token mask from the batch before forward:

```python
replay_token_mask = batch["loss_mask"].bool()
```

Then, inside the router replay hook, compute both replayed top-k and current top-k. Select replayed experts only where `replay_token_mask=True`:

```python
replay_probs = scores.gather(1, replay_indices)
current_probs, current_indices = default_compute_topk(scores, topk, num_groups=num_groups, group_topk=group_topk)
token_mask = replay_token_mask.to(scores.device).view(-1, 1)
probs = torch.where(token_mask, replay_probs, current_probs)
top_indices = torch.where(token_mask, replay_indices, current_indices)
```

The mask must be packed/dispatched through the same path as input tokens:

- flat forward: `preprocess_packed_seqs` plus sequence-parallel scatter
- Magi merged forward: pack by `merge_info`, `magi_attention.dispatch`, then sequence-parallel scatter

## Replay Result

Step85 no-sequence-mask full R3 replay:

- replayed grad norm: `0.4421454`
- target: `0.1445443`

Step85 with R3 only on `loss_mask=True` tokens:

- no-sequence-mask replayed grad norm: `0.1346995`
- active-mask replayed grad norm: `0.1284255`
- no-sequence-mask reduction vs full R3 baseline: `69.54%`
- target result: pass

Step1 active-path full R3 replay:

- replayed grad norm: `1.8579500`
- target: `0.1471065`

Step1 with R3 only on `loss_mask=True` tokens:

- replayed grad norm: `0.1941783`
- reduction vs full R3 baseline: `89.55%`
- still above target by `0.0470718`

Step1 with current routing for the first four MoE layers:

- replayed grad norm: `0.1486031`
- reduction vs full R3 baseline: `92.00%`
- still above target by `0.0014966`

## Validation Notes

For Step1 reruns, treat token-mask artifacts without either an explicit token-mask activation log or verified `routing_replay_token_mask` metadata as invalid. A stale node3 artifact reported the full-R3 value `1.8579500` for output-token masking, but its log had no token-mask activation message and it was generated before the token-mask patch was applied for token-mask-only runs. The confirmed node1 rerun logged the mask and produced `0.1941783`; the node2 step85 artifacts were verified by JSON metadata plus the patch implementation.

This method should be recorded as target-passing for step 85 and partial for Step1. If it passes one snapshot but not another, the right interpretation is that SGLang-prefill prompt/non-loss R3 can be a major spike amplifier, but some spikes still need early-layer or sign/advantage controls on the trainable output-token path.
