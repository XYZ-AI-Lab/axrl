# Enable Token Mismatch TIS Max 2

## Description

Enable trainer-side rollout/trainer token importance sampling by setting `mismatch_token_clip_max=2`. In `GrpoTrainer`, this changes `token_mismatch_tir` from all ones to `clamp(exp(old_logprobs - rollout_logprobs), max=2)`, and multiplies the policy-gradient loss by that token weight.

For the step-85 spike, this is effectively untruncated token mismatch IS: the observed max `old/rollout` ratio in rank 24 trainable tokens is `1.838`, so no token is actually clipped by the cap of `2`.

## Status

Strong mitigation evidence. On node1, this reduced the logged-config reduced-DP replay norm from `0.4447577` to `0.1480056` (`66.72%` reduction) with the active sequence mask, and from `0.4421454` to `0.1376026` (`68.88%` reduction) with sequence masking disabled. This keeps R3 routing replay enabled.

This is the current best full-R3 stabilizer for `axrl-v002-052915-math-moe-r3` step 85. Validate on additional retained snapshots before making it a default training setting.

## Spike

- Job: `axrl-v002-052915-math-moe-r3`
- Role: `moer3r1`
- Snapshot: `checkpoints/spike_snapshots-active/iter_0000085`
- Original spike norm: `0.4033715`
- Median recent norm: `0.0963629`
- Spike ratio: `4.185964`
- Parallelism: `tp=8`, `dp=4`, `pp=1`, `ep=8`
- Replay mode: reduced diagnostic replay mapping original ranks 24-31 onto 8 GPUs (`dp_size=1`)

## Evidence

The R3-on spike is driven mainly by negative-advantage policy-gradient terms through early shared parameters. Same-population logprob checks showed R3 is closer than current routing to both rollout and old logprobs for most tokens, so the cleanest mitigation should keep R3 and correct the policy-gradient weighting mismatch between rollout and trainer logprobs.

For rank 24 trainable tokens in the replay batch:

- token count: `148349`
- `old/rollout` mean: `1.000018`
- std: `0.017869`
- min: `0.693186`
- max: `1.838021`
- p95: `1.009243`
- p99: `1.062855`
- tokens above `2`: `0`

Therefore `mismatch_token_clip_max=2` does not work by clipping many extreme tokens in this snapshot. It works by enabling the `old/rollout` token mismatch factor at all.

Follow-up token and gradient ablations clarify which side helps:

- The multiplier is `old/rollout`, not `rollout/old`.
- `old/rollout < 1` downweights token scalar losses; `old/rollout > 1` upweights them.
- On source rank 24, scalar policy-gradient mass changes only from `63047.8906` to `63036.8203` (`-0.018%`), so the large gradient-norm reduction is not explained by total scalar loss mass.
- No-sequence-mask gradient ablation:
  - no TIS: `0.4421454`
  - TIS only for `old/rollout < 1`: `0.3018022`
  - TIS only for `old/rollout > 1`: `0.2354022`
  - full TIS with `mismatch_token_clip_max=2`: `0.1376026`

The `old/rollout > 1` side helps more at gradient-norm level, even though it upweights scalar loss. The best result needs both sides, which points to gradient-vector cancellation/rebalancing rather than simple loss downscaling.

## Mitigation

Set the GRPO config:

```python
mismatch_token_clip_max = 2
```

The relevant trainer logic is:

```python
token_mismatch_ir = self.get_ratio(
    logprobs=old_logprobs,
    base_logprobs=rollout_logprobs,
).detach()
token_mismatch_ir = torch.where(loss_mask, token_mismatch_ir, torch.ones_like(token_mismatch_ir))
token_mismatch_tir = torch.clamp(token_mismatch_ir, max=self.config.mismatch_token_clip_max)
pg_losses = pg_losses * token_mismatch_tir
```

Keep `is_base_logprobs="old_logprobs"` for this method. In this spike replay, switching `is_base_logprobs` to `rollout_logprobs` was worse and should not be bundled with this mitigation.

## Replay Result

Baseline logged-config reduced-DP replay:

- replayed grad norm: `0.4447577`
- original saved grad norm: `0.4033715`
- relative diff: `0.102601`
- mismatched params: `2/1875`

With `mismatch_token_clip_max=2` and active sequence mask:

- replayed grad norm: `0.1480056`
- reduction vs reduced-DP baseline: `66.72%`
- about `1.54x` the median recent norm
- mismatched params: `8/1875`

With `mismatch_token_clip_max=2` and sequence masking disabled:

- replayed grad norm: `0.1376026`
- reduction vs no-sequence-mask R3-on baseline: `68.88%`
- about `1.43x` the median recent norm
- mismatched params: `8/1875`

Selective no-sequence-mask ablations:

- TIS only for `old/rollout < 1`: replayed grad norm `0.3018022`
- TIS only for `old/rollout > 1`: replayed grad norm `0.2354022`

This makes token mismatch TIS a better first production candidate than disabling R3 globally or using advantage scaling. It keeps full R3 routing replay semantics while bringing the replayed gradient norm close to the R3-off diagnostic range.
