# Disable Routing Replay

## Description

Disable R3 routing replay for a spike replay by removing the saved routing handles and routing payloads from the replay batch. This tests whether the spike is caused by the fixed rollout routing information rather than by the token data or advantage values alone.

This is a diagnostic mitigation, not a final training fix by itself. Dropping routing information changes the replay/training semantics for MoE, so a production mitigation should preserve correctness while addressing why replayed routing produces the large gradient.

## Status

Strong diagnostic mitigation evidence. Disabling R3 reduced the logged-config reduced-DP replay norm from `0.4447577` to `0.1444974`, a `67.51%` reduction versus the reproduced baseline. A controlled rerun with sequence masking disabled still reduced the norm from `0.4421454` to `0.1377603`, a `68.84%` reduction. This passes the method-library recording threshold because it reduced the replayed gradient norm by more than `15%`.

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

Baseline logged-config reduced-DP replay reproduced the spike closely enough for mitigation experiments:

- replayed grad norm with R3: `0.4447577`
- original saved grad norm: `0.4033715`
- relative diff: `0.102601`
- mismatched params: `2/1875`

Disabling R3 produced:

- replayed grad norm without R3: `0.1444974`
- reduction vs reproduced baseline: `67.51%`
- mismatched params: `9/1875`

Because R3-off also changed the sequence-mismatch mask population, a controlled follow-up disabled sequence masking by setting `mismatch_seq_masking_low=None` and `mismatch_seq_masking_high=None`:

- R3 on, no sequence mask: replayed grad norm `0.4421454`
- R3 off, no sequence mask: replayed grad norm `0.1377603`
- reduction vs no-sequence-mask R3-on baseline: `68.84%`

This means the drop-routing diagnostic is not an artifact of R3-off filtering more sequences. Sequence masking must still be disabled for KL/logprob correctness comparisons, but the gradient-norm effect persists under the same training-token population.

The largest drops were in shared early-layer parameters, not only MoE router parameters:

- `decoder.layers.0.self_attention.linear_qkv.layer_norm_weight`: `0.386354 -> 0.057640` (`85.08%` drop)
- `embedding.word_embeddings.weight`: `0.082844 -> 0.016607` (`79.95%` drop)
- `decoder.layers.1.self_attention.linear_qkv.layer_norm_weight`: `0.074547 -> 0.017348` (`76.73%` drop)
- `decoder.layers.0.mlp.router.weight`: `0.012217 -> 0.002317` (`81.03%` drop)

Grouped by squared-norm reduction, the drop is dominated by self-attention and shared embedding parameters:

- self-attention: `0.151347`
- embedding: `0.006587`
- output layer: `0.000482`
- experts: `0.000307`
- router: `0.000178`

This indicates R3/routing replay is a high-impact causal direction, but the visible gradient amplification propagates into shared early representation parameters rather than staying isolated to router or expert weights. The no-sequence-mask comparison shows the same pattern:

- `decoder.layers.0.self_attention.linear_qkv.layer_norm_weight`: `0.384088 -> 0.043779` (`88.60%` drop)
- `embedding.word_embeddings.weight`: `0.082454 -> 0.013756` (`83.32%` drop)
- `decoder.layers.1.self_attention.linear_qkv.layer_norm_weight`: `0.074186 -> 0.014954` (`79.84%` drop)
- `decoder.layers.0.self_attention.linear_qkv.weight`: `0.019257 -> 0.002541` (`86.80%` drop)

The output-layer gradient norm barely changes in the no-sequence-mask comparison (`0.075836 -> 0.075662`). This suggests similar scalar/logprob loss can still produce very different upstream gradient vector norms because R3 changes the routed hidden-state and backward path.

## Mitigation

The replay experiment removed routing information from the temporary replay snapshot:

```python
if mitigation == "drop_routing_information":
    if "routing_handles_per_path" in batch.keys():
        del batch["routing_handles_per_path"]
    return batch
```

and skipped routing payload files for the replay snapshot:

```python
if mitigation == "drop_routing_information":
    routing_payload_path = replay_dir / f"routing_payload_rank{rank}.pt"
    if routing_payload_path.exists() or routing_payload_path.is_symlink():
        routing_payload_path.unlink()
```

## Replay Result

With R3 enabled:

- replayed grad norm: `0.4447577`
- original saved grad norm: `0.4033715`
- relative diff: `0.102601`
- mismatched params: `2/1875`

With R3 disabled:

- replayed grad norm: `0.1444974`
- reduction vs reproduced baseline: `67.51%`
- still about `1.50x` the median recent norm
- mismatched params: `9/1875`

With sequence masking disabled:

- R3-on replayed grad norm: `0.4421454`
- R3-off replayed grad norm: `0.1377603`
- reduction vs controlled R3-on baseline: `68.84%`
- conclusion: the diagnostic remains valid after controlling for sequence-mask population mismatch.

This result should motivate deeper R3-specific analysis: compare fixed replay routing versus current routing, inspect routing-handle alignment with packed tokens, and check whether SGLang-prefill prompt routes differ from MCore prefill/training routes more than SGLang decode routes differ on supervised generated tokens. Do not use this as a final mitigation unless the intended training semantics are explicitly changed.
