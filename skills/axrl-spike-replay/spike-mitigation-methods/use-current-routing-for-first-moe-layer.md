# Use Current Routing For First MoE Layer

## Description

Use current-model routing for the first global MoE/router layer, and keep R3 routing replay enabled for all later MoE layers. This is a layer-selective R3 mitigation: it keeps most rollout routing information, but avoids forcing the earliest replayed expert path where the step-85 spike is most sensitive.

## Status

Strong mitigation evidence for `axrl-v002-052915-math-moe-r3`. Skipping R3 only for the first MoE layer reduced the step-85 logged-config reduced-DP replay norm from `0.4447577` to `0.1531728` (`65.56%` reduction), close to the R3-off diagnostic norm `0.1444974`, while keeping R3 active for later layers. On the later and larger step-247 spike, the same mitigation reduced the reduced-DP replay norm from `0.3853823` to `0.0911358`, below that snapshot's spike detector threshold (`2 * 0.0490104 = 0.0980208`).

This should be validated on additional retained snapshots before becoming a default training setting. It is cleaner than disabling R3 globally, but it still changes R3 semantics for one layer.

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

The R3-on spike is dominated by early shared parameters, especially:

- `decoder.layers.0.self_attention.linear_qkv.layer_norm_weight`
- `embedding.word_embeddings.weight`
- `decoder.layers.1.self_attention.linear_qkv.layer_norm_weight`

Controlled layer-band diagnostics showed the first replayed MoE layer is the highest-impact routing-replay region:

- full R3 on, controlled PG-only/no sequence mask/no PPO clip: `0.7826452`
- full R3 off, controlled PG-only/no sequence mask/no PPO clip: `0.1345321`
- replay layer 0 only: `2.1216783`
- skip layer 0 and replay layers 1+: `0.1963565`
- replay layers 8+: `0.1571459`
- replay layers 12+: `0.1324784`

The production-like logged-config check on node1:

- full R3, active sequence mask: `0.4447577`
- R3 off diagnostic, active sequence mask: `0.1444974`
- current routing for layer 0, R3 for layers 1+: `0.1531728`

The no-sequence-mask check also stayed close to the R3-off range:

- full R3, sequence mask disabled: `0.4421454`
- R3 off diagnostic, sequence mask disabled: `0.1377603`
- current routing for layer 0, R3 for layers 1+: `0.1516417`

This mitigation also preserves most routing-replay logprob fidelity. Same-population rollout-logprob K2 mismatch:

- full R3: `0.0001682`
- current routing for layer 0, R3 for layers 1+: `0.0001689`
- R3 off: `0.0005829`

So the first-layer skip removes the large gradient amplification while retaining nearly all of the aggregate R3 logprob benefit.

Step-247 validation:

- full R3 baseline: `0.3853823`
- R3 off diagnostic: `0.0960043`
- `mismatch_token_clip_max=2` with full R3: `0.1754785`
- current routing for layer 0, R3 for layers 1+: `0.0911358`
- current routing for layer 0, R3 for layers 1+, plus `mismatch_token_clip_max=2`: `0.0908413`
- R3 for layer 0 only, current routing for layers 1+: `0.1014543`

The step-247 layer-band result says the issue is not simply "layer 0 replay alone is bad": layer0-only and layers1+-only are both near the normal range, while full R3 is high. The current hypothesis is a nonlinear interaction where first-layer replay changes the upstream hidden-state and router-probability-gradient basis, and later replayed layers then align gradients back into early shared parameters.

Step-247 route and logprob diagnostics argue against a gross layer-0 R3 implementation bug:

- layer-0 trainable saved-slot-in-current-set overlap: `98.57%`
- layer-0 no-overlap token-layer rate: `0.0`
- layer-0 mean current-minus-saved selected-logit gap: `0.00014`
- raw-loss-mask R3-on closer to rollout logprobs than R3-off: `71.16%` of tokens
- raw-loss-mask R3-on closer to old logprobs than R3-off: `71.35%` of tokens

This looks more like a high-variance/off-policy routing-gradient interaction than corrupted layer-0 route tensors.

## Mitigation

Use current routing for the first global MoE/router layer and replay routing for later layers. In replay experiments this was represented by `replay_layer_start=1`, where MoE/router layers are zero-indexed:

```python
for layer_idx, router in enumerate(RouterReplay.global_router_replay_instances):
    if layer_idx < current_first_n_layers:
        router.clear_router_replay_action()
        router.clear_indices()
```

A production implementation should avoid relying on the debug global list when pipeline or virtual-pipeline parallelism is active. Prefer a config such as `routing_replay_current_first_n_layers=1` and compute local MoE layer indices from the Megatron model layout.

## Replay Result

Baseline logged-config reduced-DP replay:

- replayed grad norm: `0.4447577`
- original saved grad norm: `0.4033715`
- relative diff: `0.102601`
- mismatched params: `2/1875`

With current routing for first MoE layer:

- replayed grad norm: `0.1531728`
- reduction vs reduced-DP baseline: `65.56%`
- still about `1.59x` the median recent norm
- close to the R3-off diagnostic norm `0.1444974`, but preserves R3 for later layers

With current routing for first MoE layer on step 247:

- replayed grad norm: `0.0911358`
- reduction vs reduced-DP baseline: `76.35%`
- below the detector threshold for the snapshot
- close to the R3-off diagnostic norm `0.0960043`, but preserves R3 for later layers

This supports the hypothesis that forced rollout routing in the earliest MoE layer creates a high-gain backward path into shared early representations. The method is a promising production mitigation if live continuation or additional snapshots confirm the same first-layer sensitivity.
