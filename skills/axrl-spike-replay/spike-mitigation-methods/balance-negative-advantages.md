# Balance Negative Advantages

## Description

Scale only meaningfully negative advantages down when their total trainable-token magnitude is larger than the meaningfully positive advantage magnitude. Near-zero advantages are ignored with an epsilon threshold so numerical noise around zero does not affect the balance estimate. This reduces gradient spikes caused by an imbalanced batch where failed, negatively rewarded responses dominate the token-weighted policy-gradient signal, while keeping every token in the loss.

## Status

Significant mitigation evidence. This method reduced the logged-config reduced-DP replay norm from `0.4447577` to `0.2362761` (`46.88%` reduction) without masking any tokens. The remaining caveat is that this replay maps one original DP group onto an 8-GPU `dp_size=1` diagnostic setup, so it should still be validated on additional snapshots or a fuller replay before becoming a default training change.

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

The step-85 gradient trace was dominated by early shared parameters:

- `decoder.layers.0.self_attention.linear_qkv.layer_norm_weight`
- `embedding.word_embeddings.weight`

The high-impact DP group had a strong token-weighted advantage imbalance:

- positive trainable-token advantage sum: `22817.03`
- negative trainable-token advantage magnitude: `40221.07`
- negative scaling factor: `22817.03 / 40221.07 = 0.56729`

The offline screening score `exp(rollout_logprob - old_logprob) * abs(advantage)` also showed negative-advantage sequences dominating the candidate contribution. Scaling negative advantages to match the positive magnitude reduced that offline score by `27.61%` without removing tokens.

## Mitigation

Scale negative trainable-token advantages so total negative magnitude matches total positive magnitude:

```python
def balance_negative_advantage(
    advantage: torch.Tensor,
    loss_mask: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    trainable_advantage = advantage[loss_mask]
    positive_sum = trainable_advantage[trainable_advantage > eps].sum()
    negative_abs_sum = -trainable_advantage[trainable_advantage < -eps].sum()
    if float(positive_sum.item()) <= 0.0 or float(negative_abs_sum.item()) <= 0.0:
        return advantage
    negative_scale = positive_sum / negative_abs_sum
    return torch.where(advantage < -eps, advantage * negative_scale, advantage)
```

## Replay Result

Baseline logged-config reduced-DP replay:

- replayed grad norm: `0.4447577`
- original saved grad norm: `0.4033715`
- relative diff: `0.102601`
- mismatched params: `2/1875`

With negative-advantage balancing:

- replayed grad norm: `0.2362761`
- reduction vs reduced-DP baseline: `46.88%`
- still about `2.45x` the median recent norm
- mismatched params: `7/1875`

This supports the hypothesis that token-weighted negative-advantage imbalance is a major cause of the spike. Continue with combinations such as advantage balancing plus absolute-advantage clipping, and validate on later snapshots before making this a default training change.
