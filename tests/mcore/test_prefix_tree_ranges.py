"""Unit tests for the prefix tree range builder.

Pure CPU, no GPU / Ray required. Verifies:
- `build_tree_packed_sequence` merges paths sharing a prefix into a single
  packed sequence with a correct parent-pointer tree; it also handles the
  no-sharing case (one branch per path), where the result reduces to
  per-sample causal attention.
- `build_prefix_tree_ranges` emits a `(q_ranges, k_ranges, attn_type_map)`
  triple whose flat attention mask matches a hand-computed reference over
  the packed sequence.
"""

from __future__ import annotations

import torch

from axrl.utils.megatron.prefix_tree import (
    PrefixTreeNode,
    build_prefix_tree_ranges,
    build_tree_packed_sequence,
)


def _reference_tree_mask(nodes: list[PrefixTreeNode], total: int) -> torch.Tensor:
    """Hand-compute the boolean attention mask implied by the tree.

    A token at packed position q can attend to a token at position k iff
    k is in the node containing q OR k is in some ancestor of the node
    containing q, AND (within the same node) q >= k (causal).
    """
    # Map each packed position to its node index.
    pos_to_node = [-1] * total
    for idx, n in enumerate(nodes):
        for p in range(n.start, n.end):
            pos_to_node[p] = idx
    mask = torch.zeros(total, total, dtype=torch.bool)
    for q in range(total):
        qn = pos_to_node[q]
        if qn < 0:
            continue
        # Walk ancestors + self, set allowed kv tokens.
        ancestors: list[int] = [qn]
        p = nodes[qn].parent
        while p >= 0:
            ancestors.append(p)
            p = nodes[p].parent
        for an in ancestors:
            nd = nodes[an]
            for k in range(nd.start, nd.end):
                if an == qn and k > q:
                    continue  # causal within self
                mask[q, k] = True
    return mask


def _mask_from_ranges(
    q_ranges: torch.Tensor,
    k_ranges: torch.Tensor,
    attn_type_map: torch.Tensor,
    total: int,
) -> torch.Tensor:
    """Reconstruct the boolean mask from Magi-style ranges."""
    mask = torch.zeros(total, total, dtype=torch.bool)
    for i in range(q_ranges.size(0)):
        qs, qe = int(q_ranges[i, 0]), int(q_ranges[i, 1])
        ks, ke = int(k_ranges[i, 0]), int(k_ranges[i, 1])
        mt = int(attn_type_map[i])
        for q in range(qs, qe):
            for k in range(ks, ke):
                if mt == 1:
                    # causal requires len(q-range) == len(k-range); the
                    # builder produces self-ranges with qs==ks so the
                    # causal test is (k - ks) <= (q - qs).
                    if (k - ks) <= (q - qs):
                        mask[q, k] = True
                else:
                    mask[q, k] = True
    return mask


def test_build_tree_single_path() -> None:
    """A single path degenerates to one linear sequence.

    One node per token run, all siblings of the root.
    """
    paths = [[10, 20, 30, 40]]
    packed, nodes, path_to_leaf = build_tree_packed_sequence(paths)
    assert packed == [10, 20, 30, 40]
    # One child under root; trie compacts runs into one node.
    assert len(nodes) == 1
    assert nodes[0].parent == -1
    assert nodes[0].start == 0 and nodes[0].end == 4
    # Leaf maps to that single node.
    assert path_to_leaf[0] == (0, 0)


def test_build_tree_merges_shared_prefix() -> None:
    """Two paths sharing tokens up to a fork collapse to a shared node.

    Label-aware: the last shared position before the fork is dropped
    because its label (the diverging next token) would conflict in the
    merged sample. So paths ``[1, 2, 3, 4, 5]`` and ``[1, 2, 3, 6, 7]``
    fork at position 3 (4 vs 6); the shared prefix is ``[1, 2]`` (the
    ``3`` lives in each branch).
    """
    paths = [
        [1, 2, 3, 4, 5],  # path A
        [1, 2, 3, 6, 7],  # path B
    ]
    packed, nodes, path_to_leaf = build_tree_packed_sequence(paths)
    # Packed: [1, 2,  3, 4, 5,  3, 6, 7] (shared prefix [1,2], two branches each starting with 3).
    assert packed == [1, 2, 3, 4, 5, 3, 6, 7]
    # Expect 3 nodes: shared prefix (len 2), branch A tail (len 3), branch B tail (len 3).
    assert len(nodes) == 3
    root, a, b = nodes[0], nodes[1], nodes[2]
    assert root.parent == -1 and root.start == 0 and root.end == 2
    assert a.parent == 0 and a.start == 2 and a.end == 5
    assert b.parent == 0 and b.start == 5 and b.end == 8
    # Path 0 ends on leaf a; path 1 ends on leaf b.
    assert path_to_leaf[0] == (1, 2)
    assert path_to_leaf[1] == (2, 2)


def test_prefix_tree_ranges_match_reference_mask() -> None:
    """Magi tree ranges reproduce a hand-computed reference mask.

    Uses a two-branch tree over the packed layout as the fixture.
    """
    paths = [
        [1, 2, 3, 4, 5],
        [1, 2, 3, 6, 7],
    ]
    packed, nodes, _ = build_tree_packed_sequence(paths)
    total = len(packed)
    q_ranges, k_ranges, attn_type_map = build_prefix_tree_ranges(nodes, device=torch.device("cpu"))
    ref = _reference_tree_mask(nodes, total)
    got = _mask_from_ranges(q_ranges, k_ranges, attn_type_map, total)
    assert torch.equal(ref, got), f"Attention masks differ.\nref=\n{ref.int()}\ngot=\n{got.int()}"


def test_single_path_tree_matches_reference_causal() -> None:
    """A tree with one path is equivalent to per-sample causal attention.

    Flattening the tree ranges should produce the standard lower-
    triangular causal mask over the single packed sequence.
    """
    path = [7, 8, 9, 10, 11]
    packed, nodes, _ = build_tree_packed_sequence([path])
    total = len(packed)

    tree_q, tree_k, tree_m = build_prefix_tree_ranges(nodes, device=torch.device("cpu"))
    tree_mask = _mask_from_ranges(tree_q, tree_k, tree_m, total)

    ref = torch.tril(torch.ones(total, total, dtype=torch.bool))
    assert torch.equal(tree_mask, ref), f"Single-path tree should reduce to standard causal.\ntree=\n{tree_mask.int()}\nref=\n{ref.int()}"


def test_prefix_tree_three_path_example() -> None:
    """Mirror the 3-turn example from context-management/task.md.

    Three samples share `[s+u, a1]`; samples 2 and 3 diverge after `a1`
    with the hide-tool-result context management.
    """
    # Use small ints to stand in for per-segment tokens; what matters is
    # the *sharing* structure.
    s_u = [1, 1, 1]  # shared [s, u]
    a1 = [2, 2, 2]  # trainable for sample 1; prefix for 2 & 3
    tr1 = [3, 3]  # tool_result_1 (only appears in sample 2)
    a2 = [4, 4, 4]  # trainable for sample 2
    a2_prime = [4, 4, 4]  # a2 prefix in sample 3 (no tr1 before it)
    tr2 = [5, 5]  # tool_result_2 in sample 3
    a3 = [6, 6, 6]  # trainable for sample 3

    paths = [
        s_u + a1,  # sample 1
        s_u + a1 + tr1 + a2,  # sample 2
        s_u + a1 + a2_prime + tr2 + a3,  # sample 3
    ]
    packed, nodes, path_to_leaf = build_tree_packed_sequence(paths)
    total = len(packed)
    # Merging: sample 1 and 2 share `s_u + a1`; sample 3 shares only
    # `s_u + a1` with them (its `a2'` follows `a1` directly, not `tr1`).
    # The trie splits at a1 into: branch-2 = tr1 + a2; branch-3 =
    # a2' + tr2 + a3. So we expect at least: {shared, branch-for-1,
    # branch-for-2, branch-for-3}; depending on tokenizer overlaps with
    # `a2 == a2'`, the trie may split further. We check the semantic
    # invariant: the three trainable *paths* decode back to the same
    # token sequences after the tree merge.
    for i, p in enumerate(paths):
        leaf_idx, _ = path_to_leaf[i]
        # Walk root → leaf, concatenate node tokens.
        chain: list[int] = []
        seen: list[int] = []
        node = leaf_idx
        while node >= 0:
            seen.append(node)
            node = nodes[node].parent
        for n_idx in reversed(seen):
            n = nodes[n_idx]
            chain.extend(packed[n.start : n.end])
        assert chain == p, f"path {i} failed round-trip: got {chain}, expected {p}"
    # Ranges are well-formed and the reference mask matches.
    q_ranges, k_ranges, attn_type_map = build_prefix_tree_ranges(nodes, device=torch.device("cpu"))
    ref = _reference_tree_mask(nodes, total)
    got = _mask_from_ranges(q_ranges, k_ranges, attn_type_map, total)
    assert torch.equal(ref, got)


def test_tree_builder_captures_subtree_sharing() -> None:
    """Subtree sharing (not just head-of-batch) must be captured by the trie.

    The 3-sample batch has path 0 diverging from paths 1 & 2 at position 1,
    but paths 1 & 2 still share a 2-token prefix ``[1, 2]``.
    """
    paths = [[1, 9, 9], [1, 2, 3, 4], [1, 2, 5, 6]]
    _, nodes, _ = build_tree_packed_sequence(paths)
    merged_len = sum(nd.end - nd.start for nd in nodes)
    naive_len = sum(len(p) for p in paths)
    assert merged_len < naive_len, f"expected trie to save tokens (got merged={merged_len}, naive={naive_len})"


def test_tree_builder_no_sharing_is_per_sample() -> None:
    """With no shared prefix, the trie reduces to one branch per path.

    Each branch is a top-level child of the root (parent=-1), so
    ``build_prefix_tree_ranges`` emits causal-self ranges only — exactly
    the per-sample causal layout.
    """
    paths = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    _, nodes, _ = build_tree_packed_sequence(paths)
    assert all(nd.parent == -1 for nd in nodes), "no-sharing trie must have only root-level branches"
    assert len(nodes) == len(paths), "one node per path when there is no sharing"
    q_ranges, k_ranges, attn_type_map = build_prefix_tree_ranges(nodes, device=torch.device("cpu"))
    assert q_ranges.size(0) == len(nodes), "one causal-self range per path"
    assert torch.equal(q_ranges, k_ranges), "self-ranges only (no ancestor ranges)"
    assert torch.all(attn_type_map == 1), "all ranges are causal"


# =====================================================================
# Label-aware merging — focused coverage
# =====================================================================


def _packed_offsets_for_path(nodes: list[PrefixTreeNode], leaf_idx: int) -> list[int]:
    """Return packed positions occupied by one path, root → leaf."""
    chain_idxs: list[int] = []
    n = leaf_idx
    while n >= 0:
        chain_idxs.append(n)
        n = nodes[n].parent
    offsets: list[int] = []
    for n_idx in reversed(chain_idxs):
        nd = nodes[n_idx]
        offsets.extend(range(nd.start, nd.end))
    return offsets


def test_label_aware_splits_before_fork() -> None:
    """Diverging next-id forces a split.

    Paths ``[1, 2, 3, 4]`` and ``[1, 2, 5, 6]`` fork at position 2.
    The merged sample's label at packed position 1 (the shared ``2``)
    is ``input_ids[2]`` = 3 vs 5 — different per path. Storing one
    label there would corrupt logprob/loss reads for the other path
    (e.g. compute_logprobs / GRPO ratios). The trie must drop the
    last shared position before the fork, leaving only token ``1``
    shared.
    """
    paths = [[1, 2, 3, 4], [1, 2, 5, 6]]
    packed, nodes, path_to_leaf = build_tree_packed_sequence(paths)
    # Shared prefix is [1] only; each branch carries its own [2, 3, 4] / [2, 5, 6].
    assert packed == [1, 2, 3, 4, 2, 5, 6]
    assert len(nodes) == 3
    root, branch_a, branch_b = nodes[0], nodes[1], nodes[2]
    assert root.parent == -1 and root.start == 0 and root.end == 1
    assert branch_a.parent == 0 and branch_a.start == 1 and branch_a.end == 4
    assert branch_b.parent == 0 and branch_b.start == 4 and branch_b.end == 7
    # Both paths' "2" token lands in their own branch, not the shared root.
    off0 = _packed_offsets_for_path(nodes, path_to_leaf[0][0])
    off1 = _packed_offsets_for_path(nodes, path_to_leaf[1][0])
    assert off0[1] != off1[1], "label-aware split: position 1 (the shared '2') must be unshared because labels differ"


def test_label_aware_keeps_share_when_one_path_is_strict_prefix_of_other() -> None:
    """One path ending inside the other's prefix doesn't trigger the label-aware drop.

    Paths ``[1, 2]`` and ``[1, 2, 3, 4]`` share ``[1, 2]`` fully — the
    shorter path simply ends, no fork at the boundary. The label-aware
    rule (drop only at a real fork) leaves the shared region intact.
    """
    paths = [[1, 2], [1, 2, 3, 4]]
    packed, nodes, path_to_leaf = build_tree_packed_sequence(paths)
    # Shared [1, 2] stays intact; the longer path adds [3, 4].
    assert packed == [1, 2, 3, 4]
    assert len(nodes) == 2
    root, tail = nodes[0], nodes[1]
    assert root.parent == -1 and root.start == 0 and root.end == 2
    assert tail.parent == 0 and tail.start == 2 and tail.end == 4
    # Shorter path's leaf is the shared root; longer path's leaf is the tail.
    assert path_to_leaf[0] == (0, 0)
    assert path_to_leaf[1] == (1, 2)


def test_merge_trajectory_samples_forbids_shared_trainable_across_paths() -> None:
    """Two paths sharing a trainable slot must raise — strict rule.

    Different paths can carry different per-token process / turn rewards
    for the same shared token, so a shared trainable slot is genuinely
    ambiguous and must not be silently picked.
    """
    import pytest

    from axrl.configs import IGNORE_INDEX
    from axrl.data import array_utils
    from axrl.data.sample import Sample
    from axrl.utils.megatron.prefix_tree import merge_trajectory_samples

    s1 = Sample(
        input_ids=array_utils.as_i32([1, 2, 3, 4, 6]),
        labels=array_utils.as_i32([2, 3, 4, 6, IGNORE_INDEX]),
        loss_mask=array_utils.as_bool([False, True, True, True, False]),
        attention_mask=array_utils.as_bool([True] * 5),
        position_ids=array_utils.as_i32([0, 1, 2, 3, 4]),
        reward=0.0,
        reward_baseline=0.0,
        advantage=array_utils.as_f32([0.0] * 5),
        rollout_logprobs=array_utils.as_f32([0.0, 0.1, 0.2, 0.3, 0.0]),
    )
    s2 = Sample(
        input_ids=array_utils.as_i32([1, 2, 3, 4, 7]),
        labels=array_utils.as_i32([2, 3, 4, 7, IGNORE_INDEX]),
        loss_mask=array_utils.as_bool([False, True, True, True, False]),
        attention_mask=array_utils.as_bool([True] * 5),
        position_ids=array_utils.as_i32([0, 1, 2, 3, 4]),
        reward=0.0,
        reward_baseline=0.0,
        advantage=array_utils.as_f32([0.0] * 5),
        rollout_logprobs=array_utils.as_f32([0.0, 0.1, 0.2, 0.4, 0.0]),
    )
    with pytest.raises(AssertionError, match="shared trainable slot"):
        merge_trajectory_samples([s1, s2])
