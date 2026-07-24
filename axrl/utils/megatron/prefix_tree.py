from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from axrl.data import array_utils

if TYPE_CHECKING:
    from collections.abc import Iterator

    from tensordict import TensorDict

    from axrl.data.sample import Sample


@dataclass
class PrefixTreeNode:
    """One node in the prefix tree over a packed sequence.

    Attributes:
        start: Inclusive start offset in `packed` tokens.
        end: Exclusive end offset in `packed` tokens.
        parent: Index of the parent node in the same node list, or `-1`
            for root-level branches.
    """

    start: int
    end: int
    parent: int


@dataclass
class PrefixMergeInfo:
    """Structural metadata for a prefix-tree-merged ``Sample``.

    The merged sample's flat fields (``input_ids``, ``labels``,
    ``loss_mask``, ``attention_mask``, ``position_ids``) hold the
    DFS-pre-order packed layout described by these nodes. The Magi
    forward consumes the q/k ranges directly to drive ``calc_attn``;
    every offset here is in the packed layout (not the per-path
    layout).

    Attributes:
        nodes: One ``PrefixTreeNode`` per tree node, ordered so
            ancestors precede descendants.
        path_to_leaf: ``(leaf_node_idx, prefix_len)`` per original path,
            used to walk leaf → root and unpack the merged sample back
            to per-turn samples.
        q_ranges: ``[(q_start, q_end), ...]`` half-open over the packed
            layout — one entry per emitted attention range.
        k_ranges: ``[(k_start, k_end), ...]`` half-open, paired with
            ``q_ranges``.
        attn_type_map: ``0`` for full-attention range (ancestor key),
            ``1`` for causal-self range. One per ``q_ranges`` entry.
        total_padded: Total length of the packed sequence including any
            alignment padding absorbed into the last node.
        max_path_len: Longest root → leaf chain length, used to size
            RoPE ``max_seqlen_q``.
    """

    nodes: list[PrefixTreeNode]
    path_to_leaf: list[tuple[int, int]]
    q_ranges: list[tuple[int, int]]
    k_ranges: list[tuple[int, int]]
    attn_type_map: list[int]
    total_padded: int
    max_path_len: int
    # ``input_ids`` length of each original turn sample, in the same order as
    # ``path_to_leaf``. ``unpack_tensor_from_merged`` uses this to slice
    # per-turn views back out of the packed buffer.
    turn_sample_lens: list[int]
    # Effective length of the merged sample (``sum(attention_mask)``);
    # always ``<= total_padded``. The trainer-side routing tensor for one
    # trajectory has length ``real_total - 1``.
    real_total: int = 0
    # Per-merged-routing-position source map: for each ``i`` in
    # ``[0, real_total - 1)``, ``routing_source_path[i]`` is the per-turn
    # path whose handle list provides the routing for merged position
    # ``i + 1``, and ``routing_source_read_pos[i]`` is the row index in
    # that path's concat-of-handles routing array. Built by
    # :func:`merge_trajectory_samples` with trainable-wins / lowest-path-idx
    # wins. ``None`` when the merge was built without R3 routing context.
    #
    # Indexing convention: a path's concat-of-handles routing array has
    # ``path_len - 1`` rows, one per *predicted* token; row ``k`` is the
    # routing used to predict the token at path position ``k + 1`` (sglang
    # never emits routing for position 0, the seed prompt's first token).
    #
    # Example — two turn samples sharing a 3-token prompt prefix, then
    # diverging; path 1 is trainable on its tail::
    #
    #     path 0 tokens: [p0 p1 p2 a0 a1]           (trainable: none)
    #     path 1 tokens: [p0 p1 p2 b0 b1 b2]        (trainable: b0 b1 b2)
    #
    #     merged packed: [p0 p1 p2 a0 a1 b0 b1 b2]  real_total = 8
    #     write_pos:       0  1  2  3  4  5  6  7
    #
    #     routing_source_path     = [0, 0, 0, 0, 1, 1, 1]   # len real_total-1
    #     routing_source_read_pos = [0, 1, 2, 3, 2, 3, 4]
    #
    #  - positions 1..4 source from path 0 (lowest-path-idx-wins on shared
    #    prompt; path 0 is the only visitor for a0/a1)
    #  - positions 5..7 source from path 1 because b0/b1/b2 are trainable
    #    there (trainable-wins)
    #  - ``read_pos[4] = 2`` because merged write_pos 5 is b0, which sits
    #    at *path 1 position 3* — and routing row ``3 - 1 = 2`` is the one
    #    that predicts path position 3 (row 0 predicts position 1, row 1
    #    predicts position 2, row 2 predicts position 3).
    routing_source_path: np.ndarray | None = None
    routing_source_read_pos: np.ndarray | None = None


class _MergingTreeNode:
    __slots__ = ("children", "path_owners", "tokens")

    def __init__(self, tokens: list[int]) -> None:
        self.tokens: list[int] = tokens
        self.children: list[_MergingTreeNode] = []
        self.path_owners: list[int] = []


class MergingTree:
    """Incremental label-aware prefix tree used to estimate packed lengths."""

    def __init__(self) -> None:
        self.root = _MergingTreeNode(tokens=[])
        self.packed_len = 0
        self.path_count = 0

    def add_path(self, path: list[int]) -> None:
        """Add one path using the same insertion rule as full prefix-tree packing."""
        self.packed_len += _insert_path(self.root, path, self.path_count)
        self.path_count += 1

    def get_packed_len_if_add_path(self, path: list[int]) -> int:
        """Return the packed length that would result from adding ``path``."""
        return self.packed_len + _added_len_if_insert(self.root, path)

    def to_packed_sequence(self) -> tuple[list[int], list[PrefixTreeNode], list[tuple[int, int]]]:
        """Emit the flattened DFS sequence and metadata for the current tree."""
        packed: list[int] = []
        nodes: list[PrefixTreeNode] = []
        path_to_leaf: list[tuple[int, int]] = [(-1, 0) for _ in range(self.path_count)]

        def _emit(node: _MergingTreeNode, parent_idx: int, prefix_len: int) -> None:
            start = len(packed)
            packed.extend(node.tokens)
            end = len(packed)
            this_idx = len(nodes)
            nodes.append(PrefixTreeNode(start=start, end=end, parent=parent_idx))
            for path_idx in node.path_owners:
                path_to_leaf[path_idx] = (this_idx, prefix_len)
            new_prefix_len = prefix_len + len(node.tokens)
            for child in node.children:
                _emit(child, this_idx, new_prefix_len)

        for child in self.root.children:
            _emit(child, parent_idx=-1, prefix_len=0)

        return packed, nodes, path_to_leaf


def _label_aware_common_prefix_len(tokens: list[int], suffix: list[int]) -> int:
    n = min(len(tokens), len(suffix))
    k = 0
    while k < n and tokens[k] == suffix[k]:
        k += 1
    # Label-aware truncation: at a natural trie fork (tokens disagree at
    # position ``k``), the LAST shared position ``k-1`` has label
    # ``input_ids[k]`` which differs across paths. Drop position ``k-1``
    # whenever both paths continue after the shared prefix.
    if k > 0 and k < len(suffix) and k < len(tokens):
        k -= 1
    return k


# NOTE: This is a *label-aware* packed trie, not a textbook prefix trie.
# Because ``_label_aware_common_prefix_len`` drops the last shared position
# whenever both paths continue past the shared run, a child that *literally*
# shares its first token with the new suffix can still produce label-aware
# ``k == 0`` (e.g. the only shared token is the divergence point itself).
# In that case the two paths must coexist as **same-leading siblings**:
# ``_insert_path`` ``continue``s past the matching child and falls through to
# append a brand-new sibling, and ``_added_len_if_insert`` /
# ``_added_len_if_insert_from_children`` mirror that ``continue`` so the
# length estimate stays consistent with the actual insertion. Same-leading
# siblings are therefore an expected invariant, not a bug.
def _insert_path(node: _MergingTreeNode, suffix: list[int], path_idx: int) -> int:
    if not suffix:
        node.path_owners.append(path_idx)
        return 0
    for child in node.children:
        if child.tokens and child.tokens[0] == suffix[0]:
            k = _label_aware_common_prefix_len(child.tokens, suffix)
            if k == 0:
                continue
            if k < len(child.tokens):
                split = _MergingTreeNode(tokens=child.tokens[k:])
                split.children = child.children
                split.path_owners = child.path_owners
                child.tokens = child.tokens[:k]
                child.children = [split]
                child.path_owners = []
            return _insert_path(child, suffix[k:], path_idx)
    new_child = _MergingTreeNode(tokens=list(suffix))
    new_child.path_owners.append(path_idx)
    node.children.append(new_child)
    return len(suffix)


def _added_len_if_insert(node: _MergingTreeNode, suffix: list[int]) -> int:
    if not suffix:
        return 0
    for child in node.children:
        if child.tokens and child.tokens[0] == suffix[0]:
            k = _label_aware_common_prefix_len(child.tokens, suffix)
            if k == 0:
                continue
            if k < len(child.tokens):
                return _added_len_after_split(child.tokens[k:], child.children, suffix[k:])
            return _added_len_if_insert(child, suffix[k:])
    return len(suffix)


def _added_len_after_split(split_tokens: list[int], split_children: list[_MergingTreeNode], suffix: list[int]) -> int:
    if not suffix:
        return 0
    k = _label_aware_common_prefix_len(split_tokens, suffix)
    if k == 0:
        return len(suffix)
    if k < len(split_tokens):
        return _added_len_after_split(split_tokens[k:], split_children, suffix[k:])
    return _added_len_if_insert_from_children(split_children, suffix[k:])


def _added_len_if_insert_from_children(children: list[_MergingTreeNode], suffix: list[int]) -> int:
    if not suffix:
        return 0
    for child in children:
        if child.tokens and child.tokens[0] == suffix[0]:
            k = _label_aware_common_prefix_len(child.tokens, suffix)
            if k == 0:
                continue
            if k < len(child.tokens):
                return _added_len_after_split(child.tokens[k:], child.children, suffix[k:])
            return _added_len_if_insert(child, suffix[k:])
    return len(suffix)


def get_packed_len_if_merge(merging_tree: MergingTree, sample: Sample) -> int:
    """Return packed length if ``sample`` were added to ``merging_tree``."""
    return merging_tree.get_packed_len_if_add_path(array_utils.to_int_list(sample.input_ids))


def add_sample_to_merging_tree(merging_tree: MergingTree, sample: Sample) -> None:
    """Mutate ``merging_tree`` by adding ``sample``."""
    merging_tree.add_path(array_utils.to_int_list(sample.input_ids))


def extract_merge_info_from_batch(batch: TensorDict) -> list[PrefixMergeInfo] | None:
    """Pull the per-microbatch ``merge_info`` non-tensor list out of ``batch``.

    Returns ``None`` when the batch carries no merge_info — i.e. the flat
    (non-merged) path. The merged forward fn requires this list; the flat
    forward fns assert it is ``None``.
    """
    if "merge_info" not in batch.keys():  # noqa: SIM118 - tensordict semantics
        return None
    return list(batch.get_non_tensor("merge_info"))


def extract_paths_from_batch(input_ids: torch.Tensor, attention_mask: torch.Tensor) -> list[list[int]]:
    """Return one list of valid token ids per batch row.

    `attention_mask[i]` selects which tokens of `input_ids[i]` belong to
    path `i`. Rows are read independently — the caller controls the
    row-order-to-path mapping.
    """
    paths: list[list[int]] = []
    for i in range(input_ids.size(0)):
        valid = input_ids[i][attention_mask[i]]
        paths.append(valid.tolist())
    return paths


def build_tree_packed_sequence(
    paths: list[list[int]],
) -> tuple[list[int], list[PrefixTreeNode], list[tuple[int, int]]]:
    """Merge paths sharing prefixes into a single DFS-pre-order packed layout.

    **Label-aware** sharing rule: two paths may share trie position
    ``p`` only when (a) at least one path ends at ``p`` (so it has no
    ``input_ids[p+1]`` to disagree about), OR (b) both paths' next
    tokens at ``p+1`` agree. The merged sample stores one label per
    packed position; sharing a position where labels would differ
    corrupts that path's logprob/loss reads.

    Practically: at the natural trie fork (tokens disagree at position
    ``k``), drop position ``k-1`` because its label is the diverging
    next token.

    A loss-aware variant (over-cautious split when either path has
    ``loss_mask[p]=True``) was previously available via a
    ``loss_masks`` argument; the GPU diagnostic
    ``test_loss_masks_diagnostic_*`` showed label-aware sharing alone
    is bit-exact-equivalent to the flat baseline (and slightly tighter
    than the loss-aware variant on numerics), so the rule was dropped
    to simplify. Two paths' trainable positions can now coincide at a
    shared slot; the scatter in :func:`merge_trajectory_samples`
    accepts those duplicate writes after asserting label/logprob
    agreement.

    Returns:
        - ``packed`` — flattened token list.
        - ``nodes`` — ``PrefixTreeNode`` per tree node, ordered so
          ancestors precede descendants and each node's tokens are
          contiguous.
        - ``path_to_leaf`` — ``(leaf_node_idx, prefix_len)`` for each
          input path, used to reconstruct per-path outputs by walking
          leaf → root.
    """
    tree = MergingTree()
    for path in paths:
        tree.add_path(path)

    # The emitted `packed` list is the flat DFS-pre-order token list;
    # `nodes` records `(start, end, parent)` for each emitted node in the
    # same order. `path_to_leaf` is seeded with `(-1, 0)` placeholders and
    # overwritten with `(leaf_idx, prefix_len)` when `_emit` reaches each
    # path's owner node.
    #
    # Example for paths = [[1, 2, 3], [1, 2, 4]]:
    #   trie: root → (node0: [1,2]) → { (node1: [3], owner=path0),
    #                                   (node2: [4], owner=path1) }
    #   after emit:
    #     packed       = [1, 2, 3, 4]
    #     nodes        = [(start=0, end=2, parent=-1),   # shared [1,2]
    #                     (start=2, end=3, parent=0),    # path-0 tail [3]
    #                     (start=3, end=4, parent=0)]    # path-1 tail [4]
    #     path_to_leaf = [(1, 2),                        # path 0 → leaf node 1, prefix_len 2
    #                     (2, 2)]                        # path 1 → leaf node 2, prefix_len 2
    return tree.to_packed_sequence()


def pack_tree_aligned_as_list(
    paths: list[list[int]],
    align_size: int,
    pad_token_id: int = 0,
    trie: tuple[list[int], list[PrefixTreeNode], list[tuple[int, int]]] | None = None,
) -> tuple[list[int], list[PrefixTreeNode], list[tuple[int, int]], int]:
    """Build (or reuse) a trie and pad the packed sequence to ``lcm(align_size, 128)``.

    The trailing alignment padding is absorbed into the last node so its
    attention range covers the tail. List variant; the GPU caller uses
    :func:`pack_tree_aligned`.
    """
    packed, nodes, path_to_leaf = trie if trie is not None else build_tree_packed_sequence(paths)
    total = len(packed)
    total_align = math.lcm(align_size, 128)
    pad = (-total) % total_align
    if pad and nodes:
        last = nodes[-1]
        nodes[-1] = PrefixTreeNode(start=last.start, end=last.end + pad, parent=last.parent)
        packed = packed + [pad_token_id] * pad
    return packed, nodes, path_to_leaf, total + pad


def build_prefix_tree_ranges_as_lists(
    nodes: list[PrefixTreeNode],
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], list[int]]:
    """Emit ``(q_ranges, k_ranges, attn_type_map)`` from a tree as Python lists.

    Each query token in node N attends to (a) every ancestor's token
    range with **full** attention (``attn_type_map == 0``) and (b) N's
    own token range with **causal** attention (``attn_type_map == 1``).
    Ancestor + self ranges don't overlap in the DFS-pre-order packed
    layout produced by :func:`build_tree_packed_sequence`.
    """
    q_list: list[tuple[int, int]] = []
    k_list: list[tuple[int, int]] = []
    mask_list: list[int] = []

    for node in nodes:
        ancestors: list[int] = []
        p = node.parent
        while p >= 0:
            ancestors.append(p)
            p = nodes[p].parent
        for a in ancestors:
            an = nodes[a]
            if an.end <= an.start:
                continue
            q_list.append((node.start, node.end))
            k_list.append((an.start, an.end))
            mask_list.append(0)  # 0 = full
        q_list.append((node.start, node.end))
        k_list.append((node.start, node.end))
        mask_list.append(1)  # 1 = causal

    return q_list, k_list, mask_list


def build_prefix_tree_ranges(
    nodes: list[PrefixTreeNode],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Tensor variant of :func:`build_prefix_tree_ranges_as_lists` for the Magi forward path.

    Shapes: ``q_ranges``, ``k_ranges`` are ``(N, 2)`` int32 ``[start, end)``
    half-open ranges; ``attn_type_map`` is ``(N,)`` int32.
    """
    q_list, k_list, mask_list = build_prefix_tree_ranges_as_lists(nodes)
    return (
        torch.tensor(q_list, dtype=torch.int32, device=device),
        torch.tensor(k_list, dtype=torch.int32, device=device),
        torch.tensor(mask_list, dtype=torch.int32, device=device),
    )


def compute_tree_rel_positions_as_list(nodes: list[PrefixTreeNode], total: int) -> list[int]:
    """Per-token position-within-path on the tree-merged packed layout.

    For RoPE: each token needs the rotary angle it would have had in its
    original (unmerged) path, not its offset in ``packed``. Entry ``t``
    is the 0-based position of the packed token at offset ``t`` inside
    the path it belongs to.
    """
    pos: list[int] = [0] * total
    for i, nd in enumerate(nodes):
        prefix_len = _node_prefix_len(nodes, i)
        for j in range(nd.end - nd.start):
            pos[nd.start + j] = prefix_len + j
    return pos


def compute_tree_rel_positions(nodes: list[PrefixTreeNode], total: int, device: torch.device) -> torch.Tensor:
    """Tensor variant of :func:`compute_tree_rel_positions_as_list` for the Magi forward path."""
    return torch.tensor(compute_tree_rel_positions_as_list(nodes, total), dtype=torch.int32, device=device)


def _node_prefix_len(nodes: list[PrefixTreeNode], idx: int) -> int:
    total = 0
    p = nodes[idx].parent
    while p >= 0:
        total += nodes[p].end - nodes[p].start
        p = nodes[p].parent
    return total


def compute_max_path_length(
    nodes: list[PrefixTreeNode],
    path_to_leaf: list[tuple[int, int]],
) -> int:
    """Length (in tokens) of the longest merged path in the trie.

    Walks each path's leaf → root chain summing segment lengths. Used
    to size RoPE `max_seqlen_q` so `freqs[position_ids]` never indexes
    past the end of the frequency table.
    """
    max_len = 0
    for leaf_idx, _ in path_to_leaf:
        chain_len = 0
        n = leaf_idx
        while n >= 0:
            chain_len += nodes[n].end - nodes[n].start
            n = nodes[n].parent
        max_len = max(max_len, chain_len)
    return max_len


def iter_path_segments_in_packed(
    nodes: list[PrefixTreeNode],
    path_to_leaf: list[tuple[int, int]],
    path_lens: list[int],
) -> Iterator[tuple[int, int, int, int]]:
    """Yield ``(path_idx, write_start, read_start, take)`` for each per-path segment.

    Walks each path's root → leaf chain. ``write_start`` is the segment's
    offset in the packed layout (``nd.start``); ``read_start`` is the
    cursor into that path's own flat fields. ``take`` is the segment
    length, capped by ``path_lens[path_idx] - read_start`` so callers
    never read past the end of an individual path's source.
    """
    for path_idx, (leaf_idx, _) in enumerate(path_to_leaf):
        path_len = path_lens[path_idx]
        read_start = 0
        for n_idx in path_chain_root_to_leaf(nodes, leaf_idx):
            nd = nodes[n_idx]
            take = min(nd.end - nd.start, path_len - read_start)
            if take <= 0:
                break
            yield path_idx, nd.start, read_start, take
            read_start += take


def path_chain_root_to_leaf(nodes: list[PrefixTreeNode], leaf_idx: int) -> list[int]:
    """Return node indices along a path, ordered root → leaf."""
    chain: list[int] = []
    n = leaf_idx
    while n >= 0:
        chain.append(n)
        n = nodes[n].parent
    chain.reverse()
    return chain


def scatter_packed_to_batch(
    packed: torch.Tensor,
    nodes: list[PrefixTreeNode],
    path_to_leaf: list[tuple[int, int]],
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Scatter packed-tree tokens back into batch layout.

    Input `packed` has shape ``(total_padded, *feat_dims)`` (the DFS-
    pre-order output of the tree-merged forward). For each input path
    `i`, walks its root → leaf chain and copies each node's segment
    into `out[i, valid_positions[...]]`, where `valid_positions` are
    the True entries of ``attention_mask[i]``. Result shape:
    ``(B, S, *feat_dims)``; padding / unused positions stay zero.

    """
    batch_size, seq_len = attention_mask.shape[:2]
    shape = [batch_size, seq_len, *packed.shape[1:]]
    out = torch.zeros(shape, dtype=packed.dtype, device=packed.device)
    valid_positions_per_path = [attention_mask[i].nonzero(as_tuple=False).flatten() for i in range(batch_size)]
    path_lens = [int(v.numel()) for v in valid_positions_per_path]
    for path_idx, write_start, read_start, take in iter_path_segments_in_packed(nodes, path_to_leaf, path_lens):
        dst = valid_positions_per_path[path_idx][read_start : read_start + take]
        out[path_idx, dst] = packed[write_start : write_start + take]
    return out


def build_prefix_merge_info(
    *,
    nodes: list[PrefixTreeNode],
    path_to_leaf: list[tuple[int, int]],
    total_padded: int,
    turn_sample_lens: list[int],
    real_total: int,
) -> PrefixMergeInfo:
    """Assemble a ``PrefixMergeInfo`` from the trie nodes + path metadata.

    ``real_total`` is the number of packed slots actually visited by at
    least one path (equivalently ``sum(attention_mask)`` of the merged
    Sample). Callers must supply this because they already know it —
    ``merge_trajectory_samples`` gets it from the scatter loop it ran,
    single-path callers know it is ``path_len``.
    """
    q_list, k_list, mask_list = build_prefix_tree_ranges_as_lists(nodes)
    assert len(turn_sample_lens) == len(path_to_leaf), f"turn_sample_lens length {len(turn_sample_lens)} != path_to_leaf length {len(path_to_leaf)}"
    return PrefixMergeInfo(
        nodes=list(nodes),
        path_to_leaf=list(path_to_leaf),
        q_ranges=q_list,
        k_ranges=k_list,
        attn_type_map=mask_list,
        total_padded=total_padded,
        max_path_len=compute_max_path_length(nodes, path_to_leaf),
        turn_sample_lens=list(turn_sample_lens),
        real_total=real_total,
    )


def merge_prefix_merge_infos(infos: list[PrefixMergeInfo]) -> PrefixMergeInfo:
    """Concatenate multiple per-trajectory ``PrefixMergeInfo`` into one.

    Each input describes a single trajectory's trie laid out at offsets
    ``[0, info.total_padded)``. The output is a forest of those tries:
    every node, leaf-pointer, and q/k range is shifted by the cumulative
    offset of its source trajectory in the concatenated layout.

    The returned info is itself a valid merged-sample metadata —
    indistinguishable to the Magi forward from a single-trajectory trie.
    """
    merged_nodes: list[PrefixTreeNode] = []
    merged_path_to_leaf: list[tuple[int, int]] = []
    merged_q: list[tuple[int, int]] = []
    merged_k: list[tuple[int, int]] = []
    merged_mask: list[int] = []
    merged_turn_lens: list[int] = []
    pos_offset = 0
    node_offset = 0
    max_path_len = 0
    for info in infos:
        for nd in info.nodes:
            new_parent = nd.parent + node_offset if nd.parent >= 0 else -1
            merged_nodes.append(PrefixTreeNode(start=nd.start + pos_offset, end=nd.end + pos_offset, parent=new_parent))
        for leaf_idx, prefix_len in info.path_to_leaf:
            merged_path_to_leaf.append((leaf_idx + node_offset, prefix_len))
        merged_q.extend((q[0] + pos_offset, q[1] + pos_offset) for q in info.q_ranges)
        merged_k.extend((k[0] + pos_offset, k[1] + pos_offset) for k in info.k_ranges)
        merged_mask.extend(info.attn_type_map)
        merged_turn_lens.extend(info.turn_sample_lens)
        max_path_len = max(max_path_len, info.max_path_len)
        pos_offset += info.total_padded
        node_offset += len(info.nodes)
    return PrefixMergeInfo(
        nodes=merged_nodes,
        path_to_leaf=merged_path_to_leaf,
        q_ranges=merged_q,
        k_ranges=merged_k,
        attn_type_map=merged_mask,
        total_padded=pos_offset,
        max_path_len=max_path_len,
        turn_sample_lens=merged_turn_lens,
        real_total=sum(info.real_total for info in infos),
    )


# --------------------------------------------------------------------- #
# Trajectory-merging helpers operating on ``Sample`` (with ``merge_info``).
# --------------------------------------------------------------------- #


def merge_trajectory_samples(per_turn_samples: list[Sample], align_size: int = 128) -> Sample:  # noqa: PLR0915
    """Trie-pack one trajectory's per-turn samples into a merged ``Sample`` with ``merge_info`` set.

    Strict shared-trainable rule: at most one path may have ``loss_mask=True``
    at any shared packed slot. Different paths can carry different per-token
    process / turn rewards for the same shared token, so a shared trainable
    slot is genuinely ambiguous and must not be silently picked.

    Routing precedence at shared positions: a row written from a trainable
    path (``loss_mask=True``) wins and is never overwritten — neither by
    another trainable path (the strict rule above) nor by a non-trainable
    path. Non-trainable paths fill remaining positions; agreement is
    guaranteed by construction (shared positions came from the same prefix).
    """
    from axrl.configs import IGNORE_INDEX  # local import to avoid cycle
    from axrl.data.sample import Sample as SampleCls

    assert per_turn_samples, "merge_trajectory_samples needs at least one sample"

    paths: list[list[int]] = [array_utils.to_int_list(s.input_ids) for s in per_turn_samples]
    packed_ids, nodes, path_to_leaf, total_padded = pack_tree_aligned_as_list(
        paths,
        align_size=align_size,
    )

    labels = [IGNORE_INDEX] * total_padded
    loss_mask = [False] * total_padded
    attention_mask = [False] * total_padded
    advantage = [0.0] * total_padded
    has_logprobs = all(s.rollout_logprobs is not None for s in per_turn_samples)
    rollout_logprobs: list[float] | None = [0.0] * total_padded if has_logprobs else None
    has_teacher_logprobs = all(s.teacher_logprobs is not None for s in per_turn_samples)
    teacher_logprobs: list[float] | None = [0.0] * total_padded if has_teacher_logprobs else None
    has_old_values = all(s.old_values is not None for s in per_turn_samples)
    old_values: list[float] | None = [0.0] * total_padded if has_old_values else None
    has_returns = all(s.returns is not None for s in per_turn_samples)
    returns: list[float] | None = [0.0] * total_padded if has_returns else None
    has_turn_index = all(s.turn_index is not None for s in per_turn_samples)
    turn_index: list[int] | None = [-1] * total_padded if has_turn_index else None
    has_turn_reward = all(s.turn_reward is not None for s in per_turn_samples)
    turn_reward: list[float] | None = [0.0] * total_padded if has_turn_reward else None

    path_lens = [len(s.input_ids) for s in per_turn_samples]

    def positions() -> Iterator[tuple[int, Sample, int, int]]:
        """Yield ``(path_idx, sample, write_pos, read_pos)`` over every per-path segment."""
        for path_idx, write_start, read_start, take in iter_path_segments_in_packed(nodes, path_to_leaf, path_lens):
            sample = per_turn_samples[path_idx]
            for j in range(take):
                yield path_idx, sample, write_start + j, read_start + j

    # Routing source map: for each merged routing position p (= merged token
    # position - 1), pick which path's per-path concat-routing array to source
    # from. Trainable always wins; among non-trainable, lowest path_idx (=
    # earliest turn) wins.
    routing_source_path = [-1] * max(total_padded - 1, 0)
    routing_source_read_pos = [-1] * max(total_padded - 1, 0)

    # Single-pass scatter. Trainable always wins: a trainable visit writes its
    # row and also flips ``loss_mask[write_pos] = True``, which then blocks any
    # later non-trainable visit from overwriting that slot. A trainable visit
    # arriving after a non-trainable visit still overwrites it (the
    # ``is_trainable or ...`` guard fires unconditionally for trainable). Any
    # path's row agrees at non-trainable slots by construction.
    for path_idx, sample, write_pos, read_pos in positions():
        attention_mask[write_pos] = True
        is_trainable = sample.loss_mask[read_pos]
        if is_trainable:
            assert not loss_mask[write_pos], (
                f"merge_trajectory_samples: shared trainable slot at packed pos {write_pos} "
                f"(path {path_idx}, read {read_pos}) — at most one path may have loss_mask=True."
            )
            labels[write_pos] = sample.labels[read_pos]
            loss_mask[write_pos] = True
            advantage[write_pos] = sample.advantage[read_pos]
            if rollout_logprobs is not None:
                assert sample.rollout_logprobs is not None
                rollout_logprobs[write_pos] = sample.rollout_logprobs[read_pos]
            if teacher_logprobs is not None:
                assert sample.teacher_logprobs is not None
                teacher_logprobs[write_pos] = sample.teacher_logprobs[read_pos]
            if old_values is not None:
                assert sample.old_values is not None
                old_values[write_pos] = sample.old_values[read_pos]
            if returns is not None:
                assert sample.returns is not None
                returns[write_pos] = sample.returns[read_pos]
            if turn_index is not None:
                assert sample.turn_index is not None
                turn_index[write_pos] = sample.turn_index[read_pos]
            if turn_reward is not None:
                assert sample.turn_reward is not None
                turn_reward[write_pos] = sample.turn_reward[read_pos]
            if write_pos > 0:
                routing_source_path[write_pos - 1] = path_idx
                routing_source_read_pos[write_pos - 1] = read_pos
        elif write_pos > 0 and routing_source_path[write_pos - 1] == -1:
            # First (= lowest path_idx) non-trainable visitor wins; trainable
            # paths always overwrite via the branch above.
            routing_source_path[write_pos - 1] = path_idx
            routing_source_read_pos[write_pos - 1] = read_pos

    position_ids = compute_tree_rel_positions_as_list(nodes, total_padded)
    real_total = sum(attention_mask)
    merge_info = build_prefix_merge_info(
        nodes=nodes,
        path_to_leaf=path_to_leaf,
        total_padded=total_padded,
        turn_sample_lens=path_lens,
        real_total=real_total,
    )
    if real_total > 1:
        # Trim the source map to ``real_total - 1`` so it matches the routing
        # tensor's row count (padding rows beyond ``real_total`` are never
        # read by the trainer).
        merge_info.routing_source_path = np.asarray(routing_source_path[: real_total - 1], dtype=np.int64)
        merge_info.routing_source_read_pos = np.asarray(routing_source_read_pos[: real_total - 1], dtype=np.int64)
    # Per-path handles: collect each per-turn sample's handle list (each one
    # already wraps a single inner list per ``Sample.routing_handles_per_path``
    # convention).
    handles_per_path: list[list[Any]] | None = None
    if any(s.routing_handles_per_path is not None for s in per_turn_samples):
        handles_per_path = []
        for s in per_turn_samples:
            assert s.routing_handles_per_path is not None and len(s.routing_handles_per_path) == 1, (
                "per-turn samples fed to merge_trajectory_samples must each carry exactly one path's handles"
            )
            handles_per_path.append(list(s.routing_handles_per_path[0]))
    return SampleCls(
        input_ids=np.asarray(packed_ids, dtype=np.int32),
        labels=np.asarray(labels, dtype=np.int32),
        loss_mask=np.asarray(loss_mask, dtype=np.bool_),
        attention_mask=np.asarray(attention_mask, dtype=np.bool_),
        position_ids=np.asarray(position_ids, dtype=np.int32),
        reward=0.0,
        reward_baseline=0.0,
        advantage=np.asarray(advantage, dtype=np.float32),
        rollout_logprobs=(None if rollout_logprobs is None else np.asarray(rollout_logprobs, dtype=np.float32)),
        teacher_logprobs=(None if teacher_logprobs is None else np.asarray(teacher_logprobs, dtype=np.float32)),
        old_values=(None if old_values is None else np.asarray(old_values, dtype=np.float32)),
        returns=(None if returns is None else np.asarray(returns, dtype=np.float32)),
        turn_index=(None if turn_index is None else np.asarray(turn_index, dtype=np.int32)),
        turn_reward=(None if turn_reward is None else np.asarray(turn_reward, dtype=np.float32)),
        routing_handles_per_path=handles_per_path,
        merge_info=merge_info,
    )


def unpack_tensor_from_merged(packed: torch.Tensor, info: PrefixMergeInfo) -> list[list[float]]:
    """Inverse of trie scatter: emit one per-token-value list per original turn sample.

    Generic 1-D unpack — the values can be logprobs, advantages, rewards,
    or any other per-token quantity stored in the trie's packed layout.
    ``packed`` has shape ``(total_padded,)`` or ``(1, total_padded)``;
    each output list has length ``info.turn_sample_lens[i]``.
    """
    if packed.dim() == 2 and packed.size(0) == 1:
        packed = packed[0]
    assert packed.dim() == 1, f"packed must be 1-D, got shape {tuple(packed.shape)}"
    out: list[list[float]] = []
    for path_idx, (leaf_idx, _) in enumerate(info.path_to_leaf):
        path_len = info.turn_sample_lens[path_idx]
        path_logprobs: list[float] = []
        read_offset = 0
        for n_idx in path_chain_root_to_leaf(info.nodes, leaf_idx):
            nd = info.nodes[n_idx]
            take = min(nd.end - nd.start, path_len - read_offset)
            if take <= 0:
                break
            path_logprobs.extend(packed[nd.start : nd.start + take].tolist())
            read_offset += take
        if len(path_logprobs) < path_len:
            path_logprobs.extend([0.0] * (path_len - len(path_logprobs)))
        out.append(path_logprobs)
    return out


# --------------------------------------------------------------------- #
# Routing scatter helpers (R3 path).
# --------------------------------------------------------------------- #


def gather_merged_routing_per_path(
    per_path_routings: list[np.ndarray],
    merge_info: PrefixMergeInfo,
) -> np.ndarray:
    """Apply the source map to per-path concat routings → merged trie routing.

    ``per_path_routings[k]`` is path ``k``'s cumulative routing array of shape
    ``(path_len_k - 1, L, K)`` — i.e. the concat of that path's ``TensorHandle``
    chunks fetched from the tensor store. The source map (built by
    :func:`merge_trajectory_samples`) maps each merged trie row to the
    ``(path_idx, read_pos)`` whose routing wins at that slot under the
    trainable-wins / lowest-path-idx-wins rule.
    """
    assert merge_info.routing_source_path is not None
    assert merge_info.routing_source_read_pos is not None
    if merge_info.real_total <= 1:
        sample0 = per_path_routings[0]
        return np.empty((0, *sample0.shape[1:]), dtype=sample0.dtype)
    src_path = merge_info.routing_source_path
    src_read = merge_info.routing_source_read_pos
    assert (src_path >= 0).all(), "every merged routing position must have a source path"
    out = np.empty((src_path.shape[0], *per_path_routings[0].shape[1:]), dtype=per_path_routings[0].dtype)
    # Group by source path: one vectorised gather per distinct path instead of
    # a Python loop over every merged row. Adjacent rows from the same path
    # fall into the same mask and resolve in one fused numpy gather.
    for path in np.unique(src_path):
        mask = src_path == path
        # ``src_read`` stores 1-indexed token positions; routing row index is read - 1.
        out[mask] = per_path_routings[int(path)][src_read[mask] - 1]
    return out
