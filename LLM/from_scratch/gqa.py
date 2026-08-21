"""
Build grouped query attention yourself.

    python LLM/from_scratch/check_gqa.py    # run this constantly
    python LLM/from_scratch/gqa.py          # print the grouping and the saving

`attention.py` builds the 2017 layer, where every query head has its own key
and value head. This is the one change that came later (Ainslie et al., 2023):
several query heads SHARE one kv head. Everything else is untouched - same
scores, same mask, same softmax, same merge.

Three stages:

    1  repeat_kv                     the expansion, and the grouping it must use
    2  GroupedQueryAttention         the layer around it, including the cache
    3  kv_cache_floats_per_token     what the whole thing is for

There is no RoPE here on purpose. Position and head-sharing are orthogonal -
you can have either without the other - and `rope.py` is where the rotation
lives. The reference these are checked against is `../common.py`'s
`CausalSelfAttention` run with an identity rotation, which is exactly this.

Notation follows "The equations - with and without MoE":

    n_h, n_kv    query heads, key/value heads                       E6, E11
    n_rep        n_h / n_kv, the group size
    head h reads kv group floor(h / n_rep)                          E11

`../common.py` is the finished version; opening it before you are done costs
you the exercise.
"""

import torch
import torch.nn as nn

from common import split_heads, merge_heads, scaled_dot_product   # noqa: F401


# ---------------------------------------------------- stage 1: the expansion
def repeat_kv(t, n_rep):
    """(B, n_kv, S, d_h) -> (B, n_kv * n_rep, S, d_h).

    One kv head has to serve `n_rep` query heads, and the scores are a single
    batched matmul over the head axis, so the kv side has to be widened to
    match before it runs.

    TODO:
      Copy each kv head n_rep times, keeping every group CONTIGUOUS: kv head 0
      becomes output heads 0 .. n_rep-1, kv head 1 becomes the next n_rep, and
      so on. n_rep == 1 must be a no-op.

    Why contiguous, and not any other order: E11 says query head h reads kv
    group floor(h / n_rep). Floor division puts heads 0 and 1 on group 0 when
    n_rep = 2 - so the expansion has to lay the groups out in that order.

    torch has two functions that both produce the right SHAPE here and only one
    of them produces that order. The other one is the modulo, h % n_kv, which
    is a different assignment of query heads to kv heads: a different model,
    trained happily, and loadable from no checkpoint. Nothing raises.

    A third route, which is what HuggingFace's `repeat_kv` does: insert an axis
    of length n_rep INSIDE the head axis, expand along it, then fold it back in
    with reshape. That lands in the same order, because the new axis sits
    inside the kv-head axis. Any of the three is fine if the order is right.
    """
    raise NotImplementedError("stage 1: repeat_kv")


# -------------------------------------------------------- stage 2: the layer
class GroupedQueryAttention(nn.Module):
    """The 2017 layer with one width changed, and one line added.

    TODO __init__:
      Four bias-free Linears. q and o are full width (n_heads * d_head); k and
      v are NARROWER - n_kv_heads * d_head. That asymmetry is the whole of it,
      and it is the only place n_kv_heads appears in the parameter count.

      Keep the names `q_proj`, `k_proj`, `v_proj`, `o_proj`, and store
      `n_rep = n_heads // n_kv_heads`.

    TODO forward(x, cache=None):  (B, T, d) -> (B, T, d)
      1. project, and split into heads - q with n_heads, k and v with
         n_kv_heads. Two different counts, from `split_heads`.
      2. if `cache` is given, concatenate the stored k/v in FRONT of the new
         ones along the token axis, and store the result back.
      3. repeat_kv on k and v.
      4. scaled_dot_product(q, k, v, causal=True), merge_heads, o_proj.

    Steps 2 and 3 are in that order for a reason the shapes will not tell you.
    The cache must hold n_kv heads, not n_h: it is the thing GQA exists to
    shrink, and expanding before you store it saves exactly nothing while
    looking completely correct. The grader checks the width of what you stored.

    At n_kv_heads == n_heads this is plain multi-head attention, with n_rep = 1
    and the expansion a no-op - GQA is a superset, not a different layer.
    """

    def __init__(self, cfg):
        super().__init__()
        raise NotImplementedError("stage 2: GroupedQueryAttention.__init__")

    def forward(self, x, cache=None):
        raise NotImplementedError("stage 2: GroupedQueryAttention.forward")


# --------------------------------------------------- stage 3: what it is for
def kv_cache_floats_per_token(n_layers, n_kv_heads, d_head):
    """How many floats the KV cache holds per token of context. E40.

    TODO: one line. Count what is actually stored: for every layer, a key and a
    value, each n_kv_heads * d_head wide.

    Note what is NOT in it: n_heads. The query side is never cached - queries
    are consumed the moment the scores are computed and never looked at again.
    That is the entire reason this trade is acceptable: the model keeps all its
    attention patterns and only the stored side shrinks.

    Note also what it does not shrink: FLOPs. `repeat_kv` puts k and v back at
    full width before the matmul, so the arithmetic costs exactly what MHA
    costs. GQA buys memory and bandwidth, not compute.
    """
    raise NotImplementedError("stage 3: kv_cache_floats_per_token")


# --------------------------------------------------------------------- look
if __name__ == "__main__":
    torch.set_printoptions(precision=4, sci_mode=False)
    n_h, n_kv = 4, 2
    n_rep = n_h // n_kv

    try:
        tagged = torch.arange(n_kv).float()[None, :, None, None].expand(1, n_kv, 3, 2)
        out = repeat_kv(tagged, n_rep)
    except NotImplementedError as e:
        raise SystemExit(f"{e} - write it first, then run this again.")

    print(f"n_h = {n_h}, n_kv = {n_kv}, n_rep = {n_rep}\n")
    print("  which kv head each query head ends up reading:")
    print(f"    yours            {[int(v) for v in out[0, :, 0, 0]]}")
    print(f"    floor(h/n_rep)   {[h // n_rep for h in range(n_h)]}   <- E11")
    print(f"    h mod n_kv       {[h % n_kv for h in range(n_h)]}   <- the trap\n")

    try:
        f = kv_cache_floats_per_token
        # a real config, not the toy above: 32 layers, 32 heads, d_head 128
        rows = [("MHA", 32), ("GQA 32:8", 8), ("MQA", 1)]
        print("  KV cache at 32 layers, 32 query heads, d_head 128:")
        for name, kv in rows:
            n = f(32, kv, 128)
            gb = n * 8192 * 32 * 2 / 1e9        # 8k context, batch 32, fp16
            print(f"    {name:<10} {n:>8,} floats/token   {gb:>6.1f} GB "
                  f"at 8k x batch 32, fp16")
        print("\n  The query side is identical in all three rows. Only the side")
        print("  you have to keep gets smaller.")
    except NotImplementedError:
        print("  (stage 3 not written yet - that table is what it is for.)")
