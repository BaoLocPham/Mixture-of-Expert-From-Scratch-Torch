"""
Build the 2017 attention layer yourself.

    python LLM/from_scratch/check_attention.py    # run this constantly
    python LLM/from_scratch/attention.py          # print your own scores and look at them

"Attention Is All You Need", exactly as written: sinusoidal position added at
the input, every query head with its own key and value, biases on all four
projections. No RoPE, no grouped query heads - those came later and are
`llm.py` stage 3.

Four stages, in the order the layer runs:

    1  sinusoidal_pe        the position table, added once before any block
    2  split_heads          one projected row read as n_h heads, and back
       merge_heads
    3  scaled_dot_product   score, scale, mask, softmax, weight the values
    4  VanillaSelfAttention the four projections around the middle three steps

Nothing here imports the rest of the track, so it can be done first.

Notation follows "The equations - with and without MoE":

    Q = xW_Q,  K = xW_K,  V = xW_V                                       E6

    S = QK^T / sqrt(d_h),   M_mn = 0 for n <= m, -inf for n > m          E9
    P = softmax(S + M)                                                   E10
    out = [head_1; ...; head_nh] W_O                                     E11

with m the query position, n the key position, d_h = d_model / n_h.

`../common.py` is the finished version of all four. Opening it before you are
done costs you the exercise; the grader does not read it either - stages 2 and
4 are checked against the code YOU wrote in the stages before them.
"""

import torch
import torch.nn as nn


# ---------------------------------------------------- stage 1: where position enters
def sinusoidal_pe(max_seq, d_model, base=10000.0, device=None):
    """The 2017 absolute positional encoding. Returns (max_seq, d_model).

    This is ADDED to the token embeddings once, before any block runs - which
    is why the attention layer below never mentions position at all.

        PE[m, 2j]   = sin(m / base**(2j/d))
        PE[m, 2j+1] = cos(m / base**(2j/d))

    TODO:
      1. one frequency per PAIR of channels: 2j = 0, 2, 4, ... up to d_model,
         which torch.arange can give you directly. Watch the exponent - it is
         2j/d, and dropping the 2 changes every wavelength in the table.
      2. the angle for position m is m / base**(2j/d). One angle per
         (position, pair), so build it as an outer product, not a broadcast
         against the wrong axis.
      3. sin into the EVEN channels, cos into the ODD ones. The table is
         d_model wide even though there are only d_model/2 angles, because
         each angle is used twice.

    Two consequences of adding this rather than rotating with it, both worth
    holding onto: the signal has to survive L blocks of residual arithmetic
    because nothing re-injects it, and position 2049 does not exist in a table
    built for 2048.

    Row 0 is a giveaway if you get the interleave right: angle 0 everywhere, so
    sin gives 0 and cos gives 1, and the row reads [0, 1, 0, 1, ...].
    """
    raise NotImplementedError("stage 1: sinusoidal_pe")


# ------------------------------------------------------------ stage 2: heads
def split_heads(t, n_heads):
    """(B, T, n_h*d_h) -> (B, n_h, T, d_h).

    There is ONE projection matrix per role, not one per head. Heads are a
    reading of its output row, and this function is that reading.

    TODO, two moves and no arithmetic:
      1. cut the last axis into n_h blocks of d_h. Head h must get the
         CONTIGUOUS block h*d_h : (h+1)*d_h, so element (h, j) is element
         h*d_h + j of the row.
      2. put the head axis in front of the token axis.

    Why the second move: torch.matmul treats the last two axes as the matrix
    and batches over everything before them. Stage 3 contracts d_h and keeps T,
    so (T, d_h) has to be the trailing pair - and with the head axis sitting
    between them it cannot be batched over.

    The trap in the first move: the obvious wrong version has the same shape
    after the transpose, runs, trains, and gives head 0 the strided columns
    0, d_h, 2*d_h, ... instead of the block. Nothing raises.
    """
    raise NotImplementedError("stage 2: split_heads")


def merge_heads(t):
    """(B, n_h, T, d_h) -> (B, T, n_h*d_h). `split_heads` run backwards.

    TODO: the same two moves in the opposite order.

    One detail the grader checks: the transpose leaves a non-contiguous tensor,
    and `view` raises on one. Use the operation that copies when it has to.
    That copy is the only data movement this pair ever causes.
    """
    raise NotImplementedError("stage 2: merge_heads")


# ------------------------------------------------- stage 3: the actual attention
def scaled_dot_product(q, k, v, causal=True):
    """E9 and E10 on already-split heads - the middle three steps.

    q: (B, H, T, d_h)   k, v: (B, H, S, d_h)   ->   (B, H, T, d_h)

    TODO:
      1. score every (query, key) pair: q @ k^T, giving (B, H, T, S).
      2. divide by sqrt(d_h). Not optional and not a constant you can fold into
         the weights: a dot product of two d_h-dimensional random vectors has
         standard deviation sqrt(d_h), so without it the scores spread wider as
         heads get wider and the softmax saturates toward a one-hot row, where
         its gradient is nearly zero. The divisor is exactly the growth rate.
      3. if causal, add E9's mask: -inf where n > m. Query row i is the
         (S - T + i)-th token overall - work that offset out before you write
         the line, because `torch.tril` is right only when S == T and this
         function is also called with S > T from stage 4's cache.
      4. softmax over the KEY axis, then weight the value rows with it.

    Order matters in step 3-4: mask BEFORE the softmax, so -inf becomes an
    exact zero and the row still normalises over the survivors. Mask after and
    every row sums to less than 1 - a quiet, position-dependent rescaling that
    nothing will ever raise about.

    Two properties the grader checks, both free consequences of doing it right:
    every row of the softmax sums to 1 (including row 0, which comes out
    [1, 0, 0, ...] because the first token has nowhere to look but itself), and
    every output row lies inside the hull of the value rows - attention
    interpolates between values, it cannot produce one outside their span.
    """
    raise NotImplementedError("stage 3: scaled_dot_product")


# --------------------------------------------------------- stage 4: the layer
class VanillaSelfAttention(nn.Module):
    """Multi-head causal self-attention, 2017 edition.

    TODO __init__:
      Four Linear layers, all d_model -> d_model, all WITH bias. Same width for
      q, k and v: in 2017 every query head has its own key and value head, and
      the narrower k/v of grouped query attention is a 2023 idea.

      Keep the names `q_proj`, `k_proj`, `v_proj`, `o_proj` - the grader
      addresses them by name.

    TODO forward(x, cos=None, sin=None, cache=None):  (B, T, d) -> (B, T, d)
      1. project x three ways. Same input, three roles: q is what this token is
         looking for, k what it offers to anyone looking, v what gets handed
         over if it is picked.
      2. split each into heads (stage 2).
      3. if `cache` is given, concatenate the stored k/v in FRONT of the new
         ones along the token axis and store the result back. k and v for a past
         token never change, so recomputing them is waste - and the mask offset
         in stage 3 is what makes the shorter query rows line up.
      4. attention (stage 3), then merge the heads and project with o_proj.

    `cos` and `sin` are accepted and ignored, deliberately: the model swaps
    this class for the RoPE one by name, so the signatures have to match. In
    this layer position never arrives here at all - it was added to the
    embedding by `sinusoidal_pe` before the first block.
    """

    def __init__(self, cfg):
        super().__init__()
        self.n_heads, self.d_head = cfg.n_heads, cfg.d_head
        # TODO: the four projections
        raise NotImplementedError("stage 4: VanillaSelfAttention.__init__")

    def forward(self, x, cos=None, sin=None, cache=None):
        raise NotImplementedError("stage 4: VanillaSelfAttention.forward")


# --------------------------------------------------------------------- look
if __name__ == "__main__":
    torch.set_printoptions(precision=4, sci_mode=False)
    T, d_model, n_heads = 5, 8, 2

    try:
        pe = sinusoidal_pe(12, d_model)
    except NotImplementedError as e:
        raise SystemExit(f"{e} - write it first, then run this again.")
    print(f"sinusoidal_pe(12, {d_model})  ->  {tuple(pe.shape)}\n")
    for m in range(4):
        print(f"  m={m}  {[f'{v:+.4f}' for v in pe[m]]}")
    print("\n  row 0 should be [0, 1, 0, 1, ...] - angle 0 in every pair.\n")

    x = (torch.arange(T * d_model).float().reshape(1, T, d_model) * 0.13 - 0.9).sin()
    try:
        q = split_heads(x, n_heads)
    except NotImplementedError:
        raise SystemExit("split_heads not written yet - that is stage 2.")
    print(f"split_heads({tuple(x.shape)}, {n_heads})  ->  {tuple(q.shape)}")
    try:
        back = merge_heads(q)
        print(f"merge_heads back  ->  {tuple(back.shape)}   "
              f"round trip max diff {(back - x).abs().max():.2e}\n")
    except NotImplementedError:
        raise SystemExit("merge_heads not written yet - also stage 2.")

    try:
        out = scaled_dot_product(q, q, q)
    except NotImplementedError:
        raise SystemExit("scaled_dot_product not written yet - that is stage 3.")
    print(f"scaled_dot_product(q, q, q)  ->  {tuple(out.shape)}")
    lo, hi = q.amin(dim=-2), q.amax(dim=-2)          # per channel, over the keys
    inside = ((out >= lo[..., None, :] - 1e-6) & (out <= hi[..., None, :] + 1e-6))
    print(f"  every output channel inside the hull of the value rows: "
          f"{bool(inside.all())}")
    print(f"  row 0 of head 0 vs value row 0 - max diff "
          f"{(out[0, 0, 0] - q[0, 0, 0]).abs().max():.2e}")
    print("\n  the first token can only attend to itself, so its output IS")
    print("  value row 0. If that difference is not ~0, look at the mask.")
