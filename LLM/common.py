"""
Reference implementation: a decoder-only transformer LM, small enough to read
in one sitting, built from the same parts every modern LLM is built from.

    embed -> [ RMSNorm -> attention -> + ] [ RMSNorm -> FFN -> + ] x L -> RMSNorm -> logits

Two models, on purpose. `TinyLLM` is that line and nothing else - no switches,
no cache, no sampling - and is the one to read first. `LLM` is the same model
with the four things a real one needs: the attention switch, the FFN switch,
weight tying, and incremental decoding. Everything that prints a number uses
`LLM`.

The FFN slot takes either a single feed-forward network (`ffn="dense"`) or a
top-k mixture of them (`ffn="moe"`). Nothing else in the model changes.

A second switch, `attn`, swaps this file's attention layer (RoPE + grouped
query heads, what a 2024 model runs) for the one "Attention Is All You Need"
wrote in 2017 (sinusoidal position added at the input, every head with its own
k and v). Same purpose as the FFN switch: make a seven-year architectural
difference a one-word diff, so it can be priced instead of argued about. That is
the whole reason this track exists: it makes "with MoE" and "without MoE" a
one-word diff, so every difference in parameters, FLOPs and loss can be
attributed to that one substitution rather than to two separately-written models.

Deliberately here: RMSNorm (not LayerNorm), pre-norm residuals, RoPE, grouped
query attention, a KV cache, weight tying. Deliberately absent: dropout,
biases, learned position embeddings - they would add lines without adding
mechanism, and every one of them is a knob rather than a part.

Symbols and equation numbers follow the notes ("The equations - with and
without MoE", E1-E43), so a line here can be read next to the line there:

    B, T          batch, sequence length          idx (B, T)
    V, d, L       vocab, model width, blocks      E1, E32
    n_h, n_kv     query heads, kv heads           E6, E11   -> n_heads, n_kv_heads
    d_h           head width, d / n_h             E7, E9    -> d_head
    d_ff          FFN inner width                 E18
    N, k          experts, experts per token      E19-E21
    m, n          query position, key position    E7-E9
    theta_i       rotation rate of pair i         E7
    b             rotation base, 10^4             E7        -> base
    s             context-extension factor        3b.2      -> scale
    alpha         aux-loss coefficient            E27       -> aux_weight
    P_tot, P_act  total, active params per token  E36, E37

Row-vector convention throughout (x is (..., d) and the maths reads xW), which
is what nn.Linear already does.

Three functions here are reference only and are never called by the model:
`rotate_half` and `apply_rope_half` (the split-half channel convention that
HuggingFace uses) and `rope_tables_scaled` (Position Interpolation). They are
the answers to stages 3 and 4 of LLM/from_scratch/rope.py.
"""

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------------- config
@dataclass
class LLMConfig:
    """Every shape in the model, in one place.

    Two fields change the architecture; everything else changes sizes.

        ffn   "dense" | "moe"        - the slot this whole track is about
        attn  "rope"  | "vanilla"    - this layer, or the one the 2017 paper wrote
    """
    vocab_size: int = 64
    d_model: int = 64
    n_layers: int = 4
    n_heads: int = 4
    n_kv_heads: int = None      # None -> = n_heads (plain MHA). Fewer -> GQA.
    d_ff: int = 128
    max_seq: int = 64
    attn: str = "rope"          # "rope" (RoPE + GQA) | "vanilla" (2017 MHA)
    ffn: str = "dense"          # "dense" | "moe"
    n_experts: int = 4          # ffn="moe" only
    k: int = 2                  # ffn="moe" only
    tie_embeddings: bool = True
    aux_weight: float = 0.01    # load-balancing loss coefficient

    def __post_init__(self):
        if self.n_kv_heads is None:
            self.n_kv_heads = self.n_heads
        assert self.d_model % self.n_heads == 0, "d_model must split evenly into heads"
        assert self.n_heads % self.n_kv_heads == 0, "each kv head must serve a whole number of q heads"
        assert self.ffn in ("dense", "moe")
        assert self.attn in ("rope", "vanilla")
        if self.attn == "vanilla":
            assert self.n_kv_heads == self.n_heads, (
                "vanilla attention is the 2017 layer: every query head has its "
                "own k and v. Set n_kv_heads = n_heads, or use attn='rope'.")

    @property
    def d_head(self):
        return self.d_model // self.n_heads


# -------------------------------------------------------------------- norm
class RMSNorm(nn.Module):
    """E4:  RMSNorm(x) = x / sqrt(mean_i x_i^2 + eps) * g.  No mean, no bias.

    LayerNorm centres *and* scales; RMSNorm only scales. Dropping the mean costs
    nothing measurable in quality and removes a reduction, which is why every
    recent model uses it. The learned gain is per-channel, so the layer can still
    decide some channels matter more - it just cannot shift them.
    """

    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x):
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()   # (..., 1)
        return x * rms * self.weight                                      # (..., d)


# -------------------------------------------------------------------- rope
def rope_tables(d_head, max_seq, base=10000.0, device=None):
    r"""Precompute cos/sin for E7, both (max_seq, d_h/2).

    RoPE reads a head's d_h channels as d_h/2 INDEPENDENT 2-D vectors:

        channel index:   0   1  |  2   3  |  4   5  |  6   7      (d_h = 8)
                        \_pair 0_/ \_pair 1_/ \_pair 2_/ \_pair 3_/

    Each pair is rotated in its own little plane, by its own angle. So there is
    one angle per pair, not per channel - which is why both tables below are
    d_h/2 wide and not d_h wide. That single fact explains every shape here.

    Pair i turns at rate

        theta_i = b ** (-2i / d_h)             b = base = 10^4

    and the angle applied at position m is just m * theta_i - the rate times how
    far along the sequence the token sits. `arange(0, d_h, 2)` IS the sequence
    2i = 0, 2, 4, ..., so the exponent below is exactly -2i/d_h from E7.

    The rates are geometric, from 1 down to 1/b. For d_h = 8:

        i    2i/d_h    theta_i        wavelength 2*pi/theta_i
        0    0.00      1.0            ~6 tokens        <- turns fast
        1    0.25      0.1            ~63 tokens
        2    0.50      0.01           ~628 tokens
        3    0.75      0.001          ~6283 tokens     <- barely turns at all

    So one head carries a whole ladder of scales at once: the low pairs resolve
    "3 tokens back" sharply and wrap around constantly, the high pairs cannot
    tell 3 from 4 but still distinguish "same paragraph" from "10k tokens ago".
    Position is not one number in one place; it is spread across the pairs.

    Returned as cos and sin separately (rather than as angles) because they are
    what the rotation actually needs, and computing them once for max_seq
    positions keeps them out of the per-layer, per-token path entirely.
    """
    two_i = torch.arange(0, d_head, 2, device=device).float()   # (d_h/2,) = 0,2,4,...
    theta = base ** (-two_i / d_head)                           # (d_h/2,) rate per pair
    m = torch.arange(max_seq, device=device).float()            # (max_seq,) positions

    # outer product: every position against every pair -> the full angle table.
    # row m, column i is the angle m * theta_i that pair i gets at position m.
    ang = torch.outer(m, theta)                                 # (max_seq, d_h/2)
    return ang.cos(), ang.sin()


def apply_rope(x, cos, sin):
    r"""Rotate each channel pair of x by its position's angle - E7.

    x: (B, H, T, d_h)   cos/sin: (T, d_h/2)   ->   (B, H, T, d_h)

    For pair i of the token at position m, this is one 2x2 rotation:

        (x'_2i )   ( cos(m.theta_i)  -sin(m.theta_i) ) (x_2i  )
        (x'_2i+1) = ( sin(m.theta_i)   cos(m.theta_i) ) (x_2i+1)

    The whole function is that line, done for every pair, position, head and
    batch element at once. Nothing is learned here and nothing is added to x -
    the vector is turned, and its length is untouched (a rotation preserves it,
    which the dissector checks).

    The payoff is E8: rotating q by m and k by n and taking the dot product
    gives q^T R_(n-m) k, so the score depends on the GAP between the tokens and
    never on where the pair sits in the sequence.

    Worked example, d_h = 4 (two pairs), one token at position m:

        x       = [ a,  b,  c,  d ]
        x1      = [ a,      c    ]        x[..., 0::2] - first of each pair
        x2      = [     b,      d]        x[..., 1::2] - second of each pair
        cos/sin = [ c0, c1 ], [ s0, s1 ]  one per PAIR, already at position m
        rot1    = [ a*c0 - b*s0,  c*c1 - d*s1 ]
        rot2    = [ a*s0 + b*c0,  c*s1 + d*c1 ]
        out     = [ a*c0 - b*s0,  a*s0 + b*c0,  c*c1 - d*s1,  c*s1 + d*c1 ]
                    \________ pair 0 ________/  \________ pair 1 ________/
    """
    # 1. Split the head dimension into the two members of every pair. Strided
    #    slicing, so these are views: no copy, and no arithmetic yet.
    x1 = x[..., 0::2]                     # (B, H, T, d_h/2)  channels 0, 2, 4, ...
    x2 = x[..., 1::2]                     # (B, H, T, d_h/2)  channels 1, 3, 5, ...

    # 2. Give the tables the two leading axes x has, so they broadcast. Both
    #    are (T, d_h/2) -> (1, 1, T, d_h/2): every batch element and every head
    #    at position m is rotated by the same angle. Position is a property of
    #    the token, not of the head - this line is where that gets asserted.
    c = cos[None, None]
    s = sin[None, None]

    # 3. The rotation, all pairs at once. Two elementwise expressions, which is
    #    the entire "matrix multiply" - a 2x2 rotation has no reason to be a
    #    matmul when both of its rows fit on one line.
    rot1 = x1 * c - x2 * s                # new FIRST coordinate of every pair
    rot2 = x1 * s + x2 * c                # new SECOND coordinate of every pair

    # 4. Interleave back into channel order. stack(dim=-1) builds
    #    (B, H, T, d_h/2, 2) - one row per pair, holding [first, second] - and
    #    flatten(-2) reads those rows in order, giving
    #        [rot1_0, rot2_0, rot1_1, rot2_1, ...]
    #    which is exactly the layout we sliced apart in step 1.
    #
    #    torch.cat([rot1, rot2], dim=-1) would be a natural-looking mistake
    #    here: it yields [rot1_0, rot1_1, ..., rot2_0, rot2_1, ...], i.e. the
    #    "split the head in half" layout that HuggingFace's LLaMA uses. That
    #    convention is equally valid - it just pairs channel j with channel
    #    j + d_h/2 instead of j with j+1, and the checkpoints permute W_Q and
    #    W_K to match; `apply_rope_half` below is it, written out. What is NOT
    #    valid is mixing the two: rotate q one way and k the other and E8
    #    quietly stops holding, with no error anywhere.
    out = torch.stack([rot1, rot2], dim=-1)                     # (…, d_h/2, 2)
    return out.flatten(-2)                                      # (…, d_h)


# ------------------------------------------------- rope: the other conventions
# The model above uses `rope_tables` + `apply_rope` and nothing else. The three
# functions below are reference implementations of the two variants every real
# codebase has to deal with, kept here because they are the answers to stages 3
# and 4 of LLM/from_scratch/rope.py. Nothing in this file calls them.


def rotate_half(x):
    r"""A quarter turn, for the split-half channel layout.

    x: (..., d_h)  ->  (..., d_h)

        [ a0 a1 a2 a3 | b0 b1 b2 b3 ]  ->  [ -b0 -b1 -b2 -b3 | a0 a1 a2 a3 ]

    Why a function shaped like this exists. E7 on a pair (a, b) is

        (a*cos - b*sin,  a*sin + b*cos)

    which is the same as

        (a, b) * cos  +  (-b, a) * sin

    and (-b, a) is just (a, b) turned 90 degrees. So ANY rotation is "the
    vector times cos, plus the vector-turned-a-quarter-turn times sin" - which
    is e^(i.phi) = cos(phi) + i.sin(phi) written without complex numbers.

    `rotate_half` is that quarter turn for a layout where a channel's partner
    sits d_h/2 away instead of 1 - so both members of every pair live in the
    same half of the vector, and one `cat` performs all d_h/2 quarter turns at
    once. Applied twice it gives -x exactly, which is the cheapest test there
    is that you have a rotation and not a shuffle.
    """
    h = x.shape[-1] // 2
    return torch.cat([-x[..., h:], x[..., :h]], dim=-1)


def apply_rope_half(x, cos, sin):
    r"""E7 again, in the split-half ("GPT-NeoX"/HuggingFace) convention.

    x: (B, H, T, d_h)   cos/sin: (T, d_h/2)   ->   (B, H, T, d_h)

    Same tables, same angles, same rotation. The only difference from
    `apply_rope` is which two channels count as a pair:

        apply_rope       pairs (0,1), (2,3), (4,5), ...     partners adjacent
        apply_rope_half  pairs (0,4), (1,5), (2,6), ...     partners d_h/2 apart

    The `cat([cos, cos])` below is where that gets decided. The tables are
    d_h/2 wide and this form needs one entry per CHANNEL, so they are
    duplicated - and duplicating means channel j and channel j + d_h/2 read the
    same theta_j. That one line IS the pairing, the way `x[..., 0::2]` is the
    pairing in `apply_rope`.

    The two are the same function on a permuted channel order:

        perm = torch.arange(d_h).reshape(2, d_h // 2).t().reshape(-1)
        inv  = torch.argsort(perm)
        apply_rope_half(x, cos, sin) == apply_rope(x[..., perm], cos, sin)[..., inv]

    exactly - max difference 0.0, not "close". Which is also the whole content
    of the checkpoint story: HuggingFace implements this form, so
    convert_llama_weights_to_hf.py bakes `perm` into the rows of W_Q and W_K
    once, and Meta's interleaved-trained weights come out agreeing with it.

    Either convention is fine on its own and E8 holds for both. Mixing them is
    not: rotate q one way and k the other and three pairs at the same gap - which
    must agree to ~1e-07 - come out 1.7 apart. Nothing raises. Same shapes, same
    dtype, a model that trains and has simply lost relative position.
    """
    c = torch.cat([cos, cos], dim=-1)[None, None]               # (1, 1, T, d_h)
    s = torch.cat([sin, sin], dim=-1)[None, None]
    return x * c + rotate_half(x) * s


def rope_tables_scaled(d_head, max_seq, base=10000.0, scale=1.0, device=None):
    r"""Position Interpolation (Chen et al., 2023): `rope_tables` with the
    positions divided by `scale`.

    A model trained at 2k tokens has never handed pair 0 (theta = 1.0) an angle
    above 2048 radians. Run it at 8k and that becomes 8192 - no error, cosine is
    happy to be evaluated anywhere, but the weights have no training signal for
    that region and quality falls off sharply.

    PI's answer is to squeeze the position axis rather than extend it: position
    m is rotated as if it were m/scale, so a `scale`x longer sequence lands
    inside the range the model was trained on. Everything else is untouched -
    same weights, same theta_i, same code path, no new parameters - which is why
    it can be applied to an already-trained checkpoint.

    The consequence, and the price, are the same measurement:

        plain tables,  gap 1 token   ->  q.k = +0.150094
        scaled by 4,   gap 4 tokens  ->  q.k = +0.150094      difference 0.0
        plain tables,  gap 4 tokens  ->  q.k = -1.149394

    Four tokens apart now produces exactly the score one token apart used to.
    That is the gain (reach) and the loss (resolution) in one line: every
    wavelength is multiplied by scale, the fast pairs included, so pair 0 goes
    from a period of ~6.3 tokens to ~25.1 and the pair that separated adjacent
    tokens now separates groups of four. Hence the fine-tuning PI needs.

    The two methods that followed spend the same budget more carefully, and
    neither is implemented here:

      NTK-aware  raise the BASE instead, base * scale ** (d_h / (d_h - 2)).
                 theta_i = base ** (-2i/d_h) has i = 0 in the exponent, so pair
                 0 is base**0 = 1 whatever the base is - the fast pairs are left
                 alone and the whole stretch lands on the slow ones.
      YaRN       a per-pair ramp between interpolating and extrapolating, plus
                 an attention temperature. 64k-128k on a fraction of PI's data.

    scale = 1.0 reproduces `rope_tables` exactly.
    """
    two_i = torch.arange(0, d_head, 2, device=device).float()
    theta = base ** (-two_i / d_head)
    m = torch.arange(max_seq, device=device).float() / scale    # <- the method
    ang = torch.outer(m, theta)                                 # (max_seq, d_h/2)
    return ang.cos(), ang.sin()


# --------------------------------------------------------------- attention
class CausalSelfAttention(nn.Module):
    """Multi-head causal attention with optional grouped query attention.

    E6 (projections), E7 (rotation), E9 (scores + mask), E10 (softmax and the
    per-head weighted sum), E11 (concat and W_O). The GQA rule of E11 - head h
    reads kv group floor(h / (n_h/n_kv)) - is the repeat_interleave below.

    n_kv_heads < n_heads means several query heads share one key/value head. The
    q side is untouched, so the model keeps all its attention patterns; what
    shrinks is the KV *cache*, which during generation is the thing that
    actually runs out of memory. GQA is a memory-bandwidth decision, not a
    quality one.
    """

    def __init__(self, cfg: LLMConfig):
        super().__init__()
        self.n_heads, self.n_kv_heads = cfg.n_heads, cfg.n_kv_heads
        self.d_head = cfg.d_head
        self.n_rep = cfg.n_heads // cfg.n_kv_heads          # n_h / n_kv, E11

        self.q_proj = nn.Linear(cfg.d_model, cfg.n_heads * cfg.d_head, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.d_head, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.d_head, bias=False)
        self.o_proj = nn.Linear(cfg.n_heads * cfg.d_head, cfg.d_model, bias=False)

    def _split(self, t, n):
        B, T, _ = t.shape
        return t.view(B, T, n, self.d_head).transpose(1, 2)  # (B, n, T, dh)

    def forward(self, x, cos, sin, cache=None):
        """x: (B, T, d) -> (B, T, d).

        cache: None, or a dict with 'k'/'v' holding everything seen so far. With
        a cache, T is usually 1 and the past is read rather than recomputed.
        """
        B, T, d = x.shape

        q = self._split(self.q_proj(x), self.n_heads)        # (B, H,   T, dh)
        k = self._split(self.k_proj(x), self.n_kv_heads)     # (B, Hkv, T, dh)
        v = self._split(self.v_proj(x), self.n_kv_heads)     # (B, Hkv, T, dh)

        # RoPE goes on q and k only. v carries content, not position - rotating
        # it would rotate the thing being retrieved instead of the address.
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        if cache is not None:
            if "k" in cache:
                k = torch.cat([cache["k"], k], dim=2)        # (B, Hkv, T_past+T, dh)
                v = torch.cat([cache["v"], v], dim=2)
            cache["k"], cache["v"] = k, v

        # One kv head serves n_rep query heads: repeat it so the einsum lines up.
        # repeat_interleave, not repeat - head h of the q side must land next to
        # its own group, and repeat would interleave the groups wrongly.
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)       # (B, H, S, dh)
            v = v.repeat_interleave(self.n_rep, dim=1)

        n_keys = k.shape[2]                                  # keys available
        att = (q @ k.transpose(-1, -2)) / self.d_head ** 0.5  # (B, H, T, n_keys)  E9

        # E9's mask: M_mn = 0 for n <= m, -inf for n > m, with m the query
        # position and n the key position. Query row i is the (n_keys-T+i)-th
        # token overall, so with a cache (T=1) the single row is all-visible -
        # that offset is what makes one line correct in training AND generation.
        m = torch.arange(n_keys - T, n_keys, device=x.device)[:, None]   # (T, 1)
        n = torch.arange(n_keys, device=x.device)[None, :]               # (1, n_keys)
        att = att.masked_fill(n > m, float("-inf"))

        p = att.softmax(dim=-1)                              # (B, H, T, n_keys)  E10
        y = p @ v                                            # (B, H, T, d_h)  head_h
        y = y.transpose(1, 2).reshape(B, T, self.n_heads * self.d_head)
        return self.o_proj(y)                                # (B, T, d)  E11


# ------------------------------------------------- attention, the 2017 version
def sinusoidal_pe(max_seq, d_model, base=10000.0, device=None):
    r"""The original absolute positional encoding (Vaswani et al., 2017).

    Returns (max_seq, d_model), to be ADDED to the token embeddings once, before
    any block runs:

        PE[m, 2j]   = sin(m / b**(2j/d)),      PE[m, 2j+1] = cos(m / b**(2j/d))

    Compare `rope_tables` above and the family resemblance is the point: same
    base, same geometric frequency ladder, same pairing of channels. What
    differs is the verb. This is ADDED to x once at the input; RoPE ROTATES q
    and k with it inside every layer - and that difference is the whole reason
    the dot product ends up seeing only n - m (E8) in one case and not the
    other.

    Two consequences of adding rather than rotating, both easy to miss:
      - the signal has to survive L blocks of residual arithmetic, because
        nothing re-injects it;
      - position 2049 does not exist in a table built for 2048, and there is no
        reason the model should read a gap of 3 the same way at m = 4 and at
        m = 400 - it had to learn each offset separately.
    """
    m = torch.arange(max_seq, device=device).float()[:, None]      # (max_seq, 1)
    two_j = torch.arange(0, d_model, 2, device=device).float()     # (d/2,)
    ang = m / base ** (two_j / d_model)                            # (max_seq, d/2)
    pe = torch.zeros(max_seq, d_model, device=device)
    pe[:, 0::2] = ang.sin()
    pe[:, 1::2] = ang.cos()
    return pe


def split_heads(t, n_heads):
    r"""(B, T, n_h*d_h) -> (B, n_h, T, d_h). Step 2 of the layer.

    Two moves and no arithmetic:

      view      cuts the row into n_h CONTIGUOUS blocks - head h is columns
                h*d_h : (h+1)*d_h, so element (h, j) is element h*d_h + j of the
                row. A view cannot move data, so which columns belong to head h
                was decided by how the projection's weight matrix was laid out,
                not by this line.
      transpose puts the head axis in front. torch.matmul treats the last two
                axes as the matrix and batches over everything before them, so
                (T, d_h) has to be the trailing pair - and in (B, T, n_h, d_h)
                the head axis sits BETWEEN the two matrix axes, where it cannot
                be batched over.

    One transpose buys all n_h head-attentions as a single batched matmul.
    """
    B, T, w = t.shape
    return t.view(B, T, n_heads, w // n_heads).transpose(1, 2)


def merge_heads(t):
    r"""(B, n_h, T, d_h) -> (B, T, n_h*d_h). `split_heads` run backwards.

    `reshape`, not `view`: transpose returned a non-contiguous tensor and view
    raises on one. That copy is the only data movement the split and its inverse
    ever cause, and it happens once per layer, on the way out.
    """
    B, n_heads, T, d_head = t.shape
    return t.transpose(1, 2).reshape(B, T, n_heads * d_head)


def scaled_dot_product(q, k, v, causal=True):
    r"""E9 and E10, on already-split heads. Steps 3 to 5 of the layer.

        S = q k^T / sqrt(d_h)                                  E9, first half
        A = S + M,  M_mn = 0 for n <= m, -inf for n > m         E9's mask
        P = softmax(A)                                         E10
        out = P v

    q: (B, H, T, d_h)   k, v: (B, H, S, d_h)   ->   (B, H, T, d_h)

    Three things this short function is carrying:

      the divisor  a dot product of two d_h-dimensional random vectors has
                   standard deviation sqrt(d_h), so without it the scores spread
                   wider as heads get wider and softmax saturates toward a
                   one-hot row, where its gradient is nearly zero. The divisor is
                   exactly the growth rate, so the score scale stops depending on
                   how wide you made the head.
      the offset   query row i is the (S - T + i)-th token overall. In training
                   S == T and the mask is a plain lower triangle; with a cache
                   T is 1 and the single row sits at the END of the keys, all
                   visible. That one term is what makes this line correct in both.
      the order    masking BEFORE the softmax makes -inf an exact zero and leaves
                   the row normalised over the survivors. Zeroing afterwards
                   leaves rows summing to less than 1 - a quiet, position-
                   dependent rescaling of every token.

    The output rows are convex combinations of v: the weights are non-negative
    and sum to 1, so attention can interpolate between value rows but never
    produce one outside their span.
    """
    T, S, d_head = q.shape[-2], k.shape[-2], q.shape[-1]
    att = (q @ k.transpose(-1, -2)) / d_head ** 0.5           # E9
    if causal:
        m = torch.arange(S - T, S, device=q.device)[:, None]  # (T, 1)  query pos
        n = torch.arange(S, device=q.device)[None, :]         # (1, S)  key pos
        att = att.masked_fill(n > m, float("-inf"))
    return att.softmax(dim=-1) @ v                            # E10


class VanillaSelfAttention(nn.Module):
    """Multi-head attention exactly as "Attention Is All You Need" wrote it.

    Same four equations as `CausalSelfAttention` - E6, E9, E10, E11 - and the
    same shapes. Four things are deliberately different, and each one is a
    decision the field made LATER:

      1. no GQA. Every query head has its own k and v, so the k/v projections
         are full width and the KV cache is n_h/n_kv times bigger. GQA (2023)
         is the only structural difference between the two classes.
      2. no RoPE. Position is added to the embedding once, by `sinusoidal_pe`,
         and this layer never hears about it. That is why forward() takes cos
         and sin and ignores them - so `attn="vanilla"` stays a one-word swap.
      3. biases on all four projections, as in the paper's implementation and
         in torch's own nn.MultiheadAttention. LLaMA and everything after it
         dropped them: they cost parameters and buy nothing measurable.
      4. no dropout on the attention weights. The paper has it (p_drop = 0.1);
         this track has no dropout anywhere, so leaving it out keeps the two
         classes comparable rather than faithful. It is the one paper feature
         missing here.

    What is NOT different: the scaling by sqrt(d_h), the causal mask, the
    softmax, the concat-and-project. Those are 2017 and unchanged.

    The cache works the same way. It is not in the paper - decoding one token
    at a time with stored k/v is an inference technique that came later - but
    it is a pure optimisation, so including it changes no output.
    """

    def __init__(self, cfg: LLMConfig):
        super().__init__()
        self.n_heads, self.d_head = cfg.n_heads, cfg.d_head
        d = cfg.d_model
        self.q_proj = nn.Linear(d, d, bias=True)              # W^Q, all heads
        self.k_proj = nn.Linear(d, d, bias=True)              # W^K
        self.v_proj = nn.Linear(d, d, bias=True)              # W^V
        self.o_proj = nn.Linear(d, d, bias=True)              # W^O

    def forward(self, x, cos=None, sin=None, cache=None):
        """x: (B, T, d) -> (B, T, d). cos/sin are ignored - see the docstring."""
        q = split_heads(self.q_proj(x), self.n_heads)         # 1 + 2
        k = split_heads(self.k_proj(x), self.n_heads)         # full width: no GQA
        v = split_heads(self.v_proj(x), self.n_heads)

        if cache is not None:
            if "k" in cache:
                k = torch.cat([cache["k"], k], dim=2)
                v = torch.cat([cache["v"], v], dim=2)
            cache["k"], cache["v"] = k, v

        y = scaled_dot_product(q, k, v, causal=True)          # 3, 4, 5   E9, E10
        return self.o_proj(merge_heads(y))                    # 6         E11


# --------------------------------------------------------------------- ffn
class FeedForward(nn.Module):
    """SwiGLU feed-forward - E18:

        FFN(x) = (SiLU(x W1) * x W3) W2,   W1, W3: (d, d_ff)   W2: (d_ff, d)

    Three matrices, not two: W1 produces the gate, W3 the value it scales, W2
    projects back. Parameters are 3*d*d_ff, which is why d_ff is conventionally
    set near (8/3)d rather than 4d - the two-matrix FFN at 4d and this one at
    (8/3)d cost the same.

    The `Expert` in DenseMoe/common.py and SparseMoe/common.py is the older
    two-matrix form, W2(SiLU(x W1)). Those tracks were studying the mixture and
    kept the expert as small as possible on purpose; this track is about the
    model real LLMs are, so the FFN is the one real LLMs use. Nothing about
    routing changes - an MoE layer does not invent a new kind of FFN, it keeps
    N of whichever one the model uses and picks between them.
    """

    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)      # gate proj
        self.w3 = nn.Linear(d_model, d_ff, bias=False)      # up proj
        self.w2 = nn.Linear(d_ff, d_model, bias=False)      # down proj

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))     # (..., d)


def switch_aux_loss(logits, topi, n_experts):
    """E26-E27 without the coefficient:  N * sum_i f_i * P_i.

    Same function as SparseMoe/common.py. P_i is the mean routing probability
    (differentiable); f_i is the mean fraction of tokens actually dispatched
    (not). alpha is applied by the model, as cfg.aux_weight, so this returns the
    bare sum. Minimised at k when the load is uniform, never at 0.
    """
    P = F.softmax(logits, dim=-1).mean(0)                    # (N,)
    f = F.one_hot(topi, n_experts).sum(1).float().mean(0)    # (N,)
    return n_experts * (f * P).sum()


class MoEFeedForward(nn.Module):
    """Top-k routed FFN: gather -> compute -> scatter.

    The dispatch loop is the one from SparseMoe/common.py. It sits here because
    the interesting question in this track is not how dispatch works - that was
    settled in SparseMoe - but what happens when you drop it into a real model:
    what it does to parameter count, to per-token compute, to the loss curve,
    and to how balanced the experts stay once the tokens are real.
    """

    def __init__(self, d_model, d_ff, n_experts, k):
        super().__init__()
        self.d_model, self.d_ff = d_model, d_ff
        self.n_experts, self.k = n_experts, k
        self.gate = nn.Linear(d_model, n_experts, bias=False)     # W_g: (d, N)  E19
        self.experts = nn.ModuleList([FeedForward(d_model, d_ff) for _ in range(n_experts)])

    def forward(self, x):
        B, T, d = x.shape
        xf = x.reshape(-1, d)                                # (S, d) routing is per token
        logits = self.gate(xf)                               # (S, N)  h = x W_g   E19

        # E21 as written there: TopK first, softmax over the survivors only
        # (Mixtral's ordering). The other ordering - softmax over all N, then
        # truncate and renormalise - is the same function, and SparseMoe's
        # dissector measures the two agreeing to 6e-08.
        topl, topi = logits.topk(self.k, dim=-1)             # (S, k)  set  T_k
        topw = F.softmax(topl, dim=-1)                       # (S, k)  g_i, sums to 1

        y = torch.zeros_like(xf)
        for e_id, expert in enumerate(self.experts):
            tok, slot = (topi == e_id).nonzero(as_tuple=True)
            if tok.numel() == 0:
                continue
            out = expert(xf[tok])                            # only the routed rows
            y.index_add_(0, tok, topw[tok, slot, None] * out)

        aux = switch_aux_loss(logits, topi, self.n_experts)
        return y.reshape(B, T, d), aux

    def load_per_expert(self, x):
        """(N,) rows routed to each expert, for imbalance reporting."""
        logits = self.gate(x.reshape(-1, self.d_model))
        topi = logits.topk(self.k, dim=-1).indices
        return F.one_hot(topi, self.n_experts).sum(dim=(0, 1))


# ------------------------------------------------------------------- block
class Block(nn.Module):
    """Pre-norm residual block - E14 and E15:

        u = x + Attn(Norm(x)),      x' = u + Slot(Norm(u))

    E15 is the entire with/without-MoE question: `Slot` is E18 (dense FFN) or
    E21 (top-k routed). Nothing else in this file changes between them.

    Pre-norm (normalise the branch input, add the raw branch output) leaves an
    unnormalised path from the embedding straight to the final norm, so the
    gradient reaches layer 0 without passing through L normalisations. Post-norm
    - the original 2017 arrangement - needs a warmup schedule to train at depth
    for exactly that reason.
    """

    def __init__(self, cfg: LLMConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model)
        self.attn = (VanillaSelfAttention(cfg) if cfg.attn == "vanilla"
                     else CausalSelfAttention(cfg))
        self.norm2 = RMSNorm(cfg.d_model)
        self.is_moe = cfg.ffn == "moe"
        self.ffn = (MoEFeedForward(cfg.d_model, cfg.d_ff, cfg.n_experts, cfg.k)
                    if self.is_moe else FeedForward(cfg.d_model, cfg.d_ff))

    def forward(self, x, cos, sin, cache=None):
        x = x + self.attn(self.norm1(x), cos, sin, cache)     # (B, T, d)
        if self.is_moe:
            out, aux = self.ffn(self.norm2(x))
            return x + out, aux
        return x + self.ffn(self.norm2(x)), None


# ------------------------------------------------------------------- model
class TinyLLM(nn.Module):
    r"""A decoder-only language model, with nothing optional in it.

        embed -> Block x L -> RMSNorm -> logits

    That is the whole architecture. Fifteen lines of forward pass, no switches,
    no cache, no sampling - so that the shape of a language model is readable
    before any of the machinery around it is.

    What it fixes, that `LLM` below lets you choose:

        RoPE            position arrives inside each attention layer
        a dense FFN     one MLP per block, not a routed mixture
        no tying        W_u is its own matrix
        no cache        every forward pass recomputes the whole prefix

    Every one of those is a real decision and every one is priced somewhere in
    run_llm.py. None of them changes what the model IS, which is why they are
    not here.

    The loss is E24, cross-entropy of the logits at position t against the
    token at t+1 - and because of the causal mask, one forward pass over T
    tokens supplies T of those, not one.
    """

    def __init__(self, cfg: LLMConfig):
        super().__init__()
        assert cfg.attn == "rope" and cfg.ffn == "dense", (
            "TinyLLM is deliberately the fixed version: RoPE and a dense FFN. "
            "For attn='vanilla' or ffn='moe', use LLM.")
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)      # E1
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        cos, sin = rope_tables(cfg.d_head, cfg.max_seq)   # buffers, not weights
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, idx, targets=None):
        """idx: (B, T) token ids -> logits (B, T, V), and the loss if asked."""
        B, T = idx.shape
        x = self.embed(idx)                                   # (B, T, d)
        cos, sin = self.cos[:T], self.sin[:T]                 # rows 0 .. T-1

        for blk in self.blocks:
            x, _ = blk(x, cos, sin)                           # E14, E15

        logits = self.lm_head(self.norm(x))                   # (B, T, V)  E32
        if targets is None:
            return logits
        return logits, F.cross_entropy(                       # E24
            logits.reshape(-1, self.cfg.vocab_size), targets.reshape(-1))


class LLM(nn.Module):
    """The configurable model: both attention layers, both FFN kinds, a KV
    cache and sampling.

    `TinyLLM` above is this class with every option removed. Read that one
    first; this one is the same twelve lines plus four things a real model
    needs and a first reading does not:

        attn="rope" | "vanilla"   which attention layer, and so where position
                                  enters - inside every block, or once at the
                                  input via `sinusoidal_pe`
        ffn="dense" | "moe"       the slot this whole repo is about
        tie_embeddings            W_u = E^T                              E32
        caches=, pos=, generate   incremental decoding, and the mask offset
                                  that makes one line correct in training AND
                                  generation

    E1 (embed) -> E14/E15 x L -> E32 (final norm and W_u) -> E33 (sampling).

    forward(idx, targets) returns (logits, loss). The loss is E24 for a dense
    model, and E29 for an MoE one:

        L = L_CE + sum_layers alpha * N * sum_i f_i P_i

    summed over MoE layers, exactly as E29 writes it - the balancing term has to
    be *in* the objective or the router has no reason to spread anything. (The
    z-loss of E28 is not here; it is a bf16 conditioning fix and this model
    trains in fp32.)
    """

    def __init__(self, cfg: LLMConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            # One matrix, two jobs: row v reads token v in, and scores token v
            # out. Not just a saving - it forces "close in embedding space" and
            # "predicted together" to be the same geometry.
            self.lm_head.weight = self.embed.weight           # W_u = E^T   E32

        cos, sin = rope_tables(cfg.d_head, cfg.max_seq)
        self.register_buffer("cos", cos, persistent=False)    # (max_seq, d_h/2)
        self.register_buffer("sin", sin, persistent=False)

        # attn="vanilla" has no rotation, so position has to enter at the input
        # instead - once, added to the embedding, exactly as in the 2017 paper.
        # The cos/sin buffers above are then built and never read.
        self.abs_pe = cfg.attn == "vanilla"
        if self.abs_pe:
            self.register_buffer("pe", sinusoidal_pe(cfg.max_seq, cfg.d_model),
                                 persistent=False)            # (max_seq, d)

        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx, targets=None, caches=None, pos=0):
        """idx: (B, T) token ids -> logits (B, T, V), loss or None.

        pos: index of the first token in idx, so a cached step rotates by the
        right absolute position instead of restarting at 0.
        """
        B, T = idx.shape
        x = self.embed(idx)                                   # (B, T, d)
        if self.abs_pe:
            x = x + self.pe[pos:pos + T]                      # 2017: add it once
        cos, sin = self.cos[pos:pos + T], self.sin[pos:pos + T]

        auxes = []
        for i, blk in enumerate(self.blocks):
            cache = None if caches is None else caches[i]
            x, aux = blk(x, cos, sin, cache)
            if aux is not None:
                auxes.append(aux)

        x = self.norm(x)
        logits = self.lm_head(x)                              # (B, T, V)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, self.cfg.vocab_size),
                                   targets.reshape(-1))
            if auxes:
                # E29 SUMS over MoE layers, it does not average - so a deeper
                # model applies more total balancing pressure, which is the
                # convention every implementation follows.
                loss = loss + self.cfg.aux_weight * torch.stack(auxes).sum()
        self.last_aux = torch.stack(auxes).sum() if auxes else None
        return logits, loss

    # ------------------------------------------------------------ generation
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None,
                 use_cache=True, generator=None):
        """Autoregressive sampling. (B, T) -> (B, T + max_new_tokens).

        With use_cache=False every step re-runs the whole prefix; with the cache
        it runs one token and reads the rest. The two must produce identical
        tokens from the same seed - which the dissector checks, because a cache
        bug is invisible until quality quietly drops.
        """
        total = idx.shape[1] + max_new_tokens
        table = "positional-encoding table" if self.abs_pe else "RoPE table"
        assert total <= self.cfg.max_seq, (
            f"generating {max_new_tokens} tokens from a {idx.shape[1]}-token prompt "
            f"needs {total} positions, but the {table} only has "
            f"{self.cfg.max_seq}. Nothing else in the model is length-limited - "
            f"raise max_seq (and expect quality to fall off past the lengths it "
            f"was trained on).")
        caches = [{} for _ in self.blocks] if use_cache else None

        for step in range(max_new_tokens):
            if use_cache:
                # First pass: whole prompt. After: only the newest token.
                inp = idx if step == 0 else idx[:, -1:]
                pos = 0 if step == 0 else idx.shape[1] - 1
            else:
                inp, pos = idx, 0                             # recompute everything

            logits, _ = self(inp, caches=caches, pos=pos)
            logits = logits[:, -1] / temperature               # (B, V) last position only

            if top_k is not None:
                kth = logits.topk(min(top_k, logits.shape[-1]), dim=-1).values[:, -1:]
                logits = logits.masked_fill(logits < kth, float("-inf"))

            probs = logits.softmax(dim=-1)
            nxt = torch.multinomial(probs, 1, generator=generator)   # (B, 1)
            idx = torch.cat([idx, nxt], dim=1)
        return idx

    # ------------------------------------------------------------ accounting
    def n_params(self, non_embedding=False):
        """P_tot of E36."""
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.embed.weight.numel()
        return n

    def macs_per_token(self, seq_len=None):
        """MACs for ONE token, split into the parts that scale differently.

        This is E38 in MACs rather than FLOPs (E38 counts 2 per MAC):

            C_fwd ~ 2 P_act + 4 L T d      ->      P_act + 2 L T d  MACs

        and the split below is exactly those two terms. The attention
        projections (E34) and the slot (E35) cost the same whatever the context
        length; the score/value matmuls are the 2LTd that grows with it. That
        split is why long-context serving and short-context training are
        different cost problems.
        """
        c = self.cfg
        T = c.max_seq if seq_len is None else seq_len
        qkv = c.d_model * (c.n_heads + 2 * c.n_kv_heads) * c.d_head     # E34
        out = c.n_heads * c.d_head * c.d_model
        scores = 2 * c.n_heads * c.d_head * T                # q.k and p.v
        if c.ffn == "moe":                                   # E35, SwiGLU: 3 matrices
            ffn = c.d_model * c.n_experts + c.k * 3 * c.d_model * c.d_ff
        else:
            ffn = 3 * c.d_model * c.d_ff
        per_layer = qkv + out + scores + ffn
        return {"attn_proj": (qkv + out) * c.n_layers,
                "attn_scores": scores * c.n_layers,
                "ffn": ffn * c.n_layers,
                "head": c.d_model * c.vocab_size,
                "total": per_layer * c.n_layers + c.d_model * c.vocab_size}

    def active_params(self):
        """P_act of E37 - the parameters a single token actually touches.

        E37 writes it as L(P_attn + (k/N) P_slot) + 2Vd; the same thing is said
        here by subtracting the experts a token skips. For ffn="dense" it equals
        P_tot, and P_tot/P_act - the sparsity ratio - is the single number that
        characterises an MoE.
        """
        c = self.cfg
        n = self.n_params()
        if c.ffn == "moe":
            per_expert = sum(p.numel() for p in self.blocks[0].ffn.experts[0].parameters())
            n -= c.n_layers * (c.n_experts - c.k) * per_expert
        return n

    def expert_loads(self, idx):
        """(n_layers, N) rows routed per expert, per layer. ffn="moe" only."""
        assert self.cfg.ffn == "moe", "no experts to count in a dense model"
        loads, x = [], self.embed(idx)
        T = idx.shape[1]
        cos, sin = self.cos[:T], self.sin[:T]
        for blk in self.blocks:
            x = x + blk.attn(blk.norm1(x), cos, sin)
            h = blk.norm2(x)
            loads.append(blk.ffn.load_per_expert(h))
            out, _ = blk.ffn(h)
            x = x + out
        return torch.stack(loads)


def build(ffn="dense", **kw):
    """Shorthand used by the dissector: build(ffn="moe", n_experts=8, k=2)."""
    return LLM(LLMConfig(ffn=ffn, **kw))
