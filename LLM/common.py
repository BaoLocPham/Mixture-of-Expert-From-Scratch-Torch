"""
Reference implementation: a decoder-only transformer LM, small enough to read
in one sitting, built from the same parts every modern LLM is built from.

    embed -> [ RMSNorm -> attention -> + ] [ RMSNorm -> FFN -> + ] x L -> RMSNorm -> logits

The FFN slot takes either a single feed-forward network (`ffn="dense"`) or a
top-k mixture of them (`ffn="moe"`). Nothing else in the model changes. That is
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
    alpha         aux-loss coefficient            E27       -> aux_weight
    P_tot, P_act  total, active params per token  E36, E37

Row-vector convention throughout (x is (..., d) and the maths reads xW), which
is what nn.Linear already does.
"""

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------------- config
@dataclass
class LLMConfig:
    """Every shape in the model, in one place.

    The only field that changes the architecture is `ffn`. Everything else
    changes sizes.
    """
    vocab_size: int = 64
    d_model: int = 64
    n_layers: int = 4
    n_heads: int = 4
    n_kv_heads: int = None      # None -> = n_heads (plain MHA). Fewer -> GQA.
    d_ff: int = 128
    max_seq: int = 64
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

        theta_i = beta ** (-2i / d_h)          beta = base = 10^4

    and the angle applied at position m is just m * theta_i - the rate times how
    far along the sequence the token sits. `arange(0, d_h, 2)` IS the sequence
    2i = 0, 2, 4, ..., so the exponent below is exactly -2i/d_h from E7.

    The rates are geometric, from 1 down to 1/beta. For d_h = 8:

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
    #    W_K to match. What is NOT valid is mixing the two: rotate q one way
    #    and k the other and E8 quietly stops holding, with no error anywhere.
    out = torch.stack([rot1, rot2], dim=-1)                     # (…, d_h/2, 2)
    return out.flatten(-2)                                      # (…, d_h)


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
        self.n_rep = cfg.n_heads // cfg.n_kv_heads          # q heads per kv head

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
        self.attn = CausalSelfAttention(cfg)
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
    """The whole model. Same class for both FFN kinds.

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
        assert total <= self.cfg.max_seq, (
            f"generating {max_new_tokens} tokens from a {idx.shape[1]}-token prompt "
            f"needs {total} positions, but the RoPE table only has "
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
    return TinyLLM(LLMConfig(ffn=ffn, **kw))
