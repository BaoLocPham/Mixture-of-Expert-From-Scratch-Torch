"""
Build the transformer LM yourself.

    python LLM/from_scratch/check.py        # run this constantly

Six stages, in dependency order: norm, position, attention, block, model,
generation. Each names the equation it implements, using the numbering from
the notes ("The equations - with and without MoE", E1-E43), so a stub can be
read next to the line it comes from. Each TODO says what the thing must do and why it matters, never how
to write it. `../common.py` is the finished version - opening it before you are
done costs you the exercise.

The grader and the dissector both reach into these modules by attribute name,
so keep them: `RMSNorm.weight`; attention's `q_proj/k_proj/v_proj/o_proj`;
`Block.norm1/attn/norm2/ffn`; the model's `embed/blocks/norm/lm_head/cos/sin`.

The model class is called `LLM`, matching `../common.py`. That file also has a
`TinyLLM` - the same model with every option stripped out, as something to read
rather than something to build. What you build here is the full one.

The FFN and the MoE layer are imported ready-made below. You built those in
DenseMoe/from_scratch and SparseMoe/from_scratch; this exercise is the
transformer they sit inside.

`split_heads`, `merge_heads` and `scaled_dot_product` come in ready-made too.
You built those in attention.py, and they are the 2017 layer unchanged - so
stage 3 here is only what a modern layer adds on top of it.
"""

import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import (FeedForward, MoEFeedForward, switch_aux_loss,   # noqa: F401
                    split_heads, merge_heads, scaled_dot_product)


@dataclass
class LLMConfig:
    vocab_size: int = 64
    d_model: int = 64
    n_layers: int = 4
    n_heads: int = 4
    n_kv_heads: int = None
    d_ff: int = 128
    max_seq: int = 64
    ffn: str = "dense"
    n_experts: int = 4
    k: int = 2
    tie_embeddings: bool = True
    aux_weight: float = 0.01

    def __post_init__(self):
        if self.n_kv_heads is None:
            self.n_kv_heads = self.n_heads
        assert self.d_model % self.n_heads == 0
        assert self.n_heads % self.n_kv_heads == 0
        assert self.ffn in ("dense", "moe")

    @property
    def d_head(self):
        return self.d_model // self.n_heads


# ------------------------------------------------------------- stage 1: norm
class RMSNorm(nn.Module):
    """E4. Scale each row to unit root-mean-square, then apply a learned gain.

    TODO:
      __init__ - one learned parameter, one gain per channel, starting at 1.0 so
                 the layer begins as a pure normalisation.
      forward  - divide each row by its own RMS (over the LAST axis only), then
                 apply the gain. eps lives INSIDE the square root, not outside:
                 it is there to stop a zero row producing inf, and adding it
                 after the fact would not.

    Note what is NOT here: no mean subtraction. Rows keep their sign structure.
    """

    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.eps = eps
        # TODO: the learned gain
        self.weight = nn.Parameter(torch.ones(d_model))
    def forward(self, x):
        # TODO: (..., d) -> (..., d), each row at unit RMS times the gain
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt() # (..., 1)
        return x * rms * self.weight


# --------------------------------------------------------- stage 2: position
def rope_tables(d_head, max_seq, base=10000.0, device=None):
    """E7. Precompute the rotation angles: (cos, sin), both (max_seq, d_h/2).

    TODO:
      One frequency per PAIR of channels, hence d_h/2 of them. Pair i gets
      theta_i = base**(-2i/d_h), so i=0 rotates once per token and the last
      pair rotates almost not at all - the head ends up carrying a range of
      wavelengths rather than one. The angle for position t is t * frequency.

    Getting the frequencies backwards (fast at the end instead of the start)
    produces a model that trains fine and generalises badly. The grader checks
    the direction.
    """
    theta = base ** (-torch.arange(0, d_head, 2, device=device).float() / d_head) # (d_h/2,)
    m = torch.arange(max_seq, device=device).float()
    ang = torch.outer(m, theta)
    return ang.cos(), ang.sin()


def apply_rope(x, cos, sin):
    """E7. Rotate each channel pair of x by its position's angle.

    x: (B, H, T, d_h)   cos, sin: (T, d_h/2)   ->   (B, H, T, d_h)

    TODO:
      Read x as dh/2 two-dimensional vectors and rotate each one:
          (a, b) -> (a*cos - b*sin, a*sin + b*cos)
      The pairing must match rope_tables: channels (0,1), (2,3), ... and the
      result must come back interleaved in the same order, or the rotation is
      applied to pairs that were never a pair.

    Two properties the grader checks, both of which fall out of it being a
    rotation: lengths are unchanged, and E8 holds - after rotating q by m and k
    by n, the dot product depends only on n - m. The second one is the point.
    """
    x1, x2 = x[..., 0::2], x[..., 1::2]                     # (B, H, T, d_h/2) each
    c, s = cos[None, None], sin[None, None]                 # (1, 1, T, d_h/2)
    out = torch.stack([x1 * c - x2 * s,                     # the 2x2 rotation matrix
                       x1 * s + x2 * c], dim=-1)            # of E7, applied pairwise
    return out.flatten(-2)                                  # interleave back to (…, d_h)

# -------------------------------------------------------- stage 3: attention
class CausalSelfAttention(nn.Module):
    """E6, E9-E11 - the modern layer: grouped query heads, RoPE, a KV cache.

    Three of the six steps arrive ready-made, imported at the top of this file:
    `split_heads`, `scaled_dot_product` (the scale, the mask, the softmax and
    the weighted sum, all four) and `merge_heads`. You built those in
    attention.py, and they are unchanged here - E9 and E10 have not moved since
    2017. What is left is exactly what a 2024 layer adds on top.

    TODO __init__:
      Four bias-free Linears: q, k, v, o. q and o are full width
      (n_heads * d_head); k and v are NARROWER when n_kv_heads < n_heads - that
      asymmetry is the whole of GQA. Keep n_rep = n_heads // n_kv_heads.

    TODO forward(x, cos, sin, cache=None):  (B, T, d) -> (B, T, d)
      1. project and split. `split_heads(self.q_proj(x), self.n_heads)`, and
         the same for k and v with self.n_kv_heads. Two different head counts
         on purpose - that is the line GQA changes.
      2. rotate q and k with apply_rope. NOT v: position belongs on the
         address, not on the payload being retrieved.
      3. if `cache` is given, concatenate the stored k/v in FRONT of the new
         ones along the token axis, and store the result back.
      4. make each kv head serve its group of q heads. Query head h must read
         kv head floor(h / n_rep), so the expansion has to keep each group
         CONTIGUOUS - one of torch's two repeat functions does that and the
         other silently does not.
      5. `scaled_dot_product(q, k, v, causal=True)`, then `merge_heads`, then
         o_proj.

    Steps 2, 3 and 4 are the entire diff against the 2017 layer; step 1 differs
    by one argument. Everything else is the same code, which is the point.

    Order matters twice in the middle, and neither mistake raises:
      - step 2 goes BEFORE step 3, so what lands in the cache is already
        rotated and the cached path needs no position bookkeeping at all.
      - step 4 goes AFTER step 3, so the cache holds n_kv heads and only the
        transient copy is full width. Do it before and you cache n_h heads,
        which throws away the entire reason GQA exists.
    """

    def __init__(self, cfg: LLMConfig):
        super().__init__()
        raise NotImplementedError("stage 3: CausalSelfAttention.__init__")

    def forward(self, x, cos, sin, cache=None):
        raise NotImplementedError("stage 3: CausalSelfAttention.forward")


# ------------------------------------------------------------ stage 4: block
class Block(nn.Module):
    """E14 and E15: attention sublayer, then FFN sublayer.

        u = x + Attn(Norm(x)),      x' = u + Slot(Norm(u))

    TODO __init__:
      Two RMSNorms, one attention, and one FFN - MoEFeedForward when
      cfg.ffn == "moe", otherwise FeedForward. Keep a flag; the MoE one returns
      a second value and the dense one does not.

    TODO forward(x, cos, sin, cache=None) -> (out, aux_or_None):
      Pre-norm, twice: normalise the input to each sublayer, add the sublayer's
      output to the UNNORMALISED stream. The residual path must stay clean from
      the embedding to the final norm - that is what lets gradients reach layer
      0 without passing through L normalisations, and it is why this arrangement
      trains at depth without a warmup schedule.

    Return the MoE aux loss (or None) so the model can add it to the objective.
    A balancing loss that is computed and dropped balances nothing.
    """

    def __init__(self, cfg: LLMConfig):
        super().__init__()
        raise NotImplementedError("stage 4: Block.__init__")

    def forward(self, x, cos, sin, cache=None):
        raise NotImplementedError("stage 4: Block.forward")


# ------------------------------------------------------------ stage 5: model
class LLM(nn.Module):
    """E1 -> E14/E15 x L -> E32. Embedding, blocks, final norm, logits.

    TODO __init__:
      An embedding table, n_layers blocks, a final RMSNorm, and a bias-free
      Linear to vocab_size. When cfg.tie_embeddings, the head and the embedding
      must be the SAME tensor - not a copy, or they drift apart after the first
      optimiser step. Register the cos/sin tables as buffers (persistent=False):
      they belong to the module and move with .to(device), but they are not
      parameters and must never receive a gradient.

    TODO forward(idx, targets=None, caches=None, pos=0) -> (logits, loss):
      idx is (B, T) integers. Slice the rotation tables from `pos`, not from 0 -
      a cached step passes the true position of its single token, and starting
      at 0 every time is a bug that only shows up during generation.
      With targets: E24, cross-entropy over the flattened batch - plus, for an
      MoE model, E29: alpha (cfg.aux_weight) times the aux losses SUMMED over
      MoE layers. Summed, not averaged; that is the convention E29 writes.

    Note the shift is NOT done here. idx and targets arrive already aligned so
    that position t predicts targets[t]; who does the shifting is a decision
    about the data, not the model.
    """

    def __init__(self, cfg: LLMConfig):
        super().__init__()
        self.cfg = cfg
        raise NotImplementedError("stage 5: LLM.__init__")

    def forward(self, idx, targets=None, caches=None, pos=0):
        raise NotImplementedError("stage 5: LLM.forward")

    # ------------------------------------------------------- stage 6: sampling
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None,
                 use_cache=True, generator=None):
        """(B, T) -> (B, T + max_new_tokens), sampled one token at a time.

        TODO (E33):
          Each step: run the model, take the logits at the LAST position only,
          divide by the temperature tau, optionally keep only the top_k entries (set the
          rest to -inf so they survive the softmax as exact zeros), sample from
          the softmax with torch.multinomial, append.

          With use_cache: the first pass sees the whole prompt at pos=0; every
          later pass sees ONE token, at pos = (length so far - 1), and reads the
          rest from the cache. Without: re-run the whole sequence each time.

        The two paths must produce identical tokens from the same generator
        seed. That equality is the only test that catches a cache bug, because
        a wrong cache does not crash - it just degrades the text.
        """
        raise NotImplementedError("stage 6: generate")

    # -------------------------------------------------- provided: accounting
    def n_params(self, non_embedding=False):
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.embed.weight.numel()
        return n

    def macs_per_token(self, seq_len=None):
        c = self.cfg
        T = c.max_seq if seq_len is None else seq_len
        qkv = c.d_model * (c.n_heads + 2 * c.n_kv_heads) * c.d_head
        out = c.n_heads * c.d_head * c.d_model
        scores = 2 * c.n_heads * c.d_head * T
        ffn = (c.d_model * c.n_experts + c.k * 2 * c.d_model * c.d_ff
               if c.ffn == "moe" else 2 * c.d_model * c.d_ff)
        return {"attn_proj": (qkv + out) * c.n_layers,
                "attn_scores": scores * c.n_layers,
                "ffn": ffn * c.n_layers,
                "head": c.d_model * c.vocab_size,
                "total": (qkv + out + scores + ffn) * c.n_layers + c.d_model * c.vocab_size}

    def active_params(self):
        c = self.cfg
        n = self.n_params()
        if c.ffn == "moe":
            per_expert = sum(p.numel() for p in self.blocks[0].ffn.experts[0].parameters())
            n -= c.n_layers * (c.n_experts - c.k) * per_expert
        return n

    def expert_loads(self, idx):
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
    return LLM(LLMConfig(ffn=ffn, **kw))
