"""
Run + dissect the transformer LM from common.py.

    python LLM/run_llm.py                  # the reference implementation
    LLM_IMPL=scratch python LLM/run_llm.py # your from_scratch/ version instead

Symbols and equation numbers follow the notes ("The equations - with and
without MoE", E1-E43): d, d_h, d_ff, N, k, T, L; m and n for query and key
position; P_tot and P_act for total and active parameters per token.

Eight sections. The first five run in a couple of seconds; section 6 trains
three small models and takes a minute or two, because a claim about whether MoE
helps is worthless without a loss curve behind it.

  1. flow + dims      - every tensor from token id to logit      (E1-E15, E32)
  2. attention        - causality, the KV cache, GQA              (E9-E11, E40)
  2b. vanilla         - the same layer as the 2017 paper wrote it (E9-E11)
  3. rope             - position as rotation, past the table      (E7, E8)
  4. where params are - the budget, and how it moves               (E34-E36)
  5. dense vs moe     - the same model with one word changed      (E18 vs E21, E37)
  6. training         - both, at matched compute                  (E24, E29)
  7. expert load      - who gets the tokens, with and without alpha (E26, E27)
  8. generation       - sampling, and the cache                   (E33)
"""

import os
import sys
import time
from pathlib import Path

from dataclasses import replace

import torch
import torch.nn.functional as F

_here = Path(__file__).resolve().parent
SCRATCH = os.environ.get("LLM_IMPL", "common") in ("scratch", "from_scratch")
if SCRATCH:
    sys.path.insert(0, str(_here / "from_scratch"))
    from llm import LLMConfig, TinyLLM, rope_tables, apply_rope, build
else:
    sys.path.insert(0, str(_here))
    from common import (LLMConfig, TinyLLM, rope_tables, apply_rope, build,
                        CausalSelfAttention, VanillaSelfAttention, sinusoidal_pe)

torch.manual_seed(0)
torch.set_printoptions(precision=4, sci_mode=False)

# These models are far too small to feed a big machine: with the default thread
# count the training section spent most of its time in thread contention (120s
# vs 5s for identical work on a 256-core box). Small models are latency-bound.
torch.set_num_threads(min(4, torch.get_num_threads()))


def hdr(s):
    print(f"\n{'=' * 72}\n{s}\n{'=' * 72}")


B, T = 2, 12
cfg = LLMConfig(vocab_size=32, d_model=64, n_layers=3, n_heads=4, n_kv_heads=2,
                d_ff=128, max_seq=32)
model = TinyLLM(cfg)
ids = torch.randint(0, cfg.vocab_size, (B, T))


# ------------------------------------------------------------ 1. flow + dims
hdr("1. flow and dims: token id to logit")

print(f"B={B}  T={T}  vocab={cfg.vocab_size}  d_model={cfg.d_model}  "
      f"heads={cfg.n_heads} (kv {cfg.n_kv_heads})  d_head={cfg.d_head}  "
      f"d_ff={cfg.d_ff}  layers={cfg.n_layers}\n")

blk = model.blocks[0]
x = model.embed(ids)
cos, sin = model.cos[:T], model.sin[:T]
h1 = blk.norm1(x)
a = blk.attn(h1, cos, sin)
h = x + a
f = blk.ffn(blk.norm2(h))
logits, _ = model(ids)

rows = [
    ("idx", ids.shape, "token ids, integers - the only discrete thing in the model"),
    ("embed(idx)", x.shape, "one row of the table per id"),
    ("norm1(x)", h1.shape, "rmsnorm, per token, over d"),
    ("attn(...)", a.shape, "heads split, rotated, masked, merged, projected"),
    ("x + attn", h.shape, "residual: the branch ADDS to the raw stream"),
    ("ffn(norm2)", f.shape, "position-wise: no token sees another here"),
    ("norm + head", logits.shape, "one score per (position, vocab entry)"),
]
for name, shape, why in rows:
    print(f"  {name:<12} {str(tuple(shape)):<14} {why}")

print("\nTwo different jobs, alternating: attention MIXES ACROSS POSITIONS and")
print("does no per-token computation worth the name (it is four linear maps and")
print("an average); the FFN does all the per-token computation and cannot see")
print("another token. Every parameter in the model is in one camp or the other.")
print(f"\nlogits[b, t] scores the token that should come AFTER position t, so one")
print(f"forward pass over {T} tokens supplies {T} training signals, not 1.")


# ------------------------------------------------------------- 2. attention
hdr("2. attention: causality, the cache, grouped query attention  (E9-E11, E40)")

model.eval()
base, _ = model(ids)
probe = ids.clone()
probe[:, 5] = (probe[:, 5] + 7) % cfg.vocab_size
alt, _ = model(probe)
delta = (alt - base).abs().amax(dim=-1)[0]
print("changed the token at position 5; per-position max logit change:")
print("  " + "  ".join(f"{t}:{v:.3f}" for t, v in enumerate(delta.tolist())))
assert delta[:5].max() == 0, "positions before the edit must be untouched"
print("\nExactly zero before position 5. Nothing after the mask can leak backwards,")
print("which is the property the whole training scheme rests on.")

# the cache: run the same sequence one token at a time
caches = [{} for _ in model.blocks]
inc = torch.cat([model(ids[:, t:t + 1], caches=caches, pos=t)[0] for t in range(T)], 1)
print(f"\nincremental (cached, one token at a time) vs one full pass:")
print(f"  max |diff| = {(inc - base).abs().max():.2e}")
print("The cache is not an approximation. k and v for a past token never change,")
print("so recomputing them is pure waste - but the *mask* has to know that the")
print("single query row sits at the END of the keys, which is the offset bug")
print("everyone writes once.")

kv_full = 2 * cfg.n_layers * cfg.n_heads * cfg.d_head
kv_gqa = 2 * cfg.n_layers * cfg.n_kv_heads * cfg.d_head
print(f"\nKV cache, floats per token: MHA would be {kv_full}, this GQA model keeps "
      f"{kv_gqa}")
print(f"  {cfg.n_heads} query heads share {cfg.n_kv_heads} kv heads "
      f"({cfg.n_heads // cfg.n_kv_heads} q heads per kv head)")
print(f"  at 8k context, batch 32, fp16: "
      f"{kv_full * 8192 * 32 * 2 / 1e9:.2f} GB -> {kv_gqa * 8192 * 32 * 2 / 1e9:.2f} GB")
print("The query side is untouched, so the model keeps all its attention")
print("patterns. What shrinks is the thing that runs out of memory at serving.")


# --------------------------------------------------------------- 2b. vanilla
hdr("2b. the same layer, 2017 edition  (attn=\"vanilla\")")

if SCRATCH:
    print("skipped: attn=\"vanilla\" is reference-only, and LLM_IMPL=scratch is")
    print("running your llm.py, which implements the RoPE + GQA layer alone.")

if not SCRATCH:
    van_cfg = replace(cfg, attn="vanilla", n_kv_heads=cfg.n_heads)
    van = TinyLLM(van_cfg).eval()
    van_base, _ = van(ids)

    print("Attention Is All You Need, unchanged: sinusoidal position ADDED to the")
    print("embedding once, every query head with its own k and v, biases on all four")
    print("projections. Same E9-E11 in the middle - the scaling, the mask, the")
    print("softmax, the concat - so the only structural difference is GQA.\n")

    p_rope = sum(p_.numel() for p_ in model.parameters())
    p_van = sum(p_.numel() for p_ in van.parameters())
    n_bias = 4 * cfg.d_model * cfg.n_layers
    n_kv_extra = 2 * cfg.d_model * (cfg.n_heads - cfg.n_kv_heads) * cfg.d_head * cfg.n_layers
    print(f"  parameters      rope {p_rope:,}     vanilla {p_van:,}   "
          f"(+{p_van - p_rope:,})")
    print(f"    of which  wider k and v {n_kv_extra:,}   biases {n_bias:,}"
          f"   {'(they agree)' if n_kv_extra + n_bias == p_van - p_rope else '(MISMATCH)'}")

    kv_rope = 2 * cfg.n_layers * cfg.n_kv_heads * cfg.d_head
    kv_van = 2 * cfg.n_layers * cfg.n_heads * cfg.d_head
    print(f"  KV floats/token rope {kv_rope:<9,} vanilla {kv_van:<9,} "
          f"({kv_van / kv_rope:.0f}x - the whole of GQA, and the reason it exists)")
    print(f"  position table  rope cos/sin {tuple(model.cos.shape)} x2, read every "
          f"layer")
    print(f"                  vanilla pe {tuple(van.pe.shape)}, read once at the input")

    # causality and the cache hold for the 2017 layer too - same mask, same offset
    probe2 = ids.clone()
    probe2[:, 5] = (probe2[:, 5] + 7) % cfg.vocab_size
    van_delta = (van(probe2)[0] - van_base).abs().amax(dim=-1)[0]
    assert van_delta[:5].max() == 0
    vc = [{} for _ in van.blocks]
    van_inc = torch.cat([van(ids[:, t:t + 1], caches=vc, pos=t)[0] for t in range(T)], 1)
    print(f"\n  causality: zero change before position 5      (max "
          f"{van_delta[:5].max():.1e})")
    print(f"  cached vs full pass: max |diff| {(van_inc - van_base).abs().max():.2e}")
    print("Both properties are the mask's, not RoPE's - which is why they survive")
    print("the swap untouched.")

    # where position actually lives: shuffle a prefix and see what survives.
    # One layer, not the stack - at depth 2 a shuffled position's own output feeds
    # the later ones, so the invariance below is a statement about the LAYER.
    xv = torch.randn(1, T, cfg.d_model) * 0.5
    perm = torch.tensor([2, 0, 3, 1] + list(range(4, T)))
    pe1 = sinusoidal_pe(cfg.max_seq, cfg.d_model)[:T]
    van_l = VanillaSelfAttention(van_cfg).eval()
    rope_l = CausalSelfAttention(cfg).eval()
    with torch.no_grad():
        def moved(fn, inp):
            return (fn(inp)[:, 4:] - fn(inp[:, perm])[:, 4:]).abs().max()
        m_none = moved(lambda t: van_l(t), xv)
        m_pe = moved(lambda t: van_l(t + pe1), xv)
        m_rope = moved(lambda t: rope_l(t, model.cos[:T], model.sin[:T]), xv)

    print("\npermute the first 4 tokens and read the outputs at positions 4..11:")
    print(f"  vanilla layer, no position signal   {m_none:.2e}   <- unchanged")
    print(f"  vanilla layer, + sinusoidal PE      {m_pe:.2e}")
    print(f"  this layer, RoPE inside             {m_rope:.2e}")
    print("The first line is what 'attention is permutation-equivariant' means,")
    print("measured: with no positional signal the output at position t is a")
    print("function of the SET of tokens before it, not their order. The causal")
    print("mask alone does not fix that - it decides who may look at whom, never")
    print("how far apart they are. The 2017 encoding and RoPE are two answers, and")
    print("the six orders of magnitude between line 1 and lines 2-3 is the whole")
    print("contribution of both.")


# ------------------------------------------------------------------ 3. rope
hdr("3. rope: the dot product only sees n - m  (E8)")

cosf, sinf = rope_tables(cfg.d_head, 64)
q, k = torch.randn(1, 1, 1, cfg.d_head), torch.randn(1, 1, 1, cfg.d_head)


def rdot(m, n):
    return (apply_rope(q, cosf[m:m + 1], sinf[m:m + 1]) *
            apply_rope(k, cosf[n:n + 1], sinf[n:n + 1])).sum().item()


print("one fixed (q, k) pair, dotted at pairs of positions:")
print(f"  {'m':>3} {'n':>3} {'n-m':>4}   q.k")
for m, n in ((3, 0), (13, 10), (43, 40), (9, 0), (29, 20)):
    print(f"  {m:>3} {n:>3} {n - m:>4}   {rdot(m, n):+.6f}")
same = [rdot(m, n) for m, n in ((3, 0), (13, 10), (43, 40))]
print(f"\nthe three pairs at n - m = -3 agree to {max(same) - min(same):.2e}")
print("Absolute position is applied to each vector and cancels in the product.")
print("That is why RoPE needs no learned position table and no extra parameters:")
print(f"  position parameters in this model: 0 "
      f"(cos/sin are buffers, {model.cos.numel() + model.sin.numel()} floats, not trained)")

print(f"\nThe table is built to max_seq={cfg.max_seq}. Past that there is no row:")
try:
    model(torch.zeros(1, cfg.max_seq + 1, dtype=torch.long))
except Exception as ex:
    print(f"  running {cfg.max_seq + 1} tokens -> {type(ex).__name__}: "
          f"{str(ex).splitlines()[0][:60]}")
print("Nothing in the architecture is length-limited - the mask, the residual")
print("and the FFN never mention T. Context length is a decision about this")
print("table and about what lengths the model was trained to expect.")


# --------------------------------------------------------- 4. where params are
hdr("4. where the parameters actually are  (E34-E36)")


def budget(m):
    c = m.cfg
    emb = m.embed.weight.numel()
    attn = sum(p.numel() for b in m.blocks for p in b.attn.parameters())
    ffn = sum(p.numel() for b in m.blocks for p in b.ffn.parameters())
    norms = m.n_params() - emb - attn - ffn
    return {"embed (tied, counted once)": emb, "attention": attn,
            "ffn": ffn, "norms": norms}


tot = model.n_params()
for name, n in budget(model).items():
    print(f"  {name:<28} {n:>9,}  {100 * n / tot:>5.1f}%")
print(f"  {'total':<28} {tot:>9,}")

# E34: P_attn = d(n_h d_h) + 2 d(n_kv d_h) + (n_h d_h) d
# E35: P_slot = 3 d d_ff (dense, SwiGLU)
# E36: P_tot  = L(P_attn + P_slot) + Vd   (tied, so the head is free)
c = cfg
p_attn = c.d_model * c.n_heads * c.d_head + 2 * c.d_model * c.n_kv_heads * c.d_head \
    + c.n_heads * c.d_head * c.d_model
p_slot = 3 * c.d_model * c.d_ff
derived = c.n_layers * (p_attn + p_slot) + c.vocab_size * c.d_model
measured = tot - sum(p.numel() for p in model.parameters() if p.dim() == 1)  # minus norms
print(f"\nE34-E36 derived {derived:,} vs measured {measured:,} (norms excluded): "
      f"{'agree' if derived == measured else 'DISAGREE'}")
assert derived == measured, "the closed form and the module disagree"

print("\nWith a toy 32-entry vocabulary the embedding is a rounding error and the")
print("weight sits in attention and the FFN. With a real vocabulary it is the")
print("other way round at small d - the embedding is the one matrix that scales")
print("with vocabulary instead of with depth, so it dominates until the model")
print("is big enough to outgrow it. Same code, vocab=32,000:\n")
print(f"  {'d_model':>8} {'layers':>7} {'total':>12} {'non-embed':>12} {'embed %':>9}")
for d, L in ((64, 3), (256, 6), (512, 12), (1024, 24)):
    m = TinyLLM(LLMConfig(vocab_size=32000, d_model=d, n_layers=L, n_heads=8,
                          d_ff=4 * d, max_seq=64))
    print(f"  {d:>8} {L:>7} {m.n_params():>12,} {m.n_params(True):>12,} "
          f"{100 * m.embed.weight.numel() / m.n_params():>8.1f}%")


# ----------------------------------------------------------- 5. dense vs moe
hdr("5. the same model with one word changed  (E35, E37)")

kw = dict(vocab_size=32, d_model=64, n_layers=3, n_heads=4, d_ff=128, max_seq=32)
dense = build(ffn="dense", **kw)
print(f"  {'model':<26} {'P_tot':>10} {'P_act':>13} {'MACs/token':>12}")
print(f"  {'dense':<26} {dense.n_params():>10,} {dense.active_params():>13,} "
      f"{dense.macs_per_token(32)['total']:>12,}")
for N, k in ((4, 1), (4, 2), (8, 1), (8, 2), (64, 2)):
    m = build(ffn="moe", n_experts=N, k=k, **kw)
    print(f"  {f'moe  N={N:<3} k={k}':<26} {m.n_params():>10,} "
          f"{m.active_params():>13,} {m.macs_per_token(32)['total']:>12,}")

eq = build(ffn="moe", n_experts=8, k=1, **kw)
print(f"\nN=8, k=1 vs dense: {eq.n_params() / dense.n_params():.2f}x the parameters, "
      f"{eq.macs_per_token(32)['total'] / dense.macs_per_token(32)['total']:.2f}x the MACs.")
print(f"P_tot / P_act = {eq.n_params() / eq.active_params():.2f} - the sparsity ratio, "
      f"and 1.00 for every dense model ever built.")
print("That ratio - and not the loss curve - is the whole reason anyone builds")
print("these. Whether the extra parameters are USED is section 6.")

parts = dense.macs_per_token(32)
print(f"\nwhere the compute goes (dense, T=32): " +
      "  ".join(f"{k}={v:,}" for k, v in parts.items() if k != "total"))
long_parts = dense.macs_per_token(4096)
print(f"                        (dense, T=4096): " +
      "  ".join(f"{k}={v:,}" for k, v in long_parts.items() if k != "total"))
print("Only the score/value matmuls grew. MoE makes the FFN term cheap per")
print("parameter; it does nothing at all for the term that grows with context.")


# -------------------------------------------------------------- 6. training
hdr("6. training: a task that cannot be solved without attention")

# Copy task: [SEP] a1..aL [SEP] a1..aL. The second half is fully determined by
# the first, but only by looking back L+1 positions - an FFN alone cannot do it
# and neither can any n-gram model. The first half is uniform noise, so the
# reported loss has a floor and the informative number is the copied half alone.
VOCAB, SYM, L = 14, 12, 6
SEP = 12
SEQ = 2 * L + 2
SEEDS = (0, 1, 2)
STEPS = 300


def batch(n=32):
    a = torch.randint(0, SYM, (n, L))
    s = torch.full((n, 1), SEP)
    return torch.cat([s, a, s, a], dim=1)                    # (n, 2L+2)


def evaluate(m, n=512):
    b = batch(n)
    half = slice(L + 1, None)                                # the copied half
    with torch.no_grad():
        lg, loss = m(b[:, :-1], b[:, 1:])
        ce = F.cross_entropy(lg[:, half].reshape(-1, VOCAB),
                             b[:, 1:][:, half].reshape(-1))
        acc = (lg[:, half].argmax(-1) == b[:, 1:][:, half]).float().mean()
    return loss.item(), ce.item(), acc.item()


def train(m, steps=STEPS, lr=3e-3):
    opt = torch.optim.AdamW(m.parameters(), lr=lr)
    t0 = time.time()
    for _ in range(steps):
        b = batch()
        _, loss = m(b[:, :-1], b[:, 1:])
        opt.zero_grad()
        loss.backward()
        opt.step()
    return time.time() - t0


shared = dict(vocab_size=VOCAB, d_model=48, n_layers=3, n_heads=3, d_ff=96,
              max_seq=SEQ)
CONFIGS = (("dense", dict(ffn="dense")),
           ("moe N=8 k=1", dict(ffn="moe", n_experts=8, k=1)),
           ("moe N=8 k=1, aux off", dict(ffn="moe", n_experts=8, k=1, aux_weight=0.0)))

print(f"{STEPS} steps, batch 32, {len(SEEDS)} seeds per config, objective E29")
print("(cross-entropy plus alpha * the aux loss, summed over MoE layers). The MoE")
print("models are matched to dense on compute (k=1), not on parameters - that is")
print("the comparison MoE claims to win.\n")
print(f"  {'config':<22} {'P_tot':>8} {'P_act':>8} {'MACs/tok':>9}  "
      f"{'copy-half CE':>22}  {'copy acc':>8}")

results = {}
for label, kwargs in CONFIGS:
    runs = []
    for seed in SEEDS:
        torch.manual_seed(seed)                              # same init AND same batches
        m = build(**kwargs, **shared)
        secs = train(m)
        full, ce, acc = evaluate(m)
        runs.append(dict(model=m, seed=seed, loss=full, ce=ce, acc=acc, secs=secs))
    results[label] = runs
    m0 = runs[0]["model"]
    ces = [r["ce"] for r in runs]
    accs = [r["acc"] for r in runs]
    print(f"  {label:<22} {m0.n_params():>8,} {m0.active_params():>8,} "
          f"{m0.macs_per_token(SEQ)['total']:>9,}  "
          f"{min(ces):.4f} - {max(ces):.4f} (mean {sum(ces)/len(ces):.4f})  "
          f"{min(accs):>8.3f}")

mean = {label: sum(r["ce"] for r in runs) / len(runs) for label, runs in results.items()}
acc = {label: min(r["acc"] for r in runs) for label, runs in results.items()}

print("\nThe three configs isolate two separate effects, and the seeds agree on")
print("the ranking:")
print(f"  routing at all     dense {mean['dense']:.4f}  ->  moe alpha=0    "
      f"{mean['moe N=8 k=1, aux off']:.4f}")
print(f"  the balancing term moe alpha=0 {mean['moe N=8 k=1, aux off']:.4f}  ->  "
      f"moe alpha=0.01 {mean['moe N=8 k=1']:.4f}")
print(f"  worst-seed copy accuracy: dense {acc['dense']:.3f}, "
      f"moe alpha=0 {acc['moe N=8 k=1, aux off']:.3f}, "
      f"moe alpha=0.01 {acc['moe N=8 k=1']:.3f}")
d_ce = sorted(r["ce"] for r in results["dense"])
m_ce = sorted(r["ce"] for r in results["moe N=8 k=1"])
print(f"  per seed: dense {['%.4f' % v for v in d_ce]}   "
      f"moe {['%.4f' % v for v in m_ce]}")

print("\nSo on this task MoE LOSES, and both halves of the loss are worth naming.")
print("Routing costs something even at alpha=0: the choice is discrete, the")
print("router is learning at the same time as the experts, and at k=1 a token")
print("gets exactly one expert's worth of FFN - the same width dense already")
print("had, from a model that also has to learn which one to pick. The")
print("balancing term then costs")
print("more on top, because E29 adds a term the language-modelling objective did")
print("not ask for and the two pull in different directions. That tension is the")
print("entire motivation for aux-loss-free balancing (E30, DeepSeek-V3).")
print("\nWhat this is NOT is evidence against MoE. The model carries "
      f"{results['moe N=8 k=1'][0]['model'].n_params() / results['dense'][0]['model'].n_params():.1f}x")
print("the parameters at "
      f"{results['moe N=8 k=1'][0]['model'].macs_per_token(SEQ)['total'] / results['dense'][0]['model'].macs_per_token(SEQ)['total']:.2f}x the compute, and copying six symbols is an")
print("attention circuit - the FFN was never the bottleneck, so the extra")
print("capacity has nothing to do while the extra machinery still has to be")
print("learned. Capacity only pays when the data is what limits you, and a task")
print("this small cannot show that either way. What it does establish is that")
print("the substitution is drop-in: same optimiser, same steps, same shapes.")


# ----------------------------------------------------------- 7. expert load
hdr("7. who gets the tokens, and what alpha is worth  (E26, E27)")

probe = batch(64)[:, :-1]
for label in ("moe N=8 k=1", "moe N=8 k=1, aux off"):
    print(f"\n{label}:")
    ratios = []
    for r in results[label]:
        loads = r["model"].expert_loads(probe)                # (layers, N)
        tot = loads.sum(0).float()
        ratio = (tot.max() / tot.min().clamp(min=1)).item()
        ratios.append(ratio)
        if r["seed"] == SEEDS[0]:
            for i, row in enumerate(loads):
                share = 100 * row.float() / row.sum()
                print(f"  layer {i}  {[f'{v:>3}' for v in row.tolist()]}"
                      f"  share {[f'{v:.0f}%' for v in share.tolist()]}")
        print(f"  seed {r['seed']}: busiest/quietest expert = {ratio:6.2f}x, "
              f"experts unused in some layer = {(loads == 0).sum().item()}"
              f" of {loads.numel()}")
    print(f"  mean imbalance over seeds: {sum(ratios) / len(ratios):.2f}x")

print("\nThat is what alpha buys: a flatter distribution, and no expert sitting")
print("idle in a layer. Nothing in the cross-entropy objective wants balance -")
print("a router that piles tokens onto two experts predicts just as well, and an")
print("expert that receives no tokens receives no gradient either, so imbalance")
print("feeds itself.")
print("\nAnd section 6 priced it: the alpha=0.01 runs are the WORST of the three")
print("on copy-half CE. That is not a bug in the coefficient, it is the shape of")
print("the trade - E29 adds a term the language model never asked for, and it is")
print("paid for out of the same parameters. Which is exactly why DeepSeek-V3")
print("replaced it with a bias updated by a rule instead of a gradient (E30):")
print("balance, without a second objective competing for the weights.")


# ------------------------------------------------------------ 8. generation
hdr("8. generation")

best = results["dense"][0]["model"].eval()
prompt = batch(1)[:, :L + 2]                                 # [SEP] a1..aL [SEP]
g = torch.Generator().manual_seed(0)
cached = best.generate(prompt.clone(), L, temperature=0.5, use_cache=True, generator=g)
g = torch.Generator().manual_seed(0)
uncached = best.generate(prompt.clone(), L, temperature=0.5, use_cache=False, generator=g)

print(f"  prompt      {prompt[0].tolist()}")
print(f"  should emit {prompt[0, 1:L + 1].tolist()}")
print(f"  generated   {cached[0, L + 2:].tolist()}")
print(f"  cached and uncached generation identical: {torch.equal(cached, uncached)}")
print("\nThat equality is the only test that matters for a cache. A cache bug does")
print("not crash - it produces slightly wrong attention and slightly worse text,")
print("which looks exactly like a model that needed more training.")

# The speedup is a function of prefix length, so measure it somewhere it shows.
big = build(ffn="dense", vocab_size=64, d_model=128, n_layers=4, n_heads=4,
            d_ff=256, max_seq=320).eval()
long_prompt = torch.randint(0, 64, (1, 128))
t0 = time.time(); big.generate(long_prompt, 64, use_cache=False); slow = time.time() - t0
t0 = time.time(); big.generate(long_prompt, 64, use_cache=True); fast = time.time() - t0
print(f"\n  64 tokens after a 128-token prompt:")
print(f"    no cache {slow * 1000:>7.0f} ms   cache {fast * 1000:>7.0f} ms   "
      f"{slow / fast:.1f}x")
print("Without a cache, step i re-runs the whole prefix, so generating n tokens")
print("costs O(n^2) forward passes worth of work over a prefix that never")
print("changed. The cache is not an optimisation detail; it is the difference")
print("between generation being linear and quadratic in what you have written.")

try:
    best.generate(prompt.clone(), 999)
except AssertionError as ex:
    print(f"\n  asking for more positions than the table has:\n    {str(ex).splitlines()[0][:96]}...")
print()
