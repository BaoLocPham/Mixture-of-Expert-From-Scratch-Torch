"""
The whole LLM at a size you can check by hand.

    python LLM/steps_llm.py

4 tokens, vocab 6, d_model 4, 2 heads of width 2, one layer. Every intermediate
tensor is printed. Nothing here is a new mechanism - it is the same model as
common.py with the numbers small enough that you can follow a single value from
the token id all the way to its logit and back.

Symbols follow the notes ("The equations - with and without MoE", E1-E43):
d, d_h, d_ff, N, k, T; m and n for query and key position; theta_i for the
rotation rate of channel pair i. Each step names the equation it is walking
through, so you can read a printed row next to the line it comes from.

Ten steps:

  1. embed       - the lookup is a matmul with a one-hot row      (E1)
  2. rmsnorm     - by hand, then against the module               (E4)
  3. q, k, v     - one matmul, then split into heads              (E6)
  4. rope        - what the rotation does, and the (n-m) property (E7, E8)
  5. scores      - scale, mask, softmax; row 0 sees only itself   (E9, E10)
  6. attn out    - weighted sum of v, merge, project, residual    (E11, E14)
  7. ffn         - the dense branch, then the routed one          (E18, E19-E21)
  8. logits      - tied head = dot product with the embedding rows (E32)
  9. loss        - the shift by one, and cross-entropy by hand    (E24)
 10. real model  - the same numbers out of TinyLLM, dense vs MoE
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (LLMConfig, RMSNorm, CausalSelfAttention, FeedForward,
                    MoEFeedForward, TinyLLM, rope_tables, apply_rope, build)

torch.manual_seed(0)
torch.set_printoptions(precision=4, sci_mode=False)


def hdr(s):
    print(f"\n{'=' * 72}\n{s}\n{'=' * 72}")


def fmt(row):
    # `v if v else 0.0` normalises -0.0, which masked sums produce and nobody means
    return "[" + "  ".join(f"{(v if v else 0.0):7.4f}" for v in row.tolist()) + "]"


def show(t, label, mark=()):
    for i, row in enumerate(t):
        tag = "   <--" if i in mark else ""
        print(f"  {label}[{i}] = {fmt(row)}{tag}")


V, D, H, DH, D_FF, T = 6, 4, 2, 2, 6, 4
N_EXP, K = 3, 2
cfg = LLMConfig(vocab_size=V, d_model=D, n_layers=1, n_heads=H, d_ff=D_FF,
                max_seq=8, n_experts=N_EXP, k=K)

# Tokens: a tiny "sentence". Ids are arbitrary labels - the model has never
# seen them and has no idea 3 comes after 2.
ids = torch.tensor([[2, 0, 4, 1]])                              # (B=1, T=4)


# ------------------------------------------------------------- 1. embedding
hdr("1. embed: a lookup table IS a matmul  (E1)")

E = (torch.arange(V * D).float().reshape(V, D) * 0.1 - 1.0)     # (V, d) readable values
x = E[ids[0]]                                                   # (T, d)

print(f"token ids  {ids[0].tolist()}\n")
print("embedding table E, one row per vocab entry:")
show(E, "E")
print("\nx = E[ids] just picks rows 2, 0, 4, 1:")
show(x, "x")

onehot = F.one_hot(ids[0], V).float()                           # (T, V)
print(f"\nsame thing as one_hot(ids) @ E:  max diff = {(onehot @ E - x).abs().max():.2e}")
print("Worth seeing once: the embedding is a linear layer whose input happens")
print("to be a basis vector, which is why a gradient reaches exactly one row.")


# --------------------------------------------------------------- 2. rmsnorm
hdr("2. rmsnorm: scale each row to unit RMS, then a learned gain  (E4)")

rms = x.pow(2).mean(-1, keepdim=True).add(1e-6).rsqrt()         # (T, 1)
xn = x * rms                                                    # gain is all-ones at init
print("row  |  raw row                          rms      normalised")
for i in range(T):
    print(f"  {i}  | {fmt(x[i])}  {1/rms[i].item():7.4f}  {fmt(xn[i])}")
print(f"\nrms of every normalised row: {[round(v, 4) for v in xn.pow(2).mean(-1).sqrt().tolist()]}")

norm = RMSNorm(D)
print(f"vs RMSNorm module: max diff = {(norm(x) - xn).abs().max():.2e}")
print("\nNo mean subtraction anywhere. A row of all-positive numbers stays")
print("all-positive - LayerNorm would have centred it around zero.")


# ------------------------------------------------------------- 3. q, k, v
hdr("3. q, k, v and the head split  (E6)")

attn = CausalSelfAttention(cfg)
with torch.no_grad():                                           # small readable weights
    # a different pattern per projection - if q, k and v shared one, the
    # scores below would be self-dots and every lesson in them would be an artefact
    for j, lin in enumerate((attn.q_proj, attn.k_proj, attn.v_proj, attn.o_proj)):
        n = lin.weight.numel()
        ramp = torch.arange(n).float().reshape(lin.weight.shape)
        w = (ramp + 2 * j) % (5 + j) + 0.3 * torch.sin(ramp + j)
        lin.weight.copy_(w * 0.1 - 0.2 - 0.05 * j)

q_flat = attn.q_proj(xn)                                        # (T, H*dh)
print(f"q_proj: Linear({D} -> {H * DH}) - ALL heads in one matmul, then reshaped")
show(q_flat, "q_flat")

q = q_flat.view(T, H, DH).transpose(0, 1)                       # (H, T, dh)
k = attn.k_proj(xn).view(T, H, DH).transpose(0, 1)
v = attn.v_proj(xn).view(T, H, DH).transpose(0, 1)
for h in range(H):
    print(f"\nhead {h}:")
    show(q[h], f"  q{h}")
print("\nThe split is a view, not a computation. 'Heads' are a reading of one")
print("matrix: columns 0-1 are head 0, columns 2-3 are head 1. They never mix")
print("again until o_proj, which is what makes them independent.")


# ------------------------------------------------------------------ 4. rope
hdr("4. rope: position enters as a rotation  (E7, E8)")

cos, sin = rope_tables(DH, cfg.max_seq)                         # (max_seq, d_h/2)

print("RoPE reads the head's channels as PAIRS, and turns each pair in its own")
print("plane. With a wider head there are more pairs, each turning at its own")
print("rate - so before the d_h=2 model, here is d_h=4 on one hand-made vector.\n")

# --- what rope_tables actually returns -----------------------------------
TAB = 8                                                         # a wider head
tcos, tsin = rope_tables(TAB, cfg.max_seq)                      # (max_seq, 4)
ttheta = 10000.0 ** (-torch.arange(0, TAB, 2).float() / TAB)
print(f"rope_tables(d_h={TAB}, max_seq) -> cos {tuple(tcos.shape)}, "
      f"sin {tuple(tsin.shape)}   ROWS are positions, COLUMNS are pairs")
print(f"  theta per pair          {fmt(ttheta)}")
print(f"\n  m |          angle = m * theta          |"
      f"              cos(angle)")
for t in range(6):
    print(f"  {t} | {fmt(t * ttheta)} | {fmt(tcos[t])}")
print(f"\nThe table is {tcos.shape[0]} x {tcos.shape[1]} - d_h/2 wide, not d_h. Column 0 "
      f"races ahead\n(theta=1) while column 3 has moved {6 * ttheta[3]:.3f} rad after six "
      f"tokens.\nA token at position m uses ROW m, and every head and every sequence in\n"
      f"the batch uses that same row. Built once; the model only ever slices it.")

DEMO = 4
dcos, dsin = rope_tables(DEMO, cfg.max_seq)
print()
vec = torch.tensor([[[[1.0, 2.0, 3.0, 4.0]]]])                  # (1,1,1,4)
pos = 3
print(f"  x            = {fmt(vec[0, 0, 0])}     channels 0 1 2 3")
print(f"  pairs        = (0,1) and (2,3)")
print(f"  x[..., 0::2] = {fmt(vec[0, 0, 0, 0::2])}     first of each pair")
print(f"  x[..., 1::2] = {fmt(vec[0, 0, 0, 1::2])}     second of each pair")
two_i = torch.arange(0, DEMO, 2).float()
print(f"\n  theta        = {fmt(10000.0 ** (-two_i / DEMO))}     one rate per pair")
print(f"  at m={pos}:      cos = {fmt(dcos[pos])}   sin = {fmt(dsin[pos])}")

a, b, c_, d_ = vec[0, 0, 0].tolist()
c0, c1 = dcos[pos].tolist()
s0, s1 = dsin[pos].tolist()
by_hand = torch.tensor([a * c0 - b * s0, a * s0 + b * c0,
                        c_ * c1 - d_ * s1, c_ * s1 + d_ * c1])
print(f"\n  pair 0: ({a:.0f}, {b:.0f}) turned by {pos}*theta_0 -> "
      f"({by_hand[0]:.4f}, {by_hand[1]:.4f})")
print(f"  pair 1: ({c_:.0f}, {d_:.0f}) turned by {pos}*theta_1 -> "
      f"({by_hand[2]:.4f}, {by_hand[3]:.4f})")
print(f"  interleaved  = {fmt(by_hand)}")
print(f"  apply_rope   = {fmt(apply_rope(vec, dcos[pos:pos+1], dsin[pos:pos+1])[0, 0, 0])}")
print(f"  max diff     = "
      f"{(apply_rope(vec, dcos[pos:pos+1], dsin[pos:pos+1])[0,0,0] - by_hand).abs().max():.2e}")
print("\nNote pair 1 barely moved and pair 0 swung right round: theta_1 is ten")
print("times slower. Note too that the two pairs never touch each other - the")
print("only thing tying them together is that they share a position.")

print(f"\nthe rates are geometric, so a head carries a ladder of scales at once:")
print(f"  {'d_h':>4} {'pair':>5} {'theta_i':>12} {'wavelength (tokens)':>21}")
for dh in (4, 64):
    th = 10000.0 ** (-torch.arange(0, dh, 2).float() / dh)
    for i in sorted({0, dh // 4, dh // 2 - 1}):
        print(f"  {dh:>4} {i:>5} {th[i]:>12.6f} {2 * torch.pi / th[i]:>21,.0f}")

print(f"\nback to the d_h={DH} model. theta_i = base**(-2i/d_h), so with one pair")
print("there is one angle per token:")
for t in range(T):
    print(f"  pos {t}:  cos={cos[t,0]:7.4f}  sin={sin[t,0]:7.4f}"
          f"  (angle {torch.atan2(sin[t,0], cos[t,0]):.4f} rad)")

qr = apply_rope(q[None], cos[:T], sin[:T])[0]                   # (H, T, d_h)
kr = apply_rope(k[None], cos[:T], sin[:T])[0]
print("\nhead 0, before and after rotation (lengths are preserved):")
for t in range(T):
    print(f"  pos {t}: {fmt(q[0, t])} -> {fmt(qr[0, t])}   "
          f"|q|={q[0, t].norm():.4f} -> {qr[0, t].norm():.4f}")

a, b = torch.randn(1, 1, 1, DH), torch.randn(1, 1, 1, DH)


def rdot(m, n):
    am = apply_rope(a, cos[m:m + 1], sin[m:m + 1])
    bn = apply_rope(b, cos[n:n + 1], sin[n:n + 1])
    return (am * bn).sum().item()


print(f"\nE8: two fixed vectors, dotted at several (m, n) with the same n-m:")
for m, n in ((5, 3), (6, 4), (7, 5)):
    print(f"  m={m} n={n}  (n-m={n-m})   q.k = {rdot(m, n):+.6f}")
print("Identical. The absolute positions cancel and only the distance survives,")
print("so attention gets relative position without anything computing a distance.")


# ---------------------------------------------------------------- 5. scores
hdr("5. scores: scale, mask, softmax  (E9, E10)")

att = (qr @ kr.transpose(-1, -2)) / DH ** 0.5                   # (H, T, T)
print("head 0 raw scores (row = query, col = key):")
show(att[0], "s")

mask = torch.arange(T)[None, :] > torch.arange(T)[:, None]      # (T, T) True = future
print("\nE9's M: 0 where n <= m, -inf where n > m (m = query row, n = key col):")
for i in range(T):
    print("   " + " ".join("  .  " if not m else " -inf" for m in mask[i]))

masked = att[0].masked_fill(mask, float("-inf"))
p = masked.softmax(dim=-1)
print("\nhead 0 attention weights after softmax:")
show(p, "p")
print(f"\nrow sums: {[round(v, 4) for v in p.sum(-1).tolist()]}")
print(f"row 0 is exactly {fmt(p[0])} - the first token has nowhere to look")
print("but itself, so its 'attention' is the identity no matter what it learns.")
print("\nWhy divide by sqrt(dh): a dot product of two dh-dim random vectors has")
print("standard deviation sqrt(dh), so without the scale the scores spread wider")
print("as heads get wider, and softmax saturates into a near-one-hot row:")
for dh in (2, 16, 128):
    a_, b_ = torch.randn(4096, dh), torch.randn(4096, dh)
    raw = (a_ * b_).sum(-1)
    print(f"  dh={dh:>4}   std(q.k) = {raw.std():6.3f}   "
          f"after /sqrt(dh) = {(raw / dh ** 0.5).std():5.3f}")
print("The divisor is exactly the growth rate, so the score scale stops")
print("depending on how wide you made the head.")


# ------------------------------------------------------- 6. attention output
hdr("6. the weighted sum, the merge, the residual  (E11, E14)")

print("head 0: each output row is p[i] . v, a convex combination of the v rows")
show(v[0], "v")
o0 = p @ v[0]
show(o0, "out")
print(f"\n  out[1] = {p[1,0]:.4f}*v[0] + {p[1,1]:.4f}*v[1] = {fmt(o0[1])}")
print("Same convex-hull argument as a dense MoE gate: non-negative weights that")
print("sum to 1, so attention can only interpolate the values it is given.")

full = attn(xn[None], cos[:T], sin[:T])[0]                      # (T, d) via the module
heads = []
for h in range(H):
    m = att[h].masked_fill(mask, float("-inf")).softmax(-1)
    heads.append(m @ v[h])
merged = torch.cat(heads, dim=-1)                               # (T, H*dh)
print("\nmerge: heads concatenated back to width d, then o_proj mixes them:")
show(merged, "cat")
show(attn.o_proj(merged), "o_proj")
print(f"\nvs the module's own forward: max diff = {(attn.o_proj(merged) - full).abs().max():.2e}")

resid = x + full
print("\nresidual x + attn(norm(x)) - note it adds to the RAW x, not the normalised one:")
show(resid, "h")


# -------------------------------------------------------------------- 7. ffn
hdr("7. the FFN slot: one network, or a routed mixture of them  (E18, E21)")

hn = RMSNorm(D)(resid)
ff = FeedForward(D, D_FF)
with torch.no_grad():
    # moduli chosen coprime with the row length, so no row repeats another
    ff.w1.weight.copy_((torch.arange(D_FF * D).float().reshape(D_FF, D) % 5) * 0.1 - 0.2)
    ff.w3.weight.copy_((torch.arange(D_FF * D).float().reshape(D_FF, D) % 7) * 0.1 - 0.3)
    ff.w2.weight.copy_((torch.arange(D * D_FF).float().reshape(D, D_FF) % 5) * 0.1 - 0.2)

gate, up = F.silu(ff.w1(hn)), ff.w3(hn)
print(f"SwiGLU FFN (E18): ({D} -> {D_FF}) twice, multiplied, then {D_FF} -> {D}")
show(gate, "silu(xW1)")
show(up, "xW3")
show(gate * up, "gate*up")
show(ff(hn), "dense_y")
print("\nW1 decides HOW MUCH of each hidden channel survives and W3 decides")
print("WHAT is in it. A plain two-matrix FFN has only the second half; the gate")
print("is the whole of what SwiGLU adds, at the cost of a third matrix - which")
print("is why d_ff is set near (8/3)d instead of 4d to keep the parameter count.")

moe = MoEFeedForward(D, D_FF, N_EXP, K)
with torch.no_grad():
    moe.gate.weight.copy_(torch.tensor([[0.6, -0.2, 0.3, 0.1],
                                        [-0.4, 0.5, 0.2, -0.3],
                                        [0.1, 0.3, -0.6, 0.4]]))
    for j, e in enumerate(moe.experts):
        e.w1.weight.copy_(ff.w1.weight + 0.05 * j)
        e.w3.weight.copy_(ff.w3.weight - 0.03 * j)
        e.w2.weight.copy_(ff.w2.weight - 0.05 * j)

logits_g = moe.gate(hn)                                         # (T, N)
topl, topi = logits_g.topk(K, dim=-1)
topw = F.softmax(topl, dim=-1)
print(f"\nMoE FFN (E19-E21): {N_EXP} copies of that same network, each token visits {K}")
print("\n  token   router logits          picks        weights")
for t in range(T):
    print(f"    {t}   {fmt(logits_g[t])}   E{topi[t,0].item()}, E{topi[t,1].item()}"
          f"     {topw[t,0]:.4f}, {topw[t,1]:.4f}")

moe_y, aux = moe(hn[None])
print("\nmoe_y (only the routed experts ran):")
show(moe_y[0], "moe_y")
loads = F.one_hot(topi, N_EXP).sum(dim=(0, 1))
print(f"\nrows per expert: {loads.tolist()}  (sums to T*k = {T*K})")
print(f"aux loss (E27 without alpha) = {aux.item():.4f}   floor (balanced) = {K}, "
      f"ceiling (all on one) ~ {N_EXP}")
print("\nThe FFN is the ONLY thing that changed. Same embed, same attention,")
print("same residual, same norm - which is why any difference downstream is")
print("attributable to this slot and nothing else.")


# ----------------------------------------------------------------- 8. logits
hdr("8. logits: the tied head is a dot product with the embedding rows  (E32)")

out = resid + ff(hn)
final = RMSNorm(D)(out)
logits = final @ E.T                                            # tied: lm_head.weight IS E
print("final hidden states:")
show(final, "z")
print("\nlogits[t, v] = z[t] . E[v] - literally 'how much does this state look")
print("like the embedding of token v':")
show(logits, "logits")

probs = logits.softmax(-1)
print("\nnext-token distribution after position 0 (token 2):")
print(f"  {fmt(probs[0])}   argmax = token {probs[0].argmax().item()}")
print(f"\nlogits are shift-invariant: adding 100 to a row changes nothing - "
      f"{(logits[0] + 100).softmax(-1).sub(probs[0]).abs().max():.2e}")
print("Only differences within a row carry information, which is why nobody")
print("normalises logits and why comparing them across models is meaningless.")


# ------------------------------------------------------------------- 9. loss
hdr("9. loss: the model predicts the NEXT token, so targets are shifted  (E24)")

inp, tgt = ids[0, :-1], ids[0, 1:]
print(f"  inputs  {inp.tolist()}")
print(f"  targets {tgt.tolist()}      (position t must predict what sits at t+1)")

lp = logits[:-1].log_softmax(-1)
per_tok = -lp[torch.arange(T - 1), tgt]
print("\n  pos   predicts   -log p")
for t in range(T - 1):
    print(f"    {t}       {tgt[t].item()}       {per_tok[t]:.4f}")
print(f"\n  mean = {per_tok.mean():.4f}")
print(f"  F.cross_entropy = {F.cross_entropy(logits[:-1], tgt):.4f}")
print(f"\nAn untrained model should sit near ln(V) = ln({V}) = {torch.tensor(float(V)).log():.4f};")
print("this one is off because the weights above were hand-set, not initialised.")
print("Cross-entropy is exactly 'the -log probability you assigned to the truth',")
print("averaged - and exp(loss) is the perplexity everyone quotes.")


# ------------------------------------------------------------- 10. the model
hdr("10. the same thing, from TinyLLM")

torch.manual_seed(0)
real_ids = torch.randint(0, 32, (1, 6))
dense = build(ffn="dense", vocab_size=32, d_model=16, n_layers=2, n_heads=2,
              d_ff=32, max_seq=16).eval()
torch.manual_seed(0)
sparse = build(ffn="moe", vocab_size=32, d_model=16, n_layers=2, n_heads=2,
               d_ff=32, max_seq=16, n_experts=4, k=2).eval()

for name, m in (("dense", dense), ("moe  ", sparse)):
    lg, loss = m(real_ids, targets=real_ids)
    mac = m.macs_per_token(6)
    print(f"{name}  logits {tuple(lg.shape)}   loss {loss.item():.4f}   "
          f"params {m.n_params():>7,}   active {m.active_params():>7,}   "
          f"MACs/token {mac['total']:>7,}")

print("\nSame shapes, same loss scale, different economics. The dense model's")
print("active parameter count is its parameter count; the MoE model's is not.")

# causality, in one line: change a token, watch which logits move
probe = real_ids.clone()
probe[0, 3] = (probe[0, 3] + 5) % 32
before, _ = dense(real_ids)
after, _ = dense(probe)
diff = (after - before).abs().amax(dim=-1)[0]
print(f"\nchanged token 3; per-position logit change: "
      f"{[f'{v:.3f}' for v in diff.tolist()]}")
print("Positions 0-2 are exactly zero. That is the causal mask doing its job,")
print("and it is the property that lets one forward pass supply T training")
print("targets instead of one.")
print()
