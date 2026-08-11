"""
Run + dissect SparseMoE from common.py.

    python SparseMoe/run_sparse_moe.py

Sparse top-k routing: what it saves, what it costs, and what it breaks.
Seven sections, each printing something you can check by hand:

  1. flow + dims   - the gather -> compute -> scatter path, tensor by tensor
  2. agreement     - trap, dispatch and (k=N) dense all compute the same function
  3. the trap      - identical numbers, zero savings. What actually gets skipped.
  4. the payoff    - params and FLOPs finally diverge. This is the whole point.
  5. tok and slot  - the routing table, and the silent bug hiding in it
  6. balance       - ragged loads, starved experts, and the auxiliary loss
  7. renormalise   - the convention that quietly kills the router's gradient
"""

import torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path
import os, sys

_here = Path(__file__).resolve().parent
if os.environ.get("MOE_IMPL", "common") in ("scratch", "from_scratch"):
    sys.path.insert(0, str(_here / "from_scratch"))
    from sparse_moe import Expert, MaskedSparseMoE, SparseMoE, switch_aux_loss, load_per_expert
else:
    sys.path.insert(0, str(_here))
    from common import Expert, MaskedSparseMoE, SparseMoE, switch_aux_loss, load_per_expert

torch.manual_seed(0)
torch.set_printoptions(precision=4, sci_mode=False)

B, T, d, d_ff, N, k = 2, 4, 8, 16, 6, 2
S = B * T
moe = SparseMoE(d, d_ff, N, k)
trap = MaskedSparseMoE(d, d_ff, N, k)
moe.copy_from(trap)                              # identical weights, different execution
x = torch.randn(B, T, d)


def hdr(s):
    print(f"\n{'=' * 70}\n{s}\n{'=' * 70}")


# --------------------------------------------------------- 1. flow + dims
hdr("1. flow and dims: gather -> compute -> scatter")

print(f"B={B} batch   T={T} tokens   S=B*T={S}   d={d}   d_ff={d_ff}   N={N} experts   k={k}\n")

xf = x.reshape(-1, d)                            # (S, d)
logits = moe.gate(xf)                            # (S, N)
topl, topi = logits.topk(k, dim=-1)              # (S, k)
topw = F.softmax(topl, dim=-1)                   # (S, k)

print(f"  x       {str(tuple(x.shape)):<12} input")
print(f"  |  reshape(-1, d) - routing is per token, so (B,T) structure is irrelevant here")
print(f"  xf      {str(tuple(xf.shape)):<12} a flat list of S rows")
print(f"  |  gate: Linear(d -> N)")
print(f"  logits  {str(tuple(logits.shape)):<12} one score per (token, expert)")
print(f"  |  topk(k) - the ONLY new operation vs dense. It produces exact zeros.")
print(f"  topl    {str(tuple(topl.shape)):<12} the k largest logits per token")
print(f"  topi    {str(tuple(topi.shape)):<12} their expert ids, in 0..N-1")
print(f"  |  softmax over the k SURVIVORS - each row sums to 1")
print(f"  topw    {str(tuple(topw.shape)):<12} the k gate weights per token")
print()
print(f"  then, per expert e:")
print(f"    tok, slot = (topi == e).nonzero()   both (S_e,)  which tokens, which rank")
print(f"    xf[tok]                             (S_e, d)     a COPY of just those rows")
print(f"    expert(xf[tok])                     (S_e, d)     a dense GEMM on S_e rows")
print(f"    y.index_add_(0, tok, w * out)                    accumulate, don't assign")
print()
print(f"  y       {str(tuple(x.shape)):<12} reshaped back to (B, T, d)")

print("\nThe N axis never appears. Dense MoE's (B,T,N,d) tensor - the thing that")
print("made it expensive - is never built at all. That absence is the whole idea.")

y, aux = moe(x)
print(f"\noutput {tuple(y.shape)}, aux loss {aux.item():.4f}")


# -------------------------------------------------------- 2. agreement
hdr("2. three implementations, one function")

yt, at = trap(x)
print(f"max |dispatch - masked-trap|  = {(y - yt).abs().max().item():.2e}")
assert torch.allclose(y, yt, atol=1e-5)

full = SparseMoE(d, d_ff, N, k=N).copy_from(trap)     # k = N means nothing is dropped
dense_w = F.softmax(moe.gate(xf), dim=-1)             # a plain dense gate
outs = torch.stack([e(xf) for e in moe.experts], dim=-2)
dense_y = (dense_w.unsqueeze(-1) * outs).sum(dim=-2).reshape(B, T, d)
print(f"max |k=N sparse - dense MoE|  = {(full(x)[0] - dense_y).abs().max().item():.2e}")
assert torch.allclose(full(x)[0], dense_y, atol=1e-5)

print("\nk=N recovers dense exactly: with nothing dropped, a softmax over the top-N")
print("logits IS the softmax over all N. Sparse MoE is a strict generalisation -")
print("dense is the k=N corner of it, which is why the dense module came first.")


# ------------------------------------------------------------ 3. the trap
hdr("3. the trap: same answers, no savings")

rows_trap = S * N                                 # every expert sees every token
rows_disp = int(load_per_expert(topi, N).sum())   # sum of S_e
print(f"expert-rows actually computed:   trap {rows_trap:>4}     dispatch {rows_disp:>4}")
print(f"                                 ratio {rows_trap / rows_disp:.1f}x  (= N/k = {N/k:.1f})")

print(f"\npeak intermediate tensor:")
print(f"  trap      (S, N, d) = ({S}, {N}, {d}) = {S*N*d:>6} floats  <- autograd keeps it")
print(f"  dispatch  (S_e, d) per expert, largest = "
      f"{int(load_per_expert(topi, N).max()) * d:>6} floats")

print("\nAt real scale that gap is the story. S=8192, d=4096, N=8, bf16:")
print(f"  trap intermediate = 8192*8*4096*2 bytes = {8192*8*4096*2/2**20:,.0f} MB")
print("  ...per MoE layer, per microbatch, held alive for the backward pass.")

print("\nThe trap is not useless: the mask genuinely zeroes the gradient to")
print("unselected experts, so its LEARNING DYNAMICS are truly sparse. It is the")
print("right tool for studying routing at small N with dense-tensor debuggability.")
print("It is only a lie when someone calls it fast.")


# ---------------------------------------------------------- 4. the payoff
hdr("4. the payoff: parameters and compute finally come apart")

params = sum(p.numel() for p in moe.parameters())
active = moe.macs_per_token()
trap_macs = trap.macs_per_token()

print(f"total parameters        {params:>8,}")
print(f"MACs/token, dispatch    {active:>8,}   ({100*active/params:.0f}% of params)")
print(f"MACs/token, trap        {trap_macs:>8,}   ({100*trap_macs/params:.0f}% of params)")

print("\nIn the dense module these two numbers were IDENTICAL - one MAC per")
print("parameter per token, which is true of every dense layer. Here they split.")
print("Parameters scale with N, compute scales with k. That decoupling is the")
print("entire architectural argument for MoE.\n")

print(f"  {'N':>5} {'k':>3} {'params':>12} {'MACs/token':>12} {'ratio':>7}")
for (nn_, kk) in [(N, k), (8, 2), (64, 8), (256, 8)]:
    m = SparseMoE(d, d_ff, nn_, kk)
    p = sum(q.numel() for q in m.parameters())
    print(f"  {nn_:>5} {kk:>3} {p:>12,} {m.macs_per_token():>12,} {p/m.macs_per_token():>6.1f}x")

print("\nReal models sit on this table: Mixtral 8x7B is N=8 k=2 (46.7B total, ~12.9B")
print("active); DeepSeek-V3 is N=256 k=8 (671B total, 37B active, ~18x).")


# --------------------------------------------------------- 5. tok and slot
hdr("5. tok and slot: the routing table, and the bug hiding in it")

print("topi (which experts each token chose, by rank):")
for s in range(S):
    print(f"  token {s}: 1st -> E{topi[s,0].item()} (w={topw[s,0]:.3f})   "
          f"2nd -> E{topi[s,1].item()} (w={topw[s,1]:.3f})")

print("\nper expert, (topi == e).nonzero() gives BOTH coordinates:")
for e_id in range(N):
    tok, slot = (topi == e_id).nonzero(as_tuple=True)
    if tok.numel():
        pairs = ", ".join(f"t{t}@rank{s}" for t, s in zip(tok.tolist(), slot.tolist()))
        print(f"  E{e_id}: S_e={tok.numel()}  {pairs}")
    else:
        print(f"  E{e_id}: S_e=0   (starved - no kernel launched)")

# the silent bug: topw[tok, 0] instead of topw[tok, slot]
wrong = torch.zeros_like(xf)
for e_id, expert in enumerate(moe.experts):
    tok, slot = (topi == e_id).nonzero(as_tuple=True)
    if tok.numel() == 0:
        continue
    wrong.index_add_(0, tok, topw[tok, 0, None] * expert(xf[tok]))   # BUG: rank 0 always
diff = (wrong.reshape(B, T, d) - y).abs().max().item()
print(f"\nusing topw[tok, 0] instead of topw[tok, slot]: max diff {diff:.4f}")
print("Not a crash, not a shape error - just wrong weights on the right experts.")
print("A token's 2nd-choice expert gets its 1st-choice weight. The model still")
print("trains; it just optimises a different function than you think.")


# ------------------------------------------------------------- 6. balance
hdr("6. load balance: ragged by default")

load = load_per_expert(topi, N)
print(f"rows per expert  {load.tolist()}   (sum {int(load.sum())} = S*k = {S*k})")
print(f"perfectly balanced would be S*k/N = {S*k/N:.1f} each")
print(f"aux loss = {aux.item():.4f}   (minimum is k = {k}, reached only at uniform load)")

# starvation: more experts than tokens makes it a certainty
many = SparseMoE(d, d_ff, 12, k)
_, _ = many(x)
mlog = many.gate(xf)
mtopi = mlog.topk(k, dim=-1).indices
mload = load_per_expert(mtopi, 12)
many.zero_grad(); many(x)[0].sum().backward()
mg = [e.w1.weight.grad.norm().item() if e.w1.weight.grad is not None else 0.0
      for e in many.experts]
starved = [j for j, v in enumerate(mload.tolist()) if v == 0]
print(f"\nwith N=12 and only S={S} tokens (k={k}):")
print(f"  loads       {mload.tolist()}")
print(f"  grad norms  {['%.2f' % v for v in mg]}")
print(f"  starved experts {starved} -> gradient exactly "
      f"{set(round(mg[j], 8) for j in starved) if starved else 'n/a'}")
print("\nAn expert nothing routes to gets no gradient, so it never improves, so it")
print("keeps losing the top-k. That loop is why the auxiliary loss is not optional")
print("decoration: it is differentiable through P even though top-k itself is not.")

# does the aux loss actually push toward balance?
print("\naux loss as routing collapses (S=4096 synthetic):")
for name, sharp in [("uniform", 0.0), ("mild skew", 1.0), ("hard collapse", 12.0)]:
    lg = torch.randn(4096, N) * 0.1
    lg[:, 0] += sharp
    ti = lg.topk(k, -1).indices
    ld = load_per_expert(ti, N).float()
    print(f"  {name:<14} aux={switch_aux_loss(lg, ti, N).item():.3f}   "
          f"busiest expert holds {100 * ld.max() / ld.sum():.0f}% of all rows "
          f"(uniform = {100 / N:.0f}%)")

print(f"\nThe load column saturates at {100/k:.0f}% - one expert can win at most one of")
print("each token's k slots, so k=2 caps any single expert at half the rows. The")
print("aux loss keeps climbing past that point anyway, because it reads P, the")
print("probability mass, not just the discrete counts. It sees the router growing")
print("more certain even when the load numbers have stopped moving - which is")
print("exactly the early-warning behaviour you want from a balancing term.")


# ---------------------------------------------------------- 7. renormalise
hdr("7. the renormalisation that kills the router")

lg = torch.randn(200, N)
tl, ti = lg.topk(k, -1)
mixtral = F.softmax(tl, -1)                                   # topk -> softmax
switch = F.softmax(lg, -1).gather(-1, ti)
switch_renorm = switch / switch.sum(-1, keepdim=True)         # softmax -> topk -> renorm
print(f"'topk then softmax' vs 'softmax then topk then renormalise':")
print(f"  max |difference| = {(mixtral - switch_renorm).abs().max().item():.2e}")
print("\nThey are the SAME function, not two conventions. Renormalising a softmax")
print("over a subset cancels the shared denominator exactly. The ordering is a")
print("matter of arithmetic cost, not semantics.")

print("\nThe real fork is renormalise vs DON'T. At k=1 it decides everything:")


def gate_grad(renorm):
    m = SparseMoE(d, d_ff, N, k=1)
    lgt = m.gate(xf)
    tv, tix = lgt.topk(1, -1)
    w = F.softmax(tv, -1) if renorm else F.softmax(lgt, -1).gather(-1, tix)
    out = torch.zeros_like(xf)
    for e_id, e in enumerate(m.experts):
        tk, sl = (tix == e_id).nonzero(as_tuple=True)
        if tk.numel():
            out.index_add_(0, tk, w[tk, sl, None] * e(xf[tk]))
    m.zero_grad(); out.sum().backward()
    return m.gate.weight.grad.norm().item()


print(f"  k=1, renormalised    gate grad norm = {gate_grad(True):.8f}")
print(f"  k=1, raw probability gate grad norm = {gate_grad(False):.8f}")
print("\nsoftmax of a single kept logit is the constant 1.0, so the weight carries")
print("no dependence on the gate at all and dL/dW_g is exactly zero - the router")
print("is frozen from step one. Switch Transformer multiplies by the UN-normalised")
print("probability precisely to keep that gradient alive. At k>=2 renormalising is")
print("harmless, because the k weights still depend on their relative logits.")
print()
