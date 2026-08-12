"""
DenseMoE, one tensor at a time - small enough to check by hand.

    python DenseMoe/steps_dense_moe.py

run_dense_moe.py runs the real layer on random data and measures it. This file
does the opposite: 3 tokens, 3 experts, d=4, every number literal, every
intermediate printed. Nothing is random and nothing hides inside a Module.

It walks the same path as DenseMoE.forward in common.py:

    w    = softmax(gate(x), dim=-1)          (T, N)   one distribution per token
    outs = stack([e(x) for e in experts])    (T, N, d)  <-- the dense cost
    y    = (w.unsqueeze(-1) * outs).sum(-2)  (T, d)   weighted average

Nine steps: the gate, the experts, the broadcast, three spellings that agree,
the convex hull, why no flatten is needed, the bottleneck, the backward pass,
and a check against the real DenseMoE and BatchedDenseMoE.

The expert outputs are the same literal numbers used in the notes, so the
arithmetic here can be compared line for line with the walkthrough there.
"""

import torch
import torch.nn.functional as F
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DenseMoE, BatchedDenseMoE                            # noqa: E402

torch.set_printoptions(precision=4, sci_mode=False)


def hdr(s):
    print(f"\n{'=' * 70}\n{s}\n{'=' * 70}")


def fmt(row):
    # `v if v else 0.0` normalises -0.0, which the sum produces and nobody means
    return "[" + "  ".join(f"{(v if v else 0.0):7.4f}" for v in row.tolist()) + "]"


def show_rows(t, label):
    for i, row in enumerate(t):
        print(f"  {label}[{i}] = {fmt(row)}")


# ------------------------------------------------------------ 0. the setup
hdr("0. three tokens, three experts, everything literal")

T, N, d = 3, 3, 4

# What each expert answered, for each token. In a real layer this is produced by
# torch.stack([e(x) for e in self.experts], dim=-2) - here the numbers are just
# written down, because nothing below depends on where they came from.
outs = torch.tensor([                                       # (T, N, d)
    [[1., 2., -1., 0.], [0., -1., 2., 1.], [2., 1., 0., -1.]],   # token 0
    [[0., 1., 1., -2.], [1., 0., -1., 1.], [-1., 2., 0., 1.]],   # token 1
    [[2., -1., 0., 1.], [1., 1., 1., 0.], [0., 0., -2., 2.]],    # token 2
])

print(f"T={T} tokens   N={N} experts   d={d}")
print(f"outs {tuple(outs.shape)}   - every expert's answer for every token\n")
for t in range(T):
    print(f"  token {t}:")
    show_rows(outs[t], "    E")


# ------------------------------------------------------------- 1. the gate
hdr("1. w = softmax(logits, dim=-1)")

# In the real layer: logits = self.gate(x), a Linear(d -> N). Written down here
# as log-probabilities, which is the one choice of logits that makes softmax
# return the round numbers 0.6 / 0.3 / 0.1 exactly.
target = torch.tensor([[.6, .3, .1], [.2, .5, .3], [.1, .2, .7]])
logits = target.log()                                       # (T, N)
w = F.softmax(logits, dim=-1)                               # (T, N)

print(f"logits {tuple(logits.shape)}")
show_rows(logits, "logits")
print(f"\nw {tuple(w.shape)}   softmax over dim=-1, the EXPERT axis")
show_rows(w, "w")
print(f"\n  row sums {[round(v, 4) for v in w.sum(-1).tolist()]}   "
      f"<- each token's weights form a distribution")

# softmax is shift-invariant: only the differences between logits matter.
print(f"\nsoftmax(logits + 5) - w:  max |diff| = "
      f"{(F.softmax(logits + 5, dim=-1) - w).abs().max():.2e}")
print("Adding a constant to a whole row changes nothing - the constant factors")
print("out of numerator and denominator. Logits have no absolute meaning; only")
print("the gaps between them do.")

# the axis bug
wrong_axis = F.softmax(logits, dim=0)                       # over TOKENS, not experts
print(f"\nsoftmax(dim=0) instead: column sums "
      f"{[round(v, 4) for v in wrong_axis.sum(0).tolist()]}, row sums "
      f"{[round(v, 4) for v in wrong_axis.sum(-1).tolist()]}")
print("Same shape, no error, and it normalises across TOKENS - each expert's")
print("share of the batch rather than each token's mixture. dim=-1 is not a")
print("stylistic choice: it is what makes routing per-token.")


# ---------------------------------------------------------- 2. the broadcast
hdr("2. (w.unsqueeze(-1) * outs).sum(dim=-2)")

print(f"w    {tuple(w.shape)}       one scalar per (token, expert)")
print(f"outs {tuple(outs.shape)}    d numbers per (token, expert)")
print("\nThey cannot multiply as they are: broadcasting aligns axes from the")
print("RIGHT, so w's N would line up against outs' d. The weight has to grow a")
print("trailing size-1 axis so it can stretch along d instead:\n")
print(f"  w.unsqueeze(-1)  {tuple(w.unsqueeze(-1).shape)}   one scalar, "
      f"stretched over all {d} dims")

scaled = w.unsqueeze(-1) * outs                             # (T, N, d)
print(f"  * outs           {tuple(scaled.shape)}   each expert's answer, scaled")

print(f"\nthe three scaled rows for token 0:")
for j in range(N):
    print(f"  w[0,{j}] = {w[0, j]:.1f}  x  {fmt(outs[0, j])}  =  {fmt(scaled[0, j])}")

y = scaled.sum(dim=-2)                                      # (T, d)
print(f"\n  .sum(dim=-2)     {tuple(y.shape)}   collapse the expert axis")
print(f"  y[0] = {fmt(y[0])}   <- add the three rows above, column by column")

print(f"\ny {tuple(y.shape)}")
show_rows(y, "y")

print(f"\nBy hand those rows are [0.8, 1.0, 0.0, 0.2], [0.2, 0.8, -0.3, 0.4] and")
print(f"[0.4, 0.1, -1.2, 1.5]. Note y[0][2] prints as -0.0000: it is actually")
print(f"{y[0, 2]:.3e}, not zero. The weights came out of log -> softmax and carry")
print("~1e-8 of rounding, which the cancellation at that coordinate then exposes.")
print("Exact in maths, approximate in float32 - worth seeing once, and the reason")
print("every check in this repo compares against a tolerance instead of ==.")

# the axis trap, and the case where it does not announce itself
bad = scaled.sum(dim=-1)
print(f"\n.sum(dim=-1) instead: {tuple(bad.shape)} - it summed over d, not over N.")
print(f"Here that is caught by the shape ({tuple(bad.shape)} is not "
      f"{tuple(y.shape)}), but only because")
sq = torch.randn(T, 3, 3)
print(f"d != N. With d == N both are {tuple(sq.sum(-1).shape)} and nothing "
      f"complains at all.")


# ------------------------------------------------------ 3. three spellings
hdr("3. three spellings of the same sum")

loop = torch.zeros(T, d)
for t in range(T):
    for j in range(N):
        loop[t] += w[t, j] * outs[t, j]                     # scalar * (d,)

eins = torch.einsum('tn,tnd->td', w, outs)                  # n summed away

print(f"max |broadcast - python loop| = {(loop - y).abs().max():.2e}")
print(f"max |broadcast - einsum|      = {(eins - y).abs().max():.2e}")

print("\nThe loop is the one to trust: every value in it is a scalar or a 1-D")
print("vector, so there is no axis to choose and no broadcast to line up.")
print("Neither mistake the vectorised version can make quietly is expressible.")
print("\nIn 'tn,tnd->td', n appears on both inputs and not on the output, which")
print("is what marks it as the summed index. That one letter replaces unsqueeze,")
print("multiply and sum - and states the contraction rather than implying it.")


# ------------------------------------------------------- 4. the convex hull
hdr("4. the output can never leave the experts' convex hull")

lo, hi = outs.min(dim=-2).values, outs.max(dim=-2).values   # (T, d)
print(f"every coordinate of y is between the min and max over experts: "
      f"{bool(((y >= lo - 1e-6) & (y <= hi + 1e-6)).all())}\n")

print("token 0, coordinate by coordinate:")
print("  dim | E0     E1     E2    |  min    max   |  y")
for i in range(d):
    es = "  ".join(f"{outs[0, j, i]:5.1f}" for j in range(N))
    print(f"   {i}  | {es} | {lo[0, i]:5.1f}  {hi[0, i]:5.1f}  | {y[0, i]:6.2f}")

print("\nThe weights are non-negative and sum to 1, so y is a weighted AVERAGE.")
print("No gate setting can push it outside the box the experts span. A dense MoE")
print("interpolates; it cannot extrapolate.\n")

uniform = torch.full((T, N), 1.0 / N)
print(f"uniform gate (1/3 each): y[0] = "
      f"{fmt((uniform.unsqueeze(-1) * outs).sum(-2)[0])}")
print(f"  the plain mean of the three experts = {fmt(outs[0].mean(0))}")
onehot = torch.tensor([[1., 0., 0.]] * T)
print(f"one-hot gate [1,0,0]:    y[0] = "
      f"{fmt((onehot.unsqueeze(-1) * outs).sum(-2)[0])}")
print(f"  expert 0's answer, unchanged        = {fmt(outs[0, 0])}")

print("\nThose are the two corners. Uniform gives one blurry FFN at N times the")
print("price; one-hot is hard routing, which is what top-k approximates. Every")
print("useful dense gate sits between them.")

# capacity lives in the experts, not the gate - measured
print("\nAnd the hull is only as big as the experts are different. Collapse them:")
collapsed = outs[:, :1].expand(-1, N, -1)                   # all experts = E0
spread = torch.stack([(torch.rand(T, N).softmax(-1).unsqueeze(-1) * collapsed)
                      .sum(-2) for _ in range(50)])
print(f"  with all N experts identical, 50 random gates give outputs that differ")
print(f"  by at most {(spread.max(0).values - spread.min(0).values).max():.2e}")
print("  - the gate has become decoration. Capacity comes from the experts")
print("  differing from each other; the gate only chooses where between them.")


# ------------------------------------------------- 5. the leading axes ride
hdr("5. B and T are along for the ride")

x3 = outs.unsqueeze(0).expand(2, -1, -1, -1)                # (B, T, N, d)
w3 = w.unsqueeze(0).expand(2, -1, -1)                       # (B, T, N)
y3 = (w3.unsqueeze(-1) * x3).sum(dim=-2)                    # (B, T, d)

print(f"same numbers with a batch axis: outs {tuple(x3.shape)} -> y {tuple(y3.shape)}")
print(f"max |y3[0] - y| = {(y3[0] - y).abs().max():.2e}   and "
      f"|y3[1] - y| = {(y3[1] - y).abs().max():.2e}")

print("\nEvery operation in dense MoE - the Linear, the softmax over dim=-1, the")
print("broadcast, the sum over dim=-2 - addresses axes from the right. The")
print("leading axes are never named, so (T, ...) and (B, T, ...) both work with")
print("the same code and give the same answer per token.")
print("\nThis is why dense MoE has no reshape anywhere, and sparse MoE opens with")
print("one: the moment routing needs to GATHER specific tokens, a token needs a")
print("single address instead of a (batch, position) pair. See SparseMoe.")


# ---------------------------------------------------------- 6. the bottleneck
hdr("6. the bottleneck: MACs per token == parameters")

D, D_FF, NE = 8, 16, 3
moe = DenseMoE(D, D_FF, NE)
params = sum(p.numel() for p in moe.parameters())
mac_gate, mac_expert = D * NE, 2 * D * D_FF

print(f"a real DenseMoE with d={D}, d_ff={D_FF}, N={NE}:\n")
print(f"  {'':<44}{'MACs/token':>12}{'params':>10}")
print(f"  {f'gate     Linear({D} -> {NE})':<44}{mac_gate:>12,}{D * NE:>10,}")
print(f"  {f'expert   Linear({D} -> {D_FF}), Linear({D_FF} -> {D})':<44}"
      f"{mac_expert:>12,}{mac_expert:>10,}")
print(f"  {f'         x {NE} experts':<44}{NE * mac_expert:>12,}{NE * mac_expert:>10,}")
print(f"  {'-' * 66}")
print(f"  {'total':<44}{moe.macs_per_token():>12,}{params:>10,}")
print(f"\n  macs_per_token() == parameters:  {moe.macs_per_token()} == {params}  "
      f"-> {moe.macs_per_token() == params}")

print("\nThat identity is not a coincidence of these numbers - it is what a dense")
print("layer IS. Every parameter is a multiply, and every token walks past every")
print("parameter exactly once. So 'active parameters' is 100% of the model, by")
print("construction, and scaling N scales memory and compute together:\n")
print(f"  {'N':>5} {'params':>12} {'MACs/token':>12} {'active':>8}")
for n in (NE, 8, 64, 512):
    m = DenseMoE(D, D_FF, n)
    p = sum(q.numel() for q in m.parameters())
    print(f"  {n:>5} {p:>12,} {m.macs_per_token():>12,} "
          f"{100 * m.macs_per_token() / p:>7.0f}%")

print("\nThe entire promise of MoE is that those two columns come apart. Here they")
print("are the same column. Dense MoE buys the routing mechanism and none of the")
print("efficiency - it is a stepping stone, not a destination.")


# ----------------------------------------------------- 7. backward is dense too
hdr("7. the backward pass is just as dense")

xr = torch.randn(2, 4, D)
moe.zero_grad()
moe(xr).sum().backward()
g = [e.w1.weight.grad.norm().item() for e in moe.experts]
print(f"||dL/dW1|| per expert: {['%.4f' % v for v in g]}")
print(f"all non-zero: {all(v > 0 for v in g)}")

print("\nEvery expert receives gradient from every token, because every expert")
print("contributed to every token. That is genuinely pleasant: no dead experts,")
print("no starvation, nothing to balance, no auxiliary loss. It is also exactly")
print("why it costs what it costs. Sparse routing trades this away and spends")
print("the rest of its complexity budget coping with the consequences.")


# ------------------------------------------------------- 8. the real modules
hdr("8. the same arithmetic against the real modules")

wr = F.softmax(moe.gate(xr), dim=-1)                        # (B, T, N)
outs_r = torch.stack([e(xr) for e in moe.experts], dim=-2)  # (B, T, N, d)
hand = (wr.unsqueeze(-1) * outs_r).sum(dim=-2)              # (B, T, d)

print(f"max |hand-rolled - DenseMoE(x)|      = {(hand - moe(xr)).abs().max():.2e}")
print(f"max |forward_loop - DenseMoE(x)|     = "
      f"{(moe.forward_loop(xr) - moe(xr)).abs().max():.2e}")
print(f"max |forward_einsum - DenseMoE(x)|   = "
      f"{(moe.forward_einsum(xr) - moe(xr)).abs().max():.2e}")
assert torch.allclose(hand, moe(xr), atol=1e-6)

batched = BatchedDenseMoE(D, D_FF, NE).copy_from(moe)
print(f"max |BatchedDenseMoE - DenseMoE|     = "
      f"{(batched(xr) - moe(xr)).abs().max():.2e}")
print(f"  MACs/token: DenseMoE {moe.macs_per_token()}, "
      f"BatchedDenseMoE {batched.macs_per_token()}  -> "
      f"{moe.macs_per_token() == batched.macs_per_token()}")

print("\nBatching the N experts into one (N, d, d_ff) tensor changes the number of")
print("kernel launches, not the number of multiplies. Same answer, same cost.")
print("It matters anyway: once the experts live in one indexed tensor, 'run")
print("expert j on token i' becomes a gather instead of a branch - which is the")
print("layout sparse dispatch is built on.")
print("\nNext: python DenseMoe/run_dense_moe.py for the same layer at a size worth")
print("measuring, then SparseMoe/ for the version that finally splits those two")
print("columns in step 6.")
print()
