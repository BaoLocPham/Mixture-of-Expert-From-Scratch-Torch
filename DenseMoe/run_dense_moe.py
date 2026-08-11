"""
Run + dissect DenseMoE from common.py.

    python DenseMoe/run_dense_moe.py

Dense MoE only - the flow, the dims, and the bottleneck. Five sections, each
printing something you can check by hand:

  1. flow + dims   - every tensor in forward(), and why each axis is there
  2. hand-recompute- the broadcast sum == a naive python loop over tokens
  3. convex hull   - the output is a *weighted average* of expert outputs
  4. bottleneck    - dense pays for all N experts on every token, fwd and bwd
  5. sanity fit    - the gate discovers structure on its own
"""

import torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path
import os, sys

_here = Path(__file__).resolve().parent
if os.environ.get("MOE_IMPL", "common") in ("scratch", "from_scratch"):
    # MOE_IMPL=scratch -> run all of this against your own from_scratch/ version
    sys.path.insert(0, str(_here / "from_scratch"))
    from dense_moe import Expert, DenseMoE
else:
    sys.path.insert(0, str(_here))
    from common import Expert, DenseMoE

torch.manual_seed(0)
torch.set_printoptions(precision=4, sci_mode=False)

B, T, d, d_ff, N = 2, 4, 8, 16, 3          # batch, tokens, d_model, d_ff, n_experts
moe = DenseMoE(d, d_ff, N)
x = torch.randn(B, T, d)


def hdr(s):
    print(f"\n{'=' * 68}\n{s}\n{'=' * 68}")


# --------------------------------------------------------- 1. flow + dims
hdr("1. flow and dims: every tensor inside forward()")

print(f"B={B} batch   T={T} tokens   d={d} d_model   d_ff={d_ff}   N={N} experts\n")

logits = moe.gate(x)                        # (B, T, N)
w = F.softmax(logits, dim=-1)               # (B, T, N)
outs = torch.stack([e(x) for e in moe.experts], dim=-2)   # (B, T, N, d)
y = (w.unsqueeze(-1) * outs).sum(dim=-2)    # (B, T, d)

print(f"  x       {str(tuple(x.shape)):<14} input tokens")
print(f"  |")
print(f"  |  gate: Linear(d -> N), weight {tuple(moe.gate.weight.shape)}")
print(f"  v")
print(f"  logits  {str(tuple(logits.shape)):<14} one score per (token, expert)")
print(f"  |")
print(f"  |  softmax over the LAST axis = over experts, per token")
print(f"  v")
print(f"  w       {str(tuple(w.shape)):<14} rows sum to 1")
print()
print(f"  x       {str(tuple(x.shape)):<14} (same input, sent to all experts)")
print(f"  |")
print(f"  |  each Expert: d -> d_ff -> silu -> d.  stack on a NEW axis -2")
print(f"  v")
print(f"  outs    {str(tuple(outs.shape)):<14} EVERY expert ran on EVERY token")
print()
print(f"  w.unsqueeze(-1)  {str(tuple(w.unsqueeze(-1).shape)):<14} broadcasts against outs")
print(f"  * outs           {str(tuple(outs.shape)):<14} elementwise, N kept")
print(f"  .sum(dim=-2)     {str(tuple(y.shape)):<14} collapse the expert axis")
print()
print(f"  y       {str(tuple(y.shape)):<14} same shape as x -> drop-in FFN replacement")

print("\nWhy unsqueeze(-1): w is (B,T,N) but outs is (B,T,N,d). The weight is one")
print("scalar per expert, and it must multiply all d output dims of that expert,")
print("so w needs a trailing size-1 axis to broadcast along d.")

print("\ngate weights for batch 0 (rows = tokens, cols = experts):")
print(w[0])
print("row sums:", w[0].sum(-1))

assert torch.allclose(y, moe(x)), "manual trace must equal module forward"
print("\nmanual trace == moe(x)  ok")


# -------------------------------------------------------- 2. hand-recompute
hdr("2. three spellings, one function")

# forward_loop is the naive triple loop; forward_einsum replaces the broadcast
# with 'btn,btnd->btd'. Both live in common.py next to forward().
slow = moe.forward_loop(x)
eins = moe.forward_einsum(x)

print("max |vectorised - python loop| =", (slow - y).abs().max().item())
print("max |vectorised - einsum|      =", (eins - y).abs().max().item())
assert torch.allclose(slow, y, atol=1e-5)
assert torch.allclose(eins, y, atol=1e-5)
print("so   y[b,t] = sum_j  softmax(W_g h)[j] * E_j(h)   ok")
print("\nThe loop is the trustworthy one: inside it every tensor is 1-D or a")
print("scalar, so there is no axis to choose and no broadcast to line up -")
print("neither bug the vectorised version can hide is expressible there.")
print("\nNote the gate is applied PER TOKEN, not per sequence: routing is a")
print("token-level decision, which is why the N axis sits next to T.")

print("\ntoken (0,0) broken into its 3 contributions (first 4 dims):")
for j in range(N):
    print(f"  w={w[0,0,j]:.4f} * E_{j}(h) = {(w[0,0,j] * outs[0,0,j])[:4]}")
print(f"  {'sum':>21} = {y[0,0][:4]}")


# ------------------------------------------------------------ 3. convex hull
hdr("3. dense MoE output lives in the convex hull of the experts")

lo = outs.min(dim=-2).values                # (B, T, d) per-dim min over experts
hi = outs.max(dim=-2).values
inside = ((y >= lo - 1e-5) & (y <= hi + 1e-5)).all()
print("every output coordinate is between min and max expert output:", bool(inside))
print("\ntoken (0,0), dim 0:")
print(f"  expert outputs {outs[0,0,:,0]}")
print(f"  weights        {w[0,0]}")
print(f"  mixture        {y[0,0,0]:.4f}   (a weighted average, never outside)")
print("\nBecause the weights are non-negative and sum to 1, a dense MoE can only")
print("INTERPOLATE its experts. Capacity comes from the experts being different,")
print("not from the gate. A gate that outputs uniform weights just gives you the")
print("average of N experts - an expensive way to build one blurry FFN.")


# ---------------------------------------------------------- 4. bottleneck
hdr("4. the bottleneck")

mac_expert = 2 * d * d_ff                   # w1: d*d_ff, w2: d_ff*d
mac_gate = d * N
tokens = B * T
p_total = sum(p.numel() for p in moe.parameters())
p_expert = sum(p.numel() for p in moe.experts.parameters())

total = moe.macs_per_token()                # derived by the model itself
assert total == mac_gate + N * mac_expert, "macs_per_token disagrees with the breakdown"

print(f"per token: gate     {mac_gate:>6} MACs")
print(f"           expert   {mac_expert:>6} MACs each  x {N} experts = {N * mac_expert}")
print(f"           total    {total:>6} MACs   (moe.macs_per_token())")
print(f"\nparams: {p_total:,} total, {p_expert:,} in experts "
      f"({100 * p_expert / p_total:.0f}%)")
print(f"ACTIVE params per token: {p_total:,}  ->  {100 * p_total / p_total:.0f}% of the model")

print("\nScaling N scales memory AND compute together, linearly:")
print(f"  {'N':>4} {'params':>10} {'MACs/token':>12}")
for n in (N, 8, 64, 512):
    mn = DenseMoE(d, d_ff, n)               # measured params vs derived MACs
    pn = sum(p.numel() for p in mn.parameters())
    print(f"  {n:>4} {pn:>10,} {mn.macs_per_token():>12,}")

print("\nThat is the whole problem. The appeal of MoE is 'more parameters without")
print("more compute per token', and dense delivers exactly none of it - N x the")
print("params costs N x the FLOPs. Dense MoE is only a stepping stone.")

# and it is just as bad backwards
moe.zero_grad()
moe(x).sum().backward()
g = [e.w1.weight.grad.norm().item() for e in moe.experts]
print(f"\nbackward, ||dL/dW1|| per expert: {['%.4f' % v for v in g]}")
print("Every expert gets a non-zero gradient from every token - nice for")
print("optimisation (no dead experts, smooth gradients, nothing to tune), and")
print("precisely why it is expensive. The sparse version trades this away.")


# ---------------------------------------------------------- 5. sanity fit
hdr("5. sanity fit: can the gate discover structure?")

# Task: two token types. Type 0 must be negated, type 1 must be doubled.
# The type is encoded in the sign of dim 0. Nobody tells the model this.
torch.manual_seed(1)
model = DenseMoE(d_model=4, d_ff=16, n_experts=2)
opt = torch.optim.Adam(model.parameters(), lr=3e-2)


def batch(n=256):
    z = torch.randn(n, 1, 4)
    typ = (z[..., 0] > 0).float().unsqueeze(-1)      # (n,1,1)
    tgt = torch.where(typ.bool(), 2.0 * z, -z)
    return z, tgt, typ


for step in range(400):
    z, tgt, _ = batch()
    loss = F.mse_loss(model(z), tgt)
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 100 == 0:
        print(f"  step {step:>3}  loss {loss.item():.4f}")

z, tgt, typ = batch(1024)
with torch.no_grad():
    print(f"  final loss {F.mse_loss(model(z), tgt).item():.4f}")
    gw = F.softmax(model.gate(z), dim=-1).squeeze(1)          # (n, 2)
    m = typ.squeeze(-1).squeeze(-1).bool()
    print(f"\n  mean gate weights, type-0 tokens: {gw[~m].mean(0)}")
    print(f"  mean gate weights, type-1 tokens: {gw[m].mean(0)}")
print("\nThose two rows come out near-opposite: the gate found the split by")
print("itself, and the two experts specialised. That is the mechanism. What")
print("is still missing is making it CHEAP - which is what sparse routing is for.")
print()
