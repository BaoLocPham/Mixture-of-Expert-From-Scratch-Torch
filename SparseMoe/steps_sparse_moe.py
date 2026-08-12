"""
SparseMoE, one tensor at a time - small enough to check by hand.

    python SparseMoe/steps_sparse_moe.py

run_sparse_moe.py dissects the real layer at realistic-ish sizes and measures
things. This file does the opposite: S=6 rows, N=3 experts, d=4, every number
literal, every intermediate printed. Nothing here is random and nothing is
hidden behind a Module - you can recompute any line with a calculator.

It walks the same path as SparseMoE.forward in common.py:

    xf   = x.reshape(-1, d)                     flatten, routing is per token
    logits, topl, topi = ...                    score, then keep the top k
    topw = softmax(topl)                        weights over the survivors
    for each expert:  tok, slot -> gather -> compute -> scale -> index_add_
    y.reshape(B, T, d)                          put the batch shape back

Along the way it builds the DENSE mixture and the MASKED one from the same six
rows, so the three sit side by side with identical numbers and very different
costs.
"""

import torch
import torch.nn.functional as F
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import Expert, SparseMoE, switch_aux_loss, load_per_expert   # noqa: E402

torch.set_printoptions(precision=4, sci_mode=False)


def hdr(s):
    print(f"\n{'=' * 70}\n{s}\n{'=' * 70}")


def fmt(row):
    return "[" + "  ".join(f"{v:7.4f}" for v in row.tolist()) + "]"


def show_rows(t, label, mark=()):
    """Print a (S, ·) tensor one row per line, arrowing the rows that just moved."""
    for i, row in enumerate(t):
        tag = "   <-- changed" if i in mark else ""
        print(f"  {label}[{i}] = {fmt(row)}{tag}")


# ------------------------------------------------------------ 0. the setup
hdr("0. six tokens, three experts, everything literal")

B, T, d, N, k = 2, 3, 4, 3, 2
S = B * T

# Values chosen so a token's identity is readable straight off the tensor:
# every element of token s is (s + 1).
x = torch.tensor([
    [[1., 1., 1., 1.], [2., 2., 2., 2.], [3., 3., 3., 3.]],   # sequence 0: t0 t1 t2
    [[4., 4., 4., 4.], [5., 5., 5., 5.], [6., 6., 6., 6.]],   # sequence 1: t0 t1 t2
])                                                            # (B, T, d) = (2, 3, 4)

print(f"B={B}  T={T}  S=B*T={S}  d={d}  N={N} experts  k={k}")
print(f"x {tuple(x.shape)}")


def expert(e_id, rows):
    """Stand-in for Expert(...) from common.py: multiply by (e_id + 1).

    A real expert is w2(silu(w1(rows))). Nothing in the routing machinery cares
    what the expert does - only that it maps (rows, d) -> (rows, d) and never
    looks across rows. Using x3 instead of an MLP keeps every output a number
    you can verify: expert(e, token s) = (e + 1) * (s + 1).
    """
    return (e_id + 1) * rows


# ---------------------------------------------------------- 1. the flatten
hdr("1. xf = x.reshape(-1, d)")

xf = x.reshape(-1, d)                                         # (S, d) = (6, 4)

print(f"x  {tuple(x.shape)}  ->  xf {tuple(xf.shape)}\n")
show_rows(xf, "xf")
print(f"\n  xf[4] is x[1, 1]:      {torch.equal(xf[4], x[1, 1])}     (row s = b*T + t)")
print(f"  same storage, no copy: {xf.data_ptr() == x.data_ptr()}")

print("\nRouting is per token: the gate scores one row at a time and never looks")
print("along T or across B. So the (B, T) split carries no information the router")
print("uses, and collapsing it costs nothing - it just makes tok a single integer")
print("instead of a (batch, position) pair. y.reshape(B, T, d) undoes it at the end.")


# ------------------------------------------------------------ 2. the logits
hdr("2. logits = gate(xf), then topk")

# In the real layer this is self.gate(xf), a Linear(d -> N). Written out here so
# the routing decisions below are chosen, not discovered.
logits = torch.tensor([
    [2.0, 1.0, 0.0],       # row 0
    [0.0, 3.0, 1.0],       # row 1
    [1.0, 0.5, 4.0],       # row 2
    [3.0, 2.0, 0.0],       # row 3
    [2.5, 1.5, 0.5],       # row 4
    [1.0, 0.0, 2.0],       # row 5
])                                                            # (S, N) = (6, 3)

topl, topi = logits.topk(k, dim=-1)                           # both (S, k) = (6, 2)

print(f"logits {tuple(logits.shape)}   topl {tuple(topl.shape)}   topi {tuple(topi.shape)}\n")
print("  row | logits              | kept (rank 0, rank 1) | dropped")
for s in range(S):
    kept = ", ".join(f"E{topi[s, j].item()}={topl[s, j]:.1f}" for j in range(k))
    dropped = [f"E{e}" for e in range(N) if e not in topi[s].tolist()]
    print(f"   {s}  | {str(logits[s].tolist()):<19} | {kept:<21} | {', '.join(dropped)}")

print("\ntopk returns values AND indices, and both matter later: topl feeds the")
print("softmax, topi says where the rows have to go. This is the only operation")
print("in the whole layer that dense MoE does not have.")


# ------------------------------------------------------------ 3. the weights
hdr("3. topw = softmax(topl) - over the k survivors, not all N")

topw = F.softmax(topl, dim=-1)                                # (S, k)

print(f"topw {tuple(topw.shape)}, each row sums to 1: "
      f"{[round(v, 4) for v in topw.sum(-1).tolist()]}\n")
for s in range(S):
    parts = "  ".join(f"E{topi[s, j].item()}: {topw[s, j]:.4f}" for j in range(k))
    print(f"  row {s}: {parts}     (logit gap {topl[s, 0] - topl[s, 1]:.1f})")

print("\nRows 0, 3, 4 and 5 all get 0.7311 / 0.2689 - softmax reads only the GAP")
print("between the kept logits, so a gap of 1.0 gives the same split whether the")
print("logits were [2, 1] or [3, 2]. Row 2's gap is 3.0, hence 0.9526 / 0.0474.")

# the same weights, the long way round: the masked convention
mask = torch.zeros_like(logits).scatter(-1, topi, 1.0)        # (S, N), 1 at the kept
w_masked = F.softmax(logits, dim=-1) * mask                   # zeros elsewhere
w_masked = w_masked / w_masked.sum(-1, keepdim=True)          # renormalise over the k
gathered = w_masked.gather(-1, topi)                          # (S, k), back to compact form
print(f"\nsoftmax-over-N then mask then renormalise gives the same numbers:")
print(f"  max |difference| = {(gathered - topw).abs().max().item():.2e}")
print("The shared denominator cancels. Two spellings, one function.")


# --------------------------------------------------- 4. what dense would do
hdr("4. for contrast: the dense mixture, all N experts on all S rows")

w_dense = F.softmax(logits, dim=-1)                           # (S, N)
outs = torch.stack([expert(i, xf) for i in range(N)], dim=-2)  # (S, N, d)  <-- the big one
dense_y = (w_dense.unsqueeze(-1) * outs).sum(dim=-2)          # (S, d)

print(f"w_dense {tuple(w_dense.shape)}   outs {tuple(outs.shape)}   dense_y {tuple(dense_y.shape)}")
print(f"\nouts[2] - what all three experts computed for token 2 (value 3):")
show_rows(outs[2], "  E")
print(f"\ndense_y[2] = sum_i w_dense[2, i] * outs[2, i] = {fmt(dense_y[2])}")
print(f"  w_dense[2] = {[round(v, 4) for v in w_dense[2].tolist()]}   (no zeros anywhere)")

masked_y = (w_masked.unsqueeze(-1) * outs).sum(dim=-2)        # (S, d)
print(f"\nmasked_y[2] = {fmt(masked_y[2])}   (E1 dropped, E0 and E2 renormalised)")
print(f"  w_masked[2] = {[round(v, 4) for v in w_masked[2].tolist()]}   "
      f"(one exact zero, and the other two rescaled to sum to 1)")

print(f"\nBoth of those built outs, an (S, N, d) = ({S}, {N}, {d}) tensor: "
      f"{S * N} expert-rows.")
print("The masked version then multiplies a third of them by zero. The FLOPs were")
print("already spent. That is the tensor the dispatch loop below never allocates.")

# Where the toy lies, measured rather than asserted. Dropping E1 barely moved the
# answer above - but x(e+1) experts are COLLINEAR, so shifting weight between them
# can only rescale one direction, never point the result somewhere new.
shift = (masked_y[2] - dense_y[2]).norm() / dense_y[2].norm()
print(f"\ncos(E0 out, E2 out) for the toy = "
      f"{F.cosine_similarity(outs[2, 0], outs[2, 2], dim=0):.4f}  <- collinear")
print(f"  dropping E1 (2.8% of the mass) moved token 2 by {100 * shift:.1f}%")

torch.manual_seed(0)
real = [Expert(64, 128) for _ in range(N)]
h = torch.randn(1, 64)
O = torch.stack([e(h)[0] for e in real]).detach()               # (N, 64)
print(f"cos for real Expert MLPs      = "
      f"{F.cosine_similarity(O[0], O[2], dim=0):.4f}  <- nearly orthogonal")
for name, lg in [("peaked [1.0, 0.5, 4.0]", torch.tensor([1., .5, 4.])),
                 ("flat   [1.0, 0.9, 1.1]", torch.tensor([1., .9, 1.1]))]:
    wd = F.softmax(lg, -1)
    top = wd.topk(k).indices
    wm = torch.zeros(N).scatter(0, top, wd[top])
    wm = wm / wm.sum()
    dy, my = (wd[:, None] * O).sum(0), (wm[:, None] * O).sum(0)
    print(f"  {name}: dropped {100 * (1 - wd[top].sum()):4.1f}% of the mass "
          f"-> output moved {100 * (my - dy).norm() / dy.norm():5.1f}%")

print("\nSo top-k's cost depends on the discarded mass AND on how differently the")
print("experts answer. The toy pins the second factor at zero, which is fine for")
print("tracing the mechanism and useless for judging the accuracy hit.")


# ------------------------------------------------------- 5. tok and slot
hdr("5. tok, slot = (topi == e_id).nonzero(as_tuple=True)")

print("topi, as a table:  rows are tokens, columns are rank\n")
print("        rank 0   rank 1")
for s in range(S):
    print(f"  row {s}:  E{topi[s, 0].item()}       E{topi[s, 1].item()}")

print("\nnonzero on (topi == e_id) reads that table down the columns instead:\n")
for e_id in range(N):
    tok, slot = (topi == e_id).nonzero(as_tuple=True)
    pairs = ", ".join(f"row {t} @ rank {s}" for t, s in zip(tok.tolist(), slot.tolist()))
    print(f"  E{e_id}:  tok={tok.tolist()}  slot={slot.tolist()}   S_e={tok.numel()}")
    print(f"        {pairs}")

load = load_per_expert(topi, N)
print(f"\nrows per expert {load.tolist()}, summing to {int(load.sum())} = S*k = {S * k}")
print("Ragged by default: nothing balances these, which is what the aux loss in")
print("step 8 is for. Note S_e is a Python int here - the shape of each expert's")
print("matmul depends on the DATA, which is exactly what makes this hard to batch.")


# ------------------------------------------------------- 6. the loop itself
hdr("6. the dispatch loop, iteration by iteration")

y = torch.zeros_like(xf)                                      # (S, d) accumulator
rows_computed = 0

for e_id in range(N):
    tok, slot = (topi == e_id).nonzero(as_tuple=True)
    print(f"\n--- e_id = {e_id} " + "-" * 52)
    print(f"  tok  = {tok.tolist()}      which rows chose E{e_id}")
    print(f"  slot = {slot.tolist()}      which rank E{e_id} held for each of them")

    if tok.numel() == 0:
        print("  S_e = 0 -> continue. No kernel is launched at all.")
        continue

    gathered_rows = xf[tok]                                   # (S_e, d)
    print(f"  {'xf[tok]':<22}{str(tuple(gathered_rows.shape)):<8} "
          f"{[r[0].item() for r in gathered_rows]}  (showing column 0 only)")

    out = expert(e_id, gathered_rows)                         # (S_e, d)
    rows_computed += tok.numel()
    print(f"  {f'expert{e_id}(xf[tok])':<22}{str(tuple(out.shape)):<8} "
          f"{[r[0].item() for r in out]}")

    w = topw[tok, slot, None]                                 # (S_e, 1)
    print(f"  {'topw[tok, slot, None]':<22}{str(tuple(w.shape)):<8} "
          f"{[round(v.item(), 4) for v in w]}")
    second = (slot != 0).nonzero()
    if second.numel():
        j = second[0, 0].item()
        print(f"    NOT topw[tok, 0]: row {tok[j].item()} picked E{e_id} second, so it "
              f"needs {topw[tok[j], 1]:.4f}, not {topw[tok[j], 0]:.4f}.")
    else:
        print(f"    every row here picked E{e_id} first, so topw[tok, 0] would happen to "
              f"be right - which is how that bug survives review.")

    y.index_add_(0, tok, w * out)
    print(f"  y.index_add_(0, tok, w * out):")
    show_rows(y, "  y", mark=set(tok.tolist()))

print(f"\nexpert-rows computed: {rows_computed} = S*k = {S * k}   "
      f"(dense/masked did {S * N})")
print("\nEach row of y is touched exactly k times, once per expert it chose - which")
print("is why this is index_add_ and not assignment. The += IS the sum over the")
print("top-k set; there is no other line where the mixture happens.")


# ---------------------------------------------------------- 7. the check
hdr("7. same answer as the masked version, one third less work")

print(f"max |dispatch - masked| = {(y - masked_y).abs().max().item():.2e}\n")
show_rows(y, "dispatch")
print()
show_rows(masked_y, "masked  ")

print(f"\nBy hand, row 0: it chose E0 at rank 0 (w=0.7311) and E1 at rank 1 (w=0.2689).")
print(f"  expert0(1) = 1 * 1 = 1      expert1(1) = 2 * 1 = 2")
print(f"  0.7311 * 1 + 0.2689 * 2 = {0.7311 * 1 + 0.2689 * 2:.4f}   "
      f"and y[0] = {y[0, 0].item():.4f}")

y_out = y.reshape(B, T, d)                                    # (B, T, d)
print(f"\ny.reshape(B, T, d) -> {tuple(y_out.shape)}, and y_out[1, 1] is row 4: "
      f"{torch.equal(y_out[1, 1], y[4])}")


# ------------------------------------------------------------- 8. the loss
hdr("8. load_per_expert and the aux loss, on these six rows")

P = F.softmax(logits, dim=-1).mean(0)                         # (N,) differentiable
f = F.one_hot(topi, N).sum(1).float().mean(0)                 # (N,) not differentiable
aux = switch_aux_loss(logits, topi, N)

print(f"  f (fraction of tokens dispatched)  {[round(v, 4) for v in f.tolist()]}   "
      f"sums to {f.sum():.1f} = k")
print(f"  P (mean routing probability)       {[round(v, 4) for v in P.tolist()]}   "
      f"sums to {P.sum():.1f}")
print(f"  aux = N * sum_i f_i * P_i = {N} * {(f * P).sum():.4f} = {aux:.4f}")
print(f"  minimum is k = {k}, reached only when both vectors are flat "
      f"(f_i = k/N = {k/N:.3f}, P_i = 1/N = {1/N:.3f})")

flat = torch.zeros(S, N)                                      # every logit equal
ti_flat = flat.topk(k, -1).indices
print(f"\n  with all-equal logits: aux = {switch_aux_loss(flat, ti_flat, N):.4f} "
      f"(= k, the floor)")
collapsed = torch.tensor([[9.0, 0.0, 0.0]] * S)               # everyone wants E0
ti_col = collapsed.topk(k, -1).indices
print(f"  with everyone on E0:   aux = {switch_aux_loss(collapsed, ti_col, N):.4f} "
      f"(ceiling is N = {N})")

print("\nf is a count, so top-k makes it non-differentiable - it acts as a fixed")
print("per-expert weight. All the gradient flows through P: dL/dP_i = N * f_i,")
print("so the busiest expert gets the strongest push DOWN on its probability.")


# --------------------------------------------------------- 9. the silent bug
hdr("9. what topw[tok, 0] would have done")

wrong = torch.zeros_like(xf)
for e_id in range(N):
    tok, slot = (topi == e_id).nonzero(as_tuple=True)
    if tok.numel() == 0:
        continue
    wrong.index_add_(0, tok, topw[tok, 0, None] * expert(e_id, xf[tok]))   # BUG

print(f"max |wrong - y| = {(wrong - y).abs().max().item():.4f}   "
      f"(no crash, no shape error)\n")
show_rows(wrong, "wrong")
print("\nEvery row still gets contributions from exactly the right two experts -")
print("the rank-1 one just arrives carrying the rank-0 weight, so a row's weights")
print("no longer sum to 1. Row 0 becomes 0.7311*1 + 0.7311*2 = 2.1932 instead of")
print("0.7311*1 + 0.2689*2 = 1.2689. It trains. It converges. It optimises")
print("something else. tok and slot are two different coordinates and the code")
print("has to say so.")


# ------------------------------------------------------- 10. the real layer
hdr("10. the same loop against the real SparseMoE")

torch.manual_seed(0)
moe = SparseMoE(d, d_ff=8, n_experts=N, k=k)
real_y, real_aux = moe(x)

# redo steps 1-7 by hand, this time with the module's own gate and MLP experts
xf2 = x.reshape(-1, d)
lg2 = moe.gate(xf2)
tl2, ti2 = lg2.topk(k, dim=-1)
tw2 = F.softmax(tl2, dim=-1)
hand = torch.zeros_like(xf2)
for e_id, e in enumerate(moe.experts):
    tok, slot = (ti2 == e_id).nonzero(as_tuple=True)
    if tok.numel() == 0:
        continue
    hand.index_add_(0, tok, tw2[tok, slot, None] * e(xf2[tok]))

print(f"max |hand-rolled - SparseMoE(x)| = "
      f"{(hand.reshape(B, T, d) - real_y).abs().max().item():.2e}")
print(f"aux: hand {switch_aux_loss(lg2, ti2, N):.4f}   module {real_aux:.4f}")
assert torch.allclose(hand.reshape(B, T, d), real_y, atol=1e-6)

load2 = load_per_expert(ti2, N)
starved = [i for i, v in enumerate(load2.tolist()) if v == 0]
print(f"\nrows per expert with an untrained gate: {load2.tolist()}"
      + (f"   -> E{starved[0]} is starved, so `if tok.numel() == 0: continue` fires"
         if starved else ""))
print(f"params {sum(p.numel() for p in moe.parameters()):,}   "
      f"MACs/token {moe.macs_per_token():,}")

print("\nSame nine lines, real weights. Swapping the x3 toy for an MLP changed")
print("nothing about the routing - which is the point of writing it this small.")
print("Next: python SparseMoe/run_sparse_moe.py, where the numbers get big enough")
print("for the savings to be worth measuring.")
print()
