"""
Grader for sparse_moe.py.

    python SparseMoe/from_scratch/check.py

Runs the stages in order and stops at the first one that isn't done or isn't
right. It tells you WHAT is wrong and usually WHY, never the answer.

The expected numbers baked in below are just outputs - reading them won't tell
you how to produce them.
"""

import sys
import traceback
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sparse_moe as sol

torch.set_printoptions(precision=6, sci_mode=False)

# ---------------------------------------------------------------- fixtures
X = torch.tensor([[[0.5, -1.0, 2.0, 0.25],
                   [-0.25, 0.75, 0.0, -1.5]]])                   # (1, 2, 4)
D, D_FF, N, K = 4, 6, 3, 2

MOE_Y = torch.tensor([[[-1.00168681, 0.08526988, 1.17222667, 2.25918341],
                       [0.09352022, -0.12050070, -0.33452165, -0.54854256]]])
AUX = 1.97472119
TOPW = torch.tensor([[0.62010640, 0.37989357], [0.56954622, 0.43045378]])
TOPI = torch.tensor([[2, 1], [0, 1]])
LOAD = torch.tensor([1, 2, 1])


class Fail(Exception):
    def __init__(self, msg, hint=None):
        super().__init__(msg)
        self.hint = hint


def need(cond, msg, hint=None):
    if not cond:
        raise Fail(msg, hint)


def close(a, b, tol=1e-5):
    return torch.allclose(torch.as_tensor(a), torch.as_tensor(b), atol=tol)


def call(fn, *a, **kw):
    try:
        out = fn(*a, **kw)
    except RuntimeError as ex:
        msg = str(ex)
        if "size" in msg or "broadcast" in msg or "shape" in msg:
            raise Fail(f"raised a shape error:\n   {msg.splitlines()[0]}",
                       "Line the shapes up on paper before writing the line. A "
                       "per-token scalar needs a trailing size-1 axis to spread "
                       "across d.")
        raise
    if out is None:
        raise Fail("returned None instead of a value",
                   "A function with no `return` hands back None.")
    return out


def fix(m):
    """Deterministic weights, independent of construction order."""
    with torch.no_grad():
        m.gate.weight.copy_(torch.arange(float(N * D)).reshape(N, D) * 0.07 - 0.4)
        for j, e in enumerate(m.experts):
            e.w1.weight.copy_(torch.arange(float(D_FF * D)).reshape(D_FF, D) * 0.05 - 0.3 + j * 0.15)
            e.w2.weight.copy_(torch.arange(float(D * D_FF)).reshape(D, D_FF) * 0.04 - 0.2 - j * 0.1)
    return m


def logits_of(m):
    return m.gate(X.reshape(-1, D))


# ------------------------------------------------------------------ stages
def stage_1():
    m = fix(sol.MaskedSparseMoE(D, D_FF, N, K))
    lg = logits_of(m)
    out = call(sol.top_k_gate, lg, K)
    need(isinstance(out, (tuple, list)) and len(out) == 2,
         "top_k_gate must return a (topw, topi) pair")
    topw, topi = out
    need(tuple(topw.shape) == (2, K) and tuple(topi.shape) == (2, K),
         f"both must be (S, k) = (2, {K}); got {tuple(topw.shape)}, {tuple(topi.shape)}",
         "If you got (2, N) you returned the full distribution instead of the top-k.")
    need(close(topi.long(), TOPI), f"wrong expert ids.\n   expected {TOPI.tolist()}\n"
         f"   got      {topi.tolist()}",
         "torch.topk returns (values, indices) in that order - check you didn't swap them.")
    s = topw.sum(-1)
    if not close(s, torch.ones_like(s)):
        raise Fail(f"each row of topw must sum to 1, got {s.tolist()}",
                   "Softmax over the k survivors, i.e. the last axis of the (S, k) "
                   "tensor - not over the original N logits.")
    need(close(topw, TOPW), f"wrong weights.\n   expected {TOPW.tolist()}\n"
         f"   got      {topw.tolist()}")


def stage_2():
    got = call(sol.load_per_expert, TOPI, N)
    need(tuple(got.shape) == (N,), f"expected shape ({N},), got {tuple(got.shape)}")
    need(close(got.long(), LOAD),
         f"expected {LOAD.tolist()}, got {got.tolist()}",
         "Each of the S*k picks contributes 1 to its expert. Your counts must sum "
         f"to S*k = {int(LOAD.sum())}.")
    big = call(sol.load_per_expert, torch.randint(0, 5, (64, 3)), 5)
    need(int(big.sum()) == 64 * 3,
         f"counts must sum to S*k = 192, got {int(big.sum())}")
    src = Path(__file__).resolve().parent.joinpath("sparse_moe.py").read_text()
    body = src.split("def load_per_expert")[1].split("def switch_aux_loss")[0]
    need("for " not in body, "do it without a python loop over experts")


def stage_3():
    m = fix(sol.MaskedSparseMoE(D, D_FF, N, K))
    got = call(sol.switch_aux_loss, logits_of(m), TOPI, N)
    need(got.dim() == 0, f"must return a scalar, got shape {tuple(got.shape)}")
    if not close(got, torch.tensor(AUX), tol=1e-4):
        hint = "P is a softmax averaged over tokens; f is the dispatch fraction."
        if close(got, torch.tensor(AUX / N), tol=1e-4):
            hint = "You forgot the leading factor of N."
        raise Fail(f"expected {AUX:.6f}, got {got.item():.6f}", hint)

    # uniform load must bottom out at exactly k
    S = 600
    uni_logits = torch.zeros(S, 8)
    uni_topi = torch.stack([torch.arange(S) % 8, (torch.arange(S) + 1) % 8], -1)
    flat = call(sol.switch_aux_loss, uni_logits, uni_topi, 8)
    need(close(flat, torch.tensor(2.0), tol=1e-3),
         f"at perfectly uniform load with k=2 the loss must be k=2.0, got {flat.item():.4f}",
         "sum_i f_i = k always. With f_i = k/N and P_i = 1/N, N*sum(f*P) = k.")

    # and it must be differentiable through P
    lg = torch.randn(50, 4, requires_grad=True)
    ti = lg.detach().topk(2, -1).indices
    call(sol.switch_aux_loss, lg, ti, 4).backward()
    need(lg.grad is not None and lg.grad.abs().sum() > 0,
         "the loss must be differentiable w.r.t. the logits",
         "If the gradient is zero you detached P, or built the whole thing out of "
         "hard counts. P is the only path the gradient has.")


def stage_4():
    m = fix(sol.MaskedSparseMoE(D, D_FF, N, K))
    out = call(m, X)
    need(isinstance(out, (tuple, list)) and len(out) == 2,
         "forward must return (y, aux)")
    y, aux = out
    need(tuple(y.shape) == (1, 2, D), f"y must be (1, 2, {D}), got {tuple(y.shape)}")
    need(close(y, MOE_Y), f"wrong values.\n   expected {MOE_Y.tolist()}\n"
         f"   got      {y.tolist()}",
         "Renormalise after masking: the surviving k weights have to sum to 1 again.")
    need(close(aux, torch.tensor(AUX), tol=1e-4), f"aux should be {AUX:.6f}, got {aux.item():.6f}")

    # the mask must genuinely cut gradient to unselected experts
    m2 = sol.MaskedSparseMoE(D, D_FF, 8, k=1)
    xr = torch.randn(1, 2, D)
    m2.zero_grad(); m2(xr)[0].sum().backward()
    lg = m2.gate(xr.reshape(-1, D))
    chosen = set(lg.topk(1, -1).indices.flatten().tolist())
    for j, e in enumerate(m2.experts):
        g = 0.0 if e.w1.weight.grad is None else e.w1.weight.grad.abs().sum().item()
        if j not in chosen:
            need(g == 0.0,
                 f"expert {j} was never selected but has gradient {g:.3e}",
                 "The mask has to zero the WEIGHT, not just hide it. If unselected "
                 "experts still get gradient, this isn't sparse in any sense.")

    need(m.macs_per_token() == 156,
         f"macs_per_token should be 156 for d=4 d_ff=6 N=3, got {m.macs_per_token()}",
         "This version runs ALL N experts. Its cost is the dense cost - that is "
         "the entire complaint about it. If you wrote k*2*d*d_ff you described "
         "the version you wish you had written.")


def stage_5():
    trap = fix(sol.MaskedSparseMoE(D, D_FF, N, K))
    m = sol.SparseMoE(D, D_FF, N, K).copy_from(trap)
    out = call(m, X)
    need(isinstance(out, (tuple, list)) and len(out) == 2, "forward must return (y, aux)")
    y, aux = out
    need(tuple(y.shape) == (1, 2, D), f"y must be (1, 2, {D}), got {tuple(y.shape)}")
    if not close(y, MOE_Y, tol=1e-5):
        hint = ("The masked version is the oracle here - it is obviously correct. "
                "If they disagree, this one is wrong.")
        # the classic: topw[tok, 0] instead of topw[tok, slot]
        xf = X.reshape(-1, D)
        lg = m.gate(xf)
        tl, ti = lg.topk(K, -1)
        tw = F.softmax(tl, -1)
        wrong = torch.zeros_like(xf)
        for e_id, e in enumerate(m.experts):
            tok, _ = (ti == e_id).nonzero(as_tuple=True)
            if tok.numel():
                wrong.index_add_(0, tok, tw[tok, 0, None] * e(xf[tok]))
        if close(y, wrong.reshape(1, 2, D), tol=1e-5):
            hint = ("Those are exactly the numbers you get from topw[tok, 0]. You "
                    "need topw[tok, slot]: for a token where this expert was the "
                    "SECOND choice, rank 0 is another expert's weight.")
        else:
            # the other classic: assignment instead of accumulation
            overwrite = torch.zeros_like(xf)
            for e_id, e in enumerate(m.experts):
                tok, slot = (ti == e_id).nonzero(as_tuple=True)
                if tok.numel():
                    overwrite[tok] = tw[tok, slot, None] * e(xf[tok])
            if close(y, overwrite.reshape(1, 2, D), tol=1e-5):
                hint = ("Those are the numbers you get from `y[tok] = ...`. Each "
                        "token appears in k iterations of this loop, so assignment "
                        "keeps only whichever expert happened to come last and "
                        "throws the other k-1 away. The writes have to ACCUMULATE - "
                        "that loop is where the sum over the top-k set happens.")
        raise Fail(f"wrong values.\n   expected {MOE_Y.tolist()}\n   got      {y.tolist()}",
                   hint)
    need(close(aux, torch.tensor(AUX), tol=1e-4), f"aux should be {AUX:.6f}, got {aux.item():.6f}")

    # must agree with the masked version on a bigger random case too
    t2 = sol.MaskedSparseMoE(8, 12, 5, k=2)
    s2 = sol.SparseMoE(8, 12, 5, k=2).copy_from(t2)
    xr = torch.randn(3, 4, 8)
    need(close(s2(xr)[0], t2(xr)[0], tol=1e-5),
         f"disagrees with MaskedSparseMoE on random input by up to "
         f"{(s2(xr)[0] - t2(xr)[0]).abs().max().item():.2e}")

    # k = N must reproduce a plain dense mixture
    t3 = sol.MaskedSparseMoE(8, 12, 4, k=4)
    s3 = sol.SparseMoE(8, 12, 4, k=4).copy_from(t3)
    xr = torch.randn(2, 3, 8)
    xf = xr.reshape(-1, 8)
    w = F.softmax(t3.gate(xf), -1)
    outs = torch.stack([e(xf) for e in t3.experts], dim=-2)
    dense = (w.unsqueeze(-1) * outs).sum(-2).reshape(2, 3, 8)
    need(close(s3(xr)[0], dense, tol=1e-5),
         "with k=N your sparse layer must reproduce a plain dense MoE exactly",
         "Nothing is dropped when k=N, so a softmax over the top-N logits is just "
         "the softmax over all N.")

    # no expert may run on rows not routed to it
    calls = {}
    probe = sol.SparseMoE(8, 12, 6, k=1)
    for j, e in enumerate(probe.experts):
        def mk(j, orig):
            def f(inp):
                calls[j] = calls.get(j, 0) + inp.shape[0]
                return orig(inp)
            return f
        e.forward = mk(j, e.forward)
    xr = torch.randn(2, 5, 8)
    probe(xr)
    lg = probe.gate(xr.reshape(-1, 8))
    want = sol.load_per_expert(lg.topk(1, -1).indices, 6)
    for j in range(6):
        need(calls.get(j, 0) == int(want[j]),
             f"expert {j} was given {calls.get(j, 0)} rows but only {int(want[j])} "
             f"were routed to it",
             "Every expert must see exactly its own S_e rows - no more (that is "
             "the trap) and no fewer. An expert with S_e=0 must not run at all.")

    need(m.macs_per_token() == 108,
         f"macs_per_token should be 108 for d=4 d_ff=6 N=3 k=2, got {m.macs_per_token()}",
         "Gate plus k experts, not N.")
    params = sum(p.numel() for p in m.parameters())
    print(f"      note: {params} params vs {m.macs_per_token()} MACs/token - "
          f"they finally diverge.")


STAGES = [
    ("1   top_k_gate", stage_1),
    ("2   load_per_expert", stage_2),
    ("3   switch_aux_loss", stage_3),
    ("4   MaskedSparseMoE", stage_4),
    ("5   SparseMoE (dispatch)", stage_5),
]


def main():
    print("\n  SparseMoE from scratch\n  " + "-" * 44)
    for name, fn in STAGES:
        try:
            fn()
        except NotImplementedError:
            print(f"  ..  {name}")
            print(f"\n  ^ next up: implement this one, then re-run.\n")
            return 1
        except Fail as ex:
            print(f"  XX  {name}")
            print(f"\n  {ex}")
            if ex.hint:
                print(f"\n  hint: {ex.hint}")
            print()
            return 1
        except Exception:
            print(f"  XX  {name}  (raised)\n")
            traceback.print_exc()
            return 1
        print(f"  ok  {name}")

    print("\n  All stages pass. Now run the dissector on YOUR code:\n")
    print("      MOE_IMPL=scratch python SparseMoe/run_sparse_moe.py\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
