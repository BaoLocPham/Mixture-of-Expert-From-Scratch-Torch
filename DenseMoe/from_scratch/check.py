"""
Grader for dense_moe.py.

    python DenseMoe/from_scratch/check.py

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
import dense_moe as sol

torch.set_printoptions(precision=6, sci_mode=False)

# ---------------------------------------------------------------- fixtures
X = torch.tensor([[[0.5, -1.0, 2.0],
                   [-0.25, 0.75, 0.0]]])                    # (1, 2, 3)

EXPERT_Y = torch.tensor([[[0.00385483, 0.02611127, 0.04836771],
                          [0.00651657, 0.00764049, 0.00876441]]])
EXPERT_Y_RELU = torch.tensor([[[-0.04800000, 0.03600000, 0.11999999],
                               [-0.01350000, 0.01050000, 0.03449999]]])
GATE_W = torch.tensor([[[0.38936079, 0.61063927],
                        [0.46257013, 0.53742981]]])
MOE_Y = torch.tensor([[[-0.10443806, -0.01300277, 0.07843254],
                       [-0.02054019, -0.00136236, 0.01781548]]])


class Fail(Exception):
    def __init__(self, msg, hint=None):
        super().__init__(msg)
        self.hint = hint


def need(cond, msg, hint=None):
    if not cond:
        raise Fail(msg, hint)


def close(a, b, tol=1e-5):
    return torch.allclose(torch.as_tensor(a), torch.as_tensor(b), atol=tol)


BROADCAST_HINT = (
    "That's a broadcasting error, and it's the classic one here. Broadcasting "
    "lines axes up from the RIGHT, so a (B,T,N) tensor meets a (B,T,N,d) one "
    "as N-vs-d - which only 'works' when N happens to equal d, and is wrong "
    "even then. The weight tensor needs a trailing size-1 axis so N meets N "
    "and the scalar spreads across all d components."
)


def call(fn, *a, **kw):
    """Run a student function, turning shape RuntimeErrors into a real hint."""
    try:
        return fn(*a, **kw)
    except RuntimeError as ex:
        msg = str(ex)
        if "size" in msg or "broadcast" in msg or "shape" in msg:
            raise Fail(f"raised a shape error:\n   {msg.splitlines()[0]}", BROADCAST_HINT)
        raise


def set_expert_weights(e, j=0):
    with torch.no_grad():
        e.w1.weight.copy_((torch.arange(12.).reshape(4, 3) * 0.05) - 0.3 + j * 0.2)
        e.w2.weight.copy_((torch.arange(12.).reshape(3, 4) * 0.04) - 0.2 - j * 0.1)


def fixed_moe():
    """A DenseMoE(3, 4, 2) with fully deterministic weights."""
    m = sol.DenseMoE(3, 4, 2)
    with torch.no_grad():
        m.gate.weight.copy_(torch.arange(6.).reshape(2, 3) * 0.1 - 0.3)
    for j, e in enumerate(m.experts):
        set_expert_weights(e, j)
    return m


# ------------------------------------------------------------------ stages
def stage_0a():
    e = sol.Expert(3, 4)
    need(hasattr(e, "w1") and hasattr(e, "w2"),
         "Expert must have attributes named w1 and w2",
         "The grader and run_dense_moe.py both reach in by name.")
    need(isinstance(e.w1, nn.Linear) and isinstance(e.w2, nn.Linear),
         "w1 and w2 must be nn.Linear modules")
    need(tuple(e.w1.weight.shape) == (4, 3),
         f"w1.weight should be (d_ff, d_model) = (4, 3), got {tuple(e.w1.weight.shape)}",
         "nn.Linear(in_features, out_features) stores .weight as (out, in). "
         "If you got (3, 4) you passed the dimensions the other way round.")
    need(tuple(e.w2.weight.shape) == (3, 4),
         f"w2.weight should be (d_model, d_ff) = (3, 4), got {tuple(e.w2.weight.shape)}")
    need(e.w1.bias is None and e.w2.bias is None,
         "the experts must be bias-free", "pass bias=False to nn.Linear.")


def stage_0b():
    e = sol.Expert(3, 4)
    set_expert_weights(e)
    y = e(X)
    need(tuple(y.shape) == (1, 2, 3),
         f"Expert.forward must preserve shape: expected (1, 2, 3), got {tuple(y.shape)}")
    if not close(y, EXPERT_Y):
        hint = "The order is w1 -> activation -> w2."
        if close(y, EXPERT_Y_RELU):
            hint = ("Those are exactly the numbers you get with ReLU. The spec "
                    "says SiLU (a.k.a. swish) - F.silu.")
        raise Fail(f"wrong values.\n   expected {EXPERT_Y.tolist()}\n   got      {y.tolist()}",
                   hint)
    big = e(torch.randn(2, 5, 3))
    need(tuple(big.shape) == (2, 5, 3),
         f"must work on any leading axes: (2,5,3) in -> got {tuple(big.shape)}")


def stage_1a():
    m = sol.DenseMoE(3, 4, 2)
    need(isinstance(getattr(m, "gate", None), nn.Linear), "self.gate must be an nn.Linear")
    need(tuple(m.gate.weight.shape) == (2, 3),
         f"gate.weight should be (n_experts, d_model) = (2, 3), got {tuple(m.gate.weight.shape)}",
         "The gate maps a d_model vector to one score per expert.")
    need(m.gate.bias is None, "the gate must be bias-free")
    need(isinstance(getattr(m, "experts", None), nn.ModuleList),
         f"self.experts must be an nn.ModuleList, got {type(getattr(m, 'experts', None)).__name__}",
         "A plain python list looks fine but silently hides the experts from "
         ".parameters(), so they never train. Worth confirming yourself: swap "
         "in a list and count len(list(m.parameters())).")
    need(len(m.experts) == 2, f"expected 2 experts, got {len(m.experts)}")
    need(all(isinstance(e, sol.Expert) for e in m.experts), "every entry must be an Expert")
    n = len(list(m.parameters()))
    need(n == 5, f"expected 5 parameter tensors (gate + 2 experts x 2), got {n}")


def stage_2():
    m = fixed_moe()
    w = call(m.gate_weights, X)
    need(tuple(w.shape) == (1, 2, 2),
         f"gate_weights must return (B, T, N) = (1, 2, 2), got {tuple(w.shape)}")
    need(bool((w >= 0).all()), "gate weights must be non-negative")
    s = w.sum(dim=-1)
    if not close(s, torch.ones_like(s)):
        hint = "Softmax over which axis?"
        if close(w.sum(dim=1), torch.ones_like(w.sum(dim=1))):
            hint = ("Your weights sum to 1 across TOKENS, not across experts. "
                    "You normalised dim=1. Routing is a per-token decision: each "
                    "token independently splits its 1.0 among the N experts.")
        raise Fail(f"rows must sum to 1 over the expert axis, got {s.tolist()}", hint)
    need(close(w, GATE_W),
         f"wrong values.\n   expected {GATE_W.tolist()}\n   got      {w.tolist()}")


def stage_3():
    m = fixed_moe()
    outs = call(m.expert_stack, X)
    need(tuple(outs.shape) == (1, 2, 2, 3),
         f"expert_stack must return (B, T, N, d) = (1, 2, 2, 3), got {tuple(outs.shape)}",
         "The expert axis goes at position -2, immediately before d. If you got "
         "(1, 2, 3, 2) you stacked on the last axis; if the 2 landed first you "
         "used the default dim=0.")
    for j, e in enumerate(m.experts):
        need(close(outs[..., j, :], e(X)),
             f"outs[..., {j}, :] should be expert {j}'s output on the whole input")


def stage_4():
    w = torch.tensor([[[0.25, 0.75],
                       [1.00, 0.00]]])                       # (1, 2, 2)
    outs = torch.tensor([[[[1., 2., 3.], [5., 6., 7.]],
                          [[10., 20., 30.], [40., 50., 60.]]]])   # (1, 2, 2, 3)
    expect = torch.tensor([[[4., 5., 6.],
                            [10., 20., 30.]]])
    y = call(sol.DenseMoE.combine, w, outs)
    if tuple(y.shape) != (1, 2, 3):
        hint = "You want (B, T, d): the expert axis has to disappear."
        if tuple(y.shape) == (1, 2, 2, 3):
            hint = "You multiplied but never summed the expert axis away."
        elif tuple(y.shape) == (1, 2, 2):
            hint = "You summed the wrong axis - collapse -2 (experts), not -1 (d)."
        raise Fail(f"expected shape (1, 2, 3), got {tuple(y.shape)}", hint)
    need(close(y, expect),
         f"wrong values.\n   expected {expect.tolist()}\n   got      {y.tolist()}",
         "Check by hand: token 0 is 0.25*[1,2,3] + 0.75*[5,6,7]. If your numbers "
         "look shuffled, the broadcast lined w's N up against outs's d.")


def stage_5():
    m = fixed_moe()
    y = call(m, X)
    need(tuple(y.shape) == (1, 2, 3), f"forward must return (1, 2, 3), got {tuple(y.shape)}")
    need(close(y, MOE_Y),
         f"wrong values.\n   expected {MOE_Y.tolist()}\n   got      {y.tolist()}")

    torch.manual_seed(0)
    r = sol.DenseMoE(4, 8, 3)
    xr = torch.randn(2, 3, 4)
    yr, outs = call(r, xr), r.expert_stack(xr)
    lo, hi = outs.min(dim=-2).values, outs.max(dim=-2).values
    need(bool(((yr >= lo - 1e-5) & (yr <= hi + 1e-5)).all()),
         "output escaped the convex hull of the expert outputs",
         "With non-negative weights summing to 1 the result is a weighted "
         "average, so it can never exceed the per-coordinate min/max. If it "
         "does, your weights aren't the normalised ones.")


def stage_6():
    m = fixed_moe()
    y = call(m.forward_loop, X)
    need(tuple(y.shape) == (1, 2, 3), f"forward_loop must return (1, 2, 3), got {tuple(y.shape)}")
    need(close(y, MOE_Y),
         f"wrong values.\n   expected {MOE_Y.tolist()}\n   got      {y.tolist()}")

    torch.manual_seed(1)
    r = sol.DenseMoE(4, 5, 3)
    xr = torch.randn(2, 3, 4)
    need(close(r.forward_loop(xr), r(xr)),
         "forward_loop and forward disagree on a random input",
         "They must be the same function. The loop is the one you can read "
         "line by line - trust it and fix the vectorised one.")


def stage_7():
    m = fixed_moe()
    y = call(m.forward_einsum, X)
    need(tuple(y.shape) == (1, 2, 3), f"forward_einsum must return (1, 2, 3), got {tuple(y.shape)}")
    need(close(y, MOE_Y),
         f"wrong values.\n   expected {MOE_Y.tolist()}\n   got      {y.tolist()}",
         "The summed index is the one on both inputs but not the output.")
    src = Path(__file__).resolve().parent.joinpath("dense_moe.py").read_text()
    body = src.split("def forward_einsum")[1].split("def macs_per_token")[0]
    need("einsum" in body, "stage 7 is meant to use torch.einsum")


def stage_8():
    for (d, d_ff, n, want) in [(3, 4, 2, 54), (8, 16, 3, 792), (512, 2048, 8, 16781312)]:
        m = sol.DenseMoE(d, d_ff, n)
        got = m.macs_per_token()
        need(isinstance(got, int), f"macs_per_token must return an int, got {type(got).__name__}")
        if got != want:
            hint = "Per token: the gate is d*N, and each of the N experts is two Linears."
            if got == want - d * n:
                hint = "Close - you forgot the gate itself."
            elif got == n * d * d_ff + d * n:
                hint = "Each expert has TWO Linear layers, d->d_ff and d_ff->d."
            elif got == 2 * want or got == 2 * (want - d * n) + d * n:
                hint = "Count multiply-accumulates, not separate multiplies and adds."
            raise Fail(f"DenseMoE({d}, {d_ff}, {n}).macs_per_token() = {got}, expected {want}",
                       hint)
    m = sol.DenseMoE(8, 16, 3)
    params = sum(p.numel() for p in m.parameters())
    print(f"      note: {m.macs_per_token()} MACs/token vs {params} params - "
          f"equal, and that is the bottleneck.")


def stage_9():
    torch.manual_seed(2)
    d, d_ff, n = 4, 6, 3
    dense = sol.DenseMoE(d, d_ff, n)
    batched = sol.BatchedDenseMoE(d, d_ff, n).copy_from(dense)
    xr = torch.randn(2, 5, d)
    yb, yd = call(batched, xr), dense(xr)
    need(tuple(yb.shape) == tuple(yd.shape),
         f"shape mismatch: batched {tuple(yb.shape)} vs dense {tuple(yd.shape)}")
    need(close(yb, yd, tol=1e-5),
         f"batched output differs from DenseMoE by up to {(yb - yd).abs().max().item():.2e}",
         "Same math, different layout. Remember copy_from transposes: W1[j] is "
         "(d_model, d_ff), so x @ W1[j] works directly with no .t().")
    src = Path(__file__).resolve().parent.joinpath("dense_moe.py").read_text()
    body = src.split("class BatchedDenseMoE")[1]
    need("for " not in body.split("def forward")[-1],
         "stage 9 should have no python loop over experts in forward")


STAGES = [
    ("0a  Expert.__init__", stage_0a),
    ("0b  Expert.forward", stage_0b),
    ("1a  DenseMoE.__init__", stage_1a),
    ("2   gate_weights", stage_2),
    ("3   expert_stack", stage_3),
    ("4   combine", stage_4),
    ("5   forward", stage_5),
    ("6   forward_loop", stage_6),
    ("7   forward_einsum", stage_7),
    ("8   macs_per_token", stage_8),
]


def main():
    print("\n  DenseMoE from scratch\n  " + "-" * 44)
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

    print("\n  All core stages pass. Now run the dissector on YOUR code:\n")
    print("      MOE_IMPL=scratch python DenseMoe/run_dense_moe.py\n")

    try:
        stage_9()
        print("  ok  9   BatchedDenseMoE  (stretch)\n")
    except NotImplementedError:
        print("  --  9   BatchedDenseMoE  (stretch, not attempted)\n")
    except Fail as ex:
        print(f"  XX  9   BatchedDenseMoE  (stretch)\n\n  {ex}")
        if ex.hint:
            print(f"\n  hint: {ex.hint}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
