"""
Grader for attention.py.

    python LLM/from_scratch/check_attention.py

Four stages, in dependency order, stopping at the first one that isn't done or
isn't right. It says WHAT is wrong and usually WHY, never the answer.

Stages 1, 3 and 4 are checked against baked-in outputs, including the outputs
of four specific WRONG implementations, so the grader can name the mistake you
made. Stage 2 is index arithmetic, so it is checked by property - and stage 4
is checked against the code YOU wrote in stages 2 and 3 as well as by value.
"""

import sys
import traceback
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import attention as sol

torch.set_printoptions(precision=6, sci_mode=False)

# ---------------------------------------------------------------- fixtures
D, H, DH, T, MAXSEQ = 8, 2, 4, 5, 12
Q = (torch.arange(1 * H * T * DH).float().reshape(1, H, T, DH) * 0.11 - 1.0).sin() * 0.9
K = (torch.arange(1 * H * T * DH).float().reshape(1, H, T, DH) * 0.07 + 0.4).cos() * 0.8
V = (torch.arange(1 * H * T * DH).float().reshape(1, H, T, DH) * 0.13 - 0.3).sin() * 1.1
X = (torch.arange(1 * T * D).float().reshape(1, T, D) * 0.13 - 0.9).sin() * 1.4


class Cfg:
    """The three fields attention.py's __init__ is allowed to read."""
    d_model, n_heads, d_head = D, H, DH


class Fail(Exception):
    def __init__(self, msg, hint=None):
        super().__init__(msg)
        self.hint = hint


def need(cond, msg, hint=None):
    if not cond:
        raise Fail(msg, hint)


def close(a, b, tol=1e-5):
    a, b = torch.as_tensor(a), torch.as_tensor(b)
    return a.shape == b.shape and torch.allclose(a, b.to(a.dtype), atol=tol)


def flat(a, b, tol=1e-5):
    a, b = torch.as_tensor(a).flatten(), torch.as_tensor(b).flatten()
    return a.shape == b.shape and torch.allclose(a, b.to(a.dtype), atol=tol)


def call(fn, *a, **kw):
    try:
        out = fn(*a, **kw)
    except NotImplementedError:
        raise
    except RuntimeError as ex:
        first = str(ex).splitlines()[0]
        raise Fail(f"raised a shape error:\n   {first}",
                   "Write the shapes next to each line before you run it.")
    if out is None:
        raise Fail("returned None", "A function with no `return` hands back None.")
    return out


def fix(module):
    """Deterministic weights, keyed by parameter NAME so construction order
    cannot change them."""
    with torch.no_grad():
        for i, (name, p) in enumerate(sorted(module.named_parameters())):
            v = torch.sin(torch.arange(p.numel()).float() * 0.7 + 1.3 * i) * 0.3
            p.copy_(v.reshape(p.shape))
    return module


# ------------------------------------------------------- expected (outputs)
PE_ROW0 = PE_ROW1 = PE_LAST = None
SDP = SDP_NOSCALE = SDP_NOMASK = SDP_MASKAFTER = None
ATT = ATT_BADMERGE = None
exec(open(Path(__file__).with_name("expected_attention.py")).read())


# ------------------------------------------------------------------ stages
def stage_1():
    pe = torch.as_tensor(call(sol.sinusoidal_pe, MAXSEQ, D))
    need(tuple(pe.shape) == (MAXSEQ, D),
         f"returned {tuple(pe.shape)}, expected {(MAXSEQ, D)}",
         "This table is added straight onto the embedding, so it is d_model "
         "wide - not d_model/2. Each angle is used twice: sin in the even "
         "channel, cos in the odd one.")

    if flat(pe[0], torch.zeros(D), 1e-6):
        raise Fail("row 0 is all zeros",
                   "Angle 0 gives sin 0 AND cos 1, so row 0 alternates "
                   "[0, 1, 0, 1, ...]. All zeros means the cosine half never "
                   "got written - check which channels you assigned.")
    need(flat(pe[0], PE_ROW0, 1e-6), "row 0 is wrong",
         "At m = 0 every angle is 0. sin(0) = 0 into the even channels, "
         "cos(0) = 1 into the odd ones.")
    need(flat(pe[1], PE_ROW1, 1e-5), "row 1 is wrong",
         "Row 1 IS the frequency ladder: sin(1/base**(2j/d)) and its cosine, "
         "pair by pair. If the first entry is not sin(1) = 0.8415 the "
         "exponent is off; if the pairs are not adjacent the interleave is.")
    need(flat(pe[MAXSEQ - 1], PE_LAST, 1e-5),
         f"row {MAXSEQ - 1} is wrong while row 1 is right",
         "The rows are m / base**(2j/d) for m = 0 .. max_seq-1. If only the "
         "later rows are off, you broadcast a position against the wrong axis.")

    pe2 = torch.as_tensor(call(sol.sinusoidal_pe, MAXSEQ, D, 100.0))
    need(not close(pe2[1], pe[1], 1e-4), "changing `base` changed nothing",
         "base sets the bottom of the frequency ladder. If it is ignored, it "
         "is hard-coded somewhere.")
    return f"position table: {(MAXSEQ, D)}, sin/cos interleaved, row 0 = [0, 1, ...]"


def stage_2():
    flat_row = torch.arange(1 * 1 * (H * DH)).float().reshape(1, 1, H * DH)
    heads = call(sol.split_heads, flat_row, H)
    need(tuple(heads.shape) == (1, H, 1, DH),
         f"split_heads returned {tuple(heads.shape)}, expected {(1, H, 1, DH)}",
         "(B, T, n_h*d_h) -> (B, n_h, T, d_h): cut the last axis into n_h "
         "blocks, then bring the head axis in front of T.")

    want = flat_row.reshape(H, DH)
    if flat(heads[0, :, 0], flat_row.reshape(DH, H).t().reshape(-1), 1e-6):
        raise Fail("head h got the STRIDED columns, not the contiguous block",
                   f"With d_h = {DH}, head 0 must be {list(range(DH))}, not "
                   f"{list(range(0, H * DH, H))[:DH]}. You cut the last axis "
                   "into (d_h, n_h) instead of (n_h, d_h) - same shape after "
                   "the transpose, no error, a different model.")
    need(flat(heads[0, :, 0], want, 1e-6), "split_heads puts the wrong numbers "
         "in the wrong heads",
         "Head h is exactly columns h*d_h : (h+1)*d_h of the row.")

    x4 = (torch.arange(2 * 3 * (H * DH)).float().reshape(2, 3, H * DH) * 0.1).sin()
    sp = call(sol.split_heads, x4, H)
    need(tuple(sp.shape) == (2, H, 3, DH),
         f"on a (2, 3, {H * DH}) input it returned {tuple(sp.shape)}")
    for h in range(H):
        need(close(sp[:, h], x4[..., h * DH:(h + 1) * DH], 1e-6),
             f"head {h} is not columns {h * DH}:{(h + 1) * DH} of the input")

    back = call(sol.merge_heads, sp)
    need(tuple(back.shape) == tuple(x4.shape),
         f"merge_heads returned {tuple(back.shape)}, expected {tuple(x4.shape)}")
    need(close(back, x4, 1e-6),
         "split then merge did not return the original",
         "They have to be exact inverses. If the numbers come back shuffled, "
         "one of the two is missing its transpose, or doing it in the wrong "
         "order relative to the reshape.")
    need(back.is_contiguous(),
         "merge_heads returned a non-contiguous tensor",
         "You transposed but never materialised the result - which means you "
         "used view() where reshape() was needed, or skipped the copy "
         "entirely. Downstream Linear layers will still work, so nothing "
         "will tell you.")
    return "heads: contiguous blocks, head axis in front, and an exact inverse"


def stage_3():
    got = call(sol.scaled_dot_product, Q, K, V)
    need(tuple(got.shape) == tuple(Q.shape),
         f"returned {tuple(got.shape)}, expected {tuple(Q.shape)}",
         "The output is one d_h-wide row per query: (B, H, T, d_h). A "
         "(B, H, T, T) result means the weighted sum never happened.")

    if flat(got, SDP_NOSCALE, 1e-5):
        raise Fail("the scores are not divided by sqrt(d_h)",
                   "Without it the score scale grows with head width and the "
                   "softmax saturates toward one-hot, where its gradient is "
                   "nearly zero. It trains, badly, and wider heads train worse.")
    if flat(got, SDP_NOMASK, 1e-5):
        raise Fail("the future is not masked",
                   "Every position can see every other one, so the model reads "
                   "the answer. It converges beautifully and is worthless. Mask "
                   "BEFORE the softmax and let -inf do the work.")
    if flat(got, SDP_MASKAFTER, 1e-5):
        raise Fail("the mask is applied AFTER the softmax",
                   "Zeroing the upper triangle afterwards leaves each row "
                   "summing to less than 1 - row 0 sums to 1/T. That is a "
                   "quiet, position-dependent rescaling of every token. Add "
                   "-inf to the scores instead, and let softmax normalise over "
                   "the survivors.")
    need(flat(got, SDP, 1e-5), "wrong values",
         "Work outwards: scores, scale, mask, softmax, weighted sum. Print the "
         "(B, H, T, S) weight tensor and look at row 0 - it must be exactly "
         "one 1.0 and the rest zeros.")

    # row 0 attends to itself alone, so its output IS value row 0
    need(close(got[:, :, 0], V[:, :, 0], 1e-5),
         "row 0 of the output is not value row 0",
         "The first token can only attend to itself, so its weights are "
         "[1, 0, 0, ...] and its output is v[0] exactly, whatever the scores "
         "were. If it isn't, the mask is not reaching the first row.")

    # convexity: every output channel inside the hull of the value rows
    lo, hi = V.amin(dim=-2), V.amax(dim=-2)
    need(bool(((got >= lo[..., None, :] - 1e-5) &
               (got <= hi[..., None, :] + 1e-5)).all()),
         "an output row lies outside the hull of the value rows",
         "The weights are non-negative and sum to 1, so every output is a "
         "convex combination of v. Landing outside means they are not - a "
         "missing softmax, or a mask that produced a negative weight.")

    # causal=False must actually change something
    nom = call(sol.scaled_dot_product, Q, K, V, causal=False)
    need(not close(nom, got, 1e-6), "causal=False changed nothing",
         "The flag has to reach the mask. With it off, row 0 sees every key.")

    # the offset: T=1 against S keys, as the cache calls it
    one = call(sol.scaled_dot_product, Q[:, :, -1:], K, V)
    need(tuple(one.shape) == (1, H, 1, DH),
         f"with T=1 and S={T} it returned {tuple(one.shape)}")
    need(close(one[:, :, 0], got[:, :, -1], 1e-5),
         "a single query row against all the keys disagrees with the last row "
         "of the full pass",
         "Query row i is the (S - T + i)-th token overall. With T = 1 that "
         "single row sits at the END of the keys and sees all of them. A mask "
         "written as a plain lower triangle makes it see only key 0 - correct "
         "while training, silently wrong while generating.")
    return "attention: scaled, masked before softmax, convex, offset-correct"


def stage_4():
    at = fix(sol.VanillaSelfAttention(Cfg()))
    names = sorted(n for n, _ in at.named_parameters())
    want = sorted(["q_proj.weight", "q_proj.bias", "k_proj.weight", "k_proj.bias",
                   "v_proj.weight", "v_proj.bias", "o_proj.weight", "o_proj.bias"])
    if names != want:
        missing, extra = sorted(set(want) - set(names)), sorted(set(names) - set(want))
        raise Fail(f"wrong parameters.\n   missing: {missing}\n   unexpected: {extra}",
                   "Four Linears named q_proj, k_proj, v_proj, o_proj, all "
                   "d_model -> d_model, all with bias. The 2017 layer has "
                   "biases; dropping them is a 2023 habit.")
    for n in ("q_proj", "k_proj", "v_proj", "o_proj"):
        w = dict(at.named_parameters())[f"{n}.weight"]
        need(tuple(w.shape) == (D, D),
             f"{n}.weight is {tuple(w.shape)}, expected {(D, D)}",
             "All four are full width. Narrower k and v is grouped query "
             "attention, which is llm.py stage 3, not this layer.")

    out = call(at, X)
    need(tuple(out.shape) == tuple(X.shape),
         f"returned {tuple(out.shape)}, expected {tuple(X.shape)}")

    if flat(out, ATT_BADMERGE, 1e-5):
        raise Fail("the heads are reshaped without being transposed back",
                   "y is (B, n_h, T, d_h) and the row you want is (B, T, "
                   "n_h*d_h). Reshaping straight from the first to the second "
                   "interleaves the heads into each token's row. Use "
                   "merge_heads - it is stage 2, already written.")
    need(flat(out, ATT, 1e-5), "wrong values",
         "The layer is stages 2 and 3 with four projections around them. If "
         "those two pass, the fault is in the order: project, split, attend, "
         "merge, project.")

    # it has to USE the pieces, not reimplement them
    q = sol.split_heads(at.q_proj(X), H)
    k = sol.split_heads(at.k_proj(X), H)
    v = sol.split_heads(at.v_proj(X), H)
    by_hand = at.o_proj(sol.merge_heads(sol.scaled_dot_product(q, k, v)))
    need(close(by_hand, out, 1e-6),
         "the layer disagrees with your own stages 2 and 3 wired together",
         "Whatever forward() is doing, it is not project -> split_heads -> "
         "scaled_dot_product -> merge_heads -> o_proj.")

    # causality, end to end
    edited = X.clone()
    edited[:, -1] += 3.0
    need(close(call(at, edited)[:, :-1], out[:, :-1], 1e-6),
         "changing the LAST token changed EARLIER outputs",
         "That is the causal mask leaking. Nothing at position p may depend "
         "on anything after p.")

    # the cache: one token at a time must equal one full pass
    cache = {}
    inc = torch.cat([at(X[:, t:t + 1], cache=cache) for t in range(T)], dim=1)
    need(close(inc, out, 1e-4),
         "the cached path disagrees with the full pass",
         "Concatenate the stored k/v in FRONT of the new ones, store the "
         "result back, and let stage 3's offset handle the mask. If stage 3 "
         "passed, the fault is the concatenation order or the missing store.")
    need("k" in cache and cache["k"].shape[2] == T,
         "the cache does not hold every key at the end",
         f"After {T} single-token steps cache['k'] should be (B, H, {T}, d_h).")
    return "layer: named, biased, causal, and cache-consistent"


STAGES = [
    (1, "sinusoidal_pe", stage_1),
    (2, "split_heads + merge_heads", stage_2),
    (3, "scaled_dot_product", stage_3),
    (4, "VanillaSelfAttention", stage_4),
]


def main():
    torch.manual_seed(0)
    print("\nchecking attention.py\n" + "-" * 60)
    for num, name, fn in STAGES:
        try:
            msg = fn()
        except NotImplementedError as ex:
            print(f"[ ] stage {num}  {name}")
            print(f"\n    not written yet: {ex}")
            print("    Read the TODO above it, then run this again.\n")
            return 1
        except Fail as ex:
            print(f"[x] stage {num}  {name}")
            print(f"\n    {ex}")
            if ex.hint:
                print(f"\n    hint: {ex.hint}")
            print()
            return 1
        except Exception:
            print(f"[x] stage {num}  {name}\n")
            traceback.print_exc()
            return 1
        print(f"[ok] stage {num}  {name:<28} {msg}")

    print("-" * 60)
    print("all four stages pass.\n")
    print("Look at what you built:\n")
    print("    python LLM/from_scratch/attention.py\n")
    print("Then the layer that replaced it - RoPE and grouped query heads:\n")
    print("    python LLM/from_scratch/check_rope.py")
    print("    python LLM/from_scratch/check.py        # stage 3\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
