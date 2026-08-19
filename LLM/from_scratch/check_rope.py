"""
Grader for rope.py.

    python LLM/from_scratch/check_rope.py

Four stages, in dependency order, stopping at the first one that isn't done or
isn't right. It says WHAT is wrong and usually WHY, never the answer.

Stages 1 and 2 are checked against baked-in outputs, including the outputs of
four specific WRONG implementations, so the grader can name the mistake you
made instead of just saying "wrong". Stages 3 and 4 are checked against the
code you wrote in stages 1 and 2 - the properties, not the values, because the
properties are the whole point of those stages.
"""

import sys
import traceback
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rope as sol

torch.set_printoptions(precision=6, sci_mode=False)

# ---------------------------------------------------------------- fixtures
DH, MAXSEQ, BASE = 8, 16, 10000.0
M = 3
Q = torch.tensor([[[[0.6, -0.2, 0.9, 0.4, -0.7, 0.1, 0.3, -0.5]]]])   # (1,1,1,d_h)


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
    """Compare ignoring shape - the goldens are stored flat."""
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
                   "Write the shapes next to each line before you run it. Most "
                   "of these come from a broadcast axis that was never added.")
    if out is None:
        raise Fail("returned None", "A function with no `return` hands back None.")
    return out


def tables():
    """Stage 1's output, for the later stages to lean on."""
    return call(sol.rope_tables, DH, MAXSEQ, BASE)


# ------------------------------------------------------- expected (outputs)
COS_ROW1 = SIN_ROW1 = COS_LAST = None
COS_ROW1_NO2 = COS_ROW1_FLIPPED = None
ROT_HALF_Q = None
Q_ROT = Q_ROT_CONJ = Q_ROT_CAT = Q_ROT_HALF = None
exec(open(Path(__file__).with_name("expected_rope.py")).read())


# ------------------------------------------------------------------ stages
def stage_1():
    out = call(sol.rope_tables, DH, MAXSEQ, BASE)
    need(isinstance(out, (tuple, list)) and len(out) == 2,
         f"expected (cos, sin), got {type(out).__name__}")
    cos, sin = out
    cos, sin = torch.as_tensor(cos), torch.as_tensor(sin)

    want = (MAXSEQ, DH // 2)
    need(tuple(cos.shape) == want, f"cos is {tuple(cos.shape)}, expected {want}",
         f"The tables are HALF as wide as the head: {DH // 2} columns for a "
         f"{DH}-wide head, because there is one angle per PAIR of channels. "
         "A table as wide as the head means one angle per channel, and the "
         "pairing is already wrong.")
    need(tuple(sin.shape) == want, f"sin is {tuple(sin.shape)}, expected {want}")

    need(close(cos[0], torch.ones(DH // 2)) and close(sin[0], torch.zeros(DH // 2)),
         "position 0 is not the identity rotation",
         "Row 0 is angle 0 for every pair, so cos is all 1 and sin all 0 - the "
         "first token comes back untouched. If yours isn't, your positions "
         "probably start at 1.")

    if flat(cos[1], COS_ROW1_NO2, 1e-4):
        raise Fail("the exponent is missing its factor of 2",
                   "theta_i = base**(-2i/d_h), not base**(-i/d_h). The "
                   "sequence 2i is 0, 2, 4, ... - which is what "
                   "arange(0, d_head, 2) gives you. Dropped, the ladder only "
                   "spans base**(-1/2) and every wavelength in the head is "
                   "wrong by a square root.")
    if flat(cos[1], COS_ROW1_FLIPPED, 1e-4):
        raise Fail("the frequency ladder runs backwards",
                   "Pair 0 must be the FAST one (theta = 1, a full turn every "
                   "~6 tokens) and the last pair the slow one. Flipped, the "
                   "model still trains - it just spends its fine resolution "
                   "where it needs the coarse kind.")
    need(flat(cos[1], COS_ROW1, 1e-5) and flat(sin[1], SIN_ROW1, 1e-5),
         "row 1 of the tables is wrong",
         "Row m is m * theta, elementwise, then cos and sin of that. Row 1 IS "
         "the rate ladder, so print it: the first entry should be cos(1) and "
         "the last should be within a rounding error of 1.")
    need(flat(cos[MAXSEQ - 1], COS_LAST, 1e-5),
         f"row {MAXSEQ - 1} is wrong while row 1 is right",
         "The rows are m * theta for m = 0 .. max_seq-1. If only the later "
         "rows are off, check that you built the angles as an OUTER product "
         "and did not broadcast a position against the wrong axis.")

    c2, _ = call(sol.rope_tables, DH, MAXSEQ, 100.0)
    need(not close(c2[1], cos[1], 1e-4),
         "changing `base` changed nothing",
         "base is the bottom of the ladder: theta ranges from 1 down to "
         "1/base. If it is ignored, it is hard-coded somewhere.")

    ratio = float(cos[1, 0]), float(cos[1, -1])
    need(ratio[0] < ratio[1],
         f"pair 0 turns slower than the last pair ({ratio[0]:.4f} vs {ratio[1]:.4f})")
    return f"tables: {want} per table, right ladder, position 0 is the identity"


def stage_2():
    cos, sin = tables()
    got = call(sol.apply_rope, Q, cos[M:M + 1], sin[M:M + 1])
    need(tuple(got.shape) == tuple(Q.shape),
         f"returned {tuple(got.shape)}, expected {tuple(Q.shape)}",
         "The rotation is in-place in shape terms: d_h channels in, d_h out. "
         "A (…, d_h/2, 2) result means the last flatten never happened.")

    need(close(got.norm(), Q.norm(), 1e-5),
         f"the length changed: {Q.norm():.6f} -> {got.norm():.6f}",
         "A rotation preserves length. If yours doesn't, you are mixing "
         "channels that are not a pair, or you dropped a term from one of the "
         "two rows of E7.")

    if flat(got, Q_ROT_CONJ, 1e-5):
        raise Fail("the rotation runs the wrong way round",
                   "You have E7's matrix transposed: the minus sign belongs on "
                   "the sin term of the FIRST row, and the plus on the second. "
                   "R(-m) instead of R(m) still preserves length, and E8 still "
                   "holds with the sign of the gap flipped - so nothing "
                   "downstream will complain.")
    if flat(got, Q_ROT_CAT, 1e-5):
        raise Fail("the channels come back in the wrong order",
                   "The arithmetic is right and the reassembly is not: `cat` "
                   "returns [all the firsts, then all the seconds], so channel "
                   "1 of the output is pair 1's first member instead of pair "
                   "0's second. `stack` on a new last axis, then flatten it.")
    if flat(got, Q_ROT_HALF, 1e-5):
        raise Fail("you paired channel j with channel j + d_h/2",
                   "That is the split-half convention, and it is stage 3 - but "
                   "these tables and this function are the interleaved one, so "
                   "here it pairs channels that share no angle. Pair 2i with "
                   "2i+1: adjacent, not half a head apart.")
    need(flat(got, Q_ROT, 1e-5), "wrong values",
         "Work one pair at a time on paper. Pair 0 is (0.6, -0.2) turned by "
         f"3*theta_0 = 3.0 radians; if that one is right and the rest are not, "
         "the tables are being broadcast against the wrong axis.")

    ident = call(sol.apply_rope, Q, cos[0:1], sin[0:1])
    need(flat(ident, Q, 1e-6), "position 0 did not come back unchanged",
         "cos 1, sin 0 makes E7 the identity matrix. If token 0 moves, the "
         "angle being applied is not the one in the row you were handed.")

    # each PAIR keeps its own length, not just the vector as a whole
    p_in = Q.reshape(-1, DH // 2, 2).norm(dim=-1)
    p_out = got.reshape(-1, DH // 2, 2).norm(dim=-1)
    need(close(p_in, p_out, 1e-5),
         "the total length survived but the individual pairs did not",
         "RoPE turns d_h/2 independent little vectors. Nothing may move "
         "between pairs - the only thing they share is the position.")

    # broadcasting: same answer whatever sits in front of (T, d_h)
    big = torch.randn(2, 3, 4, DH)
    out_big = call(sol.apply_rope, big, cos[:4], sin[:4])
    need(tuple(out_big.shape) == tuple(big.shape),
         f"on a (2, 3, 4, {DH}) input it returned {tuple(out_big.shape)}")
    need(close(out_big[1, 2], call(sol.apply_rope, big[1:2, 2:3], cos[:4], sin[:4])[0, 0], 1e-5),
         "the result depends on which batch/head slice it is in",
         "cos and sin need the batch and head axes added so they broadcast. "
         "Every head at position m gets the same angle: position belongs to "
         "the token, not the head.")

    # E8 - the reason any of this exists
    a, b = torch.randn(1, 1, 1, DH), torch.randn(1, 1, 1, DH)

    def dot(m, n):
        return float((sol.apply_rope(a, cos[m:m + 1], sin[m:m + 1]) *
                      sol.apply_rope(b, cos[n:n + 1], sin[n:n + 1])).sum())

    trio = [dot(m, n) for m, n in ((4, 1), (7, 4), (13, 10))]
    need(max(trio) - min(trio) < 1e-4,
         f"three pairs at distance 3 gave different dot products: "
         f"{['%.6f' % v for v in trio]}",
         "This is E8, and it is the property RoPE exists for. If it fails "
         "while the values above passed, the same angle is not reaching q and "
         "k, or the two disagree about which channels are a pair.")
    near, far = dot(5, 4), dot(15, 4)
    need(abs(near - far) > 1e-4,
         "gaps of 1 and 11 gave the same dot product",
         "The score has to actually depend on the gap. If it doesn't, the "
         "rotation is being applied to q and k identically and cancelling.")
    return "rotation: length-preserving, per-pair, and E8 holds"


def stage_3():
    cos, sin = tables()

    got = call(sol.rotate_half, Q)
    need(tuple(got.shape) == tuple(Q.shape),
         f"rotate_half returned {tuple(got.shape)}, expected {tuple(Q.shape)}")
    need(flat(got, ROT_HALF_Q, 1e-6), "rotate_half is wrong",
         "Second half negated, in front of the first half. Half the entries "
         "must change sign and none may change magnitude - if yours negates "
         "the wrong half you have built the quarter turn backwards.")
    need(flat(call(sol.rotate_half, got), -Q, 1e-6),
         "rotate_half applied twice is not -x",
         "Two quarter turns make a half turn. That identity is the cheapest "
         "test there is that you have a rotation and not a shuffle.")

    half = call(sol.apply_rope_half, Q, cos[M:M + 1], sin[M:M + 1])
    need(tuple(half.shape) == tuple(Q.shape),
         f"apply_rope_half returned {tuple(half.shape)}, expected {tuple(Q.shape)}")
    need(close(half.norm(), Q.norm(), 1e-5),
         f"the length changed: {Q.norm():.6f} -> {half.norm():.6f}",
         "Still a rotation, still length-preserving. The tables are d_h/2 wide "
         "and you need one entry per channel - duplicate them so that j and "
         "j + d_h/2 get the SAME angle.")
    if flat(half, Q_ROT, 1e-5):
        raise Fail("apply_rope_half is doing the interleaved pairing",
                   "It returned exactly what stage 2 returns. This convention "
                   "pairs channel j with channel j + d_h/2, which is a "
                   "different pairing of the same channels - and the whole "
                   "point of the stage.")
    need(flat(half, Q_ROT_HALF, 1e-5), "wrong values",
         "x * cos + rotate_half(x) * sin, with cos and sin duplicated to full "
         "width. If only the second half of the output is wrong, check which "
         "half your duplication put first.")

    # the point of the stage: the two conventions are the same function on a
    # permuted channel order
    perm = torch.arange(DH).reshape(2, DH // 2).t().reshape(-1)   # 0, d/2, 1, d/2+1, ...
    inv = torch.argsort(perm)
    x = torch.randn(2, 3, 4, DH)
    via_interleaved = call(sol.apply_rope, x[..., perm], cos[:4], sin[:4])[..., inv]
    need(close(call(sol.apply_rope_half, x, cos[:4], sin[:4]), via_interleaved, 1e-5),
         "the two conventions are not the same function under the permutation",
         "Permute x so that the split-half partners land next to each other, "
         "run stage 2's apply_rope, permute back: that must equal "
         "apply_rope_half exactly. If it doesn't, one of the two is pairing "
         "channels that don't share an angle.")

    def dot(fn, m, n):
        torch.manual_seed(7)
        a, b = torch.randn(1, 1, 1, DH), torch.randn(1, 1, 1, DH)
        return float((fn(a, cos[m:m + 1], sin[m:m + 1]) *
                      fn(b, cos[n:n + 1], sin[n:n + 1])).sum())

    trio = [dot(sol.apply_rope_half, m, n) for m, n in ((4, 1), (7, 4), (13, 10))]
    need(max(trio) - min(trio) < 1e-4,
         f"E8 fails for the split-half version: {['%.6f' % v for v in trio]}",
         "Either convention has to give a dot product that depends only on "
         "n - m. Only mixing them breaks that.")
    return "split-half: same function, other pairing, E8 still holds"


def stage_4():
    cos, sin = tables()

    c1, s1 = call(sol.rope_tables_scaled, DH, MAXSEQ, BASE, 1.0)
    need(close(torch.as_tensor(c1), cos, 1e-6) and close(torch.as_tensor(s1), sin, 1e-6),
         "scale = 1.0 does not reproduce rope_tables",
         "No scaling means no change. If these differ, the scale is being "
         "applied somewhere it shouldn't be - to theta, perhaps, rather than "
         "to the positions.")

    S = 4.0
    cs, ss = call(sol.rope_tables_scaled, DH, MAXSEQ, BASE, S)
    cs, ss = torch.as_tensor(cs), torch.as_tensor(ss)
    need(tuple(cs.shape) == (MAXSEQ, DH // 2),
         f"scaled cos is {tuple(cs.shape)}, expected {(MAXSEQ, DH // 2)}",
         "Same table, same size - `scale` extends the reach of the rows, not "
         "their number.")

    for m in (4, 8, 12):
        need(close(cs[m], cos[int(m / S)], 1e-5),
             f"row {m} at scale {S:.0f} is not row {int(m / S)} of the plain table",
             "Position m must be rotated as if it were m / scale. If you "
             "multiplied instead of divided you have made the problem worse: "
             "the angles now leave the trained range sooner, not later.")

    need(float(cs[MAXSEQ - 1, 0]) != float(cos[MAXSEQ - 1, 0]),
         "the scaled table is identical to the unscaled one at the far end")

    # the consequence: a gap of `scale` now buys what a gap of 1 used to
    torch.manual_seed(11)
    a, b = torch.randn(1, 1, 1, DH), torch.randn(1, 1, 1, DH)

    def dot(c, s, m, n):
        return float((sol.apply_rope(a, c[m:m + 1], s[m:m + 1]) *
                      sol.apply_rope(b, c[n:n + 1], s[n:n + 1])).sum())

    need(abs(dot(cs, ss, 0, int(S)) - dot(cos, sin, 0, 1)) < 1e-4,
         f"a gap of {S:.0f} scaled tokens does not match a gap of 1 unscaled",
         "That equality IS position interpolation: `scale`x the context, "
         "squeezed into the angular range the model was trained on. It is also "
         "the cost - the fast pairs get compressed along with the slow ones, "
         "so fine local distinctions blur. NTK-aware scaling and YaRN exist to "
         "spend that compression only where it is affordable.")
    return f"interpolation: positions divided by scale, {S:.0f}x reach"


STAGES = [
    (1, "rope_tables", stage_1),
    (2, "apply_rope", stage_2),
    (3, "rotate_half + apply_rope_half", stage_3),
    (4, "rope_tables_scaled", stage_4),
]


def main():
    torch.manual_seed(0)
    print("\nchecking rope.py\n" + "-" * 60)
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
        print(f"[ok] stage {num}  {name:<30} {msg}")

    print("-" * 60)
    print("all four stages pass.\n")
    print("Look at what you built:\n")
    print("    python LLM/from_scratch/rope.py\n")
    print("Then paste stages 1 and 2 into llm.py and carry on with the layer\n"
          "they feed:\n")
    print("    python LLM/from_scratch/check.py\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
