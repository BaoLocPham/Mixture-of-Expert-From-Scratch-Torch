"""
Grader for gqa.py.

    python LLM/from_scratch/check_gqa.py

Three stages, in dependency order, stopping at the first one that isn't done or
isn't right. It says WHAT is wrong and usually WHY, never the answer.

    1  repeat_kv                   the grouping, not just the shape
    2  GroupedQueryAttention       the widths, the layer, and what the cache
                                   is allowed to hold
    3  kv_cache_floats_per_token   the accounting the whole thing is for

The reference values come from `../common.py`'s CausalSelfAttention run with an
identity rotation, which is exactly this layer. They include the output of the
one wrong grouping that has the right shape, so the grader can name it.
"""

import sys
import traceback
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import gqa as sol

torch.set_printoptions(precision=6, sci_mode=False)

# ------------------------------------------- the fixtures every stage shares
D, H, HKV, DH, T = 16, 4, 2, 4, 5
N_REP = H // HKV
X = (torch.arange(1 * T * D).float().reshape(1, T, D) * 0.13 - 0.9).sin() * 1.4


class Cfg:
    """The fields gqa.py's __init__ is allowed to read."""
    def __init__(self, n_heads=H, n_kv_heads=HKV, d_model=D, d_head=DH):
        self.n_heads, self.n_kv_heads = n_heads, n_kv_heads
        self.d_model, self.d_head = d_model, d_head


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
        raise Fail(f"raised a shape error:\n   {str(ex).splitlines()[0]}",
                   "Write the shapes next to each line before you run it.")
    if out is None:
        raise Fail("returned None", "A function with no `return` hands back None.")
    return out


def fix(module):
    with torch.no_grad():
        for i, (name, p) in enumerate(sorted(module.named_parameters())):
            v = torch.sin(torch.arange(p.numel()).float() * 0.7 + 1.3 * i) * 0.3
            p.copy_(v.reshape(p.shape))
    return module


def tagged(n_kv=HKV, S=3, dh=DH):
    """kv head i filled with the value i, so the grouping is readable."""
    return torch.arange(n_kv).float()[None, :, None, None].expand(1, n_kv, S, dh)


# --------------------------------- expected outputs - values only, no method
GQA = GQA_MHA = GQA_REPEAT = None
exec(open(Path(__file__).with_name("expected_gqa.py")).read())


# ---------------------------------------------- stage 1: the expansion, E11
def stage_1():
    t = tagged()
    out = call(sol.repeat_kv, t, N_REP)
    need(tuple(out.shape) == (1, H, 3, DH),
         f"returned {tuple(out.shape)}, expected {(1, H, 3, DH)}",
         f"n_kv * n_rep = {HKV} * {N_REP} = {H} heads out.")

    got = [int(v) for v in out[0, :, 0, 0]]
    floor = [h // N_REP for h in range(H)]
    modulo = [h % HKV for h in range(H)]
    if got == modulo and got != floor:
        raise Fail(f"the groups are interleaved: query heads read {got}",
                   f"E11 says head h reads kv group floor(h / n_rep), which is "
                   f"{floor} - each group CONTIGUOUS. You have h mod n_kv. "
                   "That is torch's `repeat`; the one you want repeats each "
                   "head in place rather than tiling the whole tensor.")
    need(got == floor, f"query heads read kv heads {got}, expected {floor}",
         "Each kv head must be copied n_rep times in place, so the groups come "
         "out contiguous.")

    need(close(call(sol.repeat_kv, t, 1), t, 1e-7),
         "n_rep = 1 changed the tensor",
         "With one query head per kv head there is nothing to expand, and the "
         "input must come back untouched.")

    # every copy inside a group must be the SAME rows, not a shifted view
    big = torch.randn(2, HKV, 6, DH)
    ex = call(sol.repeat_kv, big, N_REP)
    for h in range(H):
        need(close(ex[:, h], big[:, h // N_REP], 1e-7),
             f"output head {h} is not a copy of kv head {h // N_REP}",
             "The values have to survive the expansion, not just the shape.")

    need(close(call(sol.repeat_kv, big, 3), big.repeat_interleave(3, dim=1), 1e-7),
         "it only works for n_rep = 2",
         "Nothing here should be hard-coded to the fixture; n_rep comes from "
         "the config and is 8 in a real model.")
    return f"expansion: {HKV} -> {H} heads, groups contiguous, n_rep=1 a no-op"


# ------------------------------ stage 2: the layer, and what the cache holds
def stage_2():
    at = fix(sol.GroupedQueryAttention(Cfg()))
    got = sorted(n for n, _ in at.named_parameters())
    want = sorted(["q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight"])
    if got != want:
        missing, extra = sorted(set(want) - set(got)), sorted(set(got) - set(want))
        raise Fail(f"wrong parameters.\n   missing: {missing}\n   unexpected: {extra}",
                   "Four bias-free Linears named q_proj, k_proj, v_proj, o_proj. "
                   "Biases are the 2017 layer; this one has none.")

    shapes = {n: tuple(p.shape) for n, p in at.named_parameters()}
    need(shapes["q_proj.weight"] == (H * DH, D),
         f"q_proj is {shapes['q_proj.weight']}, expected {(H * DH, D)}")
    need(shapes["k_proj.weight"] == (HKV * DH, D),
         f"k_proj is {shapes['k_proj.weight']}, expected {(HKV * DH, D)}",
         "k and v are NARROWER than q - n_kv_heads * d_head. If yours is full "
         "width you have built MHA with an extra expansion in the middle, and "
         "the cache saving, which is the entire point, is zero.")
    need(shapes["v_proj.weight"] == (HKV * DH, D),
         f"v_proj is {shapes['v_proj.weight']}, expected {(HKV * DH, D)}")

    out = call(at, X)
    need(tuple(out.shape) == tuple(X.shape),
         f"returned {tuple(out.shape)}, expected {tuple(X.shape)}")

    if flat(out, GQA_REPEAT, 1e-5):
        raise Fail("the kv heads are expanded with the wrong grouping",
                   "Stage 1 passed, so this layer is not calling it - it is "
                   "doing its own expansion, with `repeat` rather than a "
                   "per-head one. Call repeat_kv.")
    if flat(out, GQA_MHA, 1e-5):
        raise Fail("this is multi-head attention, not grouped",
                   "Every query head got its own k and v. Either the k/v "
                   "projections are full width, or n_rep is being ignored.")
    need(flat(out, GQA, 1e-5), "wrong values",
         "Four steps: project and split (two head counts), cache, repeat_kv, "
         "then scaled_dot_product -> merge_heads -> o_proj.")

    # n_kv == n_h has to degenerate to plain MHA
    mha = fix(sol.GroupedQueryAttention(Cfg(n_kv_heads=H)))
    need(flat(call(mha, X), GQA_MHA, 1e-5),
         "at n_kv_heads == n_heads it does not reduce to plain MHA",
         "GQA is a superset: n_rep = 1, the expansion is a no-op, and nothing "
         "else in the layer knows the difference.")

    # causality
    edited = X.clone()
    edited[:, -1] += 3.0
    need(close(call(at, edited)[:, :-1], out[:, :-1], 1e-6),
         "changing the LAST token changed EARLIER outputs",
         "scaled_dot_product masks for you, so if this fails you are passing "
         "causal=False - or not using it at all.")

    # the cache: agreement, and the width of what is stored
    cache = {}
    inc = torch.cat([at(X[:, t:t + 1], cache=cache) for t in range(T)], dim=1)
    need(close(inc, out, 1e-4),
         "the cached path disagrees with the full pass",
         "Concatenate the stored k/v in FRONT of the new ones along the token "
         "axis, and store the result back.")
    need("k" in cache, "nothing was stored in the cache")
    kv_heads = cache["k"].shape[1]
    if kv_heads == H:
        raise Fail(f"the cache is holding {H} heads, not {HKV}",
                   "You expanded before you stored. The output is right and "
                   "the saving is gone: the KV cache is the only thing GQA "
                   "shrinks, and it is now exactly the size it would be "
                   "without GQA. repeat_kv goes AFTER the cache, and what it "
                   "produces is a transient that nothing keeps.")
    need(kv_heads == HKV,
         f"the cache holds {kv_heads} heads, expected {HKV}")
    need(cache["k"].shape[2] == T,
         f"after {T} single-token steps the cache holds "
         f"{cache['k'].shape[2]} positions, expected {T}")
    return f"layer: k/v narrow, grouped, causal, cache holds {HKV} heads not {H}"


# ---------------------------------------------- stage 3: what it is all for
def stage_3():
    f = sol.kv_cache_floats_per_token
    need(call(f, 32, 8, 128) == 2 * 32 * 8 * 128,
         f"kv_cache_floats_per_token(32, 8, 128) = {call(f, 32, 8, 128)}, "
         f"expected {2 * 32 * 8 * 128}",
         "Per layer, a key and a value, each n_kv_heads * d_head wide. The 2 "
         "is k and v; forgetting it halves every number below.")
    need(call(f, 1, 1, 1) == 2, "the shape of the formula is wrong")

    mha, gqa, mqa = call(f, 32, 32, 128), call(f, 32, 8, 128), call(f, 32, 1, 128)
    need(mha == 4 * gqa and gqa == 8 * mqa,
         f"MHA {mha}, GQA {gqa}, MQA {mqa} do not scale with n_kv_heads",
         "The count is linear in n_kv_heads and in nothing else - not n_heads, "
         "which never appears, because queries are never cached.")

    # a sanity check on the units, at a serving-shaped batch
    gb = gqa * 8192 * 32 * 2 / 1e9
    need(abs(gb - 34.36) < 0.2,
         f"at 8k context, batch 32, fp16 that is {gb:.2f} GB - expected ~34.36",
         "floats/token * context * batch * bytes-per-float. If you are out by "
         "2x the k-and-v factor is missing; by 1024 or 1e9, the units are.")
    return (f"accounting: MHA {mha:,} -> GQA {gqa:,} floats/token, "
            f"{mha * 8192 * 32 * 2 / 1e9:.1f} GB -> {gb:.1f} GB at 8k x 32")


# ---------------------------------------------------------------- the runner
STAGES = [
    (1, "repeat_kv", stage_1),
    (2, "GroupedQueryAttention", stage_2),
    (3, "kv_cache_floats_per_token", stage_3),
]


def main():
    torch.manual_seed(0)
    print("\nchecking gqa.py\n" + "-" * 60)
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
        print(f"[ok] stage {num}  {name:<26} {msg}")

    print("-" * 60)
    print("all three stages pass.\n")
    print("Look at what you built:\n")
    print("    python LLM/from_scratch/gqa.py\n")
    print("Then the layer that has this AND RoPE AND the cache, together:\n")
    print("    python LLM/from_scratch/check.py        # stage 3\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
