"""
Grader for llm.py.

    python LLM/from_scratch/check.py

Runs the six stages in order and stops at the first one that isn't done or
isn't right. It says WHAT is wrong and usually WHY, never the answer.

The numbers baked in below are outputs, including the outputs of several
specific WRONG implementations - so the grader can tell you which mistake you
made. Reading them tells you nothing about how to produce them.
"""

import sys
import traceback
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import llm as sol

torch.set_printoptions(precision=6, sci_mode=False)

# ---------------------------------------------------------------- fixtures
D, H, HKV, DH, D_FF, V, LAYERS, MAXSEQ = 8, 2, 1, 4, 16, 10, 2, 12
T = 5
IDS = torch.tensor([[3, 1, 7, 0, 5]])
X = (torch.arange(T * D).float().reshape(1, T, D) * 0.13 - 0.9).sin() * 1.4


def cfg(**kw):
    base = dict(vocab_size=V, d_model=D, n_layers=LAYERS, n_heads=H,
                n_kv_heads=HKV, d_ff=D_FF, max_seq=MAXSEQ)
    base.update(kw)
    return sol.LLMConfig(**base)


def fix(module):
    """Deterministic weights, keyed by parameter NAME so construction order
    cannot change them."""
    with torch.no_grad():
        for i, (name, p) in enumerate(sorted(module.named_parameters())):
            v = torch.sin(torch.arange(p.numel()).float() * 0.7 + 1.3 * i) * 0.3
            p.copy_(v.reshape(p.shape))
    return module


def names(module):
    return sorted(n for n, _ in module.named_parameters())


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


def call(fn, *a, **kw):
    try:
        out = fn(*a, **kw)
    except NotImplementedError:
        raise
    except RuntimeError as ex:
        first = str(ex).splitlines()[0]
        raise Fail(f"raised a shape error:\n   {first}",
                   "Write the shapes next to each line before you run it. Most "
                   "of these come from a transpose that never happened or a "
                   "size-1 axis that was never added.")
    if out is None:
        raise Fail("returned None", "A function with no `return` hands back None.")
    return out


def want_names(module, expected, what):
    got = names(module)
    if got != sorted(expected):
        missing = sorted(set(expected) - set(got))
        extra = sorted(set(got) - set(expected))
        raise Fail(f"{what} has the wrong parameters.\n   missing: {missing}\n"
                   f"   unexpected: {extra}",
                   "The grader and the dissector address these by name, and the "
                   "list is in the docstring at the top of llm.py.")


# ------------------------------------------------------- expected (outputs)
NORM_OUT = None
NORM_LAYERNORM = None
ROPE_COS0 = None
ROPE_COS_LAST = None
ROPE_Q = None
ATT_OUT = None
ATT_NOMASK = None
ATT_NOSCALE = None
ATT_NOROPE = None
BLOCK_OUT = None
BLOCK_POSTNORM = None
BLOCK_NORESID = None
LOGITS = None
LOSS = None
MOE_LOSS = None
GEN = None
exec(open(Path(__file__).with_name("expected.py")).read())


# ------------------------------------------------------------------ stages
def stage_1():
    n = sol.RMSNorm(D)
    want_names(n, ["weight"], "RMSNorm")
    need(close(n.weight, torch.ones(D)),
         f"the gain starts at {n.weight.tolist()}, not all ones",
         "At init the layer should be a pure normalisation - anything else "
         "changes the scale of the residual stream before training begins.")

    out = call(n, X)
    need(out.shape == X.shape, f"returned {tuple(out.shape)}, expected {tuple(X.shape)}")
    if close(out, NORM_LAYERNORM):
        raise Fail("this is LayerNorm, not RMSNorm",
                   "You centred the rows. RMSNorm only scales - the mean of a "
                   "row is left exactly where it was.")
    need(close(out, NORM_OUT), "wrong values",
         "Each row must come out at RMS 1: divide by sqrt(mean(x^2)) over the "
         "LAST axis, with the row's own statistics and nobody else's.")

    rms = out.pow(2).mean(-1).sqrt()
    need(close(rms, torch.ones_like(rms), 1e-3), f"row RMS came out {rms.tolist()}")

    z = torch.zeros(1, D)
    need(torch.isfinite(call(n, z)).all(),
         "an all-zero row produced inf/nan",
         "eps belongs inside the square root, where it can stop a zero norm "
         "from dividing.")
    return "RMSNorm: normalises, keeps the mean, survives a zero row"


def stage_2():
    cos, sin = call(sol.rope_tables, DH, MAXSEQ)
    need(cos.shape == (MAXSEQ, DH // 2),
         f"cos is {tuple(cos.shape)}, expected {(MAXSEQ, DH // 2)}",
         "One angle per PAIR of channels, not per channel.")
    need(close(sin.shape and sin, torch.as_tensor(sin)) and sin.shape == cos.shape,
         "sin and cos must have the same shape")
    need(close(cos[0], torch.ones(DH // 2)) and close(sin[0], torch.zeros(DH // 2)),
         "position 0 must be the identity rotation (cos 1, sin 0)")
    need(close(cos[1], ROPE_COS0, 1e-4) and close(cos[MAXSEQ - 1], ROPE_COS_LAST, 1e-4),
         "the angles are wrong",
         "Pair j turns at base**(-2j/d_head) radians per position. If your "
         "first pair is the SLOW one you have the exponent's sign flipped.")

    q = torch.tensor([[[[0.6, -0.2, 0.9, 0.4]]]])
    got = call(sol.apply_rope, q, cos[3:4], sin[3:4])
    need(got.shape == q.shape, f"apply_rope returned {tuple(got.shape)}, expected {tuple(q.shape)}")
    need(close(got.norm(), q.norm(), 1e-5),
         f"the length changed: {q.norm():.6f} -> {got.norm():.6f}",
         "A rotation preserves length. If yours doesn't, you are mixing "
         "channels that are not a pair, or dropping a term.")
    need(close(got, ROPE_Q, 1e-5), "wrong values",
         "Check the pairing and the order you put the result back in - "
         "rotating (0,1),(2,3) and returning it as (0,2),(1,3) also preserves "
         "length, and is still wrong.")

    a, b = torch.randn(1, 1, 1, DH), torch.randn(1, 1, 1, DH)

    def dot(m, n):
        return (sol.apply_rope(a, cos[m:m + 1], sin[m:m + 1]) *
                sol.apply_rope(b, cos[n:n + 1], sin[n:n + 1])).sum().item()

    trio = [dot(m, n) for m, n in ((4, 1), (7, 4), (10, 7))]
    need(max(trio) - min(trio) < 1e-4,
         f"three pairs at distance 3 gave different dot products: "
         f"{['%.6f' % v for v in trio]}",
         "This is the property RoPE exists for. If it fails, the same angle is "
         "not being applied to q and k, or the pairs disagree between the two.")
    return "rope: right frequencies, length-preserving, sees only n - m"


def stage_3():
    c = cfg()
    at = fix(sol.CausalSelfAttention(c))
    want_names(at, ["q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight"],
               "CausalSelfAttention")
    shapes = {n: tuple(p.shape) for n, p in at.named_parameters()}
    need(shapes["k_proj.weight"] == (HKV * DH, D),
         f"k_proj is {shapes['k_proj.weight']}, expected {(HKV * DH, D)}",
         "With n_kv_heads < n_heads the k/v projections are NARROWER than q. "
         "If yours is full width you have built MHA with extra steps.")

    cos, sin = sol.rope_tables(DH, MAXSEQ)
    out = call(at, X, cos[:T], sin[:T])
    need(out.shape == X.shape, f"returned {tuple(out.shape)}, expected {tuple(X.shape)}")

    for bad, msg, hint in (
        (ATT_NOMASK, "the future is not masked",
         "Every position can see every other one. Mask BEFORE the softmax, and "
         "let -inf do the work."),
        (ATT_NOSCALE, "the scores are not scaled",
         "Divide by sqrt(d_head). Without it the score scale grows with head "
         "width and the softmax saturates."),
        (ATT_NOROPE, "position never entered",
         "q and k have to be rotated. Without it the layer is permutation "
         "invariant - it cannot tell 'a b' from 'b a'."),
    ):
        if close(out, bad):
            raise Fail(msg, hint)
    need(close(out, ATT_OUT), "wrong values",
         "Work outwards: heads split, rotation, mask, softmax, weighted sum, "
         "merge, project. Print the (B, H, T, S) score tensor and look at row 0 "
         "- it should be exactly one 1.0 and the rest zeros.")

    edited = X.clone()
    edited[:, -1] += 3.0
    need(close(call(at, edited, cos[:T], sin[:T])[:, :-1], out[:, :-1]),
         "changing the LAST token changed EARLIER outputs",
         "That is the causal mask leaking. Nothing at position p may depend on "
         "anything after p.")

    cache = {}
    inc = torch.cat([at(X[:, t:t + 1], cos[t:t + 1], sin[t:t + 1], cache)
                     for t in range(T)], dim=1)
    need(close(inc, out, 1e-4),
         "the cached path disagrees with the full pass",
         "The single query row is at the END of the keys, so its mask is not "
         "the first row of a triangle - it is the last. And the rotation for "
         "that token uses ITS position, not 0.")
    return "attention: masked, scaled, rotated, GQA-shaped, cache-consistent"


def stage_4():
    b = fix(sol.Block(cfg()))
    cos, sin = sol.rope_tables(DH, MAXSEQ)
    got = call(b, X, cos[:T], sin[:T])
    need(isinstance(got, (tuple, list)) and len(got) == 2,
         "forward must return (output, aux) - aux is None for a dense FFN",
         "The model adds the MoE balancing loss to the objective, so the block "
         "has to hand it up.")
    out, aux = got
    need(aux is None, f"a dense block returned aux={aux}, expected None")
    need(out.shape == X.shape, f"returned {tuple(out.shape)}, expected {tuple(X.shape)}")

    if close(out, BLOCK_NORESID):
        raise Fail("the residual connections are missing",
                   "Each sublayer ADDS to the stream. Without it the block is a "
                   "stack of transformations and the gradient has to survive "
                   "all of them.")
    if close(out, BLOCK_POSTNORM):
        raise Fail("this is post-norm, not pre-norm",
                   "You normalised the sum. Pre-norm normalises the INPUT to "
                   "each sublayer and leaves the residual path untouched.")
    need(close(out, BLOCK_OUT), "wrong values",
         "Two sublayers, each: normalise, transform, add to the unnormalised "
         "stream. The second one's input is the first one's output.")

    bm = fix(sol.Block(cfg(ffn="moe", n_experts=4, k=2)))
    _, aux = call(bm, X, cos[:T], sin[:T])
    need(aux is not None and aux.numel() == 1,
         "an MoE block must return a scalar aux loss")
    need(aux.requires_grad, "the aux loss has no gradient path",
         "It has to be differentiable through the router probabilities, or it "
         "cannot influence anything.")
    return "block: pre-norm, residual, passes the aux loss up"


def stage_5():
    m = fix(sol.TinyLLM(cfg()))
    need(hasattr(m, "cos") and hasattr(m, "sin"), "the model needs cos/sin buffers")
    need(not any(n in ("cos", "sin") for n, _ in m.named_parameters()),
         "cos/sin are parameters", "They are constants. register_buffer, and "
         "persistent=False since they can be rebuilt from the config.")
    need(m.lm_head.weight is m.embed.weight,
         "the head and the embedding are not the same tensor",
         "tie_embeddings means one tensor with two jobs. Copying the values "
         "ties them for exactly one optimiser step.")

    logits, loss = call(m, IDS, IDS)
    need(logits.shape == (1, T, V),
         f"logits are {tuple(logits.shape)}, expected {(1, T, V)}")
    need(close(logits, LOGITS, 1e-4), "wrong logits",
         "Embed, run the blocks, normalise, project. If stage 4 passed, the "
         "error is in the order of those four or in the tables you slice.")
    need(close(loss, torch.tensor(LOSS), 1e-4),
         f"loss is {loss.item():.6f}, expected {LOSS:.6f}",
         "Flatten to (B*T, V) against (B*T,) and take the mean.")

    fresh = fix(sol.TinyLLM(cfg()))
    _, no_target = call(fresh, IDS)
    need(no_target is None, "loss must be None when no targets are given")

    mm = fix(sol.TinyLLM(cfg(ffn="moe", n_experts=4, k=2)))
    _, mloss = call(mm, IDS, IDS)
    need(close(mloss, torch.tensor(MOE_LOSS), 1e-4),
         f"MoE loss is {mloss.item():.6f}, expected {MOE_LOSS:.6f}",
         "cross-entropy + aux_weight * the MEAN aux over layers. A model that "
         "computes the balancing loss and drops it trains an unbalanced router.")

    caches = [{} for _ in m.blocks]
    inc = torch.cat([call(m, IDS[:, t:t + 1], caches=caches, pos=t)[0]
                     for t in range(T)], dim=1)
    need(close(inc, logits, 1e-4),
         "feeding the sequence one token at a time gives different logits",
         "`pos` is the position of the FIRST token in this call, and the "
         "rotation tables must be sliced from there.")
    return "model: tied, buffered, correct loss, cache-consistent"


def stage_6():
    m = fix(sol.TinyLLM(cfg()))
    prompt = IDS[:, :3]

    g = torch.Generator().manual_seed(0)
    a = call(m.generate, prompt.clone(), 4, use_cache=True, generator=g)
    need(a.shape == (1, 7), f"generated {tuple(a.shape)}, expected {(1, 7)}")
    need(torch.equal(a[:, :3], prompt), "the prompt was not preserved at the front")
    need(torch.equal(a, torch.as_tensor(GEN)), "wrong tokens",
         "Sample from the LAST position's logits only, after temperature, with "
         "torch.multinomial and the generator you were handed.")

    g = torch.Generator().manual_seed(0)
    b = m.generate(prompt.clone(), 4, use_cache=False, generator=g)
    need(torch.equal(a, b),
         "cached and uncached generation produced different tokens",
         "The cache must be exact. Check that the cached step passes the true "
         "position, and that its mask lets the new query see everything stored.")

    g = torch.Generator().manual_seed(3)
    kept = m.generate(prompt.clone(), 8, top_k=1, temperature=1.0, generator=g)[0, 3:]
    with torch.no_grad():
        greedy, seq = [], prompt.clone()
        for _ in range(8):
            lg, _ = m(seq)
            nxt = lg[:, -1].argmax(-1, keepdim=True)
            greedy.append(nxt.item())
            seq = torch.cat([seq, nxt], dim=1)
    need(kept.tolist() == greedy,
         f"top_k=1 sampled {kept.tolist()}, but the argmax path is {greedy}",
         "top_k should leave exactly k candidates alive, so k=1 is greedy "
         "decoding. If yours differs, the mask is keeping too many.")
    return "generate: cached == uncached, top_k restricts, prompt preserved"


STAGES = [
    (1, "RMSNorm", stage_1),
    (2, "rope_tables + apply_rope", stage_2),
    (3, "CausalSelfAttention", stage_3),
    (4, "Block", stage_4),
    (5, "TinyLLM", stage_5),
    (6, "generate", stage_6),
]


def main():
    torch.manual_seed(0)
    print("\nchecking llm.py\n" + "-" * 60)
    for num, name, fn in STAGES:
        try:
            msg = fn()
        except NotImplementedError as ex:
            print(f"[ ] stage {num}  {name}")
            print(f"\n    not written yet: {ex}")
            print(f"    Read the TODO above it, then run this again.\n")
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
    print("all six stages pass.\n")
    print("Now run the dissector against your own code:\n")
    print("    LLM_IMPL=scratch python LLM/run_llm.py\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
