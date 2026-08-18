# The transformer from scratch

Build the model the MoE layer plugs into, with a grader that checks each piece
and diagnoses the specific mistake without giving up the answer.

```bash
python LLM/from_scratch/check.py        # run this constantly
```

Edit `llm.py`. Do `DenseMoe/from_scratch` and `SparseMoe/from_scratch` first —
the FFN and the routed FFN are imported here already built, because you built
them there. (The FFN you import is the three-matrix SwiGLU of E18, not the
two-matrix expert those tracks used; the routing around it is unchanged.)

Every stub cites the equation it implements, using the E1–E43 numbering from
the notes, so you can read the stub and the maths side by side.

## The rule that makes this worth doing

`../common.py` is the finished implementation. **Don't open it until you're
done.** Stage 3 is where the whole exercise lives; reading the answer costs you
the point of it.

## Stages

| # | What | The idea |
|---|------|----------|
| 1 | `RMSNorm` (E4) | Scale, don't centre. Two lines, and one of them has a trap in it. |
| 2 | `rope_tables`, `apply_rope` (E7, E8) | Position as a rotation. The dot product must end up seeing only `n - m`. |
| 3 | `CausalSelfAttention` (E6, E9–E11) | Heads, GQA, the mask, the cache. **The hard one.** |
| 4 | `Block` (E14, E15) | Pre-norm and two residuals. Easy to write, easy to write backwards. |
| 5 | `TinyLLM` (E1, E32, E24/E29) | Tying, buffers, the loss, and `pos`. |
| 6 | `generate` (E33) | Sampling, and a cache that must not change the answer. |

The grader knows several specific wrong implementations by their *output* and
will name them:

- **no causal mask** — trains, converges beautifully, and is worthless: the
  model has been reading the answer.
- **no `1/sqrt(d_head)`** — the scores grow with head width, softmax saturates,
  and gradients through attention go flat.
- **RoPE never applied** — the layer becomes permutation invariant. It cannot
  tell `a b` from `b a` and will never learn to copy.
- **post-norm instead of pre-norm** — normalising the sum instead of the branch
  input. Trains at 2 layers, fights you at 20.
- **missing residual** — the block computes something, but the gradient has to
  survive every sublayer to reach the embedding.
- **cache offset wrong** — cached and uncached generation disagree. Nothing
  crashes; the text just gets worse.

Stages 3, 5 and 6 all check the same equality from different angles: feeding a
sequence one token at a time through the cache must give bit-comparable results
to feeding it whole. That test is the only thing standing between you and a
plausible, subtly wrong cache.

## What you should be able to answer at the end

1. Why is RoPE applied to `q` and `k` but not `v`?
2. `n_kv_heads < n_heads` shrinks the KV cache. What does it *not* shrink, and
   why is that the reason it's an acceptable trade?
3. Why must the mask be applied before the softmax rather than after?
4. With a KV cache, `T = 1` and the mask is all-visible. What exactly makes the
   same line of code correct in both calls?
5. One forward pass over `T` tokens produces `T` training signals. Which
   property of the architecture makes that legal?
6. The embedding and the output head are the same tensor. Name one thing that
   ties beyond the parameter saving.
7. Switch `ffn="dense"` to `ffn="moe"`. Which numbers move, which don't, and
   which of them is the reason the substitution exists?
8. E29 sums the balancing loss over MoE layers rather than averaging it. What
   does that do to the effective pressure on a 3-layer model versus a 60-layer
   one, and does `α` mean the same thing in both?

## When it passes

```bash
LLM_IMPL=scratch python LLM/run_llm.py
```

The same dissector that runs the reference, driven by your code: causality and
the cache, RoPE's relative-position property, where the parameters live, dense
against MoE at matched compute, three seeds of training on a copy task, what
the load-balancing loss is actually worth, and generation.
