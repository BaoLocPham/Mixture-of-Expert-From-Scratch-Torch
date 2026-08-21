# The transformer from scratch

Build the model the MoE layer plugs into, with a grader that checks each piece
and diagnoses the specific mistake without giving up the answer.

```bash
python LLM/from_scratch/check.py        # run this constantly
```

## Two self-contained side exercises

Both are separate from `llm.py` and import nothing from the rest of the track,
so either can be done first.

### `attention.py` — the 2017 attention layer

*Attention Is All You Need*, exactly as written: sinusoidal position added at
the input, every query head with its own key and value, biases on all four
projections. Build that before the modern layer and stage 3 below stops being
four ideas at once.

```bash
python LLM/from_scratch/check_attention.py    # run this constantly
python LLM/from_scratch/attention.py          # print your own scores and look at them
```

| # | What | The idea |
|---|------|----------|
| 1 | `sinusoidal_pe` | Where position enters in 2017 — added once, before any block, sin and cos interleaved. |
| 2 | `split_heads`, `merge_heads` | One projection read as `n_h` heads, and the exact inverse. |
| 3 | `scaled_dot_product` (E9, E10) | Score, scale, mask, softmax, weight. The whole idea, in four lines. |
| 4 | `VanillaSelfAttention` (E6, E11) | Four projections around the middle three steps, plus a KV cache. |

It names nine wrong implementations by their output, among them: the position
table concatenated instead of interleaved, `split_heads` handing head 0 the
strided columns, no `1/sqrt(d_h)`, no mask, the mask applied *after* the
softmax, a `tril` mask that ignores the cache offset, and heads reshaped
without being transposed back.

## Start here if you want RoPE on its own

`rope.py` is a separate, self-contained exercise for the one piece of this
layer that is genuinely fiddly — with a grader that goes much further than
stage 2 below.

```bash
python LLM/from_scratch/check_rope.py    # run this constantly
python LLM/from_scratch/rope.py          # print your own tables and look at them
```

| # | What | The idea |
|---|------|----------|
| 1 | `rope_tables` (E7) | The frequency ladder, as an outer product. `(max_seq, d_h/2)` — half as wide as the head. |
| 2 | `apply_rope` (E7, E8) | The interleaved (GPT-J) rotation. The last check is E8, and it is the only one that matters. |
| 3 | `rotate_half`, `apply_rope_half` | The split-half (GPT-NeoX / HuggingFace) convention, and the proof that it is the *same function* on a permuted channel order. |
| 4 | `rope_tables_scaled` | Position Interpolation — how a 2k model is run at 8k, in one line. |

It names five wrong implementations by their output: the exponent missing its
factor of 2, the frequency ladder running backwards, the rotation matrix
transposed, `cat` instead of an interleave, and the two channel conventions
crossed. Stages 3 and 4 are checked against the code *you* wrote in 1 and 2,
so those have to be right first.

Nothing in `rope.py` imports the rest of the track, so it can be done before
anything else. When it passes, paste stages 1 and 2 into `llm.py`.

---

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
