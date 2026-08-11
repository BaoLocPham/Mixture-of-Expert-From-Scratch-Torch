# SparseMoE from scratch

Build top-k routing and real token dispatch yourself, with a grader that checks
each piece and diagnoses the specific mistake without giving up the answer.

```bash
python SparseMoe/from_scratch/check.py      # run this constantly
```

Edit `sparse_moe.py`. Do `DenseMoe/from_scratch` first — this assumes you can
already write the dense combine without thinking about it.

## The rule that makes this worth doing

`../common.py` is the finished implementation. **Don't open it until you're
done.** Stage 5 is the only genuinely hard thing in either module; reading the
answer costs you the entire point.

## Stages

| # | What | The idea |
|---|------|----------|
| 1 | `top_k_gate` | The one new operation. A softmax with **exact zeros**. |
| 2 | `load_per_expert` | Counting rows per expert, no loop. You'll need it to see imbalance. |
| 3 | `switch_aux_loss` | `N·Σ fᵢPᵢ`. Differentiable through P, not through f. |
| 4 | `MaskedSparseMoE` | Sparse semantics, dense compute — the trap. Easy, correct, slow. |
| 5 | `SparseMoE` | gather → compute → scatter. **The hard one.** |

Stages 4 and 5 must produce **identical numbers**. That's the whole test: 4 is
obviously correct and buys nothing; 5 is fast and easy to get subtly wrong. When
they disagree, 4 is right.

Three ways stage 5 goes wrong, all of which the grader detects by name:

- **`topw[tok, 0]` instead of `topw[tok, slot]`** — a token's second-choice
  expert gets its first-choice weight. Runs fine, trains fine, wrong function.
- **`y[tok] = ...` instead of `index_add_`** — each token appears in `k`
  iterations, so assignment silently keeps only the last expert.
- **`expert(xf)[tok]` instead of `expert(xf[tok])`** — computes everything, then
  throws most of it away. That's the trap wearing stage 5's clothes.

## What you should be able to answer at the end

1. Why can a dense softmax gate never skip an expert? (One sentence, about the
   range of `exp`.)
2. Stage 4 zeroes the weights of unselected experts. So what exactly does stage 5
   buy, if the outputs are identical?
3. Why does `nonzero` return *two* index tensors, and what breaks if you ignore
   the second?
4. `sum_i f_i = k` always. Use that to derive the aux loss's minimum value —
   why isn't it zero?
5. Why is `f` not differentiable, and why doesn't that stop the loss from working?
6. Your `macs_per_token()` and parameter count now differ, where in `DenseMoe`
   they were equal. Which quantity did sparsity change, and which did it not?

## When it passes

```bash
MOE_IMPL=scratch python SparseMoe/run_sparse_moe.py
```

The same dissector that runs the reference, driven by your code: the dispatch
path tensor by tensor, all three implementations agreeing, what the trap
actually costs, the params/FLOPs divergence, the routing table with its silent
bug, ragged loads and starved experts, and the renormalisation that freezes the
router at `k=1`.
