# DenseMoE from scratch

Build a dense Mixture-of-Experts layer yourself, one piece at a time, with a
grader that checks each piece and tells you what's wrong without telling you
the answer.

```bash
python DenseMoe/from_scratch/check.py      # run this constantly
```

Edit `dense_moe.py`. The grader runs the stages in order and stops at the first
one that isn't done, so you always know what's next.

## The rule that makes this worth doing

`../common.py` is the finished implementation. **Don't open it until you've
passed stage 6.** If you read it first you'll write it from memory in ten
minutes, pass everything, and learn nothing. The whole value here is in getting
stage 4 wrong and working out why.

## Stages

| # | What | The idea |
|---|------|----------|
| 0 | `Expert` | An expert is just an FFN. Nothing MoE-specific yet. |
| 1 | `DenseMoE.__init__` | A gate + N experts. Why `nn.ModuleList` and not a list. |
| 2 | `gate_weights` | Softmax over experts, per token. Which axis? |
| 3 | `expert_stack` | Run all N experts, stack on axis -2. This line *is* the cost. |
| 4 | `combine` | Broadcast the weights against the outputs and sum. **The hard one.** |
| 5 | `forward` | Compose 2-4. |
| 6 | `forward_loop` | The same math as explicit python loops. Proves you understand it. |
| 7 | `forward_einsum` | The combine as one `einsum`. Learn the index notation. |
| 8 | `macs_per_token` | Derive the cost on paper. The punchline is in the number. |
| 9 | `BatchedDenseMoE` | *Stretch.* No ModuleList, no loop — stacked weights and batched matmuls. |

Stage 4 is where the real lesson is. `w` is `(B, T, N)` and `outs` is
`(B, T, N, d)`; broadcasting aligns from the right, so the naive version lines
`N` up against `d`. Sketch the shapes on paper before writing it — that habit is
most of what makes einsum-heavy code tractable later.

Stage 6 exists because it's the only version you can verify by reading. When
the vectorised one disagrees with the loop, the loop is right.

## What you should be able to answer at the end

1. Why is the softmax over `dim=-1` and not `dim=1`? What would routing mean if
   it were over tokens?
2. Why does `w` need `unsqueeze(-1)` before multiplying?
3. Can the output ever be larger than every expert's output on some coordinate?
   Why not?
4. Your `macs_per_token()` and `sum(p.numel() for p in m.parameters())` return
   the same number. Why? And why is that the reason sparse MoE has to exist?
5. In stage 9, why does `copy_from` transpose `w1.weight`?

If (3) or (4) don't have crisp answers yet, run the dissector below — sections 3
and 4 are built around exactly those two questions.

## When it passes

Run the full walkthrough against **your own** implementation:

```bash
MOE_IMPL=scratch python DenseMoe/run_dense_moe.py
```

Same script that runs the reference (`MOE_IMPL` defaults to `common`), now
driven by your code: dim-by-dim flow, a loop-vs-vectorised equality check, the
convex hull property, the cost table, and a 400-step fit where the gate
discovers a token split nobody told it about.

Then read `../common.py` and diff it against what you wrote.
