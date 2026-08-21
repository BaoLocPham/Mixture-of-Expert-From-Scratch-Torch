# Mixture of Experts, from scratch

I have to admit I don't know any of this yet, so this repo is how I'm learning
it from scratch.

Small, self-contained PyTorch implementations of MoE, each one built up from
nothing and then taken apart numerically.

## Why this exists

The MoE literature is written at an altitude where the interesting parts are
already assumed. A paper will say the router is a softmax over experts and move
on — which is enough to reimplement the equation, and nowhere near enough to
know why renormalising over the top-k changes what the router can learn, or
which of the design choices in a given architecture are load-bearing and which
are inherited.

So the rule here is that nothing gets described, only built and measured. Every
claim in this repo is a script you can run: if a property is asserted, there's a
line that checks it and prints the residual. Where an intuition turned out to be
wrong, the check that caught it stays in the file.

The aim is to get close enough to the mechanism to have opinions about it.

## What's here

| Module | What it covers |
|---|---|
| `DenseMoe/` | The dense mixture: every expert on every token. Flow, dims, and why it can't scale. |
| `SparseMoe/` | Top-k routing and real token dispatch: what it saves, what it costs, what it breaks. |
| `LLM/` | The model the layer plugs into: a decoder-only transformer, built up part by part, with the FFN slot switchable between one network and a routed mixture. |

Dense first, deliberately. It's the version where nothing is hidden: no discrete
choice, no dropped tokens, no balancing loss, no dead experts. Everything that
makes sparse routing hard is *absent* there, which makes it the right place to
establish what the mixture actually computes — and to measure precisely what
sparsity buys, and at what price.

Sparse then arrives as a strict generalisation: at `k=N` it reproduces the dense
layer exactly, and the dissector checks that rather than asserting it.

`LLM/` closes the loop. An MoE layer is not a model, and studying it in
isolation quietly assumes the surrounding architecture is a detail. So the same
transformer is built from the norm up — RMSNorm, RoPE, causal attention with
grouped query heads and a KV cache, pre-norm residuals, tied embeddings — and
the FFN slot takes either a plain feed-forward network or the routed one. Every
"with MoE vs without" number in that track is a one-word diff on the same model,
same seed, same batches.

A second switch works the same way: `attn="vanilla"` replaces that layer with
the one *Attention Is All You Need* wrote — sinusoidal position added at the
input, every query head with its own k and v, biases on all four projections —
so the seven years between the two can be priced instead of argued about.

There are two model classes, and the split is deliberate. **`TinyLLM`** is
`embed → Block × L → norm → logits` and nothing else: no switches, no cache, no
sampling, so the shape of a language model is readable before the machinery
around it is. **`LLM`** is the same model with the four things a real one needs
— the attention switch, the FFN switch, weight tying, and incremental decoding
— and it is what every printed number in the track comes from.

## Notation

The `LLM/` track follows one symbol table throughout — `d`, `d_h`, `d_ff`, `N`,
`k`, `L`, `n_h`, `n_kv`, `m` and `n` for query and key position, `θ_i` for the
rotation rate of channel pair `i`, `α` for the balancing coefficient,
`P_tot`/`P_act` for total and active parameters. Every equation it implements is
cited by number (E1–E43) in the docstring next to it, so a printed row and the
line of maths it came from can be read side by side. Where the two MoE tracks
differ — their `Expert` is the older two-matrix FFN, while `LLM/` uses the
three-matrix SwiGLU (E18) that current models actually use — the code says so at
the point of difference rather than leaving it to be discovered.

## The shape of a module

Each track in this repo is built the same way:

1. **A reference implementation** (`common.py`) — short, commented at the level
   of individual tensor axes.
2. **A dissector** (`run_*_moe.py`) — runs the reference and prints the
   things you'd otherwise take on faith: intermediate shapes, a naive loop
   reproducing the vectorised version, cost accounting, a small fit. Where a
   mechanism is easier to *see* than to measure, a `steps_*.py` companion
   shrinks it to a handful of hand-written rows and prints every intermediate.
3. **A from-scratch exercise** (`from_scratch/`) — the same layer as stubs, with
   a grader that checks each stage and diagnoses the specific mistake without
   giving up the answer. Rebuilding it is the point; the reference is there to
   diff against afterwards, not to read first.

## Running it

PyTorch is the only dependency. Everything runs on CPU in seconds.

```bash
python DenseMoe/steps_dense_moe.py          # 3 tokens, 3 experts, every number printed
python DenseMoe/run_dense_moe.py            # dense MoE, dissected
python DenseMoe/from_scratch/check.py       # build it yourself, graded

python SparseMoe/steps_sparse_moe.py        # the same layer at S=6, every number printed
python SparseMoe/run_sparse_moe.py          # top-k routing + dispatch, dissected
python SparseMoe/from_scratch/check.py      # build that yourself too

python LLM/steps_llm.py                     # 4 tokens through one layer, printed
python LLM/run_llm.py                       # the whole model, dissected (~1 min: it trains)
python LLM/from_scratch/check.py            # build the transformer yourself, graded

MOE_IMPL=scratch python DenseMoe/run_dense_moe.py   # dissect your own MoE
LLM_IMPL=scratch python LLM/run_llm.py              # dissect your own transformer
```

## What the numbers have settled so far

- **A dense MoE can only interpolate.** Gate weights are non-negative and sum to
  1, so the output never leaves the convex hull of the expert outputs — verified
  per-coordinate. Capacity comes from the experts differing, not from the gate.
  A gate that outputs uniform weights buys you one blurry FFN at N times the
  price.
- **Dense MoE delivers none of the MoE promise.** MACs per token and total
  parameters come out as the *same number* (792 for `d=8, d_ff=16, N=3`), so N×
  the parameters costs N× the compute, and active params are 100% of the model.
  The entire appeal of MoE — more parameters at fixed compute per token — is
  still unpaid for at this point.
- **Routing is learned, not designed.** On a two-mode synthetic task the gate
  converges to `[0.05, 0.95]` / `[0.95, 0.05]` on the two token types, and the
  experts specialise accordingly. Nothing told it the types exist.
- **Sparsity finally splits the two numbers.** With `N=6, k=2` the layer holds
  1,584 parameters but spends 560 MACs/token — 35%, where dense was pinned at
  100%. Parameters scale with `N`, compute with `k`. That decoupling is the
  whole architectural argument, and it is the first thing in this repo that
  dense could not do.
- **Masked gating buys nothing.** Zeroing the gate *after* computing all `N`
  experts gives byte-identical outputs to real dispatch and identical sparse
  learning dynamics — while spending `N/k` times the FLOPs and materialising an
  `(S, N, d)` intermediate that autograd holds for the backward pass. It is a
  legitimate research tool and never a fast path.
- **"Top-k then softmax" and "softmax then top-k then renormalise" are the same
  function** (agreement to 6e-08) — the shared denominator cancels. The real
  fork is renormalising *at all*: at `k=1` a renormalised gate weight is the
  constant 1.0, so the router's gradient is **exactly zero** and it never learns.
  Switch multiplies by the un-normalised probability precisely to avoid this.

- **A transformer is two alternating jobs, and only one of them is MoE's
  business.** Attention mixes across positions and computes almost nothing per
  token; the FFN computes everything per token and cannot see another token. MoE
  makes the second one cheap per parameter and does not touch the first — at
  `T=4096` the score/value matmuls cost 1,572,864 MACs/token against 73,728 for
  the FFN, so a routed FFN changes nothing about long-context cost.
- **Swapping in MoE moves exactly one number.** Same model, `ffn="moe"` with
  `N=8, k=1`: 5.13× the parameters at 1.01× the MACs per token, a `P_tot/P_act`
  of 5.07 where every dense model ever built sits at 1.00.
- **And on a task with no use for capacity, MoE loses — measurably.** Three
  seeds each, matched compute: dense reaches 0.0078 mean cross-entropy on the
  part of the task that carries signal, routing with `α=0` reaches 0.0130, and
  routing with `α=0.01` reaches 0.0256. The ranking holds on every seed. Two
  separate costs, cleanly separated by those three configs: the discrete choice
  itself, and then the balancing term, which adds an objective the language
  model never asked for and is paid for out of the same weights. That second one
  is the entire motivation for aux-loss-free balancing.
- **What the balancing loss buys, in the only place it is visible.** With it,
  the busiest expert sees 2.84× the quietest and no expert sits idle in any
  layer. Without it, 7.01×, and (layer, expert) slots start going completely
  unused — which means those experts stop receiving gradient at all, and the
  imbalance feeds itself.
- **RoPE gives relative position for free.** Rotating `q` by `m` and `k` by `n`
  leaves `q·k` depending only on `n - m`, verified to 1.6e-06 across absolute
  offsets — zero learned position parameters, and the only thing bounding context
  length is the size of a precomputed table.
- **The KV cache is exact, and it is not optional.** Feeding a sequence one token
  at a time agrees with one full pass to 2.4e-07, and generating 64 tokens after a
  128-token prompt is ~2.6× faster — a gap that widens with the prefix, because
  the work it removes is quadratic. Grouped query attention halves what that
  cache holds (384 → 192 floats per token here) without touching a single
  query head.

## Next

Now that there is a model to put them in: capacity factors and token dropping,
measured on a real loss rather than on a routing table; the parts that make
sparse routing actually fast, and actually stable:
capacity factors and token dropping; grouped/block-sparse GEMM, since the naive
dispatch loop spends the right FLOPs in badly-shaped matmuls and can lose to
dense at small `N`; expert parallelism and its two all-to-alls; and the
aux-loss-free balancing DeepSeek-V3 uses to keep the balancing term from
fighting the language-modelling objective.

## Layout

```
.
├── DenseMoe/
│   ├── common.py               reference implementation
│   ├── steps_dense_moe.py      3 tokens, 3 experts, every intermediate printed
│   ├── run_dense_moe.py        runnable walkthrough
│   ├── diagrams.py             SVG sources for the notes
│   └── from_scratch/
│       ├── README.md           the exercise brief
│       ├── dense_moe.py        stubs to fill in
│       └── check.py            staged grader
├── SparseMoe/
│   ├── common.py               MaskedSparseMoE (the trap) + SparseMoE (dispatch)
│   ├── steps_sparse_moe.py     6 rows, 3 experts, every intermediate printed
│   ├── run_sparse_moe.py       runnable walkthrough
│   ├── diagrams.py             SVG sources for the notes
│   └── from_scratch/
│       ├── README.md
│       ├── sparse_moe.py
│       └── check.py
└── LLM/
    ├── common.py               RMSNorm, RoPE, both attention layers, blocks,
    │                           TinyLLM (read first) and LLM (every switch)
    ├── steps_llm.py            4 tokens through one layer, every intermediate printed
    ├── run_llm.py              the whole model, dissected - dense against MoE
    ├── diagrams.py             SVG sources for the RoPE and attention notes
    └── from_scratch/
        ├── README.md           the exercise brief
        ├── llm.py              stubs to fill in
        ├── check.py            staged grader
        └── expected.py         the numbers it checks against
```

## License

MIT.
