# Mixture of Experts, from scratch

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

Dense first, deliberately. It's the version where nothing is hidden: no discrete
choice, no dropped tokens, no balancing loss, no dead experts. Everything that
makes sparse routing hard is *absent* there, which makes it the right place to
establish what the mixture actually computes — and to measure precisely what
sparsity buys, and at what price.

Sparse then arrives as a strict generalisation: at `k=N` it reproduces the dense
layer exactly, and the dissector checks that rather than asserting it.

## The shape of a module

Each track in this repo is built the same way:

1. **A reference implementation** (`common.py`) — short, commented at the level
   of individual tensor axes.
2. **A dissector** (`run_*_moe.py`) — runs the reference and prints the
   things you'd otherwise take on faith: intermediate shapes, a naive loop
   reproducing the vectorised version, cost accounting, a small fit.
3. **A from-scratch exercise** (`from_scratch/`) — the same layer as stubs, with
   a grader that checks each stage and diagnoses the specific mistake without
   giving up the answer. Rebuilding it is the point; the reference is there to
   diff against afterwards, not to read first.

## Running it

PyTorch is the only dependency. Everything runs on CPU in seconds.

```bash
python DenseMoe/run_dense_moe.py            # dense MoE, dissected
python DenseMoe/from_scratch/check.py       # build it yourself, graded

python SparseMoe/run_sparse_moe.py          # top-k routing + dispatch, dissected
python SparseMoe/from_scratch/check.py      # build that yourself too

MOE_IMPL=scratch python DenseMoe/run_dense_moe.py   # dissect your own version
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

## Next

The parts that make sparse routing actually fast, and actually stable:
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
│   ├── run_dense_moe.py        runnable walkthrough
│   └── from_scratch/
│       ├── README.md           the exercise brief
│       ├── dense_moe.py        stubs to fill in
│       └── check.py            staged grader
└── SparseMoe/
    ├── common.py               MaskedSparseMoE (the trap) + SparseMoE (dispatch)
    ├── run_sparse_moe.py       runnable walkthrough
    └── from_scratch/
        ├── README.md
        ├── sparse_moe.py
        └── check.py
```

## License

MIT.
