# Working on this repo

Conventions for anyone — human or agent — touching this codebase.

## What this repo is

Built-and-measured implementations of Mixture-of-Experts. The point is not to
have a working MoE layer; the point is to understand one well enough to argue
with it. Optimise for clarity and verifiability, never for generality,
performance, or reusability.

## Hard rules

1. **Never commit or push filled-in exercise solutions.**
   `DenseMoe/from_scratch/dense_moe.py` is committed as *stubs*. Loc solves it
   locally. Pushing a solved copy destroys the exercise for a fresh clone.
   Stage explicit paths — never `git add -A` or `git add .` — and check
   `git status` before every commit. If a solved stub is the only change,
   commit nothing and say so.
   Improving the *exercise itself* (better hints, a new stage, a grader fix) is
   welcome; the rule is about answers, not content.

2. **Claims must be verified before they are written.**
   Every assertion in a README or docstring needs a script that prints the
   number. Run it and read the output before committing the sentence. If a
   claim can't be checked, cut it. Numbers quoted in prose (MAC counts, gate
   weights, residuals) must be copied from real output, not estimated.

3. **The exercise must not leak its solution.**
   Stub TODOs say *what* and *why*, never *how*. `check.py` hardcodes expected
   **outputs** only — never the formula that produces them. Reading the grader
   must not shortcut the exercise.

## Module shape

Each track follows the same four-part structure:

| File | Role |
|---|---|
| `common.py` | Reference implementation. Commented at the level of individual tensor axes. |
| `steps_*.py` | Printed walkthrough at hand-checkable size: every intermediate tensor. |
| `run_*.py` | Dissector. Runs the reference and prints what you'd otherwise take on faith. |
| `from_scratch/` | The same layer as stubs + a staged grader that diagnoses mistakes. |

Dissectors accept an env var to run against the from-scratch version instead of
the reference: `MOE_IMPL=scratch` for the two MoE tracks, `LLM_IMPL=scratch` for
`LLM/`. Keep that working when editing either side.

`LLM/` follows the symbol table and equation numbering of the notes' equation
page (E1–E43): `d`, `d_h`, `d_ff`, `N`, `k`, `L`, `n_h`, `n_kv`, `m`/`n` for
query and key position, `θ_i`, `α`, `P_tot`/`P_act`. Cite the equation number in
the docstring next to what implements it, and don't introduce a second name for
something that page already names. Where the code deliberately differs from the
notes (the MoE tracks' two-matrix `Expert` vs `LLM/`'s SwiGLU), say so in the
code at the point of difference.

`LLM/from_scratch/expected.py` holds the grader's constants, including the
outputs of several deliberately wrong implementations so the grader can name the
mistake. Regenerate it from `LLM/common.py` if the reference ever changes, and
never put a formula in it.

## Paths

The git root is this directory, so documented commands are repo-root-relative
(`python DenseMoe/run_dense_moe.py`). A clone gets `DenseMoe/` at top level —
don't reintroduce a `MoE/` prefix.

## Current scope

Three tracks: `DenseMoe/`, `SparseMoe/`, and `LLM/` (the transformer the layer
plugs into, with `ffn="dense" | "moe"`). `XMoE/` is a separate track (expert
collapse), deliberately untracked and out of scope: don't commit it or add it to
the README unless asked.

The exercise-solution rule covers `LLM/from_scratch/llm.py` exactly as it covers
the other two: it is committed as stubs and stays that way.

## Git

Author identity is configured **repo-locally** (`BaoLocPham
<phambaoloc163@gmail.com>`). Don't touch global git config. Pushing to `main`
is fine when asked.

## Commands

```bash
python DenseMoe/run_dense_moe.py            # dense MoE, dissected
python DenseMoe/from_scratch/check.py       # graded, stops at first unfinished stage
python LLM/run_llm.py                       # the transformer, dissected (~1 min: it trains)
python LLM/from_scratch/check.py            # six stages, same grader shape
MOE_IMPL=scratch python DenseMoe/run_dense_moe.py
LLM_IMPL=scratch python LLM/run_llm.py
```
