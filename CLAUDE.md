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

Each track follows the same three-part structure:

| File | Role |
|---|---|
| `common.py` | Reference implementation. Commented at the level of individual tensor axes. |
| `run_*.py` | Dissector. Runs the reference and prints what you'd otherwise take on faith. |
| `from_scratch/` | The same layer as stubs + a staged grader that diagnoses mistakes. |

Dissectors accept `MOE_IMPL=scratch` to run against the from-scratch version
instead of the reference. Keep that working when editing either.

## Paths

The git root is this directory, so documented commands are repo-root-relative
(`python DenseMoe/run_dense_moe.py`). A clone gets `DenseMoe/` at top level —
don't reintroduce a `MoE/` prefix.

## Current scope

Dense MoE only. `XMoE/` is a separate track (expert collapse), deliberately
untracked and out of scope: don't commit it or add it to the README unless
asked. Sparse top-k routing is next.

## Git

Author identity is configured **repo-locally** (`BaoLocPham
<phambaoloc163@gmail.com>`). Don't touch global git config. Pushing to `main`
is fine when asked.

## Commands

```bash
python DenseMoe/run_dense_moe.py            # dense MoE, dissected
python DenseMoe/from_scratch/check.py       # graded, stops at first unfinished stage
MOE_IMPL=scratch python DenseMoe/run_dense_moe.py
```
