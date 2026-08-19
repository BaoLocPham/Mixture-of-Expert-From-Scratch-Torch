"""
Build RoPE yourself - the whole of it, on its own.

    python LLM/from_scratch/check_rope.py     # run this constantly
    python LLM/from_scratch/rope.py           # print your tables and look at them

Four stages. The first two are stage 2 of `llm.py` in isolation, with a grader
that goes considerably further; the last two are the parts a real
implementation has and the toy one doesn't - the other channel convention, and
the position scaling every long-context model uses.

Nothing here imports the rest of the track, so this file can be done first.

Notation follows "The equations - with and without MoE" (E7, E8):

    theta_i = base**(-2i/d_h)            one rate per PAIR, i = 0 .. d_h/2 - 1
    angle for position m, pair i         m * theta_i

    ( q'_2i   )   ( cos(m*theta_i)  -sin(m*theta_i) ) ( q_2i   )
    (         ) = (                                 ) (        )     (E7)
    ( q'_2i+1 )   ( sin(m*theta_i)   cos(m*theta_i) ) ( q_2i+1 )

    (R_m q) . (R_n k) = q^T R_(n-m) k                                 (E8)

E8 is the reason the whole thing exists, and stage 2's last check is the only
one that actually tests it.

`../common.py` is the finished version of all four stages: `rotate_half`,
`apply_rope_half` and `rope_tables_scaled` sit at the end of its RoPE section,
reference-only and never called by the model. Opening it before you are done
costs you the exercise. The grader does not read it either - stages 3 and 4 are
checked against the code YOU wrote in stages 1 and 2, so those have to be right
first.
"""

import torch


# ------------------------------------------------------------ stage 1: tables
def rope_tables(d_head, max_seq, base=10000.0, device=None):
    """E7. Precompute the rotation angles. Returns (cos, sin).

    Both come back (max_seq, d_h/2) - HALF as wide as the head, because there
    is one angle per pair of channels, not per channel. If yours is (max_seq,
    d_h) you have one angle per channel and the pairing is already wrong.

    TODO:
      1. one rate per pair: theta_i = base**(-2i/d_h) for i = 0 .. d_h/2 - 1.
         Watch the factor of 2 - the exponent is -2i/d_h, and the sequence 2i
         is 0, 2, 4, ... up to d_h, which torch.arange can give you directly.
      2. one angle per (position, pair): the outer product of the positions
         0 .. max_seq-1 with those rates.
      3. cos and sin of that grid.

    Two ways to get this wrong that both run:
      - the exponent's sign flipped, so the LAST pair is the fast one. The
        model trains and generalises badly.
      - the factor of 2 dropped, so the ladder only spans base**(-1/2) instead
        of base**(-1). Every wavelength is wrong, none of them obviously.

    This is built once, at model construction, and afterwards only sliced:
    cos[pos : pos+T]. It holds no parameters - it is a buffer, not a weight.
    """
    raise NotImplementedError("stage 1: rope_tables")


# ---------------------------------------------------------- stage 2: rotation
def apply_rope(x, cos, sin):
    """E7. Rotate each channel pair of x by its position's angle.

    x: (B, H, T, d_h)   cos, sin: (T, d_h/2)   ->   (B, H, T, d_h)

    TODO:
      1. read x as d_h/2 two-dimensional vectors. This convention - the
         interleaved one, sometimes called GPT-J style - pairs channel 2i with
         channel 2i+1, so the two members of a pair are ADJACENT.
      2. give cos and sin the batch and head axes so they broadcast. Every head
         at position m is rotated by the same angle: position belongs to the
         token, not to the head.
      3. apply E7's matrix. It is 2x2 - both its rows fit on one line each, so
         there is no reason to build a matrix or call a matmul.
      4. put the channels back in the order you found them. `stack` then
         `flatten` interleaves; `cat` does not, and `cat` here is a silent bug.

    Four things the grader checks:
      - the shape is unchanged
      - the length is unchanged (a rotation cannot change |x|). This fails
        first if you paired channels that were never a pair.
      - position 0 is the identity (cos 1, sin 0), so token 0 comes back untouched
      - E8: rotate a fixed q at m and a fixed k at n, and the dot product must
        depend only on n - m. This is the one that matters.

    And one rule the grader cannot check from here, because it is about the
    caller: never rotate v. Position belongs on the address, not on the payload
    being retrieved.
    """
    raise NotImplementedError("stage 2: apply_rope")


# --------------------------------------------------- stage 3: the other layout
def rotate_half(x):
    """The helper the split-half convention is built out of.

    x: (..., d_h)  ->  (..., d_h), the same numbers rearranged and half of them
    negated:  [x0 .. x_{d/2-1}, x_{d/2} .. x_{d-1}]  ->  [-x_{d/2} .. -x_{d-1}, x0 .. x_{d/2-1}]

    TODO: split the last axis in half, and return the second half negated in
    front of the first half.

    Why a function shaped like this exists: with pairs written as (a, b), E7 is
        a*cos - b*sin,  a*sin + b*cos
    which is  x*cos + (-b, a)*sin  once you notice (-b, a) is the whole vector
    turned a quarter turn. `rotate_half` IS that quarter turn, for a layout
    where a's partner is d_h/2 channels away instead of 1.
    """
    raise NotImplementedError("stage 3: rotate_half")


def apply_rope_half(x, cos, sin):
    """The GPT-NeoX / HuggingFace-LLaMA convention: pair j with j + d_h/2.

    x: (B, H, T, d_h)   cos, sin: (T, d_h/2)   ->   (B, H, T, d_h)

    Same tables, same angles, different idea of which two channels form a pair.
    Instead of (0,1), (2,3), ... it uses (0, d_h/2), (1, d_h/2+1), ...

    TODO:
      1. the tables are d_h/2 wide and you now need one entry per CHANNEL, with
         channel j and channel j + d_h/2 sharing an angle. Duplicate them.
      2. x * cos + rotate_half(x) * sin. That single line is the whole rotation
         in this layout - which is why the convention is popular.
      3. broadcast over the batch and head axes as before.

    The grader checks this against `apply_rope` under the permutation that maps
    one layout onto the other. That equality is the point of the stage: the two
    conventions are the SAME function, applied to a different pairing of the
    same channels, and either is fine on its own.

    What is not fine is mixing them - rotate q one way and k the other and E8
    quietly stops holding. No shape error, no crash, just attention that no
    longer sees relative position. It is also why you cannot load a checkpoint
    from one convention into an implementation of the other: HuggingFace
    permutes W_Q and W_K at conversion time so that Meta's weights match this
    layout.
    """
    raise NotImplementedError("stage 3: apply_rope_half")


# --------------------------------------------- stage 4: past the trained length
def rope_tables_scaled(d_head, max_seq, base=10000.0, scale=1.0):
    """Position Interpolation (Chen et al., 2023). Returns (cos, sin).

    A model trained at 2k tokens has never seen an angle larger than
    2048 * theta_i. Feed it 8k and every pair is turned further than anything
    in its training distribution, and quality falls off a cliff.

    PI's answer is the blunt one: squeeze the position axis instead of
    extending it. Position m is rotated as if it were m / scale, so a
    `scale`x longer sequence lands inside the same angular range the model was
    trained on.

    TODO: the same tables as stage 1, with the positions divided by `scale`.
    `scale = 1.0` must reproduce `rope_tables` exactly.

    The grader checks the consequence rather than the code: with scale = s, a
    gap of s tokens now produces the dot product a gap of 1 token used to. That
    is exactly the trade - you keep the model inside its trained range by making
    every position finer-grained, so fine local structure gets compressed along
    with everything else. NTK-aware scaling and YaRN exist because that last
    part is a real cost: they leave the fast pairs alone and stretch only the
    slow ones.
    """
    raise NotImplementedError("stage 4: rope_tables_scaled")


# --------------------------------------------------------------------- look
if __name__ == "__main__":
    torch.set_printoptions(precision=4, sci_mode=False)
    d_h, max_seq = 8, 6

    print(f"rope_tables(d_head={d_h}, max_seq={max_seq})\n")
    try:
        cos, sin = rope_tables(d_h, max_seq)
    except NotImplementedError as e:
        raise SystemExit(f"{e} - write it first, then run this again.")

    print(f"  cos {tuple(cos.shape)}   sin {tuple(sin.shape)}"
          f"   (d_h/2 = {d_h // 2} columns, one per PAIR)\n")
    for m in range(max_seq):
        print(f"  m={m}  cos {[f'{v:+.4f}' for v in cos[m]]}")
    print()
    print("  column 0 should be visibly moving and column "
          f"{d_h // 2 - 1} should barely have started.\n")

    q = torch.tensor([[[[0.6, -0.2, 0.9, 0.4, -0.7, 0.1, 0.3, -0.5]]]])
    try:
        out = apply_rope(q, cos[3:4], sin[3:4])
    except NotImplementedError:
        raise SystemExit("apply_rope not written yet - that is stage 2.")
    print("apply_rope(q, m=3)")
    print(f"  in   {[f'{v:+.4f}' for v in q.flatten()]}")
    print(f"  out  {[f'{v:+.4f}' for v in out.flatten()]}")
    print(f"  |q| {q.norm():.6f} -> {out.norm():.6f}   (a rotation cannot change this)")
