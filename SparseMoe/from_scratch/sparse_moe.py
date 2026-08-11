"""
SparseMoE from scratch - your implementation goes here.

Fill in every `TODO`. Check your work at any point with:

    python SparseMoe/from_scratch/check.py

Assumes you have finished DenseMoe/from_scratch. The Expert is unchanged and is
given to you here; everything new is in the gate and the dispatch.

Notation:
    B, T   batch, tokens per sequence
    S      B*T, the flattened token count - routing is per token, so the (B,T)
           split is irrelevant inside the layer
    d      d_model
    d_ff   expert hidden width
    N      n_experts, held in memory
    k      experts activated per token
    S_e    rows routed to expert e. Data-dependent; sum over e of S_e = S*k

RULES
  - No peeking at ../common.py until you're done.
  - Stage 3 (MaskedSparseMoE) is deliberately the easy one and stage 4
    (SparseMoE) the hard one. They must produce IDENTICAL numbers. That is the
    whole test: one is obviously correct, the other is fast.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# Given - identical to DenseMoe's Expert. Sparse MoE does not touch the experts.
class Expert(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)))


# ============================================================ STAGE 1
def top_k_gate(logits, k):
    """Turn router logits into a sparse gate.

    logits: (S, N)
    ->      (topw, topi), both (S, k)
            topw = the gate weight for each chosen expert, each row summing to 1
            topi = the chosen expert ids, in 0..N-1

    Use the Mixtral convention: take the top-k logits, then softmax over just
    those k. (Softmaxing over all N first and renormalising afterwards gives the
    identical result - the shared denominator cancels. Worth proving to yourself
    on paper; the grader checks that your answer matches both.)
    """
    # TODO stage 1: torch.topk, then a softmax. Mind which axis.
    raise NotImplementedError("stage 1: top_k_gate")


# ============================================================ STAGE 2
def load_per_expert(topi, n_experts):
    """How many rows each expert was given.

    topi: (S, k) chosen ids
    ->    (N,) integer counts, summing to S*k

    Do it without a python loop over experts. F.one_hot is the tool.
    """
    # TODO stage 2: one_hot then sum. Which axes disappear?
    raise NotImplementedError("stage 2: load_per_expert")


# ============================================================ STAGE 3
def switch_aux_loss(logits, topi, n_experts):
    """Switch Transformer load-balancing loss:  N * sum_i f_i * P_i

    logits: (S, N)      raw router scores
    topi:   (S, k)      chosen expert ids
    ->      scalar

    Two per-expert vectors, both shape (N,):
      P = the mean routing PROBABILITY  (softmax the logits, average over tokens)
      f = the mean fraction of tokens DISPATCHED to each expert

    Then return N * sum(f * P).

    The asymmetry is the whole design. P is differentiable and carries the entire
    gradient; f comes from top-k, which is a hard choice with no gradient, so it
    acts as a detached per-expert weight. The product is minimised when the two
    are uncorrelated - i.e. when the router is not piling probability onto
    experts that are already overloaded.

    Sanity check for yourself: sum_i f_i = k always, so at uniform load
    f_i = k/N and P_i = 1/N, and the loss bottoms out at k rather than 0.
    """
    # TODO stage 3: build P and f, then combine.
    raise NotImplementedError("stage 3: switch_aux_loss")


# ============================================================ STAGE 4
class MaskedSparseMoE(nn.Module):
    """Sparse semantics, dense compute - the version everyone writes first.

    Build the top-k gate as a full (B, T, N) tensor with exact zeros in the N-k
    unchosen slots, then do the dense weighted sum exactly as DenseMoE did.

    This is genuinely useful: the zeros really do stop gradient reaching the
    unselected experts, so it has sparse LEARNING dynamics while staying one
    clean dense tensor you can print. It just saves no compute whatsoever, which
    stage 5 exists to fix.
    """

    def __init__(self, d_model, d_ff, n_experts, k=2):
        super().__init__()
        self.d_model, self.d_ff, self.n_experts, self.k = d_model, d_ff, n_experts, k
        self.gate = nn.Linear(d_model, n_experts, bias=False)
        self.experts = nn.ModuleList([Expert(d_model, d_ff) for _ in range(n_experts)])

    def forward(self, x):
        """(B, T, d) -> (y, aux) where y is (B, T, d) and aux is a scalar."""
        # TODO stage 4:
        #   1. logits = self.gate(x)                        (B, T, N)
        #   2. pick the top-k logits and build a 0/1 mask   (B, T, N)
        #      -> torch.zeros_like(...).scatter(-1, topi, 1.0)
        #   3. w = softmax(logits) * mask, then renormalise so each row sums to 1
        #   4. the dense combine, exactly as in DenseMoe
        #   5. aux via switch_aux_loss - it wants (S, N) and (S, k), so reshape
        raise NotImplementedError("stage 4: MaskedSparseMoE.forward")

    def macs_per_token(self):
        """Careful: this is NOT k*2*d*d_ff. Count what actually executes."""
        # TODO stage 4b: what does this version really cost?
        raise NotImplementedError("stage 4b: MaskedSparseMoE.macs_per_token")


# ============================================================ STAGE 5
class SparseMoE(nn.Module):
    """The real thing: gather -> compute -> scatter, so skipped work is never done.

    Must produce numerically identical output to MaskedSparseMoE on the same
    weights. If it doesn't, the masked one is right and this one is wrong.
    """

    def __init__(self, d_model, d_ff, n_experts, k=2):
        super().__init__()
        self.d_model, self.d_ff, self.n_experts, self.k = d_model, d_ff, n_experts, k
        self.gate = nn.Linear(d_model, n_experts, bias=False)
        self.experts = nn.ModuleList([Expert(d_model, d_ff) for _ in range(n_experts)])

    def forward(self, x):
        """(B, T, d) -> (y, aux). No expert may ever see a row not routed to it.

        The shape of the solution:

            xf = x.reshape(-1, d)                    (S, d)   flatten - routing
                                                              is per token
            logits = self.gate(xf)                   (S, N)
            topw, topi = top_k_gate(logits, self.k)  (S, k)
            y = torch.zeros_like(xf)                 (S, d)   accumulator

            for e_id, expert in enumerate(self.experts):
                tok, slot = (topi == e_id).nonzero(as_tuple=True)
                ... skip if empty ...
                ... run expert on ONLY those rows, scale, accumulate into y ...

            return y.reshape(B, T, d), aux

        Three things that decide whether this is right:

        1. `tok` and `slot` are the row and column of every True in the (S, k)
           mask. tok says WHICH TOKEN; slot says WHICH RANK this expert held for
           it (0 = first choice, 1 = second). You need both to look up the weight:
           topw[tok, slot]. Using topw[tok, 0] compiles, runs, trains, and is
           wrong - a token's second-choice expert would get its first-choice
           weight. The grader checks specifically for this.

        2. Accumulate, don't assign. Each token appears in k iterations of the
           loop, so its k contributions must ADD. `y[tok] = ...` keeps only the
           last one. index_add_ is the tool. This loop is where the sum over the
           top-k set actually happens - there is no explicit sum anywhere.

        3. Skip empty experts before launching them. An expert with S_e = 0 must
           not run at all; that is the entire point of the exercise.

        The scalar weight needs a trailing axis to broadcast across d - same
        lesson as DenseMoe stage 4, in a new costume.
        """
        # TODO stage 5: the dispatch.
        raise NotImplementedError("stage 5: SparseMoE.forward")

    def macs_per_token(self):
        """Gate plus the k experts a token actually visits.

        Then compare it to sum(p.numel() for p in self.parameters()). In every
        dense layer those two numbers are equal. Here, for the first time, they
        are not - and the ratio is the reason MoE exists.
        """
        # TODO stage 5b.
        raise NotImplementedError("stage 5b: SparseMoE.macs_per_token")

    @torch.no_grad()
    def copy_from(self, other):
        """Given - lets the grader compare you against MaskedSparseMoE."""
        self.gate.weight.copy_(other.gate.weight)
        for mine, theirs in zip(self.experts, other.experts):
            mine.w1.weight.copy_(theirs.w1.weight)
            mine.w2.weight.copy_(theirs.w2.weight)
        return self
