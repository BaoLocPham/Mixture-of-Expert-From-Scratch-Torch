"""
DenseMoE from scratch - your implementation goes here.

Fill in every `TODO`. Check your work at any point with:

    python DenseMoe/from_scratch/check.py

The grader runs the stages in order and stops at the first failure, so just
keep re-running it. It gives you hints, never the answer.

Notation used throughout:
    B    batch size
    T    tokens per sequence
    d    d_model, the token/residual width
    d_ff the expert's hidden width
    N    n_experts

RULES
  - No peeking at ../common.py until you've finished stage 6. That's the
    reference solution; using it early costs you the whole point of this.
  - Only torch. No einops, no loops in the vectorised versions (stages 2-4).
  - Attribute names matter: the grader (and ../run_dense_moe.py) expect
    `self.gate`, `self.experts`, and `w1`/`w2` inside Expert.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================ STAGE 0
class Expert(nn.Module):
    """A single FFN expert: d_model -> d_ff -> d_model, no biases.

    This is just the ordinary transformer FFN block. Nothing MoE-specific.
    """

    def __init__(self, d_model, d_ff):
        super().__init__()
        # TODO stage 0a: two bias-free Linear layers named w1 and w2.
        #   w1 maps d_model -> d_ff
        #   w2 maps d_ff -> d_model
        raise NotImplementedError("stage 0a: Expert.__init__")

    def forward(self, x):
        """(..., d_model) -> (..., d_model). Must preserve all leading axes."""
        # TODO stage 0b: w1, then SiLU, then w2. One line.
        raise NotImplementedError("stage 0b: Expert.forward")


# ============================================================ STAGE 1+
class DenseMoE(nn.Module):
    """Every expert runs on every token; the gate says how much each counts.

        y[b,t] = sum_j  softmax(W_g @ x[b,t])[j] * E_j(x[b,t])
    """

    def __init__(self, d_model, d_ff, n_experts):
        super().__init__()
        self.d_model, self.d_ff, self.n_experts = d_model, d_ff, n_experts
        # TODO stage 1a: create
        #   self.gate    -> a bias-free Linear producing one score per expert.
        #                   Think: what are its in/out features? Its .weight
        #                   will have shape (N, d) - PyTorch stores Linear
        #                   weights transposed.
        #   self.experts -> N Expert modules. It MUST be an nn.ModuleList, not a
        #                   plain python list, or the parameters won't register
        #                   (try `list` afterwards and see what .parameters()
        #                   returns - a genuinely useful mistake to make once).
        raise NotImplementedError("stage 1a: DenseMoE.__init__")

    # -------------------------------------------------------- STAGE 2
    def gate_weights(self, x):
        """(B, T, d) -> (B, T, N), non-negative, summing to 1 over the LAST axis.

        Each token gets its own independent distribution over experts. Routing
        is a per-token decision - nothing here mixes information across T.
        """
        # TODO stage 2: apply the gate, then softmax. Which dim do you softmax
        #   over? Getting this wrong is silent - the shape is identical either
        #   way, only the numbers differ. Ask yourself what "sums to 1" should
        #   mean here: across experts for one token, or across tokens?
        raise NotImplementedError("stage 2: gate_weights")

    # -------------------------------------------------------- STAGE 3
    def expert_stack(self, x):
        """(B, T, d) -> (B, T, N, d): every expert's output for every token.

        The N axis goes *before* d, i.e. at position -2.
        """
        # TODO stage 3: run all N experts on the full x and stack the results.
        #   A list comprehension over self.experts is fine and idiomatic here -
        #   this one loop is over modules, not tokens.
        #   Note what this costs: N full FFN passes over the entire batch. That
        #   is the dense bottleneck, and it is this line.
        raise NotImplementedError("stage 3: expert_stack")

    # -------------------------------------------------------- STAGE 4
    @staticmethod
    def combine(w, outs):
        """Weighted sum over experts.

        w:    (B, T, N)      gate weights
        outs: (B, T, N, d)   per-expert outputs
        ->    (B, T, d)

        This is the step everyone gets wrong the first time. `w` has one scalar
        per (token, expert); that scalar must scale all d components of that
        expert's output. Line up the shapes before you write anything:

            w     (B, T, N)      <- needs to broadcast against...
            outs  (B, T, N, d)

        Broadcasting aligns from the RIGHT, so as-is torch would try to match
        w's N against outs's d. Give w a trailing axis so N meets N, then sum
        the expert axis away.
        """
        # TODO stage 4: broadcast-multiply, then sum over the expert axis.
        raise NotImplementedError("stage 4: combine")

    # -------------------------------------------------------- STAGE 5
    def forward(self, x):
        """(B, T, d) -> (B, T, d). Compose stages 2-4."""
        # TODO stage 5: three lines, or one if you're feeling brave.
        raise NotImplementedError("stage 5: forward")

    # -------------------------------------------------------- STAGE 6
    def forward_loop(self, x):
        """The same thing, written as explicit python loops over b and t.

        Slow and ugly on purpose. If this doesn't match forward() to ~1e-6,
        then one of them is wrong - and this is the one you can reason about
        line by line, so trust it over the clever version.

        Build it as literally as the math reads:
            for each b, for each t:
                h  = x[b, t]                      -> (d,)
                gw = softmax(gate(h))             -> (N,)
                y[b, t] = sum over j of gw[j] * expert_j(h)
        """
        # TODO stage 6: the naive version. No broadcasting tricks allowed.
        raise NotImplementedError("stage 6: forward_loop")

    # -------------------------------------------------------- STAGE 7
    def forward_einsum(self, x):
        """The same thing again, with the combine step as a single einsum.

        Replace stage 4's unsqueeze/multiply/sum with one torch.einsum call.
        You still need gate_weights and expert_stack.

        Work out the subscripts yourself: you have (B,T,N) and (B,T,N,d) going
        to (B,T,d). Which letter appears on both inputs but NOT the output?
        That's the one being summed over.
        """
        # TODO stage 7: one einsum for the combine.
        raise NotImplementedError("stage 7: forward_einsum")

    # -------------------------------------------------------- STAGE 8
    def macs_per_token(self):
        """Analytic cost: multiply-accumulates per token for ONE forward pass.

        Return a plain int, computed from self.d_model / self.d_ff /
        self.n_experts - do NOT measure it, derive it.

        Count only the Linear layers (ignore SiLU, softmax, the weighted sum).
        A Linear with in_features=p, out_features=q costs p*q MACs per token.

        Include: the gate, and all N experts (each expert is two Linears).
        """
        # TODO stage 8: derive the formula.
        #   Then look at what it says: compare it to the total parameter count
        #   you get from sum(p.numel() for p in self.parameters()). The two
        #   numbers come out equal, and that coincidence IS the bottleneck.
        raise NotImplementedError("stage 8: macs_per_token")


# ============================================================ STAGE 9 (stretch)
class BatchedDenseMoE(nn.Module):
    """Same math, no nn.ModuleList and no python loop over experts.

    Instead of N separate Expert modules, hold ONE stacked parameter per layer:

        W1: (N, d_model, d_ff)
        W2: (N, d_ff, d_model)

    and compute all experts in a single batched op (einsum or bmm). This is how
    real MoE kernels are written - you cannot fuse N separate nn.Linear calls,
    but you can batch one stacked tensor.

    Implement forward() to match DenseMoE exactly. `copy_from` is given so the
    grader can compare the two implementations on identical weights; read it,
    since it tells you exactly how your W1/W2 must be laid out.
    """

    def __init__(self, d_model, d_ff, n_experts):
        super().__init__()
        self.d_model, self.d_ff, self.n_experts = d_model, d_ff, n_experts
        self.gate = nn.Linear(d_model, n_experts, bias=False)
        self.W1 = nn.Parameter(torch.empty(n_experts, d_model, d_ff))
        self.W2 = nn.Parameter(torch.empty(n_experts, d_ff, d_model))
        nn.init.normal_(self.W1, std=d_model ** -0.5)
        nn.init.normal_(self.W2, std=d_ff ** -0.5)

    @torch.no_grad()
    def copy_from(self, dense):
        """Load weights from a DenseMoE so the two are numerically comparable."""
        self.gate.weight.copy_(dense.gate.weight)
        for j, e in enumerate(dense.experts):
            # nn.Linear stores (out, in); we want (in, out) per expert.
            self.W1[j].copy_(e.w1.weight.t())
            self.W2[j].copy_(e.w2.weight.t())
        return self

    def forward(self, x):
        """(B, T, d) -> (B, T, d), with no loop over experts."""
        # TODO stage 9: einsum x against W1 to get (B, T, N, d_ff), SiLU,
        #   einsum against W2 to get (B, T, N, d), then combine as before.
        raise NotImplementedError("stage 9: BatchedDenseMoE.forward")
