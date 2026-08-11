import torch, torch.nn as nn, torch.nn.functional as F

class Expert(nn.Module):
    """One plain FFN. d -> d_ff -> d, so it maps a token back to its own shape.

    An "expert" is nothing special: it is exactly the FFN block a normal
    transformer already has. MoE just keeps N of them side by side.
    """

    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        # (..., d) -> (..., d_ff) -> silu -> (..., d). Shape-preserving, and the
        # leading axes are untouched, so it runs on (B, T, d) unchanged.
        return self.w2(F.silu(self.w1(x)))

class DenseMoE(nn.Module):
    def __init__(self, d_model, d_ff, n_experts):
        super().__init__()
        self.d_model, self.d_ff, self.n_experts = d_model, d_ff, n_experts
        self.gate = nn.Linear(d_model, n_experts, bias=False)   # weight (N, d)
        self.experts = nn.ModuleList([Expert(d_model, d_ff) for _ in range(n_experts)])

    def forward(self, x):
        """y[b,t] = sum_j  softmax(W_g @ x[b,t])[j] * E_j(x[b,t])

        Every expert runs on every token; the gate says how much each one counts.

            x       (B, T, d)       input tokens
              |  gate: Linear(d -> N)
            logits  (B, T, N)       one score per (token, expert)
              |  softmax over the LAST axis = over experts, per token
            w       (B, T, N)       non-negative, rows sum to 1

            x       (B, T, d)       the same x, sent to all N experts
              |  stack on a NEW axis -2
            outs    (B, T, N, d)    every expert's answer for every token

            w.unsqueeze(-1)         (B, T, N, 1)   broadcast against outs
            * outs                  (B, T, N, d)   scale each expert's answer
            .sum(dim=-2)            (B, T, d)      collapse the expert axis

            y       (B, T, d)       same shape as x -> drop-in FFN replacement

        Three things worth internalising:

        1. Routing is PER TOKEN, not per sequence. The softmax is over dim=-1
           (experts) while T is still a separate axis, so each token in the
           sequence gets its own independent mixture. That is why N sits next
           to T in every intermediate shape.

        2. The weights are non-negative and sum to 1, so y is a weighted AVERAGE
           of the expert outputs - it can never leave their convex hull. A dense
           MoE only interpolates; all the capacity lives in the experts being
           different from each other, not in the gate.

        3. This is the bottleneck. Nothing here is skipped: N experts x every
           token, forward and backward. Active params == total params, so N x
           the parameters costs N x the FLOPs, and the whole selling point of
           MoE ("more params at the same compute") is not yet delivered. Sparse
           routing is what fixes this - it keeps steps 1 and 2 and makes the
           stack in the middle only compute the k experts a token actually uses.
        """
        # x: (B, T, d)

        # (B,T,d) @ (d,N) -> (B,T,N) logits, then softmax along -1 so each token's
        # N weights form a distribution. dim=-1 is the expert axis, NOT d.
        w = F.softmax(self.gate(x), dim=-1)                     # (B, T, N)

        # Run all N experts on the full input and stack their outputs on a new
        # axis inserted before d. This list comprehension IS the dense cost.
        outs = torch.stack([e(x) for e in self.experts], dim=-2) # (B, T, N, d)

        # unsqueeze(-1) turns w into (B,T,N,1): one scalar per expert that has to
        # multiply all d of that expert's output dims, so it needs a trailing
        # size-1 axis to broadcast along d. Then sum away the expert axis (-2).
        return (w.unsqueeze(-1) * outs).sum(dim=-2)             # (B, T, d)

    # ---------------------------------------------------------------------
    # Two more spellings of forward(), kept as cross-checks rather than for
    # use. Anything that computes the same thing a second way is a test you
    # get for free - see run_dense_moe.py section 2.
    # ---------------------------------------------------------------------

    def forward_loop(self, x):
        """forward() as explicit python loops. Slow, and the one to trust.

        Every tensor in here is 1-D or scalar, which is the point: `self.gate(h)`
        is (N,), so a softmax over it has only one axis to pick, and `gw[j]` is a
        scalar, so no broadcast has to line up. Neither of the two mistakes the
        vectorised version can make silently is even expressible here. When the
        two disagree, this is the one that's right.
        """
        B, T, d = x.shape                       # B and T come from the DATA, not
        y = torch.zeros_like(x)                 # from self - the layer is
        for b in range(B):                      # agnostic to batch and length.
            for t in range(T):
                h = x[b, t]                                     # (d,)
                gw = F.softmax(self.gate(h), dim=-1)            # (N,)
                acc = torch.zeros_like(h)                       # (d,)
                for j, e in enumerate(self.experts):
                    acc = acc + gw[j] * e(h)                    # scalar * (d,)
                y[b, t] = acc
        return y

    def forward_einsum(self, x):
        """forward() with the combine written as one einsum.

        'btn,btnd->btd': n appears on both inputs and not on the output, so n is
        the summed index. That single letter replaces unsqueeze + multiply + sum.
        """
        w = F.softmax(self.gate(x), dim=-1)                     # (B, T, N)
        outs = torch.stack([e(x) for e in self.experts], dim=-2)  # (B, T, N, d)
        return torch.einsum('btn,btnd->btd', w, outs)           # (B, T, d)

    def macs_per_token(self):
        """Multiply-accumulates per token for one forward pass, derived not measured.

        Linear layers only. A Linear (p -> q) costs p*q MACs per token, each
        expert is two of them, and the gate is d*N. Compare the result to
        sum(p.numel() for p in self.parameters()): the two are equal, because a
        dense layer spends exactly one MAC per parameter per token. That identity
        is the bottleneck in one line.
        """
        per_expert = 2 * self.d_model * self.d_ff
        return self.d_model * self.n_experts + self.n_experts * per_expert
