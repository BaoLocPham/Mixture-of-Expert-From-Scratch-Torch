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
