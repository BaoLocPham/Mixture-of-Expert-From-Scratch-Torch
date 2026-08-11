import torch, torch.nn as nn, torch.nn.functional as F


class Expert(nn.Module):
    """Identical to DenseMoe/common.py's Expert, reproduced so this module
    stands alone. That it is unchanged is the entire point: sparse MoE does not
    touch the experts, only the gate.

    Note it never cares how many rows it gets - which is exactly what lets the
    dispatch version hand it S_e rows instead of all S.
    """

    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)))                      # (..., d)


def switch_aux_loss(logits, topi, n_experts):
    """Switch Transformer load-balancing loss:  N * sum_i f_i * P_i

    logits: (S, N) raw router scores
    topi:   (S, k) chosen expert ids
    -> scalar

    P is the mean routing *probability* per expert and is differentiable - it is
    the only path the gradient takes. f is the mean *fraction of tokens actually
    dispatched* per expert, which top-k makes non-differentiable, so it acts as a
    detached per-expert weight.

    Minimised at k (not 0) when load is uniform: sum_i f_i = k always, so a
    balanced router has f_i = k/N and P_i = 1/N, giving N * N * (k/N)(1/N) = k.
    The loss grows above k exactly when the router piles probability onto experts
    that are already overloaded - i.e. when f and P correlate.
    """
    P = F.softmax(logits, dim=-1).mean(0)                       # (N,) differentiable
    f = F.one_hot(topi, n_experts).sum(1).float().mean(0)       # (N,) not differentiable
    return n_experts * (f * P).sum()


def load_per_expert(topi, n_experts):
    """(S, k) chosen ids -> (N,) count of rows routed to each expert."""
    return F.one_hot(topi, n_experts).sum(dim=(0, 1))


class MaskedSparseMoE(nn.Module):
    """Sparse *semantics*, dense *compute* - the version everyone writes first.

    The gate is a genuine top-k gate: N-k weights are exactly 0, and those
    experts receive exactly zero gradient. So the learning dynamics really are
    sparse, and for studying routing behaviour or specialisation at small N this
    is the right tool - it keeps everything in one well-shaped dense tensor.

    What it does NOT do is save a single FLOP. Every expert still runs on every
    token; the zeros are applied afterwards, deleting results that already cost
    full compute and full activation memory to produce. Multiplying by zero does
    not un-spend the FLOPs.

    Keep it for experiments. Never mistake it for the fast path.
    """

    def __init__(self, d_model, d_ff, n_experts, k=2):
        super().__init__()
        self.d_model, self.d_ff, self.n_experts, self.k = d_model, d_ff, n_experts, k
        self.gate = nn.Linear(d_model, n_experts, bias=False)
        self.experts = nn.ModuleList([Expert(d_model, d_ff) for _ in range(n_experts)])

    def forward(self, x):
        B, T, d = x.shape
        logits = self.gate(x)                                   # (B, T, N)

        topl, topi = logits.topk(self.k, dim=-1)                # (B, T, k)
        mask = torch.zeros_like(logits).scatter(-1, topi, 1.0)  # (B, T, N) 1 at chosen

        w = F.softmax(logits, dim=-1) * mask                    # (B, T, N) zeros elsewhere
        w = w / w.sum(dim=-1, keepdim=True)                     # renormalise over the k

        outs = torch.stack([e(x) for e in self.experts], dim=-2)   # (B, T, N, d)  <-- all N ran
        y = (w.unsqueeze(-1) * outs).sum(dim=-2)                # (B, T, d)

        aux = switch_aux_loss(logits.reshape(-1, self.n_experts),
                              topi.reshape(-1, self.k), self.n_experts)
        return y, aux

    def macs_per_token(self):
        """Same as a DENSE MoE. That is the whole complaint."""
        return self.d_model * self.n_experts + self.n_experts * 2 * self.d_model * self.d_ff


class SparseMoE(nn.Module):
    """Real sparse MoE: gather -> compute -> scatter, so skipped work is never done.

        y[s] = sum_{i in TopK(h_s, k)}  g_i(x_s) * E_i(x_s)

    The gate output is unchanged from MaskedSparseMoE - what changes is that an
    exact zero is treated as *permission not to compute* rather than a multiplier
    applied after the fact. Each expert receives only the S_e rows routed to it,
    so its matmul shrinks in row count while the weight matrices stay the same
    size. Sum of S_e over all experts is S*k, not S*N.
    """

    def __init__(self, d_model, d_ff, n_experts, k=2):
        super().__init__()
        self.d_model, self.d_ff, self.n_experts, self.k = d_model, d_ff, n_experts, k
        self.gate = nn.Linear(d_model, n_experts, bias=False)
        self.experts = nn.ModuleList([Expert(d_model, d_ff) for _ in range(n_experts)])

    def forward(self, x):
        B, T, d = x.shape

        # Routing is per token, so the (B, T) split is irrelevant inside the
        # layer - flatten to a plain list of S rows and restore the shape at the end.
        xf = x.reshape(-1, d)                                   # (S, d)
        logits = self.gate(xf)                                  # (S, N)

        topl, topi = logits.topk(self.k, dim=-1)                # (S, k) values, ids
        topw = F.softmax(topl, dim=-1)                          # (S, k) sums to 1 per row

        y = torch.zeros_like(xf)                                # (S, d) accumulator

        for e_id, expert in enumerate(self.experts):
            # tok  = WHICH TOKENS chose this expert
            # slot = WHICH RANK it held for that token (0 = first pick, 1 = second)
            tok, slot = (topi == e_id).nonzero(as_tuple=True)   # both (S_e,)

            if tok.numel() == 0:
                continue            # nothing routed here - skip the launch entirely

            # xf[tok] copies the selected rows into a fresh contiguous buffer. That
            # copy is not waste: it turns what would be a strided gather inside the
            # kernel into a dense, well-shaped GEMM. You pay a gather to earn a matmul.
            out = expert(xf[tok])                               # (S_e, d)

            # topw[tok, slot] needs BOTH coordinates. topw[tok, 0] would take this
            # token's *first-choice* weight even when this expert was its second
            # choice - a bug that still trains, just with scrambled gate weights.
            # index_add_ (not assignment) because each token appears in k iterations
            # of this loop and its k contributions must accumulate. This is where
            # the sum over the top-k set actually happens.
            y.index_add_(0, tok, topw[tok, slot, None] * out)

        aux = switch_aux_loss(logits, topi, self.n_experts)
        return y.reshape(B, T, d), aux

    def macs_per_token(self):
        """Gate + only the k experts a token actually visits.

        Compare with sum(p.numel() for p in self.parameters()): unlike every dense
        layer, these two now DIVERGE, by a factor of about N/k. Parameters scale
        with N, compute scales with k. That decoupling is the entire architectural
        argument for MoE, and it is the number the whole DenseMoe module was
        building up to.
        """
        return self.d_model * self.n_experts + self.k * 2 * self.d_model * self.d_ff

    @torch.no_grad()
    def copy_from(self, other):
        """Load weights from a MaskedSparseMoE (or another SparseMoE) so the two
        can be compared numerically. They compute the same function."""
        self.gate.weight.copy_(other.gate.weight)
        for mine, theirs in zip(self.experts, other.experts):
            mine.w1.weight.copy_(theirs.w1.weight)
            mine.w2.weight.copy_(theirs.w2.weight)
        return self
