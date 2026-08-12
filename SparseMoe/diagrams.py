"""Diagram source for the sparse MoE notes - same style as the main page's generator.

Regenerate with:  python SparseMoe/diagrams.py
"""

RAMP = {
    "gray":   ("#F1EFE8", "#5F5E5A", "#2C2C2A", "#5F5E5A"),
    "purple": ("#EEEDFE", "#534AB7", "#3C3489", "#534AB7"),
    "coral":  ("#FAECE7", "#993C1D", "#712B13", "#993C1D"),
    "teal":   ("#E1F5EE", "#0F6E56", "#085041", "#0F6E56"),
}
PRI, SEC, ARR = "#3D3D3A", "#73726C", "#73726C"
F = 'font-family="DejaVu Sans, Helvetica, Arial, sans-serif"'

DEFS = ('<defs><marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" '
        'markerHeight="6" orient="auto-start-reverse"><path d="M2 1L8 5L2 9" fill="none" '
        'stroke="#73726C" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>'
        '</marker>'
        '<rect id="k" width="20" height="20" rx="3" stroke-width="0.5"/></defs>')


def cell(x, y, ramp, op=1.0):
    fill, stroke, _, _ = RAMP[ramp]
    o = f' opacity="{op}"' if op != 1.0 else ""
    return (f'<use href="#k" xlink:href="#k" x="{x}" y="{y}" fill="{fill}" '
            f'stroke="{stroke}"{o}/>')


def strip(x, y, ramp, n=4, op=1.0):
    return "".join(cell(x + i * 22, y, ramp, op=op) for i in range(n))


def txt(x, y, s, size=12, fill=PRI, anchor="start", weight="normal"):
    return (f'<text {F} x="{x}" y="{y}" text-anchor="{anchor}" dominant-baseline="central" '
            f'font-size="{size}" font-weight="{weight}" fill="{fill}">{s}</text>')


def box(x, y, w, h, ramp, title, sub=None, dash=False):
    fill, stroke, tc, sc = RAMP[ramp]
    d = ' stroke-dasharray="4 4"' if dash else ""
    s = (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="{fill}" '
         f'stroke="{stroke}" stroke-width="0.5"{d}/>')
    cx = x + w / 2
    if sub is None:
        s += txt(cx, y + h / 2, title, 13, tc, "middle", "500")
    else:
        s += txt(cx, y + 17, title, 13, tc, "middle", "500") + txt(cx, y + 35, sub, 11, sc, "middle")
    return s


def arrow(x1, y1, x2, y2, color=ARR, dash=False):
    d = ' stroke-dasharray="4 4" opacity="0.45"' if dash else ""
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" '
            f'stroke-width="1.5" stroke-linecap="round" marker-end="url(#arrow)"{d}/>')


def wrap(w, h, title, desc, body):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'xmlns:xlink="http://www.w3.org/1999/xlink" width="{w}" height="{h}" '
            f'viewBox="0 0 {w} {h}" role="img"><title>{title}</title><desc>{desc}</desc>'
            f'<rect width="{w}" height="{h}" fill="#FFFFFF"/>{DEFS}{body}</svg>')


# ---------------------------------------------------------------- 20. the flatten
def flatten_diagram():
    b = txt(40, 32, "x  —  (B, T, d) = (2, 3, 4)", 13, PRI, "start", "500")
    b += txt(470, 32, "xf  —  (S, d) = (6, 4)", 13, PRI, "start", "500")

    # left: two sequences, 3 tokens each
    ys = [66, 96, 126, 176, 206, 236]
    for i, y in enumerate(ys):
        seq, tok = divmod(i, 3)
        ramp = "coral" if seq == 0 else "teal"
        b += txt(96, y + 10, f"t{tok}", 11, SEC, "end")
        b += strip(104, y, ramp)
    b += txt(40, 56, "sequence 0", 11, RAMP["coral"][1])
    b += txt(40, 166, "sequence 1", 11, RAMP["teal"][1])

    # arrow across
    b += arrow(230, 151, 430, 151)
    b += txt(330, 133, "x.reshape(-1, d)", 12, PRI, "middle", "500")
    b += txt(330, 170, "one row per token", 11, SEC, "middle")

    # right: six rows, contiguous
    for i, y in enumerate([66, 96, 126, 156, 186, 216]):
        seq = 0 if i < 3 else 1
        ramp = "coral" if seq == 0 else "teal"
        b += txt(492, y + 10, f"{i}", 11, SEC, "end")
        b += strip(500, y, ramp)
        b += txt(596, y + 10, f"b{seq}·t{i % 3}", 10, SEC)

    b += txt(340, 285, "Same 24 numbers, same order, no copy — only the indexing changed.",
             12, SEC, "middle")
    b += txt(340, 305, "The router reads one row at a time, so which sequence a row came from",
             12, SEC, "middle")
    b += txt(340, 323, "is information it never uses. y.reshape(B, T, d) puts it back at the end.",
             12, SEC, "middle")
    return wrap(680, 345, "Flattening (B, T, d) to (S, d)",
                "A 2x3 batch of tokens becomes a flat list of 6 rows.", b)


# ------------------------------------------------- 21. one iteration, by shape
def iteration_diagram():
    picked = {1, 4}
    hot, cold = RAMP["coral"][1], SEC
    b = txt(340, 26, "One loop iteration — expert 1, chosen by 2 of the 6 rows",
            13, PRI, "middle", "500")

    # xf on the left
    b += txt(80, 58, "xf  —  (S, d)", 12, PRI, "middle", "500")
    for i in range(6):
        y = 85 + i * 26
        on = i in picked
        b += strip(36, y, "coral" if on else "gray", op=1.0 if on else 0.35)
        b += txt(28, y + 10, f"{i}", 10, hot if on else cold, "end")

    b += arrow(136, 160, 188, 160)
    b += txt(162, 140, "xf[tok]", 11, PRI, "middle", "500")
    b += txt(162, 180, "gather", 10, SEC, "middle")

    # gathered rows
    b += txt(240, 100, "gathered  —  (S_e, d) = (2, 4)", 11, PRI, "middle", "500")
    for j in range(2):
        b += strip(196, 137 + j * 26, "coral")
    b += txt(240, 208, "rows 1 and 4 only", 10, SEC, "middle")

    b += arrow(296, 160, 344, 160)
    b += box(348, 130, 124, 60, "purple", "Expert 1", "(2, d) @ (d, d_ff)")
    b += txt(410, 208, "2 rows, not 6", 10, SEC, "middle")

    b += arrow(476, 160, 516, 160)
    b += txt(564, 108, "× topw[tok, slot]", 11, PRI, "middle", "500")
    for j in range(2):
        b += strip(520, 137 + j * 26, "coral")

    # scatter back down into y
    b += arrow(564, 192, 564, 278)
    b += txt(556, 232, "index_add_(0, tok, ·)", 11, PRI, "end", "500")

    b += txt(564, 296, "y  —  (S, d)", 12, PRI, "middle", "500")
    for i in range(6):
        y = 316 + i * 26
        on = i in picked
        b += strip(520, y, "coral" if on else "gray", op=1.0 if on else 0.35)
        b += txt(512, y + 10, f"{i}", 10, hot if on else cold, "end")

    b += txt(300, 360, "only rows 1 and 4 change —", 12, SEC, "middle")
    b += txt(300, 380, "the other four are still whatever", 12, SEC, "middle")
    b += txt(300, 400, "earlier iterations left there", 12, SEC, "middle")

    b += txt(340, 490, "The (S, N, d) grid is never built. Work scales with S_e, not S·N.",
             12, SEC, "middle")
    return wrap(680, 512, "One expert iteration by shape",
                "Six rows in, two gathered, one GEMM, two rows added back.", b)


# ================================================================ steps_sparse_moe.py
# Diagrams for the S=6, N=3, k=2 run log. Numbers copied from the script's output.

EXPERT_RAMP = ["coral", "teal", "purple"]

LOGITS = [[2.0, 1.0, 0.0], [0.0, 3.0, 1.0], [1.0, 0.5, 4.0],
          [3.0, 2.0, 0.0], [2.5, 1.5, 0.5], [1.0, 0.0, 2.0]]
TOPI = [[0, 1], [1, 2], [2, 0], [0, 1], [0, 1], [2, 0]]
TOPW = [[0.7311, 0.2689], [0.8808, 0.1192], [0.9526, 0.0474],
        [0.7311, 0.2689], [0.7311, 0.2689], [0.7311, 0.2689]]
Y_STEPS = [
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.7311, 0.0, 0.1423, 2.9242, 3.6553, 1.6136],
    [1.2689, 3.5232, 0.1423, 5.0758, 6.3447, 1.6136],
    [1.2689, 4.2384, 8.7154, 5.0758, 6.3447, 14.7727],
]
Y_CHANGED = [set(), {0, 2, 3, 4, 5}, {0, 1, 3, 4}, {1, 2, 5}]


def vcell(x, y, w, h, ramp, label, op=1.0, size=11, weight="normal", dash=False):
    """A labelled box - the workhorse for the value grids below."""
    fill, stroke, tc, _ = RAMP[ramp]
    d = ' stroke-dasharray="3 3"' if dash else ""
    s = (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="3" fill="{fill}" '
         f'stroke="{stroke}" stroke-width="0.5"{d}/>'
         + txt(x + w / 2, y + h / 2 + 0.5, label, size, tc, "middle", weight))
    return f'<g opacity="{op}">{s}</g>' if op != 1.0 else s


# ------------------------------------------------------- 22. the routing table
def routing_table_diagram():
    b = txt(340, 24, "One gate pass: six rows scored, two kept each", 13, PRI, "middle", "500")

    CW, CH, RY = 42, 24, 30
    b += txt(147, 56, "logits  (S, N) = (6, 3)", 11, PRI, "middle", "500")
    for j in range(3):
        b += txt(80 + j * 46 + CW / 2, 74, f"E{j}", 11, SEC, "middle")

    for i in range(6):
        y = 88 + i * RY
        b += txt(72, y + CH / 2, f"row {i}", 11, SEC, "end")
        for j in range(3):
            rank = TOPI[i].index(j) if j in TOPI[i] else None
            ramp = "coral" if rank == 0 else ("teal" if rank == 1 else "gray")
            op = 1.0 if rank is not None else 0.3
            b += vcell(80 + j * 46, y, CW, CH, ramp, f"{LOGITS[i][j]:.1f}", op=op,
                       weight="500" if rank == 0 else "normal")

    b += arrow(238, 178, 292, 178)
    b += txt(265, 158, "topk(2)", 11, PRI, "middle", "500")
    b += txt(265, 198, "then", 10, SEC, "middle")
    b += txt(265, 212, "softmax", 10, SEC, "middle")

    b += txt(356, 56, "topi", 11, PRI, "middle", "500")
    b += txt(506, 56, "topw = softmax(topl)", 11, PRI, "middle", "500")
    for j, name in enumerate(["rank 0", "rank 1"]):
        b += txt(310 + j * 46 + CW / 2, 74, name, 10, SEC, "middle")
        b += txt(410 + j * 62 + 27, 74, name, 10, SEC, "middle")

    for i in range(6):
        y = 88 + i * RY
        for j in range(2):
            e = TOPI[i][j]
            b += vcell(310 + j * 46, y, CW, CH, "coral" if j == 0 else "teal", f"E{e}",
                       weight="500" if j == 0 else "normal")
            b += vcell(410 + j * 62, y, 54, CH, "coral" if j == 0 else "teal",
                       f"{TOPW[i][j]:.4f}")
        b += txt(542, y + CH / 2, f"gap {LOGITS[i][TOPI[i][0]] - LOGITS[i][TOPI[i][1]]:.1f}",
                 10, SEC)

    b += txt(340, 292, "coral = the token's first pick   ·   teal = its second   ·   "
                       "faded = dropped, never computed", 11, SEC, "middle")
    b += txt(340, 316, "topw depends only on the GAP between the two kept logits: rows 0, 3, 4",
             12, SEC, "middle")
    b += txt(340, 334, "and 5 all have gap 1.0, so all four split 0.7311 / 0.2689.",
             12, SEC, "middle")
    return wrap(680, 356, "The routing table",
                "Six rows of logits, top-2 kept, softmax over the survivors.", b)


# --------------------------------------------------- 23. two readings of topi
def two_readings_diagram():
    b = txt(340, 24, "The same table, read two ways", 13, PRI, "middle", "500")

    CW, CH, RY = 46, 24, 30
    b += txt(132, 54, "topi  —  across", 12, PRI, "middle", "500")
    b += txt(132, 70, "\"which experts did I pick?\"", 10, SEC, "middle")
    for j, name in enumerate(["rank 0", "rank 1"]):
        b += txt(86 + j * 50 + CW / 2, 88, name, 10, SEC, "middle")
    for i in range(6):
        y = 100 + i * RY
        b += txt(78, y + CH / 2, f"row {i}", 11, SEC, "end")
        for j in range(2):
            e = TOPI[i][j]
            b += vcell(86 + j * 50, y, CW, CH, EXPERT_RAMP[e], f"E{e}")

    b += arrow(224, 200, 288, 200)
    b += txt(256, 180, "(topi == e_id)", 10, PRI, "middle", "500")
    b += txt(256, 220, ".nonzero()", 10, PRI, "middle", "500")

    b += txt(482, 54, "down the columns", 12, PRI, "middle", "500")
    b += txt(482, 70, "\"which tokens picked me?\"", 10, SEC, "middle")
    rows = [
        (0, "tok  = [0, 2, 3, 4, 5]", "slot = [0, 1, 0, 0, 1]", 5),
        (1, "tok  = [0, 1, 3, 4]", "slot = [1, 0, 1, 1]", 4),
        (2, "tok  = [1, 2, 5]", "slot = [1, 0, 0]", 3),
    ]
    for e, tok_s, slot_s, s_e in rows:
        y = 92 + e * 62
        fill, stroke, tc, sc = RAMP[EXPERT_RAMP[e]]
        b += (f'<rect x="300" y="{y}" width="340" height="52" rx="8" fill="{fill}" '
              f'stroke="{stroke}" stroke-width="0.5"/>')
        b += txt(316, y + 16, f"E{e}", 12, tc, "start", "500")
        b += txt(624, y + 16, f"S_e = {s_e}", 11, sc, "end")
        b += txt(316, y + 34, tok_s, 11, tc)
        b += txt(470, y + 34, slot_s, 11, tc)

    b += txt(340, 300, "Same twelve picks either way. The loop needs the right-hand reading,",
             12, SEC, "middle")
    b += txt(340, 318, "and nonzero returns BOTH coordinates: tok says which row, slot says",
             12, SEC, "middle")
    b += txt(340, 336, "which of that row's k weights belongs to this expert.", 12, SEC, "middle")
    return wrap(680, 358, "Two readings of topi",
                "Per-token picks on the left, per-expert work lists on the right.", b)


# -------------------------------------------------------- 24. y filling up
def y_filling_diagram():
    b = txt(340, 24, "y accumulates: three iterations, six rows", 13, PRI, "middle", "500")
    b += txt(340, 44, "each row is touched exactly k = 2 times, once per expert it chose",
             11, SEC, "middle")

    PX, PW, GAP, CH, RY = 66, 140, 12, 26, 30
    heads = ["y = zeros", "after E0", "after E1", "after E2"]
    for p in range(4):
        x = PX + p * (PW + GAP)
        b += txt(x + PW / 2, 72, heads[p], 11, PRI, "middle", "500")
        for i in range(6):
            y = 90 + i * RY
            if p == 0:
                b += txt(58, y + CH / 2, f"y[{i}]", 11, SEC, "end")
            changed = i in Y_CHANGED[p]
            b += vcell(x, y, PW, CH, "coral" if changed else "gray",
                       f"{Y_STEPS[p][i]:.4f}", op=1.0 if changed else 0.4,
                       weight="500" if changed else "normal")

    b += txt(340, 292, "coral = written on this iteration   ·   faded = untouched, still "
                       "holding an earlier value", 11, SEC, "middle")
    b += txt(340, 318, "Row 2 lands on 0.1423 in the first iteration — it picked E0 SECOND, "
                       "so it", 12, SEC, "middle")
    b += txt(340, 336, "carries the small weight 0.0474 — sits out the second, then jumps to "
                       "8.7154", 12, SEC, "middle")
    b += txt(340, 354, "when E2, its first pick, finally runs. Order of arrival is not order "
                       "of size.", 12, SEC, "middle")
    return wrap(680, 376, "y filling up",
                "The accumulator after each expert iteration.", b)


# ------------------------------------------- 25. what each version computes
def work_grids_diagram():
    b = txt(340, 24, "What actually gets computed", 13, PRI, "middle", "500")

    CW, CH, RY, GX = 30, 22, 26, 34
    panels = [
        (130, "dense", "18 computed, 18 used"),
        (340, "masked", "18 computed, 12 used"),
        (550, "dispatch", "12 computed"),
    ]
    for px, name, sub in panels:
        b += txt(px, 58, name, 12, PRI, "middle", "500")
        for j in range(3):
            b += txt(px - GX + j * GX, 78, f"E{j}", 10, SEC, "middle")

    for i in range(6):
        y = 92 + i * RY
        b += txt(130 - GX - CW / 2 - 10, y + CH / 2, f"row {i}", 10, SEC, "end")
        for px, name, _ in panels:
            for j in range(3):
                x = px - GX - CW / 2 + j * GX
                chosen = j in TOPI[i]
                if name == "dense" or chosen:
                    b += vcell(x, y, CW, CH, EXPERT_RAMP[j], "")
                elif name == "masked":
                    # computed at full cost, then multiplied by zero
                    b += vcell(x, y, CW, CH, EXPERT_RAMP[j], "0", weight="500")
                else:
                    b += (f'<rect x="{x}" y="{y}" width="{CW}" height="{CH}" rx="3" '
                          f'fill="#FFFFFF" stroke="#C9C7C0" stroke-width="0.5" '
                          f'stroke-dasharray="3 3"/>')

    for px, name, sub in panels:
        b += txt(px, 262, sub, 11, SEC, "middle")

    b += txt(340, 290, "filled = computed   ·   0 = computed, then multiplied by zero   ·   "
                       "dashed = never allocated", 11, SEC, "middle")
    b += txt(340, 316, "The masked version's six zeroed cells cost full compute and full "
                       "activation", 12, SEC, "middle")
    b += txt(340, 334, "memory before being deleted. Dispatch never allocates them: the dashed "
                       "cells", 12, SEC, "middle")
    b += txt(340, 352, "are work that does not happen. Same output, 12 expert-rows, not 18.",
             12, SEC, "middle")
    return wrap(680, 374, "Dense vs masked vs dispatch",
                "Which cells of the token x expert grid each version computes.", b)


if __name__ == "__main__":
    for fname, fn in [
        ("20-flatten-to-rows.svg", flatten_diagram),
        ("21-one-iteration-shapes.svg", iteration_diagram),
        ("22-routing-table.svg", routing_table_diagram),
        ("23-two-readings.svg", two_readings_diagram),
        ("24-y-filling.svg", y_filling_diagram),
        ("25-work-grids.svg", work_grids_diagram),
    ]:
        open(fname, "w").write(fn())
        print("wrote", fname)
