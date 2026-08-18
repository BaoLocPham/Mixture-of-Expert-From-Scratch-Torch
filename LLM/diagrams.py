"""Diagram sources for the LLM notes - RoPE, in three pictures.

Regenerate with:  python LLM/diagrams.py

Same palette and helpers as the MoE tracks' generators, reproduced so this
module stands alone. Every number drawn here is computed below, not typed in.
"""

import math

RAMP = {
    "gray":   ("#F1EFE8", "#5F5E5A", "#2C2C2A", "#5F5E5A"),
    "purple": ("#EEEDFE", "#534AB7", "#3C3489", "#534AB7"),
    "coral":  ("#FAECE7", "#993C1D", "#712B13", "#993C1D"),
    "teal":   ("#E1F5EE", "#0F6E56", "#085041", "#0F6E56"),
}
PRI, SEC, ARR = "#3D3D3A", "#73726C", "#73726C"
F = 'font-family="DejaVu Sans, Helvetica, Arial, sans-serif"'
PAIR_RAMP = ["coral", "teal", "purple", "gray"]          # one colour per pair

DEFS = ('<defs><marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" '
        'markerHeight="6" orient="auto-start-reverse"><path d="M2 1L8 5L2 9" fill="none" '
        'stroke="#73726C" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>'
        '</marker></defs>')


def txt(x, y, s, size=12, fill=PRI, anchor="start", weight="normal"):
    return (f'<text {F} x="{x}" y="{y}" text-anchor="{anchor}" dominant-baseline="central" '
            f'font-size="{size}" font-weight="{weight}" fill="{fill}">{s}</text>')


def vcell(x, y, w, h, ramp, label, op=1.0, size=11, weight="normal", dash=False):
    fill, stroke, tc, _ = RAMP[ramp]
    d = ' stroke-dasharray="3 3"' if dash else ""
    s = (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="3" fill="{fill}" '
         f'stroke="{stroke}" stroke-width="0.5"{d}/>'
         + txt(x + w / 2, y + h / 2 + 0.5, label, size, tc, "middle", weight))
    return f'<g opacity="{op}">{s}</g>' if op != 1.0 else s


def arrow(x1, y1, x2, y2, color=ARR, dash=False):
    d = ' stroke-dasharray="4 4" opacity="0.45"' if dash else ""
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" '
            f'stroke-width="1.5" stroke-linecap="round" marker-end="url(#arrow)"{d}/>')


def wrap(w, h, title, desc, body):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'xmlns:xlink="http://www.w3.org/1999/xlink" width="{w}" height="{h}" '
            f'viewBox="0 0 {w} {h}" role="img"><title>{title}</title><desc>{desc}</desc>'
            f'<rect width="{w}" height="{h}" fill="#FFFFFF"/>{DEFS}{body}</svg>')


def brace(x1, x2, y, colour, label, up=False):
    """A flat brace under (or over) a span of cells, with a label."""
    d = -1 if up else 1
    mid = (x1 + x2) / 2
    p = (f'<path d="M{x1} {y} L{x1} {y + 5*d} L{mid - 6} {y + 5*d} L{mid} {y + 10*d} '
         f'L{mid + 6} {y + 5*d} L{x2} {y + 5*d} L{x2} {y}" fill="none" stroke="{colour}" '
         f'stroke-width="1.2" stroke-linejoin="round"/>')
    return p + txt(mid, y + (24 * d), label, 10, colour, "middle")


# ================================================================== the numbers
D_H = 8                                                   # channels in the head
BASE = 10000.0
N_PAIRS = D_H // 2
THETA = [BASE ** (-2 * i / D_H) for i in range(N_PAIRS)]  # E7 rate per pair
WAVELEN = [2 * math.pi / t for t in THETA]


# ------------------------------------------------- 30. pairs, rotate, interleave
def pairing_diagram():
    """What apply_rope actually does to the index layout."""
    CW, CH, GAP = 56, 26, 4
    x0 = 44
    b = txt(340, 24, "apply_rope, one token, one head, d_h = 8", 13, PRI, "middle", "500")
    b += txt(340, 44, "four pairs, four angles, and the same channel order coming out",
             11, SEC, "middle")

    # --- input row -----------------------------------------------------------
    y = 78
    b += txt(x0 - 10, y + CH / 2, "x", 12, PRI, "end", "500")
    for c in range(D_H):
        b += vcell(x0 + c * (CW + GAP), y, CW, CH, PAIR_RAMP[c // 2], f"x{c}")
    for i in range(N_PAIRS):
        xa = x0 + (2 * i) * (CW + GAP)
        xb = x0 + (2 * i + 1) * (CW + GAP) + CW
        b += brace(xa, xb, y + CH + 4, RAMP[PAIR_RAMP[i]][1], f"pair {i}")

    # --- split ---------------------------------------------------------------
    y2 = 196                                              # top of the split cells
    b += arrow(250, y + CH + 40, 176, y2 - 10)
    b += arrow(430, y + CH + 40, 504, y2 - 10)
    b += txt(44, y2 - 30, "x[..., 0::2]", 11, PRI, "start", "500")
    b += txt(44, y2 - 12, "first of each pair", 10, SEC)
    b += txt(636, y2 - 30, "x[..., 1::2]", 11, PRI, "end", "500")
    b += txt(636, y2 - 12, "second of each pair", 10, SEC, "end")
    for i in range(N_PAIRS):
        b += vcell(44 + i * (CW + GAP), y2, CW, CH, PAIR_RAMP[i], f"x{2*i}")
        b += vcell(384 + i * (CW + GAP), y2, CW, CH, PAIR_RAMP[i], f"x{2*i+1}")

    # --- the rotation --------------------------------------------------------
    y3 = 274
    b += arrow(190, y2 + CH + 8, 300, y3 - 16)
    b += arrow(490, y2 + CH + 8, 380, y3 - 16)
    b += txt(340, y3, "each pair turned by its own angle  m · θᵢ", 12, PRI, "middle", "500")
    b += txt(330, y3 + 22, "rot1 = x1·cos − x2·sin", 12, SEC, "end")
    b += txt(350, y3 + 22, "rot2 = x1·sin + x2·cos", 12, SEC, "start")
    b += txt(340, y3 + 42,
             "cos, sin are (T, d_h/2) — one angle per PAIR, never per channel",
             11, SEC, "middle")

    # --- stacked grid --------------------------------------------------------
    y4 = 344                                              # title line
    b += txt(340, y4, "torch.stack([rot1, rot2], dim=-1)   →   (d_h/2, 2)",
             11, PRI, "middle", "500")
    gx, gw = 232, 88
    top = y4 + 16
    for i in range(N_PAIRS):
        yy = top + i * (CH + 3)
        b += txt(gx - 10, yy + CH / 2, f"pair {i}", 10, SEC, "end")
        b += vcell(gx, yy, gw, CH, PAIR_RAMP[i], f"rot1[{i}]")
        b += vcell(gx + gw + 3, yy, gw, CH, PAIR_RAMP[i], f"rot2[{i}]")
    bottom = top + N_PAIRS * (CH + 3)

    # --- flatten back --------------------------------------------------------
    y5 = bottom + 56                                      # top of the out row
    b += arrow(340, bottom + 8, 340, y5 - 8)
    b += txt(356, bottom + 26, ".flatten(-2) reads those rows in order",
             11, PRI, "start", "500")
    b += txt(x0 - 10, y5 + CH / 2, "out", 12, PRI, "end", "500")
    labels = []
    for i in range(N_PAIRS):
        labels += [(f"rot1[{i}]", i), (f"rot2[{i}]", i)]
    for c, (lab, i) in enumerate(labels):
        b += vcell(x0 + c * (CW + GAP), y5, CW, CH, PAIR_RAMP[i], lab, size=9)
    b += txt(340, y5 + CH + 20,
             "same colours, same positions as the input row — channel order restored",
             11, SEC, "middle")

    # --- the other convention ------------------------------------------------
    y6 = y5 + CH + 56
    b += txt(x0 - 10, y6 + CH / 2, "cat", 12, RAMP["gray"][1], "end", "500")
    alt = [(f"rot1[{i}]", i) for i in range(N_PAIRS)] + \
          [(f"rot2[{i}]", i) for i in range(N_PAIRS)]
    for c, (lab, i) in enumerate(alt):
        b += vcell(x0 + c * (CW + GAP), y6, CW, CH, PAIR_RAMP[i], lab, size=9,
                   op=0.55, dash=True)
    b += txt(340, y6 + CH + 20,
             "torch.cat([rot1, rot2], -1) instead: pairs channel j with j + d_h/2.",
             11, SEC, "middle")
    b += txt(340, y6 + CH + 38,
             "That is HuggingFace LLaMA's layout — equally valid, with W_Q and W_K permuted",
             11, SEC, "middle")
    b += txt(340, y6 + CH + 56,
             "to match. Mixing the two conventions between q and k breaks E8 silently.",
             11, SEC, "middle")
    return wrap(680, y6 + CH + 80, "RoPE: pairs, rotation, interleave",
                "How apply_rope slices a head into pairs, rotates each, and puts "
                "the channels back in order.", b)


# ------------------------------------------------------- 31. the frequency ladder
def ladder_diagram():
    """Each pair turns at its own rate: one head carries several scales at once."""
    b = txt(340, 24, "One head, four pairs, four clocks", 13, PRI, "middle", "500")
    b += txt(340, 44, "θᵢ = 10000^(−2i/d_h) — the arrow is the SAME vector, "
             "drawn at positions m = 0…5", 11, SEC, "middle")

    R = 52
    cy = 140
    for i in range(N_PAIRS):
        cx = 108 + i * 158
        ramp = PAIR_RAMP[i]
        line = RAMP[ramp][1]
        b += (f'<circle cx="{cx}" cy="{cy}" r="{R}" fill="{RAMP[ramp][0]}" '
              f'stroke="{line}" stroke-width="0.6"/>')
        for m in range(6):
            a = m * THETA[i]
            x2 = cx + R * math.cos(a)
            y2 = cy - R * math.sin(a)
            op = 0.28 + 0.72 * (m / 5)
            b += (f'<line x1="{cx}" y1="{cy}" x2="{x2}" y2="{y2}" stroke="{line}" '
                  f'stroke-width="{1.0 + 1.2 * (m / 5):.1f}" opacity="{op:.2f}" '
                  f'stroke-linecap="round"/>')
        if i == 0:
            b += txt(cx, cy - R - 16, "faint = m 0   ·   bold = m 5", 10, SEC, "middle")
        b += txt(cx, cy + R + 24, f"pair {i}", 12, RAMP[ramp][2], "middle", "500")
        b += txt(cx, cy + R + 42, f"θ = {THETA[i]:g}", 11, SEC, "middle")
        b += txt(cx, cy + R + 60, f"turns once per", 10, SEC, "middle")
        b += txt(cx, cy + R + 74, f"{WAVELEN[i]:,.0f} tokens", 10, SEC, "middle")

    b += txt(340, 300, "Pair 0 sweeps most of the circle in six tokens; pair 3 has "
             "not visibly moved.", 12, SEC, "middle")
    b += txt(340, 320, "Fine distance lives in the fast pairs, coarse distance in the "
             "slow ones, and", 12, SEC, "middle")
    b += txt(340, 340, "a real head (d_h = 128) spreads 64 of these across the whole "
             "range at once.", 12, SEC, "middle")
    b += txt(340, 372, "This is also where long-context scaling acts: PI, NTK-aware "
             "and YaRN all", 11, SEC, "middle")
    b += txt(340, 390, "change these rates — nothing else in attention moves.",
             11, SEC, "middle")
    return wrap(680, 412, "RoPE frequency ladder",
                "Four pairs of one head, each rotating at its own rate across six "
                "positions.", b)


# ------------------------------------------------------ 32. only the gap survives
def relative_diagram():
    """Two absolute rotations, one relative angle - E8, drawn."""
    q_ang, k_ang = math.radians(35), math.radians(90)     # a fixed q and k, one plane
    theta = 0.35                                          # EXAGGERATED, see the caption
    cases = [(1, 3), (4, 6), (7, 9)]                      # (m, n), all with n - m = 2

    b = txt(340, 24, "Why the score only sees n − m   (E8)", 13, PRI, "middle", "500")
    b += txt(340, 44, "q sits at position m, k at position n — three different (m, n), "
             "always two apart", 11, SEC, "middle")

    # key
    for dx, ramp, lab in ((-190, "coral", "R_m q  —  q, turned by m"),
                          (40, "teal", "R_n k  —  k, turned by n")):
        b += (f'<line x1="{340 + dx}" y1="70" x2="{340 + dx + 18}" y2="70" '
              f'stroke="{RAMP[ramp][1]}" stroke-width="2.4" stroke-linecap="round"/>')
        b += txt(340 + dx + 26, 70, lab, 10, RAMP[ramp][2], "start")

    R = 62
    cy = 168
    for j, (m, n) in enumerate(cases):
        cx = 132 + j * 208
        qa, ka = q_ang + m * theta, k_ang + n * theta
        b += (f'<circle cx="{cx}" cy="{cy}" r="{R}" fill="#FFFFFF" stroke="{SEC}" '
              f'stroke-width="0.6"/>')
        large = 0 if abs(ka - qa) <= math.pi else 1
        rw = R * 0.46
        x1, y1 = cx + rw * math.cos(qa), cy - rw * math.sin(qa)
        x2, y2 = cx + rw * math.cos(ka), cy - rw * math.sin(ka)
        b += (f'<path d="M{cx} {cy} L{x1:.1f} {y1:.1f} A{rw:.1f} {rw:.1f} 0 '
              f'{large} 0 {x2:.1f} {y2:.1f} Z" fill="{RAMP["purple"][0]}" '
              f'stroke="{RAMP["purple"][1]}" stroke-width="0.5"/>')
        for ang, ramp in ((qa, "coral"), (ka, "teal")):
            ex, ey = cx + R * math.cos(ang), cy - R * math.sin(ang)
            b += (f'<line x1="{cx}" y1="{cy}" x2="{ex:.1f}" y2="{ey:.1f}" '
                  f'stroke="{RAMP[ramp][1]}" stroke-width="2.4" stroke-linecap="round"/>')
        gap = ka - qa
        b += txt(cx, cy + R + 26, f"m = {m},  n = {n}", 12, PRI, "middle", "500")
        b += txt(cx, cy + R + 46, f"angle between = {math.degrees(gap):.1f}°",
                 11, RAMP["purple"][2], "middle")
        b += txt(cx, cy + R + 64, f"cos = {math.cos(gap):+.4f}", 11, SEC, "middle")

    b += txt(340, 320, "Both arrows moved. The wedge between them did not.",
             12, PRI, "middle", "500")
    b += txt(340, 348, "⟨R_m q, R_n k⟩ = |q||k| cos(angle),  and the angle is "
             "(∠k − ∠q) + (n − m)·θ,", 12, SEC, "middle")
    b += txt(340, 368, "so m and n reach the score only through their difference. "
             "Absolute position is", 12, SEC, "middle")
    b += txt(340, 388, "applied to each vector and cancels in the product — that is "
             "the whole trick.", 12, SEC, "middle")
    b += txt(340, 418, f"θ is drawn at {theta} rad/token to make the movement visible; "
             "a real θ ranges from", 11, SEC, "middle")
    b += txt(340, 436, "1 down to 1/10000, and the wedge is exactly as constant at "
             "every one of them.", 11, SEC, "middle")
    b += txt(340, 464, "Nothing here is learned and nothing is stored: no position "
             "table, no bias, no", 11, SEC, "middle")
    b += txt(340, 482, "extra parameters at any context length.", 11, SEC, "middle")
    return wrap(680, 508, "RoPE: only the gap survives",
                "Three pairs of rotated vectors with the same positional gap and "
                "the same angle between them.", b)


# --------------------------------------------- 33. the table, built and read
def table_diagram():
    """What rope_tables actually returns, with the numbers in the cells."""
    POS = 6                                               # positions to show
    CW, CH, GAP = 62, 24, 3
    b = txt(340, 24, "rope_tables(d_h = 8, max_seq) — one outer product, then cos and sin",
            13, PRI, "middle", "500")
    b += txt(340, 44, "rows are POSITIONS, columns are PAIRS. Both tables are d_h/2 = 4 "
             "wide, never 8.", 11, SEC, "middle")

    # --- theta row ------------------------------------------------------------
    gx, gy = 232, 96
    b += txt(gx - 12, gy - 34, "θᵢ", 11, PRI, "end", "500")
    for i in range(N_PAIRS):
        b += vcell(gx + i * (CW + GAP), gy - 46, CW, CH, PAIR_RAMP[i],
                   f"{THETA[i]:g}", size=10)
    b += txt(gx + N_PAIRS * (CW + GAP) + 14, gy - 34,
             "rate per pair", 10, SEC, "start")

    # --- m column -------------------------------------------------------------
    b += txt(gx - 74, gy - 12, "m", 11, PRI, "middle", "500")
    for t in range(POS):
        b += vcell(gx - 104, gy + t * (CH + GAP), 60, CH, "gray", f"{t}", size=10)

    # --- the angle grid -------------------------------------------------------
    b += txt(gx + 2 * (CW + GAP), gy - 12, "ang = outer(m, θ) = m · θᵢ   (radians)",
             11, PRI, "middle", "500")
    for t in range(POS):
        for i in range(N_PAIRS):
            a = t * THETA[i]
            b += vcell(gx + i * (CW + GAP), gy + t * (CH + GAP), CW, CH,
                       PAIR_RAMP[i], f"{a:.3f}", size=10,
                       op=1.0 if t == 3 else 0.62)
    b += txt(gx + N_PAIRS * (CW + GAP) + 14, gy + 3 * (CH + GAP),
             "row m = 3", 10, PRI, "start", "500")

    # --- cos / sin ------------------------------------------------------------
    ty = gy + POS * (CH + GAP) + 46
    b += arrow(300, ty - 34, 210, ty - 8)
    b += arrow(380, ty - 34, 470, ty - 8)
    for j, (fn, name, x0) in enumerate(((math.cos, "cos(ang)", 44),
                                        (math.sin, "sin(ang)", 372))):
        b += txt(x0 + 2 * (CW + GAP), ty, name, 12, PRI, "middle", "500")
        for i in range(N_PAIRS):
            b += txt(x0 + i * (CW + GAP) + CW / 2, ty + 2 + 12,
                     f"pair {i}", 9, RAMP[PAIR_RAMP[i]][2], "middle")
        for t in range(POS):
            for i in range(N_PAIRS):
                v = fn(t * THETA[i])
                b += vcell(x0 + i * (CW + GAP), ty + 28 + t * (CH + GAP), CW, CH,
                           PAIR_RAMP[i], f"{v:+.4f}", size=10,
                           op=1.0 if t == 3 else 0.62)
        for t in range(POS):
            b += txt(x0 - 8, ty + 28 + t * (CH + GAP) + CH / 2, f"m={t}", 9, SEC, "end")

    by = ty + 28 + POS * (CH + GAP) + 26
    b += txt(340, by, "These two arrays are the whole of what rope_tables returns.",
             12, PRI, "middle", "500")
    b += txt(340, by + 22, "A token at position m uses ROW m — four cosines and four "
             "sines, one per pair —", 11, SEC, "middle")
    b += txt(340, by + 40, "and every head and every sequence in the batch uses the "
             "same row.", 11, SEC, "middle")
    b += txt(340, by + 66, "Built once for max_seq positions, then sliced: "
             "cos[pos : pos+T].", 11, PRI, "middle", "500")
    b += txt(340, by + 84, "A full forward pass takes rows 0…T−1; a cached "
             "generation step takes the single", 11, SEC, "middle")
    b += txt(340, by + 102, "row for the token it is adding. That slice is the only "
             "place position enters.", 11, SEC, "middle")
    return wrap(680, by + 126, "The RoPE tables",
                "The angle table as an outer product of positions and rates, and "
                "the cos and sin tables built from it.", b)


# -------------------------------------------------- 34. the whole table at once
def diverge(v):
    """-1 -> coral, 0 -> white, +1 -> teal, as the palette's own two ends."""
    if v >= 0:
        return (255 - 240 * v, 255 - 145 * v, 255 - 169 * v)
    t = -v
    return (255 - 102 * t, 255 - 195 * t, 255 - 226 * t)


def table_heatmap_diagram():
    """cos(m·theta_i) over a real head: 32 pairs x 128 positions."""
    DH, MAXPOS = 64, 128
    pairs = DH // 2
    theta = [BASE ** (-2 * i / DH) for i in range(pairs)]
    CW, CH = 3.5, 9.0
    x0, y0 = 84, 104

    b = txt(340, 24, "The cos table of a real head — d_h = 64, 128 positions",
            13, PRI, "middle", "500")
    b += txt(340, 44, "every cell is cos(m · θᵢ): one row per pair, one column per "
             "position", 11, SEC, "middle")

    for i in range(pairs):
        for m in range(MAXPOS):
            r, g, bl = diverge(math.cos(m * theta[i]))
            b += (f'<rect x="{x0 + m * CW:.1f}" y="{y0 + i * CH:.1f}" '
                  f'width="{CW:.1f}" height="{CH:.1f}" '
                  f'fill="rgb({r:.0f},{g:.0f},{bl:.0f})"/>')

    W = MAXPOS * CW
    H = pairs * CH
    b += (f'<rect x="{x0}" y="{y0}" width="{W:.1f}" height="{H:.1f}" fill="none" '
          f'stroke="{SEC}" stroke-width="0.6"/>')

    # axes
    b += txt(x0 + W / 2, y0 + H + 22, "position m  →", 11, PRI, "middle", "500")
    for m in (0, 32, 64, 96, 127):
        b += txt(x0 + (m + 0.5) * CW, y0 + H + 6, f"{m}", 9, SEC, "middle")
    b += txt(x0 - 58, y0 - 30, "pair i", 11, PRI, "start", "500")
    b += txt(x0 - 58, y0 - 16, "(fast at the top)", 9, SEC, "start")
    for i, lab in ((0, "0"), (8, "8"), (16, "16"), (24, "24"), (31, "31")):
        b += txt(x0 - 8, y0 + (i + 0.5) * CH, lab, 9, SEC, "end")

    # callouts
    b += arrow(x0 + W + 44, y0 + 4, x0 + W + 8, y0 + 4)
    b += txt(x0 + W + 50, y0 + 4, "θ = 1", 10, PRI, "start", "500")
    b += txt(x0 + W + 50, y0 + 18, "wraps every", 9, SEC, "start")
    b += txt(x0 + W + 50, y0 + 30, "~6 tokens", 9, SEC, "start")
    b += arrow(x0 + W + 44, y0 + H - 6, x0 + W + 8, y0 + H - 6)
    last = theta[-1]
    b += txt(x0 + W + 50, y0 + H - 6, f"θ = {last:.1e}".replace("e-0", "e−"),
             10, PRI, "start", "500")
    b += txt(x0 + W + 50, y0 + H + 8, "one turn per", 9, SEC, "start")
    b += txt(x0 + W + 50, y0 + H + 20, f"~{2 * math.pi / last / 1000:.0f}k tokens",
             9, SEC, "start")

    # legend
    ly = y0 + H + 46
    for j in range(41):
        r, g, bl = diverge(-1 + 2 * j / 40)
        b += (f'<rect x="{250 + j * 4}" y="{ly}" width="4" height="10" '
              f'fill="rgb({r:.0f},{g:.0f},{bl:.0f})"/>')
    b += txt(244, ly + 5, "cos = −1", 9, SEC, "end")
    b += txt(420, ly + 5, "+1", 9, SEC, "start")

    b += txt(340, ly + 34, "Top rows stripe: the fast pairs cycle several times "
             "inside 128 tokens, so they", 11, SEC, "middle")
    b += txt(340, ly + 52, "pin down local order but repeat — on their own they "
             "cannot tell m from m+6.", 11, SEC, "middle")
    b += txt(340, ly + 74, "Bottom rows are flat: over this whole window the slow "
             "pairs have barely moved,", 11, SEC, "middle")
    b += txt(340, ly + 92, "so they carry \"roughly where in the document\" and "
             "nothing finer.", 11, SEC, "middle")
    b += txt(340, ly + 116, "Read a COLUMN and you have one token's positional "
             "signature: 32 numbers, one per", 11, PRI, "middle", "500")
    b += txt(340, ly + 134, "pair. Read a ROW and you have one pair's clock. The sin "
             "table is this picture,", 11, PRI, "middle", "500")
    b += txt(340, ly + 152, "shifted a quarter turn — together they give each pair an "
             "angle, not just a cosine.", 11, PRI, "middle", "500")
    return wrap(680, ly + 176, "The RoPE cos table as a heatmap",
                "cos(m theta_i) for 32 pairs across 128 positions, fast pairs at "
                "the top.", b)


DIAGRAMS = {
    "30_rope_pairing.svg": pairing_diagram,
    "31_rope_ladder.svg": ladder_diagram,
    "32_rope_relative.svg": relative_diagram,
    "33_rope_table.svg": table_diagram,
    "34_rope_table_heatmap.svg": table_heatmap_diagram,
}

if __name__ == "__main__":
    from pathlib import Path
    out = Path(__file__).resolve().parent / "diagrams"
    out.mkdir(exist_ok=True)
    for name, fn in DIAGRAMS.items():
        (out / name).write_text(fn())
        print("wrote", out / name)
