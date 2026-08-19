"""Diagram sources for the LLM notes: RMSNorm, RoPE, and attention.

Regenerate with:  python LLM/diagrams.py

Same palette and helpers as the MoE tracks' generators, reproduced so this
module stands alone. Every number drawn here is computed below, not typed in.
"""

import math
import re

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
    # a bare & is not valid XML, and "Zhang & Sennrich" broke the renderer once
    s = re.sub(r"&(?!(#\d+|#x[0-9a-fA-F]+|[a-zA-Z]+);)", "&amp;", str(s))
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


# ================================================================== RMSNorm
# steps_llm.py section 2 prints these rows; the LayerNorm column is the same
# row through (x - mean) / std, computed here rather than typed.
ROW = [0.6, 0.7, 0.8, 0.9]
RMS = math.sqrt(sum(v * v for v in ROW) / len(ROW))
RMS_OUT = [v / RMS for v in ROW]
MU = sum(ROW) / len(ROW)
SD = math.sqrt(sum((v - MU) ** 2 for v in ROW) / len(ROW))
LN_OUT = [(v - MU) / SD for v in ROW]


def numberline(x, y, w, vals, ramp, lo=-2.0, hi=2.0):
    """A small axis with the values marked, so 'where the row sits' is visible."""
    def sx(v):
        return x + w * (v - lo) / (hi - lo)
    b = f'<line x1="{x}" y1="{y}" x2="{x + w}" y2="{y}" stroke="{SEC}" stroke-width="1"/>'
    for t in (-2, -1, 0, 1, 2):
        b += (f'<line x1="{sx(t):.1f}" y1="{y - 4}" x2="{sx(t):.1f}" y2="{y + 4}" '
              f'stroke="{SEC}" stroke-width="{1.6 if t == 0 else 0.8}"/>')
        b += txt(sx(t), y + 15, f"{t}", 9, SEC, "middle")
    for v in vals:
        b += (f'<circle cx="{sx(v):.1f}" cy="{y}" r="4" fill="{RAMP[ramp][0]}" '
              f'stroke="{RAMP[ramp][1]}" stroke-width="1.4"/>')
    return b


# ------------------------------------------------------- 40. one row, two norms
def norm_row_diagram():
    """What each normalisation does to a single row of the residual stream."""
    CW, CH, GAP = 76, 26, 4
    b = txt(340, 24, "One row of x, through both normalisations", 13, PRI, "middle", "500")
    b += txt(340, 44, "the same four numbers, taken from the run in steps_llm.py",
             11, SEC, "middle")

    x0 = 190
    b += txt(x0 - 12, 84, "x", 12, PRI, "end", "500")
    for i, v in enumerate(ROW):
        b += vcell(x0 + i * (CW + GAP), 72, CW, CH, "gray", f"{v:.1f}")
    b += numberline(190, 128, 320, ROW, "gray")
    b += txt(526, 128, "all positive", 10, SEC, "start")

    b += arrow(290, 158, 208, 188)
    b += arrow(396, 158, 478, 188)

    # LayerNorm branch
    b += txt(160, 196, "LayerNorm", 12, RAMP["coral"][2], "middle", "500")
    b += txt(160, 214, f"subtract μ = {MU:.2f}, divide by σ = {SD:.4f}", 10, SEC, "middle")
    b += txt(160, 230, "then ⊙ γ and + β", 10, SEC, "middle")
    for i, v in enumerate(LN_OUT):
        b += vcell(24 + i * 68, 246, 64, CH, "coral", f"{v:+.3f}", size=10)
    b += numberline(24, 300, 268, LN_OUT, "coral")
    b += txt(158, 330, "mean 0, std 1 — the row was MOVED", 10, RAMP["coral"][2], "middle")

    # RMSNorm branch
    b += txt(516, 196, "RMSNorm", 12, RAMP["teal"][2], "middle", "500")
    b += txt(516, 214, f"divide by RMS = {RMS:.4f}", 10, SEC, "middle")
    b += txt(516, 230, "then ⊙ g", 10, SEC, "middle")
    for i, v in enumerate(RMS_OUT):
        b += vcell(384 + i * 68, 246, 64, CH, "teal", f"{v:+.3f}", size=10)
    b += numberline(384, 300, 268, RMS_OUT, "teal")
    b += txt(518, 330, "RMS 1 — the row was only RESIZED", 10, RAMP["teal"][2], "middle")

    b += txt(340, 366, "Both put the row at unit scale. Only LayerNorm also drags it "
             "onto zero.", 12, PRI, "middle", "500")
    b += txt(340, 392, "That difference is the whole of what RMSNorm gives up: "
             "invariance to a constant", 11, SEC, "middle")
    b += txt(340, 410, "shift added to every channel. An all-positive row stays "
             "all-positive, so whatever", 11, SEC, "middle")
    b += txt(340, 428, "the offset of a hidden state meant, the next sub-layer still "
             "sees it.", 11, SEC, "middle")
    b += txt(340, 456, "Zhang & Sennrich's claim is that transformers never needed "
             "that invariance —", 11, SEC, "middle")
    b += txt(340, 474, "and the models that followed them agree.", 11, SEC, "middle")
    return wrap(680, 500, "One row through LayerNorm and RMSNorm",
                "The same four values re-centred and rescaled by LayerNorm, only "
                "rescaled by RMSNorm.", b)


# ------------------------------------------------- 41. what the two terms cost
def norm_cost_diagram():
    """Why removing the mean is a memory-traffic win, not a FLOP win."""
    b = txt(340, 24, "Why dropping two terms is worth anything at all",
            13, PRI, "middle", "500")
    b += txt(340, 44, "normalisation is 0.17% of a layer's arithmetic and 25.5% of "
             "its runtime (Ivanov et al.)", 11, SEC, "middle")

    steps_ln = [("read x", "gray"), ("reduce → μ", "coral"), ("reduce → σ²", "coral"),
                ("(x−μ)/σ", "gray"), ("⊙ γ", "coral"), ("+ β", "coral")]
    steps_rms = [("read x", "gray"), ("reduce → Σx²", "teal"), ("x / RMS", "gray"),
                 ("⊙ g", "teal")]

    for y, name, steps, note in ((116, "LayerNorm", steps_ln,
                                  "2 reductions, 2 parameter tensors"),
                                 (218, "RMSNorm", steps_rms,
                                  "1 reduction, 1 parameter tensor")):
        b += txt(44, y - 16, name, 12, PRI, "start", "500")
        for i, (lab, ramp) in enumerate(steps):
            x = 44 + i * 100
            b += vcell(x, y, 92, 26, ramp, lab, size=10)
            if i < len(steps) - 1:
                b += arrow(x + 92, y + 13, x + 100, y + 13)
        b += txt(44, y + 44, note, 10, SEC, "start")

    b += txt(340, 292, "Every box is a pass over the activation tensor: read it, do "
             "almost no arithmetic,", 11, SEC, "middle")
    b += txt(340, 310, "write it back. On an accelerator that is bandwidth, not FLOPs "
             "— which is why an op", 11, SEC, "middle")
    b += txt(340, 328, "worth 0.17% of the maths can cost a quarter of the clock, and "
             "why deleting two of", 11, SEC, "middle")
    b += txt(340, 346, "its terms shows up as real speed.", 11, SEC, "middle")
    b += txt(340, 376, "Zhang & Sennrich measured 7–64% off normalisation runtime, at "
             "indistinguishable quality.", 11, PRI, "middle", "500")
    b += txt(340, 402, "The same argument, one level further out, is why LLaMA-era "
             "models drop Linear biases too.", 11, SEC, "middle")
    return wrap(680, 428, "LayerNorm vs RMSNorm, as memory traffic",
                "The op sequence of each normalisation, annotated with the reductions "
                "and parameters each needs.", b)


# ================================================================ attention
# The dissector's config, so the shapes here are the ones run_llm.py prints.
B_, T_, D_, NH, NKV, DH_ = 2, 12, 64, 4, 2, 16

# steps_llm.py section 5, head 0, after masking and softmax.
ATT_P = [[1.0000, 0.0, 0.0, 0.0],
         [0.5311, 0.4689, 0.0, 0.0],
         [0.3450, 0.3139, 0.3412, 0.0],
         [0.2358, 0.2795, 0.2405, 0.2443]]


# ------------------------------------------------- 50. shapes through attention
def attention_shapes_diagram():
    """Every tensor in CausalSelfAttention.forward, with its shape."""
    b = txt(340, 24, "One attention layer, shape by shape", 13, PRI, "middle", "500")
    b += txt(340, 44, f"B={B_}  T={T_}  d={D_}  n_h={NH}  n_kv={NKV}  d_h={DH_} "
             f"\u2014 the config run_llm.py prints", 11, SEC, "middle")

    rows = [
        ("x", f"({B_}, {T_}, {D_})", "gray", "the residual stream, normalised"),
        ("q_proj / k_proj / v_proj",
         f"d \u2192 n_h\u00b7d_h = {NH*DH_}   /   d \u2192 n_kv\u00b7d_h = {NKV*DH_}",
         "purple", "q is full width, k and v are not \u2014 that asymmetry is all of GQA"),
        ("_split(\u00b7)  =  view + transpose",
         f"({B_}, {NH}, {T_}, {DH_})   /   ({B_}, {NKV}, {T_}, {DH_})",
         "purple", "a re-reading of one matrix, not a computation \u2014 see the next two figures"),
        ("apply_rope(q), apply_rope(k)", f"({B_}, \u00b7, {T_}, {DH_})", "coral",
         "position enters here, on q and k only \u2014 never on v"),
        ("k, v repeat_interleave", f"({B_}, {NH}, {T_}, {DH_})", "coral",
         f"each kv head is copied to serve {NH // NKV} query heads"),
        ("scores = q @ k\u1d40 / \u221ad_h", f"({B_}, {NH}, {T_}, {T_})", "teal",
         "the only tensor quadratic in T \u2014 the one long context pays for"),
        ("+ mask, then softmax", f"({B_}, {NH}, {T_}, {T_})", "teal",
         "mask before softmax, so \u2212inf becomes an exact 0 and rows still sum to 1"),
        ("@ v", f"({B_}, {NH}, {T_}, {DH_})", "teal",
         "a convex combination of value rows: attention can only interpolate"),
        ("transpose + reshape", f"({B_}, {T_}, {D_})", "gray",
         "heads are merged back \u2014 they have not talked to each other until now"),
        ("o_proj", f"({B_}, {T_}, {D_})", "gray",
         "the mix across heads, and back onto the residual stream"),
    ]
    y = 76
    for name, shape, ramp, note in rows:
        b += vcell(36, y, 214, 24, ramp, name, size=10, weight="500")
        b += vcell(256, y, 200, 24, ramp, shape, size=10)
        b += txt(36, y + 34, note, 9.5, SEC, "start")
        y += 46

    yy = y + 12
    b += txt(340, yy, "Attention is the only place tokens see each other; everything "
             "else in the block", 11, PRI, "middle", "500")
    b += txt(340, yy + 18, "is applied to one token at a time. That is why the FFN "
             "slot can be routed per", 11, SEC, "middle")
    b += txt(340, yy + 36, "token and attention cannot.", 11, SEC, "middle")
    return wrap(680, yy + 62, "Attention, shape by shape",
                "Every tensor inside one causal self-attention layer with grouped "
                "query heads.", b)


# ------------------------------------------------------- 51. the causal mask
def mask_diagram():
    """Scores, mask, softmax - and why row 0 is the identity."""
    CW, CH, G = 42, 26, 3
    b = txt(340, 24, "The causal mask, and what softmax does with it",
            13, PRI, "middle", "500")
    b += txt(340, 44, "T = 4, one head. m is the query row, n is the key column",
             11, SEC, "middle")

    def grid(x0, y0, cells, title, sub, rowlab=False):
        g = txt(x0 + 2 * (CW + G), y0 - 26, title, 12, PRI, "middle", "500")
        g += txt(x0 + 2 * (CW + G), y0 - 10, sub, 9.5, SEC, "middle")
        for i in range(4):
            if rowlab:
                g += txt(x0 - 8, y0 + i * (CH + G) + CH / 2, f"m={i}", 9, SEC, "end")
            for j in range(4):
                lab, ramp, op = cells(i, j)
                g += vcell(x0 + j * (CW + G), y0 + i * (CH + G), CW, CH, ramp, lab,
                           size=9.5, op=op)
        for j in range(4):
            g += txt(x0 + j * (CW + G) + CW / 2, y0 - 42, f"n={j}", 9, SEC, "middle")
        return g

    y0 = 116
    b += grid(64, y0, lambda i, j: ("s", "gray", 1.0),
              "scores", "every pair scored", rowlab=True)
    b += grid(266, y0, lambda i, j: ("−inf", "coral", 1.0) if j > i else ("s", "gray", 1.0),
              "+ mask", "M = 0 for n ≤ m")
    b += grid(468, y0, lambda i, j: (f"{ATT_P[i][j]:.2f}" if j <= i else "0",
                                     "teal" if j <= i else "gray",
                                     1.0 if j <= i else 0.4),
              "softmax", "rows sum to 1")

    yy = y0 + 4 * (CH + G) + 26
    b += txt(340, yy, "Row m = 0 comes out [1, 0, 0, 0] no matter what the model "
             "learned:", 12, PRI, "middle", "500")
    b += txt(340, yy + 20, "the first token has nothing to look at but itself, so its "
             "attention is the identity.", 11, SEC, "middle")
    b += txt(340, yy + 46, "Masking before the softmax is what keeps the rows "
             "normalised. Zeroing afterwards", 11, SEC, "middle")
    b += txt(340, yy + 64, "would leave each row summing to less than 1 — a quiet "
             "rescaling of every token.", 11, SEC, "middle")
    b += txt(340, yy + 90, "This triangle is also the whole reason one forward pass "
             "over T tokens yields T", 11, PRI, "middle", "500")
    b += txt(340, yy + 108, "training signals instead of one.", 11, PRI, "middle", "500")
    return wrap(680, yy + 134, "The causal mask",
                "Scores, the mask, and the softmax weights for a four-token sequence.", b)


# --------------------------------------------------------------------- 52. GQA
def gqa_diagram():
    """Which query head reads which kv head, and what that saves."""
    b = txt(340, 24, "Grouped query attention: fewer keys, all the queries",
            13, PRI, "middle", "500")
    b += txt(340, 44, f"n_h = {NH} query heads, n_kv = {NKV} kv heads — head h reads "
             f"group ⌊h / (n_h/n_kv)⌋", 11, SEC, "middle")

    qy, ky = 92, 190
    for h in range(NH):
        x = 130 + h * 110
        ramp = PAIR_RAMP[h // (NH // NKV)]
        b += vcell(x, qy, 92, 28, ramp, f"q head {h}", size=10, weight="500")
        kx = 186 + (h // (NH // NKV)) * 220
        b += arrow(x + 46, qy + 30, kx + 46, ky - 4)
    for g in range(NKV):
        x = 186 + g * 220
        b += vcell(x, ky, 92, 28, PAIR_RAMP[g], f"k,v head {g}", size=10, weight="500")

    b += txt(340, 244, "repeat_interleave, not repeat — the groups have to stay "
             "contiguous", 11, PRI, "middle", "500")
    b += txt(340, 262, "repeat would hand head 1 the wrong group and train perfectly "
             "happily.", 10, SEC, "middle")

    kv_mha = 2 * 3 * NH * DH_
    kv_gqa = 2 * 3 * NKV * DH_
    rows = [("what each query head does", "unchanged", "unchanged"),
            ("floats of KV cache per token", f"{kv_mha}", f"{kv_gqa}"),
            ("at 8k ctx, batch 32, fp16", "0.20 GB", "0.10 GB")]
    y = 300
    b += vcell(150, y, 230, 24, "gray", "", size=10)
    b += txt(265, y + 12, "3 layers, d_h = 16, k and v both", 10, SEC, "middle")
    b += vcell(382, y, 120, 24, "coral", "MHA", size=10, weight="500")
    b += vcell(504, y, 120, 24, "teal", f"GQA {NH}:{NKV}", size=10, weight="500")
    for i, (lab, a, c) in enumerate(rows):
        yy = y + 27 + i * 27
        b += vcell(150, yy, 230, 24, "gray", lab, size=9.5)
        b += vcell(382, yy, 120, 24, "coral", a, size=9.5)
        b += vcell(504, yy, 120, 24, "teal", c, size=9.5)

    yy = y + 27 + len(rows) * 27 + 16
    b += txt(340, yy, "The query side is untouched, so the model keeps every attention "
             "pattern it had.", 12, PRI, "middle", "500")
    b += txt(340, yy + 22, "What halves is the thing that actually runs out of memory "
             "at serving time — and", 11, SEC, "middle")
    b += txt(340, yy + 40, "the cache is read once per generated token, so it is "
             "bandwidth as well as space.", 11, SEC, "middle")
    b += txt(340, yy + 62, "n_kv = n_h is plain MHA; n_kv = 1 is MQA. GQA is the dial "
             "between them.", 11, SEC, "middle")
    return wrap(680, yy + 88, "Grouped query attention",
                "Four query heads sharing two key/value heads, and what that does to "
                "the KV cache.", b)


# ------------------------------------------------------------- 53. the KV cache
def kv_cache_diagram():
    """What the cache removes, and the mask offset that makes it correct."""
    CW, CH, G = 40, 26, 4
    b = txt(340, 24, "Generating token 5, with and without a KV cache",
            13, PRI, "middle", "500")
    b += txt(340, 44, "each cell is one token's k and v for one layer", 11, SEC, "middle")

    def strip_row(x0, y, n, done, new, label):
        g = txt(x0 - 10, y + CH / 2, label, 10, PRI, "end", "500")
        for i in range(n):
            if i in new:
                ramp, op, lab = "coral", 1.0, "new"
            elif i in done:
                ramp, op, lab = "teal", 1.0, "read"
            else:
                ramp, op, lab = "gray", 0.5, ""
            g += vcell(x0 + i * (CW + G), y, CW, CH, ramp, lab, size=8.5, op=op)
        for i in range(n):
            g += txt(x0 + i * (CW + G) + CW / 2, y - 14, f"{i}", 8.5, SEC, "middle")
        return g

    b += txt(80, 96, "no cache", 12, RAMP["coral"][2], "start", "500")
    b += strip_row(140, 112, 6, set(), set(range(6)), "recompute")
    b += txt(140, 156, "every k and v is projected again, every step — the prefix has "
             "not changed,", 10, SEC, "start")
    b += txt(140, 172, "so all but the last column is work already done once before.",
             10, SEC, "start")

    b += txt(80, 216, "with cache", 12, RAMP["teal"][2], "start", "500")
    b += strip_row(140, 232, 6, set(range(5)), {5}, "k, v")
    b += txt(140, 276, "five columns are read from the store; one is computed. The "
             "model runs on a", 10, SEC, "start")
    b += txt(140, 292, "single token and attends over six keys.", 10, SEC, "start")

    b += txt(340, 330, "The mask is where this goes wrong", 12, PRI, "middle", "500")
    y = 352
    b += txt(120, y + 13, "T = 1, S = 6:", 10, SEC, "end")
    for i in range(6):
        b += vcell(140 + i * (CW + G), y, CW, CH, "teal", "ok", size=8.5)
    b += txt(140 + 6 * (CW + G) + 12, y + 13, "one query row, all keys visible",
             10, SEC, "start")
    b += txt(340, y + 46, "The single query sits at the END of the key range, so its "
             "row is all-visible —", 11, SEC, "middle")
    b += txt(340, y + 64, "not the first row of a triangle. A mask written as a plain "
             "lower triangle is", 11, SEC, "middle")
    b += txt(340, y + 82, "correct while training and silently wrong while generating.",
             11, SEC, "middle")

    b += txt(340, y + 112, "Measured in run_llm.py: cached and uncached logits agree to "
             "2.4e-07, and", 11, PRI, "middle", "500")
    b += txt(340, y + 130, "64 tokens after a 128-token prompt run ~2.6× faster. The "
             "gap widens with the", 11, PRI, "middle", "500")
    b += txt(340, y + 148, "prefix, because the work the cache removes is quadratic in "
             "it.", 11, PRI, "middle", "500")
    return wrap(680, y + 176, "The KV cache",
                "What is recomputed without a cache, what is read with one, and the "
                "mask offset it needs.", b)


# ------------------------------------------------- 54. _split, part 1: view
SN_H, SD_H = 4, 4                       # drawing config: n_h = 4, d_h = 4, d = 16


def split_view_diagram():
    """The .view(B, T, n, d_h) half of _split: one flat row becomes n head rows."""
    d = SN_H * SD_H
    b = txt(340, 24, "_split, part 1:  .view(B, T, n_h, d_h)", 13, PRI, "middle", "500")
    b += txt(340, 44, f"drawn at n_h={SN_H}, d_h={SD_H} so d={d}; the model in "
             f"run_llm.py uses n_h={NH}, d_h={DH_}, d={D_}", 11, SEC, "middle")

    CW, G = 36, 2
    x0, y = 36, 96
    b += txt(36, 78, "one token of q_proj(x)  \u2014  a flat row of "
             f"n_h\u00b7d_h = {d} numbers", 10.5, PRI, "start", "500")
    for i in range(d):
        b += vcell(x0 + i * (CW + G), y, CW, 28, PAIR_RAMP[i // SD_H], f"{i}", size=10)
    for g in range(SN_H):
        x1 = x0 + g * SD_H * (CW + G)
        b += brace(x1, x1 + SD_H * (CW + G) - G, y + 32,
                   RAMP[PAIR_RAMP[g]][1], f"head {g}")

    ay = y + 74
    b += arrow(120, ay, 120, ay + 26)
    b += txt(132, ay + 13, ".view(B, T, n_h, d_h)", 10.5, PRI, "start", "500")

    gy = ay + 40
    CW2, CH2, G2 = 54, 28, 4
    gx = 340 - (SD_H * (CW2 + G2) - G2) / 2
    for j in range(SD_H):
        b += txt(gx + j * (CW2 + G2) + CW2 / 2, gy - 12, f"j={j}", 9.5, SEC, "middle")
    for h in range(SN_H):
        b += txt(gx - 10, gy + h * (CH2 + G2) + CH2 / 2, f"h={h}", 9.5, SEC, "end")
        for j in range(SD_H):
            b += vcell(gx + j * (CW2 + G2), gy + h * (CH2 + G2), CW2, CH2,
                       PAIR_RAMP[h], f"{h * SD_H + j}", size=10)

    yy = gy + SN_H * (CH2 + G2) + 22
    b += txt(340, yy, "Element (h, j) IS element h\u00b7d_h + j of the flat row.",
             12, PRI, "middle", "500")
    b += txt(340, yy + 20, "Nothing moved. view() only re-labels the same memory with "
             "new strides \u2014 it", 11, SEC, "middle")
    b += txt(340, yy + 38, "cannot reorder anything, so the head that owns columns "
             "0\u2013" + str(SD_H - 1) + " is decided by", 11, SEC, "middle")
    b += txt(340, yy + 56, "how q_proj's weight matrix was laid out, not by this line.",
             11, SEC, "middle")
    b += txt(340, yy + 84, "view(B, T, d_h, n_h) would run, and be wrong: head 0 would "
             "get columns", 11, RAMP["coral"][2], "middle", "500")
    b += txt(340, yy + 102, "0, 4, 8, 12 instead of 0, 1, 2, 3. Same shape, different "
             "model, no error.", 11, RAMP["coral"][2], "middle")
    return wrap(680, yy + 130, "_split part 1: view",
                "A flat projection row of d numbers re-read as n_h rows of d_h.", b)


# -------------------------------------------- 55. _split, part 2: transpose(1,2)
def split_transpose_diagram():
    """The .transpose(1, 2) half: the head axis becomes a batch axis."""
    b = txt(340, 24, "_split, part 2:  .transpose(1, 2)", 13, PRI, "middle", "500")
    b += txt(340, 44, "same numbers, regrouped: token-major \u2192 head-major "
             "(drawn with T = 3)", 11, SEC, "middle")

    T3, CW, CH, G = 3, 60, 22, 4

    def stack(x0, y0, cols, rows_of, colour_of, header):
        g = ""
        for c in range(cols):
            cx = x0 + c * (CW + 10)
            g_txt, cells = header(c)
            g += txt(cx + CW / 2, y0 - 12, g_txt, 9.5, SEC, "middle")
            for r, lab in enumerate(cells):
                g += vcell(cx, y0 + r * (CH + G), CW, CH, colour_of(c, r), lab, size=9.5)
        return g

    y0 = 108
    b += txt(48, 84, "(B, T, n_h, d_h)", 11, PRI, "start", "500")
    b += txt(48, y0 + 4 * (CH + G) + 10, "the head axis sits between the two matrix "
             "axes", 9.5, SEC, "start")
    b += stack(48, y0, T3, None, lambda c, r: PAIR_RAMP[r],
               lambda c: (f"token {c}", [f"h{h}" for h in range(SN_H)]))

    b += arrow(292, y0 + 46, 336, y0 + 46)
    b += txt(314, y0 + 30, "transpose", 9.5, SEC, "middle")

    x1 = 358
    b += txt(x1, 84, "(B, n_h, T, d_h)", 11, PRI, "start", "500")
    b += txt(x1, y0 + 3 * (CH + G) + 10, "each head now owns a whole (T, d_h) matrix",
             9.5, SEC, "start")
    b += stack(x1, y0, SN_H, None, lambda c, r: PAIR_RAMP[c],
               lambda c: (f"head {c}", [f"t{t}" for t in range(T3)]))

    yy = y0 + 4 * (CH + G) + 44
    b += txt(340, yy, "Why bother: torch.matmul batches over every leading dimension "
             "and treats", 12, PRI, "middle", "500")
    b += txt(340, yy + 18, "only the last two as the matrix. q @ k\u1d40 has to "
             "contract d_h and keep T \u2014 so", 11, SEC, "middle")
    b += txt(340, yy + 36, "(T, d_h) must be last, and the head axis must be in front "
             "with the batch.", 11, SEC, "middle")
    b += txt(340, yy + 58, "One transpose buys all n_h head-attentions as a single "
             "batched matmul.", 11, SEC, "middle")

    b += txt(340, yy + 88, "transpose returns a non-contiguous view, which is why the "
             "merge at the end", 11, RAMP["coral"][2], "middle", "500")
    b += txt(340, yy + 106, "is  y.transpose(1, 2).reshape(B, T, n_h\u00b7d_h)  \u2014 "
             "reshape copies when it must;", 11, RAMP["coral"][2], "middle")
    b += txt(340, yy + 124, "view on that same tensor raises. And the merge is exactly "
             "these two figures", 11, RAMP["coral"][2], "middle")
    b += txt(340, yy + 142, "run backwards.", 11, RAMP["coral"][2], "middle")
    return wrap(680, yy + 170, "_split part 2: transpose",
                "Token-major head blocks regrouped into head-major matrices.", b)


# --------------------------------------------- 56. the two channel conventions
CONV_DH = 8
CONV_Q = [0.6, -0.2, 0.9, 0.4, -0.7, 0.1, 0.3, -0.5]
CONV_COS3 = [-0.9900, 0.9553, 0.9996, 1.0000]      # rope_tables(8, 16), row m = 3
CONV_SIN3 = [0.1411, 0.2955, 0.0300, 0.0030]
CONV_INTER = [-0.5658, 0.2827, 0.7416, 0.6481, -0.7027, 0.0790, 0.3015, -0.4991]
CONV_HALF = [-0.4952, -0.2206, 0.8906, 0.4015, 0.7777, 0.0364, 0.3269, -0.4988]
CONV_ROTH = [0.7, -0.1, -0.3, 0.5, 0.6, -0.2, 0.9, 0.4]
CONV_PERM = [0, 4, 1, 5, 2, 6, 3, 7]


def arc(x1, x2, y, height, colour, label, dash=False):
    """A shallow arc above the row, joining two channels that share an angle."""
    mid = (x1 + x2) / 2
    d = ' stroke-dasharray="4 3"' if dash else ""
    g = (f'<path d="M{x1} {y} Q{mid} {y - height} {x2} {y}" fill="none" '
         f'stroke="{colour}" stroke-width="1.4" stroke-linecap="round"{d}/>')
    return g + txt(mid, y - height * 0.62, label, 9.5, colour, "middle")


def conventions_diagram():
    """Same channels, same angles, two different ideas of which two are a pair."""
    n = CONV_DH
    CW, G = 56, 4
    x0 = (680 - (n * (CW + G) - G)) / 2

    b = txt(340, 24, "Two channel conventions, one rotation", 13, PRI, "middle", "500")
    b += txt(340, 44, f"one head, d_h = {n}, so {n // 2} pairs and "
             f"{n // 2} angles \u2014 the question is only which two channels share one",
             11, SEC, "middle")

    def row(y, pairs, title, sub, colour_of, lift=36):
        g = txt(x0, y - lift - 16, title, 12, PRI, "start", "500")
        g += txt(x0, y - lift, sub, 9.5, SEC, "start")
        for j in range(n):
            g += vcell(x0 + j * (CW + G), y, CW, 26, colour_of(j), f"ch {j}", size=9.5)
        for i, (a, c) in enumerate(pairs):
            xa = x0 + a * (CW + G) + CW / 2
            xc = x0 + c * (CW + G) + CW / 2
            h = 18 + 12 * abs(c - a)
            g += arc(xa, xc, y - 2, h, RAMP[PAIR_RAMP[i]][1], f"\u03b8{i}")
        return g

    y1 = 148
    b += row(y1, [(2 * i, 2 * i + 1) for i in range(n // 2)],
             "interleaved \u2014 \u201cGPT-J style\u201d",
             "pairs (0,1), (2,3), \u2026 \u2014 partners are adjacent. "
             "LLM/common.py and rope.py stage 2.",
             lambda j: PAIR_RAMP[j // 2])

    y2 = 336
    b += row(y2, [(i, i + n // 2) for i in range(n // 2)],
             "split-half \u2014 \u201cGPT-NeoX style\u201d, the rotate_half form",
             "pairs (0,4), (1,5), \u2026 \u2014 partners are half a head apart. "
             "HuggingFace LLaMA, and rope.py stage 3.",
             lambda j: PAIR_RAMP[j % (n // 2)], lift=80)

    yy = y2 + 52
    b += txt(340, yy, "Colour is the angle. Both rows use the SAME four thetas and "
             "the same tables;", 12, PRI, "middle", "500")
    b += txt(340, yy + 20, "they disagree only about which two channels get turned "
             "together.", 11, SEC, "middle")
    b += txt(340, yy + 46, "Neither is more correct. A model trained under one is "
             "simply a model whose", 11, SEC, "middle")
    b += txt(340, yy + 64, "W_Q and W_K columns are laid out for that pairing \u2014 "
             "which is why converting a", 11, SEC, "middle")
    b += txt(340, yy + 82, "checkpoint between them is a permutation of those two "
             "matrices, and nothing more.", 11, SEC, "middle")
    return wrap(680, yy + 110, "The two RoPE channel conventions",
                "Interleaved and split-half pairings of the same eight channels.", b)


# ------------------------------------------------------------ 57. rotate_half
def rotate_half_diagram():
    """What rotate_half does, and why one line then suffices."""
    n = CONV_DH
    CW, G = 56, 4
    x0 = (680 - (n * (CW + G) - G)) / 2

    b = txt(340, 24, "rotate_half: a quarter turn, done by moving numbers",
            13, PRI, "middle", "500")
    b += txt(340, 44, "every value in these rows is printed by "
             "LLM/from_scratch/check_rope.py", 11, SEC, "middle")

    def vals(y, data, colour_of, label):
        g = txt(x0 - 10, y + 13, label, 10, PRI, "end", "500")
        for j, v in enumerate(data):
            g += vcell(x0 + j * (CW + G), y, CW, 26, colour_of(j), f"{v:+.2f}", size=9.5)
        return g

    b += txt(x0, 82, "the halves, and what happens to them", 10, SEC, "start")
    y1 = 98
    b += vals(y1, CONV_Q, lambda j: "teal" if j < n // 2 else "coral", "x")
    b += arrow(340, y1 + 30, 340, y1 + 52)
    b += txt(352, y1 + 41, "rotate_half", 10.5, PRI, "start", "500")
    y2 = y1 + 60
    b += vals(y2, CONV_ROTH, lambda j: "coral" if j < n // 2 else "teal",
              "rotate_half(x)")
    b += txt(340, y2 + 42, "second half moves to the front and changes sign; first "
             "half follows unchanged", 10, SEC, "middle")
    b += txt(340, y2 + 60, "\u2014 do it twice and you get \u2212x exactly "
             "(max diff 0.0), which is what makes it a turn", 10, SEC, "middle")

    y3 = y2 + 96
    b += txt(340, y3, "Why that is enough", 12, PRI, "middle", "500")
    b += txt(340, y3 + 22, "E7 on a pair (a, b) is  (a\u00b7cos \u2212 b\u00b7sin,  "
             "a\u00b7sin + b\u00b7cos).", 11, SEC, "middle")
    b += txt(340, y3 + 40, "The vector (\u2212b, a) IS the pair turned a quarter "
             "turn, so the whole thing collapses to", 11, SEC, "middle")

    y4 = y3 + 66
    b += vcell(150, y4, 380, 30, "purple",
               "out = x \u00b7 cos\u0303  +  rotate_half(x) \u00b7 sin\u0303",
               size=12, weight="500")
    b += txt(340, y4 + 46, "with cos\u0303 = cat([cos, cos]) \u2014 the tables are "
             f"{n // 2} wide and a channel needs its own entry,", 10, SEC, "middle")
    b += txt(340, y4 + 64, f"so channel j and channel j+{n // 2} end up sharing "
             "\u03b8_j. That duplication IS the pairing.", 10, SEC, "middle")

    y5 = y4 + 92
    a0, b0 = CONV_Q[0], CONV_Q[n // 2]
    c0, s0 = CONV_COS3[0], CONV_SIN3[0]
    b += txt(340, y5, "One pair, by hand \u2014 channels 0 and 4 at m = 3, "
             f"\u03b8\u2080 = 1.0", 11, PRI, "middle", "500")
    lines = [
        f"a = {a0:+.2f} (ch 0)   \u00b7   b = {b0:+.2f} (ch 4)   \u00b7   "
        f"cos = {c0:+.4f}   \u00b7   sin = {s0:+.4f}",
        f"ch 0  =  a\u00b7cos \u2212 b\u00b7sin  =  {a0 * c0:+.4f} \u2212 "
        f"({b0 * s0:+.4f})  =  {a0 * c0 - b0 * s0:+.4f}",
        f"ch 4  =  a\u00b7sin + b\u00b7cos  =  {a0 * s0:+.4f} + "
        f"({b0 * c0:+.4f})  =  {a0 * s0 + b0 * c0:+.4f}",
    ]
    for i, ln in enumerate(lines):
        b += txt(340, y5 + 24 + i * 18, ln, 10, SEC, "middle")
    b += txt(340, y5 + 84, "and those are exactly entries 0 and 4 of the split-half "
             f"output, {CONV_HALF[0]:+.4f} and {CONV_HALF[n // 2]:+.4f}.",
             10.5, PRI, "middle", "500")
    return wrap(680, y5 + 112, "rotate_half",
                "The quarter turn that lets the split-half rotation fit on one "
                "line.", b)


# ------------------------------------------- 58. the permutation, and mixing them
def convention_swap_diagram():
    """The two are one function on a permuted channel order - unless you mix them."""
    n = CONV_DH
    CW, G = 56, 4
    x0 = (680 - (n * (CW + G) - G)) / 2

    b = txt(340, 24, "The same function, and the one way to get it wrong",
            13, PRI, "middle", "500")
    b += txt(340, 44, "x[perm] gathers each split-half partner next to its own \u2014 "
             "then it IS the interleaved case", 11, SEC, "middle")

    y1 = 92
    b += txt(x0 - 10, y1 + 13, "perm", 10, PRI, "end", "500")
    for j, src in enumerate(CONV_PERM):
        b += vcell(x0 + j * (CW + G), y1, CW, 26, PAIR_RAMP[j // 2], f"ch {src}",
                   size=9.5)
    for i in range(n // 2):
        xa = x0 + (2 * i) * (CW + G) + CW / 2
        xc = x0 + (2 * i + 1) * (CW + G) + CW / 2
        b += arc(xa, xc, y1 - 2, 30, RAMP[PAIR_RAMP[i]][1], f"\u03b8{i}")

    yy = y1 + 46
    b += vcell(96, yy, 488, 28, "purple",
               "apply_rope_half(x)  ==  apply_rope(x[perm])[inv]", size=11.5,
               weight="500")
    b += txt(340, yy + 44, "Checked on a (2, 3, 4, 8) tensor: max difference "
             "0.00e+00. Not close \u2014 equal.", 11, PRI, "middle", "500")
    b += txt(340, yy + 62, "Either convention on its own is exactly E7, and E8 holds "
             "for both.", 11, SEC, "middle")

    y2 = yy + 96
    b += txt(340, y2, "What happens if q and k disagree", 12, PRI, "middle", "500")
    rows = [("q and k both interleaved", "+0.139683", "1.8e-07", "teal"),
            ("q and k both split-half", "+0.614924", "1.8e-07", "teal"),
            ("q interleaved, k split-half", "varies", "1.7e+00", "coral")]
    hy = y2 + 22
    b += vcell(96, hy, 250, 24, "gray", "one fixed (q, k), three gaps of 3", size=9.5)
    b += vcell(348, hy, 118, 24, "gray", "q\u00b7k", size=9.5)
    b += vcell(468, hy, 116, 24, "gray", "spread", size=9.5)
    for i, (lab, val, spread, ramp) in enumerate(rows):
        ry = hy + 27 + i * 27
        b += vcell(96, ry, 250, 24, ramp, lab, size=9.5)
        b += vcell(348, ry, 118, 24, ramp, val, size=9.5)
        b += vcell(468, ry, 116, 24, ramp, spread, size=9.5)

    y3 = hy + 27 + len(rows) * 27 + 20
    b += txt(340, y3, "E8 says three pairs at the same gap must give the same score. "
             "Mixed, they", 11, PRI, "middle", "500")
    b += txt(340, y3 + 18, "differ by 1.7 \u2014 seven orders of magnitude past float "
             "noise. Nothing raises:", 11, SEC, "middle")
    b += txt(340, y3 + 36, "same shapes, same dtype, a model that trains and has "
             "simply lost relative position.", 11, SEC, "middle")
    b += txt(340, y3 + 62, "This is also the checkpoint story. HuggingFace implements "
             "the split-half form and", 11, SEC, "middle")
    b += txt(340, y3 + 80, "permutes W_Q and W_K at conversion "
             "(convert_llama_weights_to_hf.py) so Meta's", 11, SEC, "middle")
    b += txt(340, y3 + 98, "interleaved weights come out agreeing with it.",
             11, SEC, "middle")
    return wrap(680, y3 + 126, "Mixing the conventions",
                "The permutation that makes them equal, and what happens when q and "
                "k disagree.", b)


# ------------------------------------------------ 60. position interpolation
PI_SCALE = 4
PI_LEN = 16
PI_TRAIN = 4                                   # positions the model was trained on
# check_rope.py's fixture: one fixed (q, k), scored at two gaps
PI_DOT_PLAIN_1 = 0.150094
PI_DOT_SCALED_4 = 0.150094
PI_DOT_PLAIN_4 = -1.149394
# a real head: wavelengths in tokens, d_h = 64, base 10000, and NTK base 41829
PI_PAIRS = [0, 8, 16, 24, 31]
PI_LAM = [6.28, 62.8, 628.3, 6283.2, 47117.2]
PI_LAM_NTK = [6.28, 89.9, 1285.1, 18377.7, 188469.0]


def pi_positions_diagram():
    """PI squeezes the position axis instead of extending it."""
    n, sc, tr = PI_LEN, PI_SCALE, PI_TRAIN
    x0, x1 = 60, 620
    step = (x1 - x0) / (n - 1)

    b = txt(340, 24, "Position Interpolation: squeeze the axis, do not extend it",
            13, PRI, "middle", "500")
    b += txt(340, 44, f"a model trained on {tr} positions, asked to read {n} \u2014 "
             f"scale = {sc}", 11, SEC, "middle")

    def axis(y, label):
        g = (f'<line x1="{x0 - 14}" y1="{y}" x2="{x1 + 14}" y2="{y}" '
             f'stroke="{ARR}" stroke-width="1"/>')
        g += txt(x0 - 14, y - 18, label, 10, PRI, "start", "500")
        for m in (0, 4, 8, 12, 15):
            x = x0 + m * step
            g += (f'<line x1="{x}" y1="{y - 4}" x2="{x}" y2="{y + 4}" '
                  f'stroke="{ARR}" stroke-width="1"/>')
            g += txt(x, y + 16, f"{m}", 9, SEC, "middle")
        return g

    # the band of angles the model has actually seen
    bw = (tr - 0.1) * step
    band = (f'<rect x="{x0 - 8}" y="86" width="{bw + 16}" height="176" rx="4" '
            f'fill="{RAMP["teal"][0]}" stroke="{RAMP["teal"][1]}" '
            f'stroke-width="0.8" stroke-dasharray="4 3"/>')
    b += band
    b += txt(x0 + bw / 2 + 4, 76, "the range the model was trained on", 9.5,
             RAMP["teal"][2], "middle", "500")

    y1 = 130
    b += axis(y1, "unscaled")
    for m in range(n):
        inside = m < tr
        col = RAMP["teal"][1] if inside else RAMP["coral"][1]
        b += (f'<circle cx="{x0 + m * step}" cy="{y1}" r="4.5" fill="{col}" '
              f'opacity="{1.0 if inside else 0.85}"/>')
    b += txt(x1 + 14, y1 - 18, f"{n - tr} past the edge", 9.5,
             RAMP["coral"][2], "end", "500")

    y2 = 226
    b += axis(y2, f"m \u00f7 {sc}")
    for m in range(n):
        b += (f'<circle cx="{x0 + (m / sc) * step}" cy="{y2}" r="4.5" '
              f'fill="{RAMP["teal"][1]}"/>')
    b += txt(x1 + 14, y2 - 18, "all sixteen, inside", 9.5,
             RAMP["teal"][2], "end", "500")

    yy = 300
    b += txt(340, yy, f"Row m of the scaled table IS row m/{sc} of the plain one.",
             12, PRI, "middle", "500")
    b += txt(340, yy + 20, f"check_rope.py verifies it at m = 4, 8, 12 \u2014 and that "
             f"scale = 1 reproduces", 11, SEC, "middle")
    b += txt(340, yy + 38, "rope_tables exactly. Nothing else in the model changes: "
             "same weights, same", 11, SEC, "middle")
    b += txt(340, yy + 56, "thetas, same code path. Only the number handed to the "
             "table moves.", 11, SEC, "middle")
    b += txt(340, yy + 84, "A real case: pair 0 reaches 8192 radians at 8k tokens, "
             "against the 2048 it", 11, PRI, "middle", "500")
    b += txt(340, yy + 102, f"saw in training. Divide the positions by 4 and it is "
             "back at 2048.", 11, PRI, "middle", "500")
    return wrap(680, yy + 130, "Position Interpolation",
                "Sixteen positions squeezed into the four the model was trained on.",
                b)


# ------------------------------------------------------------- 61. what it costs
def pi_tradeoff_diagram():
    """The same equality, read as the price."""
    sc = PI_SCALE
    b = txt(340, 24, "What interpolation actually trades", 13, PRI, "middle", "500")
    b += txt(340, 44, "one fixed (q, k), scored at two gaps \u2014 printed by "
             "check_rope.py stage 4", 11, SEC, "middle")

    rows = [("plain tables", "1 token", f"{PI_DOT_PLAIN_1:+.6f}", "gray"),
            (f"scaled by {sc}", f"{sc} tokens", f"{PI_DOT_SCALED_4:+.6f}", "teal"),
            ("plain tables", f"{sc} tokens", f"{PI_DOT_PLAIN_4:+.6f}", "coral")]
    y = 84
    b += vcell(120, y, 180, 24, "gray", "tables", size=10, weight="500")
    b += vcell(302, y, 140, 24, "gray", "gap", size=10, weight="500")
    b += vcell(444, y, 140, 24, "gray", "q \u00b7 k", size=10, weight="500")
    for i, (a, g, v, ramp) in enumerate(rows):
        yy = y + 27 + i * 27
        b += vcell(120, yy, 180, 24, ramp, a, size=9.5)
        b += vcell(302, yy, 140, 24, ramp, g, size=9.5)
        b += vcell(444, yy, 140, 24, ramp, v, size=9.5)

    yy = y + 27 + len(rows) * 27 + 22
    b += txt(340, yy, "The first two rows agree to 0.00e+00. That is the whole "
             "mechanism:", 12, PRI, "middle", "500")
    b += txt(340, yy + 20, f"after scaling, {sc} tokens apart produces exactly the "
             "score 1 token apart", 11, SEC, "middle")
    b += txt(340, yy + 38, "used to. The context window grows because the model is "
             "being told", 11, SEC, "middle")
    b += txt(340, yy + 56, "everything is closer together than it is.", 11, SEC, "middle")

    yy2 = yy + 86
    b += txt(340, yy2, "And that is also the price", 12, RAMP["coral"][2],
             "middle", "500")
    b += txt(340, yy2 + 20, f"The third row is what {sc} tokens apart used to mean: "
             f"{PI_DOT_PLAIN_4:+.4f}, not "
             f"{PI_DOT_SCALED_4:+.4f}.", 11, SEC, "middle")
    b += txt(340, yy2 + 38, "Every wavelength is multiplied by the scale, the fast "
             "pairs included \u2014 pair 0", 11, SEC, "middle")
    b += txt(340, yy2 + 56, f"goes from {PI_LAM[0]} tokens to "
             f"{PI_LAM[0] * PI_SCALE:.1f}, so the resolution that separated "
             "adjacent", 11, SEC, "middle")
    b += txt(340, yy2 + 74, "tokens now separates groups of four. Long range fits; "
             "local detail blurs.", 11, SEC, "middle")
    b += txt(340, yy2 + 102, "Which is exactly the observation NTK-aware scaling and "
             "YaRN are built on.", 11, PRI, "middle", "500")
    return wrap(680, yy2 + 130, "The interpolation trade",
                "Identical scores at scaled gaps, and the local resolution it "
                "costs.", b)


# ---------------------------------------------- 62. the family of scaling methods
def scaling_family_diagram():
    """PI, NTK-aware and YaRN as three edits to the same ladder."""
    sc = PI_SCALE
    b = txt(340, 24, "Three ways to stretch the same ladder", 13, PRI, "middle", "500")
    b += txt(340, 44, f"wavelength in tokens, d_h = 64, base 10000 \u2192 "
             f"{sc}\u00d7 context", 11, SEC, "middle")

    cols = [(64, 60, "pair"), (128, 108, "plain"), (240, 108, f"PI \u00d7{sc}"),
            (352, 108, "NTK-aware"), (464, 156, "what NTK did to it")]
    y = 82
    for x, w, lab in cols:
        b += vcell(x, y, w, 24, "gray", lab, size=10, weight="500")

    for i, pr in enumerate(PI_PAIRS):
        yy = y + 27 + i * 27
        lam, ntk = PI_LAM[i], PI_LAM_NTK[i]
        ratio = ntk / lam
        b += vcell(64, yy, 60, 24, "gray", f"{pr}", size=9.5)
        b += vcell(128, yy, 108, 24, "gray", f"{lam:,.0f}" if lam >= 100
                   else f"{lam:.1f}", size=9.5)
        b += vcell(240, yy, 108, 24, "coral", f"{lam * sc:,.0f}" if lam * sc >= 100
                   else f"{lam * sc:.1f}", size=9.5)
        b += vcell(352, yy, 108, 24, "teal", f"{ntk:,.0f}" if ntk >= 100
                   else f"{ntk:.1f}", size=9.5)
        bw = 96 * (ratio - 1) / (sc - 1)
        b += (f'<rect x="{470}" y="{yy + 5}" width="{max(bw, 1.5)}" height="14" '
              f'rx="2" fill="{RAMP["teal"][1]}" opacity="0.8"/>')
        b += txt(470 + max(bw, 1.5) + 6, yy + 12, f"\u00d7{ratio:.2f}", 9,
                 RAMP["teal"][2], "start")

    yy = y + 27 + len(PI_PAIRS) * 27 + 20
    b += txt(340, yy, "PI multiplies every wavelength by 4, including pair 0's six "
             "tokens.", 12, PRI, "middle", "500")
    b += txt(340, yy + 20, "NTK-aware raises the BASE instead (10000 \u2192 41829), "
             "which leaves the fast", 11, SEC, "middle")
    b += txt(340, yy + 38, "pairs almost untouched and puts the whole stretch on the "
             "slow ones \u2014 exactly", 11, SEC, "middle")
    b += txt(340, yy + 56, "where a 4\u00d7 context needs it.", 11, SEC, "middle")

    y2 = yy + 88
    items = [("PI", "m \u2192 m / s",
              "all pairs alike; needs fine-tuning"),
             ("NTK-aware", "b \u2192 b\u00b7s^(d_h/(d_h\u22122))",
              "fast pairs kept; often no fine-tuning"),
             ("YaRN", "ramp per pair + attn temperature",
              "the current 64k\u2013128k answer")]
    for i, (name, formula, note) in enumerate(items):
        yy2 = y2 + i * 40
        b += vcell(48, yy2, 92, 26, PAIR_RAMP[i], name, size=10, weight="500")
        b += vcell(146, yy2, 214, 26, "gray", formula, size=9.5)
        b += txt(370, yy2 + 13, note, 9.5, SEC, "start")

    y3 = y2 + len(items) * 40 + 14
    b += txt(340, y3, "All three change one thing: what angle position m gets. "
             "Not the attention", 11, PRI, "middle", "500")
    b += txt(340, y3 + 18, "formula, not a weight shape, not a single parameter "
             "count \u2014 the same kind of", 11, SEC, "middle")
    b += txt(340, y3 + 36, "single-slot intervention MoE makes to the FFN.",
             11, SEC, "middle")
    return wrap(680, y3 + 64, "PI, NTK-aware and YaRN",
                "What each method does to the wavelength ladder of a 64-wide head.",
                b)


# ------------------------------------------------------- 63. cos[None, None]
NN_B, NN_H, NN_T, NN_P = 2, 4, 12, 4        # d_h = 8, so d_h/2 = 4


def newaxis_diagram():
    """What [None, None] does to the tables, and why the line has it."""
    b = txt(340, 24, "cos[None, None]: giving the table the axes x already has",
            13, PRI, "middle", "500")
    b += txt(340, 44, "None in an index is torch.newaxis \u2014 it inserts an axis "
             "of length 1", 11, SEC, "middle")

    # --- the shape ladder, right-aligned the way broadcasting reads it ------
    CW, G = 104, 6
    x0 = 236
    heads = ["batch", "head", "position", "pair"]
    for j, h in enumerate(heads):
        b += txt(x0 + j * (CW + G) + CW / 2, 78, h, 9.5, SEC, "middle")

    rows = [
        ("x1 = x[..., 0::2]", [f"B={NN_B}", f"H={NN_H}", f"T={NN_T}",
                               f"d_h/2={NN_P}"], ["purple"] * 4, "the head, split"),
        ("cos", [None, None, f"T={NN_T}", f"d_h/2={NN_P}"],
         [None, None, "gray", "gray"], "two axes short"),
        ("cos[None, None]", ["1", "1", f"T={NN_T}", f"d_h/2={NN_P}"],
         ["coral", "coral", "gray", "gray"], "the two Nones"),
        ("x1 * cos[None, None]", [f"{NN_B}", f"{NN_H}", f"{NN_T}", f"{NN_P}"],
         ["teal"] * 4, "each 1 stretched"),
    ]
    y = 96
    for name, cells, ramps, note in rows:
        b += txt(x0 - 12, y + 13, name, 10, PRI, "end", "500")
        for j, (lab, ramp) in enumerate(zip(cells, ramps)):
            xx = x0 + j * (CW + G)
            if lab is None:
                b += (f'<rect x="{xx}" y="{y}" width="{CW}" height="26" rx="3" '
                      f'fill="none" stroke="{SEC}" stroke-width="0.5" '
                      f'stroke-dasharray="3 3" opacity="0.5"/>')
            else:
                b += vcell(xx, y, CW, 26, ramp, lab, size=10)
        y += 32

    yy = y + 6
    b += txt(340, yy, "Broadcasting reads shapes from the RIGHT. An axis of length "
             "1 is reused for", 11, PRI, "middle", "500")
    b += txt(340, yy + 18, "every index along it \u2014 so one table of angles serves "
             "every head, and every", 11, SEC, "middle")
    b += txt(340, yy + 36, "sequence in the batch. That is the claim the two Nones "
             "are making.", 11, SEC, "middle")

    # --- the picture: one table, reused across the (B, H) grid -------------
    y2 = yy + 66
    b += txt(340, y2, f"One (T, d_h/2) table \u2192 all {NN_B}\u00d7{NN_H} = "
             f"{NN_B * NN_H} (batch, head) slices", 11.5, PRI, "middle", "500")
    ty = y2 + 22
    b += vcell(64, ty + 18, 116, 44, "gray", "cos", size=11, weight="500")
    b += txt(122, ty + 74, f"({NN_T}, {NN_P})", 9, SEC, "middle")
    b += arrow(186, ty + 40, 232, ty + 40)
    CW2, CH2, G2 = 92, 26, 6
    for r in range(NN_B):
        for c in range(NN_H):
            b += vcell(248 + c * (CW2 + G2), ty + r * (CH2 + G2), CW2, CH2,
                       "gray", "cos", size=9, op=0.55)
    b += txt(248, ty + NN_B * (CH2 + G2) + 8, f"b=0..{NN_B - 1}, "
             f"h=0..{NN_H - 1} \u2014 the same numbers, not {NN_B * NN_H} copies",
             9, SEC, "start")

    y3 = ty + NN_B * (CH2 + G2) + 34
    b += txt(340, y3, "Three things this is not", 12, PRI, "middle", "500")
    facts = [
        ("not a copy", "cos and cos[None, None] share storage \u2014 same "
         "data_ptr, 256 bytes either way"),
        ("not required", "x1 * cos gives the identical result: torch prepends "
         "the 1s by itself"),
        ("not a guard", "a mis-shaped (T,) table raises the same error with the "
         "Nones as without"),
    ]
    for i, (name, note) in enumerate(facts):
        yy3 = y3 + 22 + i * 30
        b += vcell(56, yy3, 108, 24, PAIR_RAMP[i], name, size=9.5, weight="500")
        b += txt(176, yy3 + 12, note, 9.5, SEC, "start")

    y4 = y3 + 22 + len(facts) * 30 + 14
    b += txt(340, y4, "So the brackets are documentation, and worth the two "
             "characters:", 11.5, PRI, "middle", "500")
    b += txt(340, y4 + 20, "they put the intended alignment on the line instead of "
             "leaving it to a rule the", 11, SEC, "middle")
    b += txt(340, y4 + 38, "reader has to remember. In a file where every bug is a "
             "silent shape bug, saying", 11, SEC, "middle")
    b += txt(340, y4 + 56, "which axes you meant is the point.", 11, SEC, "middle")
    b += txt(340, y4 + 82, "cos[None, None]  ==  cos.unsqueeze(0).unsqueeze(0)  ==  "
             "cos[None, None, :, :]", 10, RAMP["purple"][2], "middle", "500")
    return wrap(680, y4 + 110, "cos[None, None]",
                "Inserting two length-1 axes so one table of angles broadcasts "
                "over batch and head.", b)


# ------------------------------------------ 64. apply_rope_half, the duplication
ARH_COS = [-0.9900, 0.9553, 0.9996, 1.0000]        # rope_tables(8, 16) row m = 3
ARH_SIN = [0.1411, 0.2955, 0.0300, 0.0030]
ARH_X = [0.6, -0.2, 0.9, 0.4, -0.7, 0.1, 0.3, -0.5]
ARH_RH = [0.7, -0.1, -0.3, 0.5, 0.6, -0.2, 0.9, 0.4]
ARH_XC = [-0.5940, -0.1911, 0.8996, 0.4000, 0.6930, 0.0955, 0.2999, -0.5000]
ARH_RS = [0.0988, -0.0296, -0.0090, 0.0015, 0.0847, -0.0591, 0.0270, 0.0012]
ARH_OUT = [-0.4952, -0.2206, 0.8906, 0.4015, 0.7777, 0.0364, 0.3269, -0.4988]


def rope_half_cat_diagram():
    """cat([cos, cos]) is what decides the split-half pairing."""
    n, half = 8, 4
    CW, G = 68, 5
    x0 = (680 - (n * (CW + G) - G)) / 2

    b = txt(340, 24, "apply_rope_half step 1: cat([cos, cos], dim=-1)",
            13, PRI, "middle", "500")
    b += txt(340, 44, "the tables are d_h/2 wide; this form needs one entry per "
             "CHANNEL", 11, SEC, "middle")

    # the (T, d_h/2) row, centred over the left half
    y1 = 96
    b += txt(x0 - 12, y1 + 13, "cos", 10.5, PRI, "end", "500")
    for j in range(half):
        b += vcell(x0 + j * (CW + G), y1, CW, 26, PAIR_RAMP[j],
                   f"{ARH_COS[j]:+.4f}", size=9)
        b += txt(x0 + j * (CW + G) + CW / 2, y1 - 14, f"\u03b8{j}", 9,
                 RAMP[PAIR_RAMP[j]][2], "middle", "500")
    b += txt(x0 + half * (CW + G) + 8, y1 + 13, f"(T, d_h/2) \u2014 {half} columns",
             9.5, SEC, "start")

    b += arrow(110, y1 + 32, 110, y1 + 58)
    b += txt(122, y1 + 45, "cat([cos, cos], dim=-1)", 10.5, PRI, "start", "500")

    y2 = y1 + 84
    b += txt(x0 - 12, y2 + 13, "c", 10.5, PRI, "end", "500")
    for j in range(n):
        b += vcell(x0 + j * (CW + G), y2, CW, 26, PAIR_RAMP[j % half],
                   f"{ARH_COS[j % half]:+.4f}", size=9)
        b += txt(x0 + j * (CW + G) + CW / 2, y2 - 14, f"ch {j}", 9, SEC, "middle")
    for j in range(half):
        xa = x0 + j * (CW + G) + CW / 2
        xc = x0 + (j + half) * (CW + G) + CW / 2
        b += arc(xa, xc, y2 + 30, -(16 + 9 * j), RAMP[PAIR_RAMP[j]][1], "")

    yy = y2 + 106
    b += txt(340, yy, "Channel j and channel j + d_h/2 now hold the SAME number.",
             12, PRI, "middle", "500")
    b += txt(340, yy + 20, "That is the split-half pairing \u2014 not a separate "
             "decision made elsewhere, just", 11, SEC, "middle")
    b += txt(340, yy + 38, "the consequence of duplicating a table that was one "
             "entry per pair.", 11, SEC, "middle")
    b += txt(340, yy + 60, "apply_rope makes the same decision with a slice "
             "instead: x[..., 0::2] takes", 11, SEC, "middle")
    b += txt(340, yy + 78, "the firsts, x[..., 1::2] the seconds, so partners end "
             "up adjacent.", 11, SEC, "middle")

    y3 = yy + 108
    b += txt(340, y3, "The four lines, with shapes", 12, PRI, "middle", "500")
    rows = [("cos, sin", "(T, d_h/2)", "gray", "one angle per pair"),
            ("cat([cos, cos], -1)", "(T, d_h)", "coral",
             "one per channel; j and j+d_h/2 agree"),
            ("[None, None]", "(1, 1, T, d_h)", "purple",
             "batch and head axes, to broadcast"),
            ("x * c", "(B, H, T, d_h)", "teal", "the cosine term"),
            ("rotate_half(x) * s", "(B, H, T, d_h)", "teal", "the sine term"),
            ("their sum", "(B, H, T, d_h)", "teal", "= E7, every pair at once")]
    yy3 = y3 + 22
    for name, shape, ramp, note in rows:
        b += vcell(48, yy3, 176, 24, ramp, name, size=9.5, weight="500")
        b += vcell(230, yy3, 136, 24, ramp, shape, size=9.5)
        b += txt(376, yy3 + 12, note, 9.5, SEC, "start")
        yy3 += 27
    return wrap(680, yy3 + 20, "apply_rope_half: the duplication",
                "How cat([cos, cos]) makes channel j and j + d_h/2 share an "
                "angle.", b)


# ------------------------------- 65. apply_rope_half, the arithmetic row by row
def rope_half_trace_diagram():
    """Every intermediate row of `x * c + rotate_half(x) * s`, with numbers."""
    n, half = 8, 4
    CW, G = 62, 4
    x0 = 142

    b = txt(340, 24, "apply_rope_half: x \u00b7 c + rotate_half(x) \u00b7 s, "
            "row by row", 13, PRI, "middle", "500")
    b += txt(340, 44, "d_h = 8, one token at m = 3 \u2014 every value printed by "
             "check_rope.py", 11, SEC, "middle")

    rows = [("x", ARH_X, "gray", None),
            ("c", [ARH_COS[j % half] for j in range(n)], "purple", None),
            ("x \u00b7 c", ARH_XC, "purple", "the cosine term"),
            ("rotate_half(x)", ARH_RH, "gray", "second half negated, moved front"),
            ("s", [ARH_SIN[j % half] for j in range(n)], "coral", None),
            ("rotate_half(x) \u00b7 s", ARH_RS, "coral", "the sine term"),
            ("out", ARH_OUT, "teal", "their sum")]
    y = 84
    for j in range(n):
        b += txt(x0 + j * (CW + G) + CW / 2, y - 12, f"ch {j}", 9, SEC, "middle")
    for name, vals, ramp, note in rows:
        b += txt(x0 - 12, y + 13, name, 10, PRI, "end", "500")
        for j, v in enumerate(vals):
            b += vcell(x0 + j * (CW + G), y, CW, 26, ramp, f"{v:+.4f}", size=8.5)
        if note:
            b += txt(340, y + 40, note, 9, SEC, "middle")
            y += 50
        else:
            y += 32

    yy = y + 8
    b += txt(340, yy, "Follow one pair: channels 0 and 4, angle \u03b8\u2080",
             12, PRI, "middle", "500")
    a0, b0 = ARH_X[0], ARH_X[half]
    c0, s0 = ARH_COS[0], ARH_SIN[0]
    lines = [
        f"ch 0:   x\u00b7c = {a0:+.2f}\u00d7({c0:+.4f}) = {a0 * c0:+.4f}   \u2502   "
        f"rh\u00b7s = ({-b0:+.2f})\u00d7{s0:+.4f} = {(-b0) * s0:+.4f}   \u2502   "
        f"sum {ARH_OUT[0]:+.4f}",
        f"ch 4:   x\u00b7c = {b0:+.2f}\u00d7({c0:+.4f}) = {b0 * c0:+.4f}   \u2502   "
        f"rh\u00b7s = ({a0:+.2f})\u00d7{s0:+.4f} = {a0 * s0:+.4f}   \u2502   "
        f"sum {ARH_OUT[half]:+.4f}",
    ]
    for i, ln in enumerate(lines):
        b += txt(340, yy + 24 + i * 18, ln, 9, SEC, "middle")

    y2 = yy + 72
    b += vcell(96, y2, 488, 28, "teal",
               "ch 0 = a\u00b7cos \u2212 b\u00b7sin   \u2502   "
               "ch 4 = a\u00b7sin + b\u00b7cos   \u2502   which is E7",
               size=11, weight="500")

    y3 = y2 + 46
    b += txt(340, y3, "Two coincidences that are not coincidences:", 11.5, PRI,
             "middle", "500")
    b += txt(340, y3 + 20, "the MINUS on channel 0 is rotate_half's negation "
             "(it put \u2212x[4] at slot 0), and", 11, SEC, "middle")
    b += txt(340, y3 + 38, "channel 4 reads cos[0] rather than a fourth angle "
             "because the cat put it there.", 11, SEC, "middle")
    b += txt(340, y3 + 60, "Neither line mentions a pair. The pairing lives "
             "entirely in those two moves.", 11, SEC, "middle")

    b += txt(340, y3 + 90, "Sanity: each pair keeps its own length, to 1e-07. "
             "A rotation cannot change it,", 11, PRI, "middle", "500")
    b += txt(340, y3 + 108, "and a wrong pairing almost always does.",
             11, PRI, "middle", "500")
    return wrap(680, y3 + 136, "apply_rope_half, traced",
                "Every intermediate row of the split-half rotation, and one pair "
                "worked out.", b)


DIAGRAMS = {
    "30_rope_pairing.svg": pairing_diagram,
    "31_rope_ladder.svg": ladder_diagram,
    "32_rope_relative.svg": relative_diagram,
    "33_rope_table.svg": table_diagram,
    "34_rope_table_heatmap.svg": table_heatmap_diagram,
    "40_norm_row.svg": norm_row_diagram,
    "41_norm_cost.svg": norm_cost_diagram,
    "50_attention_shapes.svg": attention_shapes_diagram,
    "54_split_view.svg": split_view_diagram,
    "55_split_transpose.svg": split_transpose_diagram,
    "56_rope_conventions.svg": conventions_diagram,
    "57_rotate_half.svg": rotate_half_diagram,
    "58_convention_swap.svg": convention_swap_diagram,
    "60_pi_positions.svg": pi_positions_diagram,
    "61_pi_tradeoff.svg": pi_tradeoff_diagram,
    "62_scaling_family.svg": scaling_family_diagram,
    "63_newaxis.svg": newaxis_diagram,
    "64_rope_half_cat.svg": rope_half_cat_diagram,
    "65_rope_half_trace.svg": rope_half_trace_diagram,
    "51_causal_mask.svg": mask_diagram,
    "52_gqa.svg": gqa_diagram,
    "53_kv_cache.svg": kv_cache_diagram,
}

if __name__ == "__main__":
    from pathlib import Path
    out = Path(__file__).resolve().parent / "diagrams"
    out.mkdir(exist_ok=True)
    for name, fn in DIAGRAMS.items():
        (out / name).write_text(fn())
        print("wrote", out / name)
