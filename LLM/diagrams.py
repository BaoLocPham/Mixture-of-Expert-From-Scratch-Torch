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
