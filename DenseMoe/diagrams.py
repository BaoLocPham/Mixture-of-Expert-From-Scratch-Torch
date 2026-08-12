"""Diagram source for the dense MoE notes.

Regenerate with:  python DenseMoe/diagrams.py

The helpers below are the same ones SparseMoe/diagrams.py uses, reproduced so
this module stands alone - the same reason common.py carries its own copy of
Expert. Numbers come from steps_dense_moe.py's output.
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
        '</marker></defs>')


def txt(x, y, s, size=12, fill=PRI, anchor="start", weight="normal"):
    return (f'<text {F} x="{x}" y="{y}" text-anchor="{anchor}" dominant-baseline="central" '
            f'font-size="{size}" font-weight="{weight}" fill="{fill}">{s}</text>')


def vcell(x, y, w, h, ramp, label, op=1.0, size=12, weight="500", dash=False):
    fill, stroke, tc, _ = RAMP[ramp]
    d = ' stroke-dasharray="3 3"' if dash else ""
    s = (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="4" fill="{fill}" '
         f'stroke="{stroke}" stroke-width="0.5"{d}/>'
         + txt(x + w / 2, y + h / 2 + 0.5, label, size, tc, "middle", weight))
    return f'<g opacity="{op}">{s}</g>' if op != 1.0 else s


def arrow(x1, y1, x2, y2, color=ARR, dash=False):
    d = ' stroke-dasharray="4 4" opacity="0.5"' if dash else ""
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" '
            f'stroke-width="1.5" stroke-linecap="round" marker-end="url(#arrow)"{d}/>')


def wrap(w, h, title, desc, body):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'xmlns:xlink="http://www.w3.org/1999/xlink" width="{w}" height="{h}" '
            f'viewBox="0 0 {w} {h}" role="img"><title>{title}</title><desc>{desc}</desc>'
            f'<rect width="{w}" height="{h}" fill="#FFFFFF"/>{DEFS}{body}</svg>')


# ------------------------------------------------------ 26. the broadcast
def broadcast_diagram():
    """Why w needs unsqueeze(-1) before it can multiply outs."""
    SW, GAP = 74, 10
    cols = [318, 318 + SW + GAP, 318 + 2 * (SW + GAP)]     # rightmost = last axis

    def shape_row(y, label, vals, ramps, names=None, start=0):
        s = txt(300, y + 17, label, 12, PRI, "end", "500")
        for i, v in enumerate(vals):
            x = cols[start + i]
            s += vcell(x, y, SW, 34, ramps[i], str(v))
            if names:
                s += txt(x + SW / 2, y + 48, names[i], 10, SEC, "middle")
        return s

    mid = (cols[0] + cols[2] + SW) / 2
    b = txt(340, 26, "Broadcasting lines axes up from the RIGHT", 13, PRI, "middle", "500")

    # --- the mismatch
    b += txt(340, 56, "w (3, 3)  against  outs (3, 3, 4)", 12, RAMP["coral"][1],
             "middle", "500")
    b += shape_row(78, "w", [3, 3], ["gray", "coral"], ["T", "N"], start=1)
    b += shape_row(146, "outs", [3, 3, 4], ["gray", "gray", "coral"], ["T", "N", "d"])
    b += (f'<rect x="{cols[2] - 4}" y="74" width="{SW + 8}" height="110" rx="6" '
          f'fill="none" stroke="#993C1D" stroke-width="1" stroke-dasharray="4 3"/>')
    b += txt(340, 212, "w has no third axis, so its N lands on outs' d: 3 against 4. "
                       "Neither is 1, so there is nothing to stretch.",
             11, RAMP["coral"][1], "middle")

    # --- the fix
    b += txt(340, 246, "w.unsqueeze(-1) (3, 3, 1)  against  outs (3, 3, 4)", 12,
             RAMP["teal"][1], "middle", "500")
    b += shape_row(268, "w.unsqueeze(-1)", [3, 3, 1], ["gray", "gray", "teal"],
                   ["T", "N", "—"])
    b += shape_row(336, "outs", [3, 3, 4], ["gray", "gray", "teal"], ["T", "N", "d"])
    b += (f'<rect x="{cols[2] - 4}" y="264" width="{SW + 8}" height="110" rx="6" '
          f'fill="none" stroke="#0F6E56" stroke-width="1" stroke-dasharray="4 3"/>')
    b += txt(340, 402, "Now the last axis is 1 against 4: one weight stretches across all "
                       "d dims of that expert's answer.", 11, RAMP["teal"][1], "middle")

    b += arrow(mid, 420, mid, 448)
    b += shape_row(456, "product", [3, 3, 4], ["gray", "gray", "gray"])
    b += txt(300, 541, ".sum(dim=-2)", 12, PRI, "end", "500")
    b += vcell(cols[1], 524, SW, 34, "purple", "3")
    b += vcell(cols[2], 524, SW, 34, "purple", "4")
    b += txt(cols[1] + SW / 2, 572, "T", 10, SEC, "middle")
    b += txt(cols[2] + SW / 2, 572, "d", 10, SEC, "middle")
    b += txt(cols[2] + SW + 16, 541, "N is gone", 11, SEC)

    b += txt(340, 602, "The weight is one scalar per (token, expert), and it has to multiply",
             12, SEC, "middle")
    b += txt(340, 620, "all d numbers of that expert's answer. unsqueeze(-1) adds exactly the",
             12, SEC, "middle")
    b += txt(340, 638, "trailing axis that lets it stretch along d, and nothing else.",
             12, SEC, "middle")
    return wrap(680, 660, "Why unsqueeze(-1)",
                "Shape alignment for w * outs, before and after unsqueeze.", b)


# ----------------------------------------------------- 27. the convex hull
def hull_diagram():
    """Token 0: y sits inside the min/max band of the experts, coordinate by coordinate."""
    E = [[1., 2., -1., 0.], [0., -1., 2., 1.], [2., 1., 0., -1.]]      # E0, E1, E2
    Y = [0.8, 1.0, 0.0, 0.2]
    RAMPS = ["coral", "teal", "purple"]

    def sx(v):
        return 168 + (v + 1.5) * 100                    # -1.5 -> 168, 2.5 -> 568

    b = txt(340, 26, "Token 0: the output cannot leave the experts' span", 13, PRI,
            "middle", "500")

    for i in range(4):
        y = 96 + i * 68
        lo, hi = min(E[j][i] for j in range(3)), max(E[j][i] for j in range(3))
        b += txt(150, y, f"dim {i}", 11, SEC, "end")
        # the band
        b += (f'<rect x="{sx(lo)}" y="{y - 13}" width="{sx(hi) - sx(lo)}" height="26" rx="4" '
              f'fill="#F1EFE8" stroke="#C9C7C0" stroke-width="0.5"/>')
        b += (f'<line x1="{sx(-1.5)}" y1="{y}" x2="{sx(2.5)}" y2="{y}" stroke="#C9C7C0" '
              f'stroke-width="0.5"/>')
        for j in range(3):
            fill, stroke, _, _ = RAMP[RAMPS[j]]
            b += (f'<circle cx="{sx(E[j][i])}" cy="{y}" r="6" fill="{fill}" '
                  f'stroke="{stroke}" stroke-width="1.2"/>')
        # y is marked from ABOVE - it often lands exactly on an expert value
        # (dims 1 and 2 here), and a filled marker would hide the dot underneath
        X = sx(Y[i])
        b += f'<line x1="{X}" y1="{y - 17}" x2="{X}" y2="{y + 17}" stroke="#3D3D3A" stroke-width="1.6"/>'
        b += f'<path d="M{X - 6} {y - 30} L{X + 6} {y - 30} L{X} {y - 20} Z" fill="#3D3D3A"/>'
        b += txt(X, y - 41, f"{Y[i]:.1f}", 10, PRI, "middle", "500")
        b += txt(596, y, f"[{lo:.0f}, {hi:.0f}]", 10, SEC)

    for v in (-1, 0, 1, 2):
        b += (f'<line x1="{sx(v)}" y1="368" x2="{sx(v)}" y2="374" stroke="#C9C7C0" '
              f'stroke-width="1"/>')
        b += txt(sx(v), 386, str(v), 10, SEC, "middle")

    for j, (lbl, r) in enumerate(zip(["E0", "E1", "E2"], RAMPS)):
        x = 220 + j * 80
        fill, stroke, tc, _ = RAMP[r]
        b += f'<circle cx="{x}" cy="416" r="6" fill="{fill}" stroke="{stroke}" stroke-width="1.2"/>'
        b += txt(x + 12, 416, lbl, 11, tc)
    b += '<line x1="470" y1="404" x2="470" y2="428" stroke="#3D3D3A" stroke-width="1.6"/>'
    b += '<path d="M464 400 L476 400 L470 410 Z" fill="#3D3D3A"/>'
    b += txt(484, 416, "y — the mixture", 11, PRI)

    b += txt(340, 452, "The weights are non-negative and sum to 1, so y is a weighted average:",
             12, SEC, "middle")
    b += txt(340, 470, "a point inside the grey band, never outside it. A dense MoE",
             12, SEC, "middle")
    b += txt(340, 488, "interpolates its experts and cannot extrapolate past them.",
             12, SEC, "middle")
    b += txt(340, 514, "Which means the capacity is in the WIDTH of those bands — how",
             12, SEC, "middle")
    b += txt(340, 532, "differently the experts answer. Identical experts collapse every",
             12, SEC, "middle")
    b += txt(340, 550, "band to a point, and the gate stops mattering at all.",
             12, SEC, "middle")
    return wrap(680, 572, "The convex hull of the experts",
                "Per-coordinate min/max over experts, with the mixture inside.", b)


# ------------------------------------------------ 28. MACs per token == params
def bottleneck_diagram():
    """The dense identity: one MAC per parameter per token."""
    SEGS = [("gate", 24, "purple"), ("expert 0", 256, "coral"),
            ("expert 1", 256, "teal"), ("expert 2", 256, "gray")]
    TOTAL, H, TOP = 792, 250, 96
    LX, RX, BW = 214, 386, 84

    b = txt(340, 26, "A dense layer spends one multiply per parameter, per token", 13,
            PRI, "middle", "500")
    b += txt(340, 50, "d = 8,  d_ff = 16,  N = 3", 11, SEC, "middle")
    b += txt(LX + BW / 2, 76, "parameters", 12, PRI, "middle", "500")
    b += txt(RX + BW / 2, 76, "MACs / token", 12, PRI, "middle", "500")

    y = TOP
    for name, n, ramp in SEGS:
        h = n / TOTAL * H
        for x in (LX, RX):
            b += vcell(x, y, BW, h, ramp, "", size=10)
        b += txt(LX - 12, y + h / 2, name, 11, SEC, "end")
        b += txt(RX + BW + 12, y + h / 2, f"{n:,}", 11, SEC)
        y += h

    b += txt(340, TOP + H / 2, "=", 20, PRI, "middle", "500")
    b += (f'<line x1="{LX}" y1="{TOP + H + 8}" x2="{LX + BW}" y2="{TOP + H + 8}" '
          f'stroke="#C9C7C0" stroke-width="1"/>')
    b += (f'<line x1="{RX}" y1="{TOP + H + 8}" x2="{RX + BW}" y2="{TOP + H + 8}" '
          f'stroke="#C9C7C0" stroke-width="1"/>')
    b += txt(LX + BW / 2, TOP + H + 24, "792", 12, PRI, "middle", "500")
    b += txt(RX + BW / 2, TOP + H + 24, "792", 12, PRI, "middle", "500")

    rows = [("3", "792", "792", "100%"), ("8", "2,112", "2,112", "100%"),
            ("64", "16,896", "16,896", "100%"), ("512", "135,168", "135,168", "100%")]
    ty = TOP + H + 64
    b += txt(196, ty, "N", 11, SEC, "end", "500")
    b += txt(330, ty, "params", 11, SEC, "end", "500")
    b += txt(450, ty, "MACs/token", 11, SEC, "end", "500")
    b += txt(546, ty, "active", 11, SEC, "end", "500")
    for i, (n, p, m, a) in enumerate(rows):
        yy = ty + 22 + i * 20
        b += txt(196, yy, n, 11, PRI, "end")
        b += txt(330, yy, p, 11, PRI, "end")
        b += txt(450, yy, m, 11, PRI, "end")
        b += txt(546, yy, a, 11, RAMP["coral"][1], "end")

    b += txt(340, ty + 128, "Two columns, one number. Scaling N buys parameters and pays",
             12, SEC, "middle")
    b += txt(340, ty + 146, "for every one of them on every token. The whole point of MoE is",
             12, SEC, "middle")
    b += txt(340, ty + 164, "to pull these apart — and dense is where they are still welded.",
             12, SEC, "middle")
    return wrap(680, ty + 190, "MACs per token equals parameters",
                "The dense identity, drawn as two identical stacks.", b)


if __name__ == "__main__":
    for fname, fn in [
        ("26-broadcast-align.svg", broadcast_diagram),
        ("27-convex-hull.svg", hull_diagram),
        ("28-macs-equals-params.svg", bottleneck_diagram),
    ]:
        open(fname, "w").write(fn())
        print("wrote", fname)
