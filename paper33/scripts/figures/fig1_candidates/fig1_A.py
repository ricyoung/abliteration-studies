#!/usr/bin/env python3
"""fig1_A.py -- Figure 1, candidate A: the study workflow as horizontal swimlanes.

    Search  (dev prompts)       1 model pairs -> 2 optimise (Heretic 2.0) -> 3 select one trial
    Release                     4 export & release      (+ community models, dashed)
    Evaluate (held-out prompts) 5 refusal | teacher-forced divergence | capability
    Statistics                  6 Wilson, McNemar, mixed-effects (H1, H2), kappa (H3), Spearman, Holm

Run:  scripts/figures/.venv/bin/python scripts/figures/fig1_candidates/fig1_A.py
Out:  scripts/figures/fig1_candidates/fig1_A.pdf              vector, 7.0 in wide, TrueType (Type 42) fonts
      paper/figures/preview/fig1_candidates/fig1_A.png        200 dpi preview

Encoding: grey cards = core stages, numbered 1-6; white dashed tags = additional conditions, attached to the
stage they extend; colour marks method identity only (house palette); arrows = flow of models.
Coordinates are inches on the final 7.0 in canvas (y grows downwards), so every font size is its printed size.
The script checks text/text and text/arrow collisions and that all text stays on the canvas (exit 1 if not).
"""
from __future__ import annotations

import math
import re
import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_agg import RendererAgg  # noqa: E402
from matplotlib.font_manager import FontProperties  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Circle, FancyBboxPatch, Polygon  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PDF = HERE / "fig1_A.pdf"
PNG = ROOT / "paper" / "figures" / "preview" / "fig1_candidates" / "fig1_A.png"
SHOW_HARDWARE = True         # GPU notes (stage 2, stage 5); False drops ~20 words for a leaner figure

# ------------------------------------------------------------------------------------------------ palette
INK, INK2 = "#0b0b0b", "#52514e"
GREY, GREY_L = "#8a8984", "#e6e5e1"
CARD = "#f2f1ee"            # GREY_L at ~55 % on white: core-stage card fill
WHITE = "#ffffff"
EDGE = INK2                 # arrows
MC = {"dim": "#2a78d6", "mpoa": "#eb6834", "ara": "#1baf7a", "som": "#eda100", "jspace": "#e87ba4",
      "community": "#6250d6", "gamma": "#e34948"}
DASH = (0, (2.6, 1.6))

FAMILY = "Liberation Sans"
mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": [FAMILY], "pdf.fonttype": 42,
                     "ps.fonttype": 42, "svg.fonttype": "none", "axes.unicode_minus": False})
ST = {  # text styles: size (pt), weight, colour
    "title": (8.0, "bold", INK),     # stage titles
    "lane": (8.0, "bold", INK),      # lane tabs
    "head": (7.0, "bold", INK),      # panel heads, hypothesis chips
    "body": (7.0, "normal", INK),
    "note": (7.0, "normal", INK2),
    "badge": (7.0, "bold", WHITE),
}

# ------------------------------------------------------------------------------------------------ metrics
LH = 0.122                   # in, body line pitch (7 pt)
TITLE_BL = 0.165             # in, title baseline below a card top
TITLE_GAP = 0.158            # in, title baseline -> first body baseline
PAD = 0.09                   # in, card inner padding (left / right)
R = 0.045                    # in, corner radius of every card, panel and tag
CAP7 = 7 * 0.716 / 72        # in, cap height of 7 pt Liberation Sans
SQ = 0.072                   # in, colour square side
SQ_GAP = 0.035               # in, square -> label

_R = RendererAgg(10, 10, 72)  # 72 dpi: pixels == points


def tw(s: str, style: str = "body") -> float:
    """Advance width of `s` in inches."""
    size, weight, _ = ST[style]
    fp = FontProperties(family=FAMILY, size=size, weight=weight)
    return _R.get_text_width_height_descent(s, fp, ismath=False)[0] / 72


def parts_width(parts) -> float:
    w = 0.0
    for p in parts:
        if isinstance(p, float):
            w += p
        elif isinstance(p, tuple) and len(p) == 1:
            w += SQ + SQ_GAP
        else:
            s, st = (p, "body") if isinstance(p, str) else p
            w += tw(s, st)
    return w


# ------------------------------------------------------------------------------------------------ canvas
W = 7.0
X0, X1 = 0.30, 6.985         # content area (left of X0: lane tabs)
CW = X1 - X0
GAP = 0.28                   # in, gap between the three Search cards (arrow room)
COLW = (CW - 2 * GAP) / 3
COLX = [X0 + i * (COLW + GAP) for i in range(3)]
COLC = [x + COLW / 2 for x in COLX]
VGAP_A, VGAP_B = 0.09, 0.16  # in, card bottom -> lane boundary, lane boundary -> next card top

# vertical plan (y downwards)
L1_TOP = 0.0
C1_TOP, C1_H = 0.05, 1.25                      # Search cards
TAG_H = 0.30                                   # two-line dashed tags
L1_BOT = C1_TOP + C1_H + TAG_H / 2 + VGAP_A
C4_TOP, C4_H = L1_BOT + VGAP_B, 0.67           # Release card
L2_BOT = C4_TOP + C4_H + VGAP_A
C5_TOP = L2_BOT + VGAP_B                       # Evaluate card
P_TOP = 0.255                                  # panels below the card title row (relative)
P_H = 0.76
C5_H = P_TOP + P_H + 0.075
L3_BOT = C5_TOP + C5_H + VGAP_A
C6_TOP = L3_BOT + VGAP_B                       # Statistics band
C6_H = 0.50
H = C6_TOP + C6_H + 0.03

fig = plt.figure(figsize=(W, H))
ax = fig.add_axes((0, 0, 1, 1))
ax.set_xlim(0, W)
ax.set_ylim(H, 0)
ax.set_axis_off()

TEXTS: list = []
ARROWS: list = []


def text(x, y, s, style="body", ha="left", va="baseline", **kw):
    size, weight, color = ST[style]
    kw.setdefault("zorder", 5)
    t = ax.text(x, y, s, fontsize=size, fontweight=weight, color=color, ha=ha, va=va, family=FAMILY, **kw)
    TEXTS.append(t)
    return t


def card(x, y, w, h, fc=CARD, ec="none", lw=0.0, ls="-", z=1):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={R}", fc=fc, ec=ec, lw=lw,
                                ls=ls, zorder=z, joinstyle="round"))


def square(x, baseline, color):
    """Colour square standing on the text baseline (cap height); returns x after square + gap."""
    ax.add_patch(FancyBboxPatch((x, baseline - SQ), SQ, SQ, boxstyle="round,pad=0,rounding_size=0.012",
                                fc=color, ec="none", zorder=4))
    return x + SQ + SQ_GAP


def runs(x, baseline, parts):
    """parts: str (body) | (str, style) | ("#rrggbb",) colour square | float gap (in). Returns end x."""
    for p in parts:
        if isinstance(p, float):
            x += p
        elif isinstance(p, tuple) and len(p) == 1:
            x = square(x, baseline, p[0])
        else:
            s, st = (p, "body") if isinstance(p, str) else p
            text(x, baseline, s, st)
            x += tw(s, st)
    return x


def badge(x, baseline, n):
    """Stage number: ink rounded square centred on the 8 pt cap height; returns x after it."""
    s = 0.135
    cy = baseline - 8 * 0.716 / 72 / 2
    ax.add_patch(FancyBboxPatch((x, cy - s / 2), s, s, boxstyle="round,pad=0,rounding_size=0.025",
                                fc=INK, ec="none", zorder=4))
    text(x + s / 2, cy + CAP7 / 2, str(n), "badge", ha="center")
    return x + s + 0.055


def title(x, top, n, s):
    bl = top + TITLE_BL
    text(badge(x + PAD, bl, n), bl, s, "title")
    return bl


def chip(x, baseline, s, w=None, style="body", fc=GREY_L, ec="none", h=0.104, padx=0.045):
    """Small rounded label (outcome classes, hypotheses); returns x after it."""
    w = w or tw(s, style) + 2 * padx
    ax.add_patch(FancyBboxPatch((x, baseline - CAP7 / 2 - h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.03",
                                fc=fc, ec=ec, lw=0.6, zorder=3))
    text(x + w / 2, baseline, s, style, ha="center")
    return x + w


def tag(x, top, line1, line2, w):
    """Additional condition: white tag, dashed outline, two lines (identity, scope)."""
    card(x, top, w, TAG_H, fc=WHITE, ec=GREY, lw=0.7, ls=DASH, z=3)
    bl = top + 0.128
    runs(x + 0.06, bl, line1)
    text(x + 0.06, bl + LH, line2, "note")


def tag_w(line1, line2):
    return max(parts_width(line1), tw(line2, "note")) + 0.12


def arrow(pts, dashed=False, color=EDGE, lw=0.8, hl=0.062, hw=0.030):
    (xa, ya), (xb, yb) = pts[-2], pts[-1]
    L = math.hypot(xb - xa, yb - ya)
    ux, uy = (xb - xa) / L, (yb - ya) / L
    body = list(pts[:-1]) + [(xb - ux * hl * 0.8, yb - uy * hl * 0.8)]
    ax.add_line(Line2D([p[0] for p in body], [p[1] for p in body], color=color, lw=lw, zorder=2,
                       ls=DASH if dashed else "-", solid_joinstyle="round", solid_capstyle="butt"))
    bx, by = xb - ux * hl, yb - uy * hl
    ax.add_patch(Polygon([(xb, yb), (bx - uy * hw, by + ux * hw), (bx + uy * hw, by - ux * hw)], closed=True,
                         fc=color, ec="none", zorder=2))
    ARROWS.append(pts)


def lane_tab(top, bot, parts):
    """Rotated lane label on a light tab at the left edge; parts [(str, style) | float gap] read bottom-up."""
    card(0.0, top + 0.05, 0.215, bot - top - 0.10, fc=GREY_L)
    total = sum(p if isinstance(p, float) else tw(*p) for p in parts)
    y = (top + bot) / 2 + total / 2          # bottom end of the rotated label
    for p in parts:
        if isinstance(p, float):
            y -= p
            continue
        s, st = p
        size, weight, color = ST[st]
        TEXTS.append(ax.text(0.1075, y, s, fontsize=size, fontweight=weight, color=color, rotation=90,
                             rotation_mode="anchor", ha="left", va="center", family=FAMILY, zorder=5))
        y -= tw(s, st)


# ================================================================================================ lane 1: Search
lane_tab(L1_TOP, L1_BOT, [("Search", "lane"), 0.07, ("dev prompts", "note")])
top = C1_TOP
FOOT = top + C1_H - 0.088                      # baseline of the last line in a Search card

# --- 1 model pairs ------------------------------------------------------------------------------
x = COLX[0]
card(x, top, COLW, C1_H)
bl = title(x, top, 1, "Model pairs")
pairs = [("Qwen", "2.5-32B", "3.8-27B"), ("Gemma", "2-27B", "4-31B"), ("OLMo", "2-32B", "3.1-32B"),
         ("Mistral-Small", "2409", "3.2 (24B)")]
small = ("Qwen", "2.5-7B", "3.5-4B")
xf = x + PAD
xe = xf + max(tw(p[0]) for p in pairs) + 0.13 + max(tw(p[1]) for p in pairs + [small])   # right edge, earlier
xa = xe + 0.07
xl = xa + tw("→") + 0.07


def pair_row(y, fam, old, new):
    text(xf, y, fam)
    text(xe, y, old, ha="right")
    text(xa, y - 0.012, "→", "note")          # lift the arrow glyph to the middle of the digits
    text(xl, y, new)


y = bl + TITLE_GAP
for k, p in enumerate(pairs):
    pair_row(y + k * LH, *p)
y += 3 * LH + 0.155
text(xf, y, "small pair", "note")
ax.add_line(Line2D([xf + tw("small pair", "note") + 0.06, x + COLW - PAD], [y - CAP7 / 2] * 2, color=GREY,
                   lw=0.5, zorder=2))
pair_row(y + LH, *small)
text(xf, FOOT, "instruction-tuned · bf16", "note")

# --- 2 optimise --------------------------------------------------------------------------------
x = COLX[1]
card(x, top, COLW, C1_H)
bl = title(x, top, 2, "Optimise · Heretic 2.0")
y = bl + TITLE_GAP
runs(x + PAD, y, [(MC["dim"],), "DiM", 0.10, (MC["mpoa"],), "MPOA", 0.10, (MC["ara"],), "ARA", 0.10,
                  (MC["som"],), "SOM"])
y += LH
runs(x + PAD, y, [(MC["jspace"],), "J-space", 0.06, ("small pair", "note")])
y += LH + 0.025
text(x + PAD, y, "Optuna TPE · 100 trials · seed 1")
y += LH
runs(x + PAD, y, [("min", "note"), 0.045, "keyword refusals", 0.045, ("100 harmful", "note")])
y += LH
runs(x + PAD, y, [("min", "note"), 0.045, "first-token KL", 0.045, ("100 harmless", "note")])
y += LH + 0.025
if SHOW_HARDWARE:
    text(x + PAD, y, "H200: main study · RTX 4090: small pair", "note")
# additional conditions straddle the bottom edge of stage 2
t_gamma = ([(MC["gamma"],), "γ-corrected MPOA"], "Gemma, OLMo pairs")
t_seed = (["seed-2 replicate"], "small pair")
wg, ws = tag_w(*t_gamma), tag_w(*t_seed)
tg = 0.06
tx = COLC[1] - (wg + ws + tg) / 2
ty = top + C1_H - TAG_H / 2
tag(tx, ty, *t_gamma, wg)
tag(tx + wg + tg, ty, *t_seed, ws)

# --- 3 select ----------------------------------------------------------------------------------
x = COLX[2]
card(x, top, COLW, C1_H)
bl = title(x, top, 3, "Select one trial")
y = bl + TITLE_GAP
text(x + PAD, y, "fewest keyword refusals with KL ≤ 0.1")
y += LH + 0.025
text(x + PAD, y, "sensitivity: KL ≤ 0.05 · 0.2 · 0.5", "note")
y += LH
text(x + PAD, y, "same rule for every study", "note")
# pictogram: trials in (KL, refusals); eligible = filled, left of the KL threshold; the pick is ringed
px0 = x + PAD + tw("refusals", "note") + 0.07
pw, py1 = x + COLW - PAD - px0, FOOT - 0.115
ph = py1 - (y + 0.075)
ax.add_line(Line2D([px0, px0, px0 + pw], [py1 - ph, py1, py1], color=GREY, lw=0.6, zorder=2))
thr = 0.40
ax.add_line(Line2D([px0 + thr * pw] * 2, [py1 - ph, py1], color=INK2, lw=0.6, ls=(0, (2, 1.5)), zorder=2))
cloud = [(0.05, 0.90), (0.10, 0.76), (0.14, 0.84), (0.18, 0.64), (0.23, 0.70), (0.27, 0.53), (0.32, 0.58),
         (0.37, 0.46), (0.50, 0.30), (0.57, 0.36), (0.64, 0.22), (0.73, 0.26), (0.82, 0.14), (0.92, 0.10)]
pick = (0.335, 0.30)
for u, v in cloud:
    ax.add_patch(Circle((px0 + u * pw, py1 - v * ph), 0.017, fc=GREY if u <= thr else WHITE, ec=GREY, lw=0.5,
                        zorder=3))
ax.add_patch(Circle((px0 + pick[0] * pw, py1 - pick[1] * ph), 0.022, fc=INK, ec="none", zorder=4))
ax.add_patch(Circle((px0 + pick[0] * pw, py1 - pick[1] * ph), 0.042, fc="none", ec=INK, lw=0.6, zorder=4))
text(px0 - 0.06, py1 - ph + CAP7, "refusals", "note", ha="right")
text(px0 + thr * pw, FOOT, "0.1", "note", ha="center")
text(px0 + pw, FOOT, "KL", "note", ha="right")

# --- arrows 1 -> 2 -> 3 --------------------------------------------------------------------------
ym = top + C1_H / 2
for i in range(2):
    arrow([(COLX[i] + COLW + 0.04, ym), (COLX[i + 1] - 0.04, ym)])

# ================================================================================================ lane 2: Release
lane_tab(L1_BOT, L2_BOT, [("Release", "lane")])
x, top = COLX[2], C4_TOP
card(x, top, COLW, C4_H)
bl = title(x, top, 4, "Export & release")
y = bl + TITLE_GAP
text(x + PAD, y, "adapter merged → standalone weights")
y += LH
text(x + PAD, y, "public Hugging Face repos")
y += LH
runs(x + PAD, y, [("reproduce manifest:", "note"), 0.04, "DiM, MPOA, ARA"])
arrow([(COLC[2], C1_TOP + C1_H + 0.04), (COLC[2], C4_TOP - 0.04)])

# community models: dashed tag in the Release lane, feeding the evaluation like the selected models
t_comm = ([(MC["community"],), "8 community models"], "public abliterated releases")
wc_ = tag_w(*t_comm)
cmx, cmy = COLC[1] - wc_ / 2, C4_TOP + (C4_H - TAG_H) / 2
tag(cmx, cmy, *t_comm, wc_)

# ================================================================================================ lane 3: Evaluate
lane_tab(L2_BOT, L3_BOT, [("Evaluate", "lane"), 0.07, ("held-out prompts", "note")])
x, top = X0, C5_TOP
card(x, top, CW, C5_H)
bl = title(x, top, 5, "Evaluation")
if SHOW_HARDWARE:
    text(X1 - PAD, bl, "RTX PRO 6000: Qwen, Gemma · H200: OLMo, Mistral · RTX 4090: small pair", "note",
         ha="right")

# inputs land on the top edge of stage 5
arrow([(COLC[0], C1_TOP + C1_H + 0.04), (COLC[0], C5_TOP - 0.04)])
text(COLC[0] + 0.07, C4_TOP + C4_H / 2 + CAP7 / 2, "originals", "note")
arrow([(COLC[1], cmy + TAG_H), (COLC[1], C5_TOP - 0.04)], dashed=True)
arrow([(COLC[2], C4_TOP + C4_H + 0.04), (COLC[2], C5_TOP - 0.04)])

# panels (white, inside stage 5)
ptop = top + P_TOP
pin, pout, pg = X0 + 0.07, X1 - 0.07, 0.06
hb = ptop + 0.15                                   # panel head baseline
r1 = hb + TITLE_GAP                                # first body row
prompts = [("1,322 prompts", "note"), ("972 harmful-type", "body"), ("350 benign", "body"),
           ("greedy, ≤ 512 tokens", "note")]
judges = [("judges", "note"), ("WildGuard (primary)", "body"), ("HarmBench, Qwen3Guard", "body"),
          ("2 encoders, keywords", "body")]
outcomes = ["refusal", "compliance", "degenerate", "incomplete"]
cap_lines = [["MMLU-Pro", ("1,000", "note"), 0.08, "GSM8K", ("500", "note")],
             ["IFEval", ("541", "note"), 0.08, "HumanEval", ("164", "note")],
             ["MBPP", ("500", "note")]]
div_lines = ["log-prob shift on the original", "model's answers (350 benign)"]
w_i = max(tw(s, st) for s, st in prompts)
w_ii = max(tw(s, st) for s, st in judges)
w_iii = max(tw(s) for s in outcomes) + 2 * 0.05
ag = 0.22                                          # arrow gap between refusal columns
wa = 2 * PAD + w_i + ag + w_ii + ag + w_iii
wc = 2 * PAD + max(parts_width(ln) + 0.04 * sum(isinstance(q, str) for q in ln) for ln in cap_lines)
wb = (pout - pin) - wa - wc - 2 * pg
panels = [(pin, wa), (pin + wa + pg, wb), (pin + wa + pg + wb + pg, wc)]
for px, pw_ in panels:
    card(px, ptop, pw_, P_H, fc=WHITE, z=2)

# (a) refusal: prompts -> judges -> outcome classes
px, _ = panels[0]
text(px + PAD, hb, "Refusal", "head")
xi = px + PAD
xii = xi + w_i + ag
xiii = xii + w_ii + ag
for k, (s, st) in enumerate(prompts):
    text(xi, r1 + k * LH, s, st)
for k, (s, st) in enumerate(judges):
    text(xii, r1 + k * LH, s, st)
for k, s in enumerate(outcomes):
    chip(xiii, r1 + k * LH, s, w=w_iii)
yarr = r1 + 1.5 * LH - CAP7 / 2
arrow([(xi + w_i + 0.06, yarr), (xii - 0.06, yarr)])
arrow([(xii + w_ii + 0.06, yarr), (xiii - 0.05, yarr)])
# (b) teacher-forced divergence
px, _ = panels[1]
text(px + PAD, hb, "Teacher-forced divergence", "head")
for k, s in enumerate(div_lines):
    text(px + PAD, r1 + k * LH, s)
# (c) capability
px, _ = panels[2]
text(px + PAD, hb, "Capability", "head")
for k, ln in enumerate(cap_lines):
    parts = []
    for q in ln:
        parts += [q, 0.04] if isinstance(q, str) else [q]
    runs(px + PAD, r1 + k * LH, parts)
text(px + PAD, r1 + 3 * LH, "lm-eval on vLLM", "note")

# ================================================================================================ lane 4: Statistics
lane_tab(L3_BOT, H, [("Statistics", "lane")])
x, top = X0, C6_TOP
card(x, top, CW, C6_H)
bl = title(x, top, 6, "Statistics")
text(x + PAD, bl + LH + 0.02, "Holm correction", "note")
arrow([(COLC[1], C5_TOP + C5_H + 0.04), (COLC[1], C6_TOP - 0.04)])


def hyps(x, baseline, labels):
    for lab in labels:
        x = chip(x, baseline, lab, style="head", fc=WHITE, ec=INK, padx=0.035) + 0.035
    return x + 0.015


items = [
    [(None, "Wilson intervals"), (None, "McNemar vs original")],
    [(["H1", "H2"], "mixed-effects logistic"), (None, "generation × method class"),
     (None, ("random effects: prompt, family", "note"))],
    [(["H3"], "keyword vs judge"), (None, "agreement κ")],
    [(None, "Spearman"), (None, "dev vs held-out")],
]


def item_w(it):
    w = 0.0
    for hs, s in it:
        s, st = (s, "body") if isinstance(s, str) else s
        hw_ = sum(tw(h, "head") + 0.07 + 0.035 for h in hs) + 0.015 if hs else 0.0
        w = max(w, hw_ + tw(s, st))
    return w


wid = [item_w(it) for it in items]
t_end = x + PAD + 0.19 + tw("Statistics", "title")
g_ = (X1 - PAD - t_end - sum(wid)) / len(wid)
xs, cur = [], t_end + g_
for w_ in wid:
    xs.append(cur)
    cur += w_ + g_
for xx, it in zip(xs, items):
    for k, (hs, s) in enumerate(it):
        s, st = (s, "body") if isinstance(s, str) else s
        x_ = hyps(xx, bl + k * LH, hs) if hs else xx
        text(x_, bl + k * LH, s, st)


# ================================================================================================ checks + save
def check():
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()

    def to_in(bb):
        return bb.x0 / fig.dpi, H - bb.y1 / fig.dpi, bb.x1 / fig.dpi, H - bb.y0 / fig.dpi

    boxes = [(t.get_text(), to_in(t.get_window_extent(rend))) for t in TEXTS]
    problems = []
    for i, (s, (a0, b0, a1, b1)) in enumerate(boxes):
        if a0 < 0 or a1 > W or b0 < 0 or b1 > H:
            problems.append(f"outside canvas: {s!r}")
        for s2, (c0, d0, c1, d1) in boxes[i + 1:]:
            if a0 < c1 - 0.004 and c0 < a1 - 0.004 and b0 < d1 - 0.004 and d0 < b1 - 0.004:
                problems.append(f"text overlap: {s!r} / {s2!r}")
        for pts in ARROWS:
            for (p, q) in zip(pts[:-1], pts[1:]):
                hit = any(a0 - 0.012 < p[0] + (q[0] - p[0]) * k / 60 < a1 + 0.012 and
                          b0 - 0.012 < p[1] + (q[1] - p[1]) * k / 60 < b1 + 0.012 for k in range(61))
                if hit:
                    problems.append(f"arrow touches text: {s!r}")
    sizes = sorted({t.get_fontsize() for t in TEXTS})
    toks = [w for t in TEXTS for w in re.split(r"\s+", t.get_text()) if re.search(r"[A-Za-z0-9]", w)]
    words = [w for w in toks if re.search(r"[A-Za-z]", w)]
    print(f"canvas {W:.2f} x {H:.2f} in; font sizes {sizes} pt; {len(TEXTS)} text items; "
          f"{len(toks)} tokens ({len(words)} words with letters, {len(toks) - len(words)} numbers)")
    for p in dict.fromkeys(problems):
        print("  !", p)
    return not problems


ok = check()
PNG.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(PDF)
fig.savefig(PNG, dpi=200)
print(f"wrote {PDF}\nwrote {PNG}")
sys.exit(0 if ok else 1)
