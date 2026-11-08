#!/usr/bin/env python3
"""fig1_B.py -- Figure 1 candidate B: the study workflow in two tiers.

Concept B
  top tier     the search pipeline as one slim row: 1 model pairs -> 2 optimise -> 3 select -> 4 export
  between      additional conditions (dashed pills) plug in where they enter; the originals (from 1) and the
               selected models (from 4) meet in the stage-5 hub
  bottom tier  the hub fans out into the three measurement streams (refusal + judges, divergence, capability),
               which converge into 6 statistics / hypotheses

Width: drawn at the template's full text width S.TEXT_W (PRIMEarxiv: 6.5 in), so \\includegraphics[width=
\\textwidth] prints it at scale 1.0 and 7 pt stays 7 pt; included at 7.0 in it only grows (7.5 pt).

Grid: four columns, two wide (1, 2) and two narrow (3, 4), gap 0.18 in. Every card edge sits on this grid:
the top cards on the four columns; the refusal stream (primary outcome, three-step mini pipeline) spans
columns 1-2, divergence and capability sit on columns 3 and 4; the statistics cells start on the same columns.

Encoding
  light-grey filled card   core stage          dashed pill, grey text   additional condition
  dark numbered disc       stage number        coloured square          method identity (house colours)
  grey arrows              flow (checked: no arrow comes within 0.02 in of any text)
  earlier generation grey, later generation bold (no extra colour scheme for generations)

Run:
    scripts/figures/.venv/bin/python scripts/figures/fig1_candidates/fig1_B.py
Writes
    scripts/figures/fig1_candidates/fig1_B.pdf                 vector, TrueType (Type 42) fonts
    paper/figures/preview/fig1_candidates/fig1_B.png           200 dpi preview
and prints a QA report: font sizes, word count, text/text and arrow/text collisions, text too close to its card.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.font_manager import FontProperties  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch  # noqa: E402
from matplotlib.path import Path as MPath  # noqa: E402
from matplotlib.transforms import Bbox  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import style as S  # noqa: E402

ROOT = HERE.parent.parent.parent
PDF_OUT = HERE / "fig1_B.pdf"
PNG_OUT = ROOT / "paper" / "figures" / "preview" / "fig1_candidates" / "fig1_B.png"

# ============================================================================================ design tokens
FONT = "Liberation Sans"
INK, INK2 = S.TEXT, S.TEXT_SECONDARY            # #0b0b0b, #52514e
GREY, RULE = "#8a8984", "#e6e5e1"               # neutral greys
CARD = "#f2f1ee"                                # card fill: #e6e5e1 at ~half strength
WHITE = "#ffffff"
MC = S.METHOD_COLORS

FS_TITLE = 8.5      # pt, stage titles
FS_BODY = 7.0       # pt, everything else (house minimum)

W = getattr(S, "TEXT_W", S.DOUBLE_COL)          # 6.5 in (template text width)
MX = 0.05           # outer margin
R_CARD = 0.055      # corner radius of every card (in)
PADX = 0.09         # inner left/right padding of cards
LH = 0.125          # line pitch of 7 pt body text (in)
T_BASE = 0.185      # card top -> title baseline
B1 = 0.205          # title baseline -> first body baseline
BOT = 0.092         # last body baseline -> card bottom
LW_ARROW = 0.9      # pt
LW_DASH = 0.75      # pt
DASH = (0, (2.6, 1.7))
R_BADGE = 0.071


def card_h(n_rows: float) -> float:
    return T_BASE + B1 + (n_rows - 1) * LH + BOT


# ============================================================================================ vertical rhythm
H_TOP = card_h(6.3)     # card 2 carries one 0.3-line gap between its two groups
H_STREAM = card_h(6)
H_STATS = card_h(3)
H_HUB = 0.38
GAP_TOP_HUB = 0.42      # top cards -> hub top (the dashed additional conditions live here)
GAP_FAN = 0.27          # hub bottom -> stream tops
GAP_CONV = 0.25         # stream bottoms -> statistics top
H = MX + H_STATS + GAP_CONV + H_STREAM + GAP_FAN + H_HUB + GAP_TOP_HUB + H_TOP + MX

S.apply()
mpl.rcParams["figure.constrained_layout.use"] = False
fig = plt.figure(figsize=(W, H), dpi=200)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")
renderer = fig.canvas.get_renderer()

TEXTS: list[tuple[mpl.text.Text, Bbox | None]] = []     # (artist, container) for QA
ARROWS: list[FancyArrowPatch] = []


# ============================================================================================ primitives
def fp(size=FS_BODY, bold=False):
    return FontProperties(family=FONT, size=size, weight="bold" if bold else "normal")


def typeset(s: str) -> str:
    """House typography: thin spaces around the middle-dot separator."""
    return s.replace(" · ", " · ")


def tw(s: str, size=FS_BODY, bold=False) -> float:
    """Advance width of a string in inches."""
    w, _, _ = renderer.get_text_width_height_descent(typeset(s), fp(size, bold), ismath=False)
    return w / fig.dpi


def text(x, y, s, size=FS_BODY, color=INK, bold=False, ha="left", va="baseline", box=None, **kw):
    t = ax.text(x, y, typeset(s), fontproperties=fp(size, bold), color=color, ha=ha, va=va, zorder=6, **kw)
    TEXTS.append((t, box))
    return t


def rbox(x, y, w, h, fill=CARD, ec="none", lw=0.0, ls="-", r=R_CARD, z=1):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fill, ec=ec,
                                lw=lw, ls=ls, zorder=z, joinstyle="round"))
    return Bbox.from_bounds(x, y, w, h)


def badge(cx, cy, n):
    ax.add_patch(Circle((cx, cy), R_BADGE, fc=INK, ec="none", zorder=6))
    t = ax.text(cx, cy, str(n), fontproperties=fp(FS_BODY, True), color=WHITE, ha="center",
                va="center_baseline", zorder=7)
    TEXTS.append((t, None))


def title_w(title):
    return 2 * R_BADGE + 0.065 + tw(title, FS_TITLE, True)


def title_line(x, tb, n, title, box):
    """Badge + bold title on baseline tb, starting at x."""
    badge(x + R_BADGE, tb + 0.030, n)
    text(x + 2 * R_BADGE + 0.065, tb, title, FS_TITLE, INK, bold=True, box=box)


def card(x, y, w, h, n=None, title=None, sub=None):
    """Filled card with optional stage badge + title (+ grey note right-aligned on the title line).
    Returns (bbox, baseline of the first body row)."""
    box = rbox(x, y, w, h)
    tb = y + h - T_BASE
    if n is not None:
        title_line(x + PADX, tb, n, title, box)
    else:
        text(x + PADX, tb, title, FS_TITLE, INK, bold=True, box=box)
    if sub:
        text(x + w - PADX, tb, sub, FS_BODY, INK2, ha="right", box=box)
    return box, tb - B1


SQ = 0.064           # chip square
SQ_GAP = 0.026       # square -> label


def chip_w(label):
    return SQ + SQ_GAP + tw(label)


def chip(x, yb, label, color, txt_color=INK, box=None):
    """Coloured square + label on baseline yb. Returns the x after the label."""
    ax.add_patch(FancyBboxPatch((x, yb - 0.002), SQ, SQ, boxstyle="round,pad=0,rounding_size=0.013",
                                fc=color, ec="none", zorder=6))
    text(x + SQ + SQ_GAP, yb, label, FS_BODY, txt_color, box=box)
    return x + chip_w(label)


def chip_row(x0, x1, yb, items, box=None, gap=None):
    """Chips spread evenly so that the row spans exactly x0..x1 (or with a fixed gap). Returns the gap."""
    if gap is None:
        gap = (x1 - x0 - sum(chip_w(lab) for lab, _ in items)) / (len(items) - 1)
    assert gap > 0.045, f"chip row too tight ({gap:.3f} in)"
    x = x0
    for lab, key in items:
        x = chip(x, yb, lab, MC[key], box=box) + gap
    return gap


DASH_PAD_X, DASH_H = 0.055, 0.165


def dashed_w(label, with_square):
    return (chip_w(label) if with_square else tw(label)) + 2 * DASH_PAD_X


def dashed_chip(x, yb, label, color):
    """Additional condition: dashed pill, coloured square (method identity) if any, grey label."""
    y0 = yb - 0.050
    box = rbox(x, y0, dashed_w(label, bool(color)), DASH_H, fill=WHITE, ec=GREY, lw=LW_DASH, ls=DASH,
               r=DASH_H / 2, z=4)
    if color:
        chip(x + DASH_PAD_X, yb, label, color, txt_color=INK2, box=box)
    else:
        text(x + DASH_PAD_X, yb, label, color=INK2, box=box)
    return box


def rounded_path(pts, r=0.06):
    """Orthogonal polyline through pts with rounded corners -> matplotlib Path."""
    verts, codes = [pts[0]], [MPath.MOVETO]
    for i in range(1, len(pts) - 1):
        (x0, y0), (x1, y1), (x2, y2) = pts[i - 1], pts[i], pts[i + 1]
        d1, d2 = np.hypot(x1 - x0, y1 - y0), np.hypot(x2 - x1, y2 - y1)
        rr = min(r, d1 / 2, d2 / 2)
        a = (x1 - (x1 - x0) / d1 * rr, y1 - (y1 - y0) / d1 * rr)
        b = (x1 + (x2 - x1) / d2 * rr, y1 + (y2 - y1) / d2 * rr)
        verts += [a, (x1, y1), b]
        codes += [MPath.LINETO, MPath.CURVE3, MPath.CURVE3]
    verts.append(pts[-1])
    codes.append(MPath.LINETO)
    return MPath(verts, codes)


def arrow(pts, dashed=False, head=True, color=GREY, r=0.06, lw=LW_ARROW, check=True):
    a = FancyArrowPatch(path=rounded_path(pts, r), arrowstyle="-|>" if head else "-", mutation_scale=6.5,
                        color=color, lw=lw, ls=DASH if dashed else "-", zorder=3, shrinkA=0, shrinkB=0,
                        capstyle="butt", joinstyle="round")
    ax.add_patch(a)
    if check:
        ARROWS.append(a)
    return a


def line(pts, **kw):
    return arrow(pts, head=False, **kw)


# ============================================================================================ grid
CW = W - 2 * MX                                     # 6.4 in
G = 0.18                                            # column gap
WN = 1.36                                           # narrow column (3, 4)
WW = (CW - 3 * G - 2 * WN) / 2                      # wide column (1, 2): 1.57 in
CWS = [WW, WW, WN, WN]
X = [MX + sum(CWS[:i]) + i * G for i in range(4)]
XC = [x + w / 2 for x, w in zip(X, CWS)]            # column centres

y_top = H - MX - H_TOP                              # bottom edge of the top cards
y_hub = y_top - GAP_TOP_HUB - H_HUB
y_str = y_hub - GAP_FAN - H_STREAM
y_sta = MX

# ============================================================================================ top tier
# ---- 1 model pairs: family | earlier -> later (later generation bold)
b1, yb = card(X[0], y_top, CWS[0], H_TOP, 1, "Model pairs")
PAIRS = [("Qwen", "2.5-32B", "3.8-27B"), ("Gemma", "2-27B", "4-31B"), ("OLMo", "2-32B", "3.1-32B"),
         ("Mistral-Small", "2409", "3.2-24B"), ("Qwen", "2.5-7B", "3.5-4B")]
x_fam = X[0] + PADX
x_new_r = X[0] + CWS[0] - PADX
w_new = max(tw(p[2], bold=True) for p in PAIRS)
x_new = x_new_r - w_new                             # later column, flush with the card's right padding
x_arr = x_new - 0.068
x_old_r = x_arr - 0.068
fam_w = [tw(p[0]) + (tw(" small") if i == 4 else 0) for i, p in enumerate(PAIRS)]
assert all(x_fam + fw + 0.04 < x_old_r - tw(p[1]) for fw, p in zip(fam_w, PAIRS)), "pairs table too tight"
for i, (fam, old, new) in enumerate(PAIRS):
    y = yb - (i if i < 4 else 4.3) * LH              # the small pair sits after a 0.3-line gap
    text(x_fam, y, fam, color=INK, box=b1)
    if i == 4:
        text(x_fam + tw(fam), y, " small", color=INK2, box=b1)
    text(x_old_r, y, old, color=INK2, ha="right", box=b1)
    text(x_arr, y + 0.007, "→", color=GREY, ha="center", box=b1)    # optical centre on the digits
    text(x_new, y, new, color=INK, bold=True, box=b1)
y_rule = yb - 3 * LH - 0.062                         # hairline between the main and the small pair
line([(x_fam, y_rule), (x_new_r, y_rule)], color=RULE, lw=0.6, check=False)
text(x_fam, yb - 5.3 * LH, "instruction-tuned · bf16", color=INK2, box=b1)

# ---- 2 optimise
b2, yb = card(X[1], y_top, CWS[1], H_TOP, 2, "Optimise", sub="Heretic 2.0")
x0, x1 = X[1] + PADX, X[1] + CWS[1] - PADX
CHIP_GAP = chip_row(x0, x1, yb, [("DiM", "dim"), ("MPOA", "mpoa"), ("ARA", "ara"), ("SOM", "som")], box=b2)
e = chip(x0, yb - LH, "J-space", MC["jspace"], box=b2)
text(e + 0.07, yb - LH, "small pair only", color=INK2, box=b2)
text(x0, yb - 2 * LH, "Optuna TPE · 100 trials · seed 1", box=b2)
text(x0, yb - 3.3 * LH, "minimise on dev prompts", color=INK2, box=b2)
text(x0, yb - 4.3 * LH, "keyword refusals · 100 harmful", box=b2)
text(x0, yb - 5.3 * LH, "first-token KL · 100 harmless", box=b2)

# ---- 3 select: pictogram of the rule (trials in the KL x refusals plane, KL budget, chosen trial)
b3, yb = card(X[2], y_top, CWS[2], H_TOP, 3, "Select")
gx0, gx1 = X[2] + PADX + 0.13, X[2] + CWS[2] - PADX - 0.17
gy0, gy1 = yb - 2 * LH - 0.01, yb + 0.075
rng = np.random.default_rng(7)
kl = np.exp(rng.uniform(np.log(0.022), np.log(1.5), 22))
ref = np.clip(0.88 - 0.40 * np.log10(kl / 0.022) + rng.normal(0, 0.10, kl.size), 0.06, 0.92)
THR = 0.1
gx = lambda k: gx0 + (np.log(k) - np.log(0.016)) / (np.log(2.0) - np.log(0.016)) * (gx1 - gx0)  # noqa: E731
gy = lambda r: gy0 + r * (gy1 - gy0)  # noqa: E731
ax.add_patch(FancyBboxPatch((gx0, gy0), gx(THR) - gx0, gy1 - gy0, boxstyle="square,pad=0", fc=WHITE,
                            ec="none", zorder=2))
ax.plot([gx0, gx1], [gy0, gy0], color=GREY, lw=0.6, zorder=3, solid_capstyle="butt")
ax.plot([gx0, gx0], [gy0, gy1], color=GREY, lw=0.6, zorder=3, solid_capstyle="butt")
ax.plot([gx(THR)] * 2, [gy0, gy1], color=INK2, lw=0.7, ls=(0, (2, 1.5)), zorder=3)
feas = kl <= THR
sel = int(np.argmin(np.where(feas, ref, 9)))
ax.scatter(gx(kl), gy(ref), s=5, c=[INK2 if f else GREY for f in feas], lw=0, zorder=4)
ax.scatter([gx(kl[sel])], [gy(ref[sel])], s=30, facecolor="none", edgecolor=INK, lw=0.9, zorder=5)
ax.scatter([gx(kl[sel])], [gy(ref[sel])], s=7, c=INK, lw=0, zorder=5)
text(gx1 + 0.035, gy0 - 0.012, "KL", color=INK2, box=b3)
text(gx0 - 0.045, (gy0 + gy1) / 2, "refusals", color=INK2, rotation=90, ha="right", va="center", box=b3)
x0 = X[2] + PADX
text(x0, yb - 3.3 * LH, "fewest refusals, KL ≤ 0.1", box=b3)
text(x0, yb - 4.3 * LH, "sensitivity 0.05 · 0.2 · 0.5", color=INK2, box=b3)
text(x0, yb - 5.3 * LH, "same rule, every study", color=INK2, box=b3)

# ---- 4 export
b4, yb = card(X[3], y_top, CWS[3], H_TOP, 4, "Export")
x0 = X[3] + PADX
text(x0, yb, "adapter merged into", box=b4)
text(x0, yb - LH, "standalone weights", box=b4)
text(x0, yb - 2 * LH, "public on Hugging Face", box=b4)
text(x0, yb - 4.3 * LH, "reproduce manifest", color=INK2, box=b4)
chip_row(x0, None, yb - 5.3 * LH, [("DiM", "dim"), ("MPOA", "mpoa"), ("ARA", "ara")], box=b4, gap=CHIP_GAP)

ym = y_top + H_TOP / 2
for i in range(3):
    arrow([(X[i] + CWS[i] + 0.035, ym), (X[i + 1] - 0.035, ym)])

# ============================================================================================ hub (stage 5)
HUB_T, HUB_S = "Held-out evaluation", "never used for search or selection"
hub_w = max(title_w(HUB_T), tw(HUB_S)) + 2 * 0.21
hub_x = W / 2 - hub_w / 2
hub_x1 = hub_x + hub_w
bh = rbox(hub_x, y_hub, hub_w, H_HUB, fill=WHITE, ec=INK, lw=0.9, r=H_HUB / 2, z=2)
hb = y_hub + H_HUB - 0.160
title_line(W / 2 - title_w(HUB_T) / 2, hb, 5, HUB_T, bh)
text(W / 2, hb - 0.142, HUB_S, color=INK2, ha="center", box=bh)

y_rail = y_hub + H_HUB / 2
arrow([(XC[0], y_top - 0.035), (XC[0], y_rail), (hub_x - 0.035, y_rail)])
arrow([(XC[3], y_top - 0.035), (XC[3], y_rail), (hub_x1 + 0.035, y_rail)])
text(XC[0] + 0.12, y_rail - 0.13, "originals", color=INK2)
text(XC[3] - 0.12, y_rail - 0.13, "selected models", color=INK2, ha="right")

# ---- additional conditions (dashed pills), on one line between the tiers
y_dc = y_top - 0.285                                 # baseline of the dashed pills
G_LAB, S_LAB, C_LAB = "γ-MPOA · Gemma, OLMo", "seed 2 · small pair", "8 community releases"
gap_dc = 0.10
x_g = XC[1] - (dashed_w(G_LAB, True) + gap_dc + dashed_w(S_LAB, False)) / 2
gam = dashed_chip(x_g, y_dc, G_LAB, MC["gamma_mpoa"])
sd2 = dashed_chip(gam.x1 + gap_dc, y_dc, S_LAB, None)
for bx in (gam, sd2):                                # both feed the optimisation (card 2)
    xa = min(max(bx.x0 + bx.width / 2, X[1] + 0.12), X[1] + CWS[1] - 0.12)
    arrow([(xa, bx.y1), (xa, y_top - 0.02)], dashed=True)
x_c = max(sd2.x1 + 0.22, hub_x1 - 0.30)              # community releases join the selected models' rail
com = dashed_chip(x_c, y_dc, C_LAB, MC["community"])
x_cj = max(com.x0 + com.width / 2, hub_x1 + 0.16)
arrow([(x_cj, com.y0), (x_cj, y_rail + 0.012)], dashed=True)

# ============================================================================================ streams (stage 5)
XA0, XA1 = X[0], X[1] + CWS[1]                       # refusal spans columns 1-2
SX = [(XA0 + XA1) / 2, XC[2], XC[3]]                 # stream centres
y_fan = y_str + H_STREAM + GAP_FAN * 0.5
line([(W / 2, y_hub - 0.03), (W / 2, y_fan)])
line([(SX[0], y_fan), (SX[2], y_fan)])
for sx in SX:
    arrow([(sx, y_fan), (sx, y_str + H_STREAM + 0.035)])

# ---- refusal: prompts -> judges -> outcome class
bA, yb = card(XA0, y_str, XA1 - XA0, H_STREAM, None, "Refusal")
COLS = [("1,322 prompts", ["972 harmful-type", "350 benign", "greedy, ≤ 512 tokens"]),
        ("judges", ["WildGuard (primary)", "HarmBench classifier", "Qwen3Guard", "2 encoder classifiers",
                    "keywords"]),
        ("outcome", ["refusal", "compliance", "degenerate", "incomplete"])]
widths = [max([tw(h, bold=True)] + [tw(s) for s in rows]) for h, rows in COLS]
x0, x1 = XA0 + PADX, XA1 - PADX
gap_c = (x1 - x0 - sum(widths)) / 2
assert gap_c > 0.30, f"refusal columns too tight ({gap_c:.3f})"
cx = x0
for j, (hd, rows) in enumerate(COLS):
    text(cx, yb, hd, bold=True, box=bA)
    for k, s in enumerate(rows):
        text(cx, yb - (k + 1) * LH, s, color=INK2, box=bA)
    if j < 2:
        arrow([(cx + widths[j] + 0.09, yb + 0.028), (cx + widths[j] + gap_c - 0.09, yb + 0.028)], lw=0.7)
    cx += widths[j] + gap_c

# ---- divergence
bB, yb = card(X[2], y_str, CWS[2], H_STREAM, None, "Divergence")
x0 = X[2] + PADX
text(x0, yb, "350 benign prompts", bold=True, box=bB)
text(x0, yb - LH, "original model's answers", color=INK2, box=bB)
arrow([(x0 + 0.04, yb - LH - 0.045), (x0 + 0.04, yb - 3 * LH + 0.105)], lw=0.7)
text(x0, yb - 3 * LH, "teacher-forced", bold=True, box=bB)
text(x0, yb - 4 * LH, "log-prob shift", color=INK2, box=bB)

# ---- capability
bC, yb = card(X[3], y_str, CWS[3], H_STREAM, None, "Capability")
x0 = X[3] + PADX
BEN = [("MMLU-Pro", "1,000"), ("GSM8K", "500"), ("IFEval", "541"), ("HumanEval", "164"), ("MBPP", "500")]
x_n = x0 + max(tw(b) for b, _ in BEN) + 0.10 + tw("1,000")
for k, (b, n) in enumerate(BEN):
    text(x0, yb - k * LH, b, box=bC)
    text(x_n, yb - k * LH, n, color=INK2, ha="right", box=bC)
text(x0, yb - 5 * LH, "lm-eval on vLLM", color=INK2, box=bC)

# ============================================================================================ statistics (6)
y_cv = y_sta + H_STATS + GAP_CONV * 0.5
line([(SX[0], y_str - 0.03), (SX[0], y_cv), (SX[2], y_cv), (SX[2], y_str - 0.03)])
line([(SX[1], y_str - 0.03), (SX[1], y_cv)])
arrow([(W / 2, y_cv), (W / 2, y_sta + H_STATS + 0.035)])

bS, yb = card(MX, y_sta, CW, H_STATS, 6, "Statistics", sub="Holm-corrected")
CELLS = [("Refusal rates", ["Wilson intervals", "McNemar vs original"]),
         ("H1 · H2", ["mixed-effects logistic", "generation × method class"]),
         ("H3", ["keyword vs judge", "agreement κ"]),
         ("Dev vs held-out", ["Spearman ρ"])]
for i, (hd, rows) in enumerate(CELLS):
    cx = X[i] + PADX
    text(cx, yb, hd, bold=True, box=bS)
    for k, r in enumerate(rows):
        text(cx, yb - (k + 1) * LH, r, color=INK2, box=bS)

# ============================================================================================ QA + save
fig.canvas.draw()
inv = ax.transData.inverted()


def bb(t):
    return Bbox(inv.transform(t.get_window_extent(renderer)))


issues = []
sizes = sorted({round(t.get_fontsize(), 2) for t, _ in TEXTS})
toks = [w for t, _ in TEXTS if not re.fullmatch(r"\d", t.get_text()) for w in t.get_text().split()
        if re.search(r"[A-Za-z0-9]", w)]
numerals = sum(1 for w in toks if not re.search(r"[A-Za-z]", w))
for t, box in TEXTS:
    b = bb(t)
    if b.x0 < 0.01 or b.x1 > W - 0.01 or b.y0 < 0.01 or b.y1 > H - 0.01:
        issues.append(f"outside figure: {t.get_text()!r}")
    if box is not None and not (b.x0 >= box.x0 + 0.03 and b.x1 <= box.x1 - 0.03 and b.y0 >= box.y0 + 0.012
                                and b.y1 <= box.y1 - 0.012):
        issues.append(f"too close to / outside its card: {t.get_text()!r}")
for i in range(len(TEXTS)):
    for j in range(i + 1, len(TEXTS)):
        a, c = bb(TEXTS[i][0]), bb(TEXTS[j][0])
        if min(a.x1, c.x1) - max(a.x0, c.x0) > 0.004 and min(a.y1, c.y1) - max(a.y0, c.y0) > 0.004:
            issues.append(f"text overlap: {TEXTS[i][0].get_text()!r} / {TEXTS[j][0].get_text()!r}")
for a in ARROWS:
    pts = inv.transform(a.get_path().transformed(a.get_transform()).interpolated(60).vertices)
    for t, _ in TEXTS:
        b = bb(t)
        b = Bbox.from_extents(b.x0 - 0.02, b.y0 - 0.02, b.x1 + 0.02, b.y1 + 0.02)
        if any(b.contains(x, y) for x, y in pts):
            issues.append(f"arrow within 0.02 in of text {t.get_text()!r}")
print(f"fig1_B: {W:.2f} x {H:.2f} in; font sizes {sizes} pt at final size; {len(toks)} words "
      f"({numerals} of them pure numbers; stage badges not counted)")
print("QA:", "clean" if not issues else "\n  " + "\n  ".join(dict.fromkeys(issues)))

PNG_OUT.parent.mkdir(parents=True, exist_ok=True)
meta = {"CreationDate": None, "Creator": "paper33 scripts/figures/fig1_candidates/fig1_B.py"}
fig.savefig(PDF_OUT, metadata=meta)
fig.savefig(PNG_OUT, dpi=200)
print(f"wrote {PDF_OUT}\nwrote {PNG_OUT}")
