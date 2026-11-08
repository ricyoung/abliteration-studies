#!/usr/bin/env python3
"""fig1_final.py -- Figure 1 (final): the study workflow of paper 33.

Built on candidate B (two tiers around a focal "held-out evaluation" hub), with the judges' fixes and grafts:

  top tier     1 model pairs -> 2 optimise -> 3 select -> 4 export, four equal-height cards whose last rows
               (footers) share one baseline
  funnel       the originals (from 1) and the selected models (from 4) drop into the TOP of the stage-5 hub;
               the two optimisation-side additional conditions hang under card 2 as dashed boxes (dashed
               tethers without heads = attached to that stage, so nothing points against the reading direction)
  hub row      dashed community box -> left end of the hub (ends on the hub border, not on a connector);
               evaluation hardware as a grey margin note right of the hub
  bottom tier  the hub fans out into refusal (prompts -> judges -> outcome chips), divergence and capability,
               which converge into 6 statistics (title column + H1/H2, H3, dev-vs-held-out columns)

Width: S.TEXT_W = 6.5 in, the PRIMEarxiv text width (PRIMEarxiv.sty: textwidth=6.5in), so
\\includegraphics[width=\\textwidth] prints it at scale 1.0 and 7 pt stays 7 pt.

Encoding
  light-grey card, black disc     core stage, numbered        white dashed box      additional condition
  outlined H1/H2/H3 tag           hypothesis                  white chip            outcome class
  coloured square                 method identity (house colours, nothing else is coloured)
  ink / grey / bold               key fact / secondary fact / later generation (column header says so)
  grey arrows                     flow of models (QA: no arrow within 0.02 in of text, none through a box)

Run:
    scripts/figures/.venv/bin/python scripts/figures/fig1_candidates/fig1_final.py
Writes
    scripts/figures/fig1_candidates/fig1_final.pdf                 vector, TrueType (Type 42) fonts
    paper/figures/preview/fig1_candidates/fig1_final.png           200 dpi preview
and prints a QA report (font sizes, word count, text/text, text/arrow, arrow/box and text/card checks);
exits 1 if any check fails.
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
PDF_OUT = HERE / "fig1_final.pdf"
PNG_OUT = ROOT / "paper" / "figures" / "preview" / "fig1_candidates" / "fig1_final.png"

# ============================================================================================ design tokens
FONT = "Liberation Sans"
INK, INK2 = S.TEXT, S.TEXT_SECONDARY            # #0b0b0b, #52514e
GREY, RULE = "#8a8984", "#e6e5e1"               # neutral greys
CARD = "#f2f1ee"                                # card fill: #e6e5e1 at ~half strength
WHITE = "#ffffff"
MC = S.METHOD_COLORS

FS_TITLE = 8.5      # pt, stage titles
FS_BODY = 7.0       # pt, everything else (house minimum)

W = S.TEXT_W        # 6.5 in (template text width)
MX = 0.05           # outer margin
R_CARD = 0.055      # corner radius of every card and dashed box (in)
PADX = 0.09         # inner left/right padding of cards
LH = 0.125          # line pitch of 7 pt body text (in)
T_BASE = 0.185      # card top -> title baseline
B1 = 0.205          # title baseline -> first body baseline
BOT = 0.092         # last body baseline -> card bottom
LW_ARROW = 0.9      # pt
LW_DASH = 0.75      # pt
DASH = (0, (2.6, 1.7))
R_BADGE = 0.071
GAP_IN = 0.035      # arrow tip / tail -> box edge


def card_h(n_rows: float) -> float:
    return T_BASE + B1 + (n_rows - 1) * LH + BOT


# ============================================================================================ vertical rhythm
N_TOP = 7.3                 # card 1: column header, 4 pairs, 0.3-line gap, small pair, footer
H_TOP = card_h(N_TOP)
H_STREAM = card_h(6)
H_STATS = card_h(3)
H_HUB = 0.38
DB_H = 0.30                 # two-line dashed box
TETHER = 0.11               # card 2 bottom -> dashed boxes
DB_RAIL = 0.13              # dashed boxes -> rails
RAIL_HUB = 0.13             # rails -> hub top
GAP_TOP_HUB = TETHER + DB_H + DB_RAIL + RAIL_HUB
GAP_FAN = 0.27              # hub bottom -> stream tops
GAP_CONV = 0.25             # stream bottoms -> statistics top
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
ARROWS: list[tuple[FancyArrowPatch, str]] = []
BOXES: list[tuple[Bbox, str]] = []                       # every card / box / chip, for the arrow-through-box check


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


def runs(x, y, parts, box=None):
    """Consecutive text runs on one baseline: parts = [(string, colour, bold), ...]. Returns the end x."""
    for s, c, b in parts:
        text(x, y, s, color=c, bold=b, box=box)
        x += tw(s, bold=b)
    return x


def rbox(x, y, w, h, fill=CARD, ec="none", lw=0.0, ls="-", r=R_CARD, z=1, name="box"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fill, ec=ec,
                                lw=lw, ls=ls, zorder=z, joinstyle="round"))
    b = Bbox.from_bounds(x, y, w, h)
    BOXES.append((b, name))
    return b


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


def card(x, y, w, h, n=None, title=None, sub=None, name="card"):
    """Filled card with optional stage badge + title (+ grey note right-aligned on the title line).
    Returns (bbox, baseline of the first body row)."""
    box = rbox(x, y, w, h, name=name)
    tb = y + h - T_BASE
    if n is not None:
        title_line(x + PADX, tb, n, title, box)
    else:
        text(x + PADX, tb, title, FS_TITLE, INK, bold=True, box=box)
    if sub:
        text(x + w - PADX, tb, sub, FS_BODY, INK2, ha="right", box=box)
    return box, tb - B1


SQ = 0.064           # method square
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
    """Method chips spread evenly so that the row spans exactly x0..x1 (or with a fixed gap). Returns the gap."""
    if gap is None:
        gap = (x1 - x0 - sum(chip_w(lab) for lab, _ in items)) / (len(items) - 1)
    assert gap > 0.045, f"chip row too tight ({gap:.3f} in)"
    x = x0
    for lab, key in items:
        x = chip(x, yb, lab, MC[key], box=box) + gap
    return gap


DB_PAD = 0.075       # dashed box inner padding (x)


def dashed_w(lines, color):
    return max(tw(s) for s in lines) + (SQ + SQ_GAP if color else 0) + 2 * DB_PAD


def dashed_box(x, y_top, lines, color, name):
    """Additional condition: white dashed box, two lines (name in ink, scope in grey); method square if any."""
    w = dashed_w(lines, color)
    box = rbox(x, y_top - DB_H, w, DB_H, fill=WHITE, ec=GREY, lw=LW_DASH, ls=DASH, z=4, name=name)
    xt = x + DB_PAD
    b1 = y_top - 0.122
    if color:
        ax.add_patch(FancyBboxPatch((xt, b1 - 0.002), SQ, SQ, boxstyle="round,pad=0,rounding_size=0.013",
                                    fc=color, ec="none", zorder=6))
        xt += SQ + SQ_GAP
    text(xt, b1, lines[0], color=INK, box=box)
    text(xt, b1 - LH, lines[1], color=INK2, box=box)
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


def arrow(pts, dashed=False, head=True, color=GREY, r=0.06, lw=LW_ARROW, check=True, name="arrow"):
    a = FancyArrowPatch(path=rounded_path(pts, r), arrowstyle="-|>" if head else "-", mutation_scale=6.5,
                        color=color, lw=lw, ls=DASH if dashed else "-", zorder=3, shrinkA=0, shrinkB=0,
                        capstyle="butt", joinstyle="round")
    ax.add_patch(a)
    if check:
        ARROWS.append((a, name))
    return a


def line(pts, **kw):
    return arrow(pts, head=False, **kw)


# ============================================================================================ grid
CW = W - 2 * MX                                     # 6.4 in
G = 0.18                                            # column gap
CWS = [1.58, 1.62, 1.33, 1.33]                      # widths follow content: 2 carries the objectives
assert abs(sum(CWS) + 3 * G - CW) < 1e-9
X = [MX + sum(CWS[:i]) + i * G for i in range(4)]
XR = [x + w for x, w in zip(X, CWS)]
XC = [x + w / 2 for x, w in zip(X, CWS)]            # column centres

y_top = H - MX - H_TOP                              # bottom edge of the top cards
y_hub = y_top - GAP_TOP_HUB - H_HUB
y_str = y_hub - GAP_FAN - H_STREAM
y_sta = MX


def row(yb, r):
    """Baseline of body row r (0 = first body row) below first-body baseline yb."""
    return yb - r * LH


R_FOOT = N_TOP - 1                                  # footer row of the top cards (6.3)

# ============================================================================================ top tier
# ---- 1 model pairs: family | earlier -> later (column header; later generation bold)
b1, yb = card(X[0], y_top, CWS[0], H_TOP, 1, "Model pairs", name="card 1")
PAIRS = [("Qwen", "2.5-32B", "3.8-27B"), ("Gemma", "2-27B", "4-31B"), ("OLMo", "2-32B", "3.1-32B"),
         ("Mistral-Small", "2409", "3.2-24B"), ("Qwen", "2.5-7B", "3.5-4B")]
x_fam = X[0] + PADX
x_new_r = XR[0] - PADX
w_new = max(tw(p[2], bold=True) for p in PAIRS)
x_new = x_new_r - w_new                             # later column, flush with the card's right padding
x_arr = x_new - 0.064
x_old_r = x_arr - 0.064
fam_w = [tw(p[0]) + (tw(" small") if i == 4 else 0) for i, p in enumerate(PAIRS)]
assert all(x_fam + fw + 0.04 < x_old_r - tw(p[1]) for fw, p in zip(fam_w, PAIRS)), "pairs table too tight"
text(x_old_r, row(yb, 0), "earlier", color=INK2, ha="right", box=b1)
text(x_arr, row(yb, 0) + 0.007, "→", color=GREY, ha="center", box=b1)
text(x_new, row(yb, 0), "later", color=INK2, box=b1)
for i, (fam, old, new) in enumerate(PAIRS):
    y = row(yb, 1 + i if i < 4 else 5.3)            # the small pair sits after a 0.3-line gap
    text(x_fam, y, fam, color=INK, box=b1)
    if i == 4:
        text(x_fam + tw(fam), y, " small", color=INK2, box=b1)
    text(x_old_r, y, old, color=INK2, ha="right", box=b1)
    text(x_arr, y + 0.007, "→", color=GREY, ha="center", box=b1)    # optical centre on the digits
    text(x_new, y, new, color=INK, bold=True, box=b1)
y_rule = row(yb, 4.65) + 0.030                       # hairline between the main and the small pair
line([(x_fam, y_rule), (x_new_r, y_rule)], color=RULE, lw=0.6, check=False)
text(x_fam, row(yb, R_FOOT), "instruction-tuned · bf16", color=INK2, box=b1)

# ---- 2 optimise
b2, yb = card(X[1], y_top, CWS[1], H_TOP, 2, "Optimise", sub="Heretic 2.0", name="card 2")
x0, x1 = X[1] + PADX, XR[1] - PADX
CHIP_GAP = chip_row(x0, x1, row(yb, 0), [("DiM", "dim"), ("MPOA", "mpoa"), ("ARA", "ara"), ("SOM", "som")],
                    box=b2)
e = chip(x0, row(yb, 1), "J-space", MC["jspace"], box=b2)
text(e + 0.07, row(yb, 1), "small pair only", color=INK2, box=b2)
text(x0, row(yb, 2), "Optuna TPE · 100 trials · seed 1", box=b2)
runs(x0, row(yb, 3.3), [("min ", INK2, False), ("keyword refusals", INK, False), (" · 100 dev", INK2, False)],
     box=b2)
runs(x0, row(yb, 4.3), [("min ", INK2, False), ("first-token KL", INK, False), (" · 100 harmless", INK2, False)],
     box=b2)
text(x0, row(yb, R_FOOT), "H200 (small pair: RTX 4090)", color=INK2, box=b2)

# ---- 3 select: pictogram of the rule (trials in the KL x dev-refusals plane, KL budget, chosen trial)
b3, yb = card(X[2], y_top, CWS[2], H_TOP, 3, "Select", name="card 3")
x0 = X[2] + PADX
text(x0, row(yb, 0), "dev refusals", color=INK2, box=b3)
gx0, gx1 = x0 + 0.012, XR[2] - PADX - tw("KL") - 0.05
gy1, gy0 = row(yb, 0) - 0.055, row(yb, 3.25)
# schematic trials (KL, dev refusals as a fraction of the axis); the ringed one is the rule's pick
TRIALS = [(0.013, 0.92), (0.016, 0.83), (0.020, 0.88), (0.025, 0.74), (0.031, 0.79), (0.038, 0.64),
          (0.046, 0.69), (0.055, 0.41), (0.066, 0.56), (0.13, 0.36), (0.16, 0.44), (0.20, 0.29),
          (0.25, 0.33), (0.32, 0.21), (0.41, 0.25), (0.53, 0.14), (0.70, 0.17), (0.92, 0.08), (1.25, 0.10)]
kl, ref = np.array(TRIALS).T
THR = 0.1
gx = lambda k: gx0 + 0.03 + (np.log(k) - np.log(0.011)) / (np.log(1.5) - np.log(0.011)) * (gx1 - gx0 - 0.04)  # noqa: E731
gy = lambda r: gy0 + 0.02 + r * (gy1 - gy0 - 0.02)  # noqa: E731
ax.add_patch(FancyBboxPatch((gx0, gy0), gx(THR) - gx0, gy1 - gy0, boxstyle="square,pad=0", fc=WHITE,
                            ec="none", zorder=2))
ax.plot([gx0, gx1], [gy0, gy0], color=GREY, lw=0.6, zorder=3, solid_capstyle="butt")
ax.plot([gx0, gx0], [gy0, gy1], color=GREY, lw=0.6, zorder=3, solid_capstyle="butt")
ax.plot([gx(THR)] * 2, [gy0, gy1], color=INK2, lw=0.7, ls=(0, (2, 1.5)), zorder=3)
feas = kl <= THR
sel = int(np.argmin(np.where(feas, ref, 9)))
ax.scatter(gx(kl), gy(ref), s=5.5, c=[INK2 if f else GREY for f in feas], lw=0, zorder=4)
ax.scatter([gx(kl[sel])], [gy(ref[sel])], s=34, facecolor="none", edgecolor=INK, lw=0.9, zorder=5)
ax.scatter([gx(kl[sel])], [gy(ref[sel])], s=8, c=INK, lw=0, zorder=5)
text(gx1 + 0.035, gy0 - 0.012, "KL", color=INK2, box=b3)
text(x0, row(yb, 4.3), "fewest refusals at KL ≤ 0.1", box=b3)
text(x0, row(yb, 5.3), "sensitivity 0.05 · 0.2 · 0.5", color=INK2, box=b3)
text(x0, row(yb, R_FOOT), "same rule, every study", color=INK2, box=b3)

# ---- 4 export
b4, yb = card(X[3], y_top, CWS[3], H_TOP, 4, "Export", name="card 4")
x0 = X[3] + PADX
text(x0, row(yb, 0), "adapter merged into", box=b4)
text(x0, row(yb, 1), "standalone weights", box=b4)
text(x0, row(yb, 2), "public on Hugging Face", box=b4)
text(x0, row(yb, R_FOOT - 1), "reproduce manifest", color=INK2, box=b4)
chip_row(x0, None, row(yb, R_FOOT), [("DiM", "dim"), ("MPOA", "mpoa"), ("ARA", "ara")], box=b4,
         gap=min(CHIP_GAP, 0.11))

ym = y_top + H_TOP / 2
for i in range(3):
    arrow([(XR[i] + GAP_IN, ym), (X[i + 1] - GAP_IN, ym)], name=f"{i + 1}->{i + 2}")

# ============================================================================================ hub (stage 5)
HUB_T, HUB_S = "Held-out evaluation", "separate from search prompts · 44 overlaps flagged"
hub_w = max(title_w(HUB_T), tw(HUB_S)) + 2 * 0.21
hub_x = W / 2 - hub_w / 2
hub_x1 = hub_x + hub_w
y_hub_top = y_hub + H_HUB
y_mid = y_hub + H_HUB / 2
bh = rbox(hub_x, y_hub, hub_w, H_HUB, fill=WHITE, ec=INK, lw=0.9, r=H_HUB / 2, z=2, name="hub")
hb = y_hub + H_HUB - 0.160
title_line(W / 2 - title_w(HUB_T) / 2, hb, 5, HUB_T, bh)
text(W / 2, hb - 0.142, HUB_S, color=INK2, ha="center", box=bh)

# ---- the two model streams drop into the hub's top: originals (from 1), selected models (from 4)
y_rail = y_hub_top + RAIL_HUB
IN_DX = 0.30                                         # entry points, symmetric about the hub centre
arrow([(XC[0], y_top - GAP_IN), (XC[0], y_rail), (hub_x + IN_DX, y_rail), (hub_x + IN_DX, y_hub_top + GAP_IN)],
      name="originals")
arrow([(XC[3], y_top - GAP_IN), (XC[3], y_rail), (hub_x1 - IN_DX, y_rail), (hub_x1 - IN_DX, y_hub_top + GAP_IN)],
      name="selected")
text(XC[0] + 0.07, y_rail + 0.045, "originals", color=INK2)
text(XC[3] - 0.07, y_rail + 0.045, "selected models", color=INK2, ha="right")

# ---- additional conditions on the optimisation side: dashed boxes hanging under card 2
G_L = ("γ-corrected MPOA control", "post-norm Gemma, OLMo")
S_L = ("seed-2 replicate", "small pair, 5 methods")
db_gap = 0.10
db_top = y_top - TETHER
w_g, w_s = dashed_w(G_L, True), dashed_w(S_L, False)
x_g = min(XC[1] - (w_g + db_gap + w_s) / 2, XR[1] - R_CARD - 0.06 - db_gap - w_s / 2 - w_g)
x_g = max(x_g, XC[0] + 0.07 + tw("originals") + 0.12)
gam = dashed_box(x_g, db_top, G_L, MC["gamma_mpoa"], "gamma box")
sd2 = dashed_box(gam.x1 + db_gap, db_top, S_L, None, "seed-2 box")
for bx in (gam, sd2):                                # both are extra optimisation studies (card 2)
    xa = min(max(bx.x0 + bx.width / 2, X[1] + R_CARD + 0.06), XR[1] - R_CARD - 0.06)
    assert bx.x0 + 0.08 < xa < bx.x1 - 0.08
    line([(xa, bx.y1 + 0.004), (xa, y_top - 0.012)], dashed=True, name="tether")   # attachment, no head

# ---- community releases join the hub from the left; evaluation hardware as a margin note on the right
C_L = ("8 community models", "public abliterated releases")
w_c = dashed_w(C_L, True)
com = dashed_box(hub_x - 0.24 - w_c, y_mid + DB_H / 2, C_L, MC["community"], "community box")
arrow([(com.x1 + 0.03, y_mid), (hub_x - 0.03, y_mid)], dashed=True, name="community")

HW = ["Qwen, Gemma: RTX PRO 6000", "OLMo, Mistral: H200", "small pair: RTX 4090"]
x_hw = hub_x1 + 0.15
for k, s in enumerate(HW):
    text(x_hw, y_mid - 0.035 + (1 - k) * LH, s, color=INK2)

# ============================================================================================ streams (stage 5)
XA0, XA1 = X[0], XR[1]                               # refusal spans columns 1-2
SX = [(XA0 + XA1) / 2, XC[2], XC[3]]                 # stream centres
y_fan = y_str + H_STREAM + GAP_FAN * 0.5
line([(W / 2, y_hub - 0.03), (W / 2, y_fan)], name="fan stem")
line([(SX[0], y_fan), (SX[2], y_fan)], name="fan bus")
for sx in SX:
    arrow([(sx, y_fan), (sx, y_str + H_STREAM + GAP_IN)], name="fan")

# ---- refusal: prompts -> judges -> outcome classes (chips)
bA, yb = card(XA0, y_str, XA1 - XA0, H_STREAM, None, "Refusal", name="refusal")
OUTC = ["refusal", "compliance", "degenerate", "incomplete"]
CH_PADX, CH_H = 0.065, 0.118
w_chip = max(tw(s) for s in OUTC) + 2 * CH_PADX
COLS = [("1,322 prompts", [("972 harmful", INK2, 1), ("350 benign", INK2, 2),
                           ("greedy, ≤ 512 tokens", INK2, 5)]),
        ("judges", [("WildGuard (primary)", INK, 1), ("HarmBench classifier", INK2, 2), ("Qwen3Guard", INK2, 3),
                    ("2 encoder classifiers", INK2, 4), ("keywords", INK2, 5)]),
        ("outcome", [])]
widths = [max([tw(h, bold=True)] + [tw(s) for s, _, _ in rows]) for h, rows in COLS]
widths[2] = max(widths[2], w_chip)
x0, x1 = XA0 + PADX, XA1 - PADX
gap_c = (x1 - x0 - sum(widths)) / 2
assert gap_c > 0.30, f"refusal columns too tight ({gap_c:.3f})"
cx = x0
for j, (hd, rows) in enumerate(COLS):
    text(cx, yb, hd, bold=True, box=bA)
    for s, c, r in rows:
        text(cx, row(yb, r), s, color=c, box=bA)
    if j < 2:
        arrow([(cx + widths[j] + 0.09, yb + 0.028), (cx + widths[j] + gap_c - 0.09, yb + 0.028)], lw=0.7,
              name="refusal step")
    cx += widths[j] + gap_c
x_chip = cx - gap_c - widths[2]
pitch = (row(yb, 1.15) - row(yb, 5)) / (len(OUTC) - 1)  # chips span the judges' rows 1..5
for k, s in enumerate(OUTC):
    ycb = row(yb, 1.15) - k * pitch                  # chip text baseline
    cbox = rbox(x_chip, ycb - 0.033, w_chip, CH_H, fill=WHITE, r=CH_H / 2, z=4, name=f"chip {s}")
    text(x_chip + w_chip / 2, ycb, s, color=INK2, ha="center", box=cbox)

# ---- divergence
bB, yb = card(X[2], y_str, CWS[2], H_STREAM, None, "Divergence", name="divergence")
x0 = X[2] + PADX
text(x0, row(yb, 0), "350 benign prompts", bold=True, box=bB)
text(x0, row(yb, 1), "original model's answers", color=INK2, box=bB)
arrow([(x0 + 0.04, row(yb, 1) - 0.05), (x0 + 0.04, row(yb, 4) + 0.105)], lw=0.7, name="divergence step")
text(x0, row(yb, 4), "teacher-forced", bold=True, box=bB)
text(x0, row(yb, 5), "log-prob shift", color=INK2, box=bB)

# ---- capability
bC, yb = card(X[3], y_str, CWS[3], H_STREAM, None, "Capability", name="capability")
x0 = X[3] + PADX
BEN = [("MMLU-Pro", "1,000"), ("GSM8K", "500"), ("IFEval", "541"), ("HumanEval", "164"), ("MBPP", "500")]
x_n = x0 + max(tw(b) for b, _ in BEN) + 0.10 + tw("1,000")
for k, (b, n) in enumerate(BEN):
    text(x0, row(yb, k), b, box=bC)
    text(x_n, row(yb, k), n, color=INK2, ha="right", box=bC)
text(x0, row(yb, 5), "lm-eval on vLLM", color=INK2, box=bC)

# ============================================================================================ statistics (6)
y_cv = y_sta + H_STATS + GAP_CONV * 0.5
line([(SX[0], y_str - 0.03), (SX[0], y_cv), (SX[2], y_cv), (SX[2], y_str - 0.03)], name="converge")
line([(SX[1], y_str - 0.03), (SX[1], y_cv)], name="converge")
arrow([(W / 2, y_cv), (W / 2, y_sta + H_STATS + GAP_IN)], name="to statistics")

bS = rbox(MX, y_sta, CW, H_STATS, name="statistics")
tb = y_sta + H_STATS - T_BASE                        # title / column-head baseline
title_line(X[0] + PADX, tb, 6, "Statistics", bS)


def htag(x, yb, label):
    """Outlined hypothesis tag on baseline yb. Returns the x after the tag."""
    w = tw(label, bold=True) + 2 * 0.032
    ax.add_patch(FancyBboxPatch((x, yb - 0.030), w, 0.122, boxstyle="round,pad=0,rounding_size=0.026",
                                fc=WHITE, ec=INK, lw=0.7, zorder=5))
    text(x + w / 2, yb, label, color=INK, bold=True, ha="center", box=bS)
    return x + w


CELLS = [(None, [("Wilson intervals", INK), ("McNemar vs original", INK), ("Holm-corrected", INK2)]),
         (["H1", "H2"], [("McNemar per pair · GEE pooled", INK), ("generation × method class", INK2),
                         ("mixed-effects logistic (VB, no p)", INK2)]),
         (["H3"], [("keyword vs judge", INK), ("agreement κ", INK2)]),
         ("Dev vs held-out", [("Spearman ρ", INK)])]
for i, (hd, rows) in enumerate(CELLS):
    cx = X[i] + PADX
    if isinstance(hd, list):
        x = cx
        for lab in hd:
            x = htag(x, tb, lab) + 0.045
    elif hd:
        text(cx, tb, hd, bold=True, box=bS)
    for k, (s, c) in enumerate(rows):
        text(cx, tb - B1 - k * LH, s, color=c, box=bS)

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
    if b.x0 < MX - 0.005 or b.x1 > W - MX + 0.005 or b.y0 < MX - 0.005 or b.y1 > H - MX + 0.005:
        issues.append(f"outside the margins: {t.get_text()!r}")
    mx, my = (0.03, 0.012) if box is None or box.height > 0.2 else (0.02, 0.0)   # chips: tighter
    if box is not None and not (b.x0 >= box.x0 + mx and b.x1 <= box.x1 - mx and b.y0 >= box.y0 + my
                                and b.y1 <= box.y1 - my):
        issues.append(f"too close to / outside its box: {t.get_text()!r}")
for i in range(len(TEXTS)):
    for j in range(i + 1, len(TEXTS)):
        a, c = bb(TEXTS[i][0]), bb(TEXTS[j][0])
        if min(a.x1, c.x1) - max(a.x0, c.x0) > 0.004 and min(a.y1, c.y1) - max(a.y0, c.y0) > 0.004:
            issues.append(f"text overlap: {TEXTS[i][0].get_text()!r} / {TEXTS[j][0].get_text()!r}")
for a, name in ARROWS:
    pts = inv.transform(a.get_path().transformed(a.get_transform()).interpolated(80).vertices)
    for t, _ in TEXTS:
        b = bb(t)
        b = Bbox.from_extents(b.x0 - 0.02, b.y0 - 0.02, b.x1 + 0.02, b.y1 + 0.02)
        if any(b.contains(x, y) for x, y in pts):
            issues.append(f"{name}: within 0.02 in of text {t.get_text()!r}")
    for bx, bname in BOXES:
        inner = Bbox.from_extents(bx.x0 + 0.01, bx.y0 + 0.01, bx.x1 - 0.01, bx.y1 - 0.01)
        if bname not in ("refusal", "divergence") and any(inner.contains(x, y) for x, y in pts):
            issues.append(f"{name}: runs through {bname}")
for i in range(len(BOXES)):                          # boxes must not touch each other (chips sit inside cards)
    for j in range(i + 1, len(BOXES)):
        (a, na), (c, nc) = BOXES[i], BOXES[j]
        if na.startswith("chip") or nc.startswith("chip"):
            continue
        if min(a.x1, c.x1) - max(a.x0, c.x0) > -0.05 and min(a.y1, c.y1) - max(a.y0, c.y0) > -0.05:
            issues.append(f"boxes closer than 0.05 in: {na} / {nc}")
print(f"fig1_final: {W:.2f} x {H:.2f} in; font sizes {sizes} pt at final size; {len(toks)} words "
      f"({numerals} of them pure numbers, {len(toks) - numerals} others; stage badges not counted)")
print("QA:", "clean" if not issues else "\n  " + "\n  ".join(dict.fromkeys(issues)))

PNG_OUT.parent.mkdir(parents=True, exist_ok=True)
meta = {"CreationDate": None, "Creator": "paper33 scripts/figures/fig1_candidates/fig1_final.py"}
fig.savefig(PDF_OUT, metadata=meta)
fig.savefig(PNG_OUT, dpi=200)
print(f"wrote {PDF_OUT}\nwrote {PNG_OUT}")
sys.exit(1 if issues else 0)
