#!/usr/bin/env python3
"""Paper 30 v3 figures (reads only analysis/*.csv written by analysis/assemble_p30.py).

Run:
    /data/heretic_2/33_2611_does_refusal_removal_get_harder/scripts/figures/.venv/bin/python \
        /data/heretic_2/paper30/v3_2611/scripts/make_figures.py

Uses paper 33's style module (fonts, sizes, ink, save) but writes into v3_2611/figures/ (PDF for the paper,
preview/*.png for visual review).

Revision 2026-11-07 (publication pass; previous version: make_figures.py.bak-20261107-fig):
  * one visual system for all figures: one colour FAMILY per tool (Original grey; Heretic 1.x teal; Heretic 2.0 and
    the fork plug-ins blue; community releases crimson; OBLITERATUS orange; jwest33 near-black); lightness and
    marker shape separate the members of a family; light members get a darker edge so they print on white.
    Palette checked with CIEDE2000 under normal vision and Machado (2009) protan/deutan/tritan simulation; within a
    family the marker shape is the redundant channel. Half-filled marker = a variant run of the same tool (1.x
    general re-exported at dev KL <= 0.1; jwest33 with the hybrid fix).
  * legends grouped by tool family (bold family headers);
  * fig1: value column right of panel (a) (no label/line overlaps), alternating model bands, key for (b);
  * fig2: capability panels show only the horizontal (capability) intervals (the vertical ones repeat panel a/c);
    lighter interval ink; re-placed model labels;
  * fig3: 1.x code in the 1.x family (light teal), model separators, shorter legend;
  * fig4: distinct dash pattern per 2.0 method plus direct end labels; cap labels moved to the top;
  * fig5 (new): headline result on Qwen2.5-32B, 1.x general at the two selection caps versus the 2.0 methods,
    rates (cells.csv) and paired differences (tests_refusal.csv families kl_cap_1x, h1_1x_vs_2.0, h1_kl_matched).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, "/data/heretic_2/33_2611_does_refusal_removal_get_harder/scripts/figures")
import style as S  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import to_rgb  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

V3 = Path("/data/heretic_2/paper30/v3_2611")
A = V3 / "analysis"
FIG = V3 / "figures"

S.apply()

# ------------------------------------------------------------------------------------------------ encodings
FAMILY_COLORS = {   # one hue family per tool
    "original": ["#8f8e88"],
    "heretic1x": ["#00695f", "#52b3a0"],
    "heretic2": ["#163d8c", "#2b78d4", "#7ab8ee", "#5b5fa8", "#a9acd9"],
    "community": ["#a3123a", "#e27a9c"],
    "obliteratus": ["#8f3a00", "#e0701a", "#f2b05f"],
    "jwest33": ["#262626"],
}
F = FAMILY_COLORS
COLOR = {"original": F["original"][0],
         "1x_general": F["heretic1x"][0], "1x_general_kl01": F["heretic1x"][0], "1x_code": F["heretic1x"][1],
         "dim": F["heretic2"][0], "mpoa": F["heretic2"][1], "ara": F["heretic2"][2],
         "som": F["heretic2"][3], "jspace": F["heretic2"][4],
         "huihui": F["community"][0], "zetasepic": F["community"][1],
         "obliteratus_spectral": F["obliteratus"][0], "obliteratus_advanced": F["obliteratus"][1],
         "obliteratus_basic": F["obliteratus"][2],
         "jwest33": F["jwest33"][0], "jwest33_hybridfix": F["jwest33"][0]}
LIGHT = {"1x_code", "ara", "jspace", "zetasepic", "obliteratus_basic", "obliteratus_advanced"}  # get a dark edge
MARKER = {"original": "*", "1x_general": "h", "1x_code": "p", "1x_general_kl01": "h", "dim": "o", "mpoa": "s",
          "som": "D", "jspace": "v", "ara": "^", "huihui": "P", "zetasepic": "X",
          "obliteratus_basic": "<", "obliteratus_advanced": ">", "obliteratus_spectral": "d",
          "jwest33": "8", "jwest33_hybridfix": "8"}
HALF = {"1x_general_kl01", "jwest33_hybridfix"}
MSCALE = {"*": 1.4, "D": 0.85, "X": 1.05, "P": 1.05, "h": 1.08, "p": 1.08, "<": 1.0, ">": 1.0, "d": 1.15,
          "8": 1.0, "v": 1.0, "^": 1.0}
DASH = {"dim": (0, (5, 1.6)), "mpoa": (0, (2.2, 1.4)), "ara": (0, (6, 1.5, 1.2, 1.5)), "som": (0, (1, 1.2)),
        "jspace": (0, (3.5, 1.2, 1, 1.2, 1, 1.2))}
# SOM and J-space are plug-ins of the project fork (paper33-plugins, commit 8fd4d76), not part of Heretic 2.0
# releases; labelled "(fork)" in every figure.
LABEL = {"original": "Original", "1x_general": "1.x general", "1x_code": "1.x code",
         "1x_general_kl01": "1.x general, KL ≤ 0.1", "dim": "2.0 DiM",
         "mpoa": "2.0 MPOA", "som": "SOM (fork)", "jspace": "J-space (fork)", "ara": "2.0 ARA",
         "huihui": "huihui-ai", "zetasepic": "zetasepic v2",
         "obliteratus_basic": "OBLITERATUS basic", "obliteratus_advanced": "OBLITERATUS advanced",
         "obliteratus_spectral": "OBLITERATUS spectral cascade", "jwest33": "jwest33 recommended",
         "jwest33_hybridfix": "jwest33 + hybrid fix"}
# Cells first reported in the companion study (paper 33); marked with a dagger in Figure 1.
COMPANION = {"qwen2.5-7b__base", "qwen2.5-7b__dim", "qwen2.5-7b__mpoa", "qwen2.5-7b__som", "qwen2.5-7b__jspace",
             "qwen2.5-7b__ara", "qwen2.5-32b__base", "qwen2.5-32b__dim", "qwen2.5-32b__mpoa", "qwen2.5-32b__som",
             "qwen2.5-32b__ara", "qwen2.5-32b__community-huihui", "qwen2.5-32b__community-zetasepic-v2"}
METHOD_ROWS = ["original", "1x_general", "1x_general_kl01", "1x_code", "dim", "mpoa", "som", "jspace", "ara",
               "huihui", "zetasepic", "obliteratus_basic", "obliteratus_advanced", "obliteratus_spectral", "jwest33",
               "jwest33_hybridfix"]
EXTERNAL = {"obliteratus_basic", "obliteratus_advanced", "obliteratus_spectral", "jwest33", "jwest33_hybridfix"}
MODEL_ORDER = ["qwen2.5-7b", "qwen2.5-coder-7b", "qwen2.5-14b", "qwen2.5-coder-14b", "qwen2.5-32b",
               "qwen2.5-coder-32b"]
MODEL_LABEL = {"qwen2.5-7b": "Qwen2.5-7B", "qwen2.5-coder-7b": "Qwen2.5-Coder-7B", "qwen2.5-14b": "Qwen2.5-14B",
               "qwen2.5-coder-14b": "Qwen2.5-Coder-14B", "qwen2.5-32b": "Qwen2.5-32B",
               "qwen2.5-coder-32b": "Qwen2.5-Coder-32B", "qwen3.5-4b": "Qwen3.5-4B"}
MODEL_SHORT = {"qwen2.5-7b": "7B", "qwen2.5-coder-7b": "C-7B", "qwen2.5-14b": "14B", "qwen2.5-coder-14b": "C-14B",
               "qwen2.5-32b": "32B", "qwen2.5-coder-32b": "C-32B", "qwen3.5-4b": "4B"}
ADDED_SIGN = " ‡"  # 2026-11-08 (review S3): was the section sign
BAND = "#f4f3ef"          # alternating model band
CI_ALPHA = 0.85


def is_added(c) -> bool:
    return c["tool"] == "external" or c["method"] == "1x_general_kl01" or c["model"] == "qwen3.5-4b"


def darker(hex_, f=0.62):
    r, g, b = to_rgb(hex_)
    return (r * f, g * f, b * f)


def edge(method):
    return darker(COLOR[method], 0.6) if method in LIGHT else COLOR[method]


def mkw(method, size=S.MS, filled=True, **extra):
    m = MARKER[method]
    c = COLOR[method]
    kw = dict(marker=m, markersize=size * MSCALE.get(m, 1.0), markeredgecolor=edge(method),
              markerfacecolor=c if filled else S.SURFACE, markeredgewidth=0.7 if filled else 1.1,
              linestyle="none", color=c)
    if filled and method in HALF:
        kw.update(fillstyle="left", markerfacecoloralt=S.SURFACE, markeredgewidth=1.0)
    kw.update(extra)
    return kw


def handle(method, label=None, **extra):
    return Line2D([], [], label=label or LABEL[method], **mkw(method, **extra))


def header(text):
    return Line2D([], [], linestyle="none", marker=None, label=text)


def blank():
    return Line2D([], [], linestyle="none", marker=None, label=" ")


def family_legend(fig, cols, **kw):
    """cols: list of (family header, [handles]); one legend column per family, header in bold."""
    n = max(len(h) for _, h in cols)
    handles, bold = [], set()
    for title, hs in cols:
        bold.add(len(handles))
        handles.append(header(title))
        handles += hs + [blank()] * (n - len(hs))
    kw = {"handletextpad": 0.35, "columnspacing": 1.1, "handlelength": 1.3, **kw}
    leg = fig.legend(handles=handles, ncol=len(cols), **kw)
    for i, t in enumerate(leg.get_texts()):
        if i in bold:
            t.set_fontweight("bold")
            t.set_color(S.TEXT)
    return leg


def load():
    cells = pd.read_csv(A / "cells.csv")
    tests = pd.read_csv(A / "tests_refusal.csv")
    dp = pd.read_csv(A / "dev_points.csv")
    return cells, tests, dp


def save(fig, name):
    S.save(fig, name, out_dir=FIG)


def vs_base(tests):
    """Paired change from the original, WildGuard, harmful_all: pre-stated family vs_base + added family vs_base_added."""
    v = tests[tests["family"].isin(["vs_base", "vs_base_added"]) & (tests["judge"] == "wg")
              & (tests["group"] == "harmful_all")]
    return v.set_index("a")


def ci_line(ax, xs, ys, meth, lw=1.3, alpha=CI_ALPHA, z=2):
    ax.plot(xs, ys, color=COLOR[meth], linewidth=lw, alpha=alpha, solid_capstyle="butt", zorder=z)


# ================================================================================================ Figure 1
def fig1(cells, tests):
    """Residual refusal (WildGuard) and keyword-minus-WildGuard gap per Qwen2.5 cell, grouped by model.
    Gap tests: family kw_vs_judge (pre-stated cells) and kw_vs_judge_added (added cells; separate Holm family)."""
    gap = tests[tests["family"].isin(["kw_vs_judge", "kw_vs_judge_added"]) & (tests["judge"] == "kw_minus_wg")
                & (tests["group"] == "harmful_all")].set_index("a")
    rows = []  # (y, kind, payload)
    blocks = []  # (model, y0, y1)
    y = 0.0
    for m in MODEL_ORDER:
        y0 = y - 0.55
        rows.append((y, "header", m))
        y += 1.0
        sub = cells[cells["model"] == m].copy()
        sub["o"] = sub["method"].map({k: i for i, k in enumerate(METHOD_ROWS)})
        prev_ext = False
        for _, c in sub.sort_values("o").iterrows():
            ext = c["method"] in EXTERNAL
            if ext and not prev_ext:
                rows.append((y - 0.5, "sep", m))
            prev_ext = ext
            rows.append((y, "cell", c))
            y += 1.0
        blocks.append((m, y0, y - 0.45))
        y += 0.2
    ymax = y - 0.2
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(S.TEXT_W, 0.148 * ymax + 0.95), sharey=True,
                                 gridspec_kw={"width_ratios": [2.0, 1.0], "wspace": 0.16})
    for a in (ax, bx):
        a.set_ylim(ymax, -0.75)
        a.grid(axis="y", visible=False)
    for i, (m, y0, y1) in enumerate(blocks):
        if i % 2 == 0:
            for a in (ax, bx):
                a.add_patch(Rectangle((0.0, y0), 1.0, y1 - y0, transform=a.get_yaxis_transform(),
                                      facecolor=BAND, edgecolor="none", zorder=0))
    ticks, ticklabels = [], []
    for yy, kind, p in rows:
        if kind == "header":
            ax.text(-0.012, yy, MODEL_LABEL[p], transform=ax.get_yaxis_transform(), ha="right", va="center",
                    fontsize=S.FONT, fontweight="bold", color=S.TEXT, clip_on=False)
            continue
        if kind == "sep":
            for a in (ax, bx):
                a.axhline(yy, color=S.AXIS, linewidth=S.LW_THIN, linestyle=(0, (1.5, 2)), zorder=1)
            continue
        c = p
        ticks.append(yy)
        tag = " †" if c["label"] in COMPANION else (ADDED_SIGN if is_added(c) else "")
        ticklabels.append(LABEL[c["method"]] + tag)
        meth = c["method"]
        r, lo, hi = 100 * c["wg_rate"], 100 * c["wg_lo"], 100 * c["wg_hi"]
        ci_line(ax, [lo, hi], [yy, yy], meth)
        ax.plot([r], [yy], zorder=3, **mkw(meth))
        ax.text(1.012, yy, f"{r:.1f}", transform=ax.get_yaxis_transform(), va="center", ha="left",
                fontsize=S.FONT_SMALL, color=S.TEXT_SECONDARY, clip_on=False)
        if c["label"] in gap.index:
            g = gap.loc[c["label"]]
            d, glo, ghi = 100 * g["diff"], 100 * g["lo"], 100 * g["hi"]
            sig = bool(g["p_holm"] < 0.05)
            ci_line(bx, [glo, ghi], [yy, yy], meth)
            kw = mkw(meth, filled=sig)
            kw.pop("fillstyle", None)          # (b): fill encodes significance only
            kw.pop("markerfacecoloralt", None)
            bx.plot([d], [yy], zorder=3, **kw)
    # original-rate reference line per model block in (a)
    for m in MODEL_ORDER:
        b = cells[(cells["model"] == m) & (cells["method"] == "original")]
        ys = [yy for yy, k, p in rows if k == "cell" and p["model"] == m]
        if b.empty or not ys:
            continue
        ax.plot([100 * b.iloc[0]["wg_rate"]] * 2, [min(ys) - 0.45, max(ys) + 0.45], color=COLOR["original"],
                linestyle=(0, (2.5, 2)), linewidth=0.8, zorder=1)
    ax.set_yticks(ticks)
    ax.set_yticklabels(ticklabels)
    ax.tick_params(axis="y", length=0, pad=3)
    ax.set_xlim(0, 100)
    ax.set_xticks(range(0, 101, 20))
    ax.set_xlabel("WildGuard residual refusal on 972 held-out harmful prompts\n(%; Wilson 95% CI)")
    ax.set_title("(a) Residual refusal; dashed line = original", loc="left", fontsize=S.FONT)
    bx.axvline(0, color=S.TEXT_SECONDARY, linewidth=S.LW_THIN, zorder=1)
    bx.set_xlim(-6, 25.5)
    bx.set_xticks([-5, 0, 5, 10, 15, 20, 25])
    bx.set_xlabel("Keyword minus WildGuard\n(pp; Newcombe 95% CI)")
    bx.set_title("(b) Keyword minus judge", loc="left", fontsize=S.FONT)
    kfill = Line2D([], [], marker="o", linestyle="none", markersize=S.MS - 0.5, markerfacecolor=S.TEXT_SECONDARY,
                   markeredgecolor=S.TEXT_SECONDARY, label="Holm p < 0.05")
    khol = Line2D([], [], marker="o", linestyle="none", markersize=S.MS - 0.5, markerfacecolor=S.SURFACE,
                  markeredgecolor=S.TEXT_SECONDARY, markeredgewidth=1.1, label="n.s.")
    bx.legend(handles=[kfill, khol], loc="upper right", ncol=2, handletextpad=0.1, columnspacing=0.6,
              borderaxespad=0.15, frameon=True, framealpha=1.0, edgecolor="none", facecolor=BAND)
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02)
    save(fig, "fig1_residual_refusal")
    plt.close(fig)


# ================================================================================================ Figure 2
# direct-label placement for panel (a): label -> (dx pt, dy pt, ha)
LAB2 = {
    "qwen2.5-7b__heretic1x-general": (6, 0, "left"), "qwen2.5-7b__heretic1x-code": (0, 7, "center"),
    "qwen2.5-coder-7b__heretic1x-general": (-6, 0, "right"), "qwen2.5-coder-7b__heretic1x-code": (-6, 0, "right"),
    "qwen2.5-14b__heretic1x-general": (6, 0, "left"), "qwen2.5-14b__heretic1x-code": (-6, 0, "right"),
    "qwen2.5-coder-14b__heretic1x-general": (6, 0, "left"), "qwen2.5-coder-14b__heretic1x-code": (6, 0, "left"),
    "qwen2.5-32b__heretic1x-general": (6, 0, "left"), "qwen2.5-32b__heretic1x-code": (6, 0, "left"),
    "qwen2.5-coder-32b__heretic1x-general": (0, -8, "center"), "qwen2.5-coder-32b__heretic1x-code": (6, 0, "left"),
    "qwen2.5-32b__heretic1x-general-kl0.1": (-6, 0, "right"),
    "qwen2.5-7b__dim": (-6, 0, "right"), "qwen2.5-7b__mpoa": (-6, 0, "right"), "qwen2.5-7b__som": (6, 0, "left"),
    "qwen2.5-7b__jspace": (6, 0, "left"), "qwen2.5-7b__ara": (6, -2, "left"),
    "qwen2.5-32b__dim": (6, 0, "left"), "qwen2.5-32b__mpoa": (0, 8, "center"), "qwen2.5-32b__som": (6, 0, "left"),
    "qwen2.5-32b__ara": (6, -3, "left"),
    "qwen2.5-32b__community-huihui": (6, 0, "left"), "qwen2.5-32b__community-zetasepic-v2": (0, 8, "center"),
    "qwen2.5-7b__obliteratus-basic": (-6, 0, "right"), "qwen2.5-7b__obliteratus-advanced": (6, 0, "left"),
    "qwen2.5-7b__obliteratus-spectral_cascade": (-6, 0, "right"),
    "qwen2.5-7b__jwest33-recommended": (6, 0, "left"),
}
LAB2C = {"dim": (6, 0, "left"), "mpoa": (-6, 0, "right"), "som": (6, -2, "left"), "jspace": (6, 0, "left"),
         "ara": (6, 0, "left"), "jwest33": (-6, 0, "right"), "jwest33_hybridfix": (6, 0, "left")}
SHORT4B = {"dim": "DiM", "mpoa": "MPOA", "som": "SOM", "jspace": "J-space", "ara": "ARA",
           "jwest33": "jwest33, as released", "jwest33_hybridfix": "jwest33 + hybrid fix"}


def fig2(cells, tests):
    """Refusal-cost trade-off: reduction in residual refusal vs teacher-forced divergence and vs capability change.
    Top row: Qwen2.5 (pre-stated cells, the 1.x KL <= 0.1 re-export and the external tools on Qwen2.5-7B);
    bottom row: Qwen3.5-4B (companion-study Heretic 2.0 cells for context and the two jwest33 runs)."""
    vb = vs_base(tests)
    d = cells[(cells["status"] == "judged") & (cells["method"] != "original")].copy()
    d["red"] = -100 * d["label"].map(vb["diff"])
    d["red_lo"] = -100 * d["label"].map(vb["hi"])
    d["red_hi"] = -100 * d["label"].map(vb["lo"])
    assert d["red"].notna().all(), d.loc[d["red"].isna(), "label"].tolist()
    fig, axs = plt.subplots(2, 2, figsize=(S.TEXT_W, 6.0), sharey=True, sharex="col",
                            gridspec_kw={"width_ratios": [1.3, 1.0], "height_ratios": [1.6, 1.0]})
    (ax, bx), (cx, dx_) = axs
    for row_axes, sel in [((ax, bx), d[d["model"] != "qwen3.5-4b"]), ((cx, dx_), d[d["model"] == "qwen3.5-4b"])]:
        a1, a2 = row_axes
        for _, c in sel.iterrows():
            meth = c["method"]
            x1, x2 = c["tf_divergence"], c["capability_mean_delta_pp"]
            ci_line(a1, [x1, x1], [c["red_lo"], c["red_hi"]], meth, lw=1.0, alpha=0.7)
            a1.plot([x1], [c["red"]], zorder=3, **mkw(meth))
            ci_line(a2, [c["capability_mean_delta_lo"], c["capability_mean_delta_hi"]], [c["red"]] * 2, meth,
                    lw=1.0, alpha=0.45, z=1)
            a2.plot([x2], [c["red"]], zorder=3, **mkw(meth))
            if c["model"] == "qwen3.5-4b":
                dx, dy, ha = LAB2C.get(meth, (6, 0, "left"))
                txt = SHORT4B[meth]
            else:
                dx, dy, ha = LAB2.get(c["label"], (6, 0, "left"))
                txt = MODEL_SHORT[c["model"]]
            a1.annotate(txt, (x1, c["red"]), xytext=(dx, dy), textcoords="offset points",
                        fontsize=S.FONT_SMALL, color=S.TEXT_SECONDARY, ha=ha, va="center", zorder=5)
    for a in (ax, cx):
        a.set_xscale("log")
        a.set_xlim(0.003, 0.65)
        a.set_xticks([0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5])
        a.set_xticklabels(["0.005", "0.01", "0.02", "0.05", "0.1", "0.2", "0.5"])
        a.minorticks_off()
    for a in (bx, dx_):
        a.axvline(0, color=S.TEXT_SECONDARY, linewidth=S.LW_THIN, zorder=1)
        a.set_xlim(-7.3, 2.2)
        a.set_xticks([-7, -6, -5, -4, -3, -2, -1, 0, 1, 2])
    ax.set_ylim(-5, 93)
    ax.set_yticks(range(0, 91, 20))
    for a in axs.flat:
        a.axhline(0, color=S.AXIS, linewidth=S.LW_THIN, zorder=0)
    cx.set_xlabel("Teacher-forced divergence from the original\n(nats/token, 350 benign prompts; log scale)")
    dx_.set_xlabel("Capability change, mean of five benchmarks\n(pp; paired 95% CI)")
    ax.set_ylabel("Refusal removed: original minus edited\n(WildGuard, pp; Newcombe 95% CI)")
    cx.set_ylabel("Refusal removed (pp)")
    ax.set_title("(a) Qwen2.5: refusal removed vs distribution shift", loc="left", fontsize=S.FONT)
    bx.set_title("(b) Qwen2.5: refusal removed vs capability", loc="left", fontsize=S.FONT)
    cx.set_title("(c) Qwen3.5-4B: vs distribution shift", loc="left", fontsize=S.FONT)
    dx_.set_title("(d) Qwen3.5-4B: vs capability", loc="left", fontsize=S.FONT)
    ax.text(0.02, 0.98, "Labels: base model (C = Coder)", transform=ax.transAxes, fontsize=S.FONT_SMALL,
            color=S.TEXT_SECONDARY, va="top", ha="left")
    cx.text(0.98, 0.04, "OBLITERATUS: no model (Qwen3.5-4B not supported)", transform=cx.transAxes,
            fontsize=S.FONT_SMALL, color=S.TEXT_SECONDARY, va="bottom", ha="right")
    H = handle
    cols = [("Heretic 1.x", [H("1x_general"), H("1x_general_kl01", "1.x general, KL ≤ 0.1 ‡"), H("1x_code")]),
            ("Heretic 2.0 / fork", [H("dim"), H("mpoa"), H("ara"), H("som"), H("jspace")]),
            ("Community", [H("huihui"), H("zetasepic")]),
            ("OBLITERATUS ‡", [H("obliteratus_basic", "basic"), H("obliteratus_advanced", "advanced"),
                               H("obliteratus_spectral", "spectral cascade")]),
            ("jwest33 ‡", [H("jwest33", "as released"), H("jwest33_hybridfix", "+ hybrid fix")])]
    family_legend(fig, cols, loc="outside lower center")
    save(fig, "fig2_tradeoff")
    plt.close(fig)


# ================================================================================================ Figure 3
def fig3(cells):
    """Heretic 1.x general vs code variant on non-code and code-as-weapon prompts."""
    fig, axs = plt.subplots(1, 2, figsize=(S.TEXT_W, 2.75), sharey=False)
    specs = [("wg_noncode", "(a) Non-code harmful prompts (n = 772)", (0, 102)),
             ("wg_code", "(b) Code-as-weapon prompts (n = 200), other scale", (0, 42))]
    meths = ["original", "1x_general", "1x_code"]
    off = {"original": -0.24, "1x_general": 0.0, "1x_code": 0.24}
    for a, (col, title, ylim) in zip(axs, specs):
        for i, m in enumerate(MODEL_ORDER):
            g = cells[cells["model"] == m].set_index("method")
            if i:
                a.axvline(i - 0.5, color=S.GRID, linewidth=S.LW_THIN, zorder=0)
            for k in meths:
                c = g.loc[k]
                r, lo, hi = 100 * c[f"{col}_rate"], 100 * c[f"{col}_lo"], 100 * c[f"{col}_hi"]
                ci_line(a, [i + off[k]] * 2, [lo, hi], k, lw=1.4)
                a.plot([i + off[k]], [r], zorder=3, **mkw(k, size=S.MS + 0.5))
        a.set_xticks(range(len(MODEL_ORDER)))
        a.set_xticklabels([MODEL_LABEL[m].replace("Qwen2.5-", "") for m in MODEL_ORDER])
        a.set_xlim(-0.5, len(MODEL_ORDER) - 0.5)
        a.set_ylim(*ylim)
        a.set_yticks(range(0, 101, 20) if ylim[1] > 50 else range(0, 41, 10))
        a.grid(axis="x", visible=False)
        a.set_title(title, loc="left", fontsize=S.FONT)
        a.tick_params(axis="x", length=0)
    axs[0].set_ylabel("WildGuard residual refusal (%)")
    handles = [handle(k, size=S.MS + 0.5) for k in meths]
    fig.legend(handles=handles, loc="outside lower center", ncol=4, handletextpad=0.3, columnspacing=1.6,
               handlelength=1.3)
    save(fig, "fig3_general_vs_code")
    plt.close(fig)


# ================================================================================================ Figure 4
def step_curve(pts: pd.DataFrame, xmax=0.5):
    p = pts.sort_values(["kl", "refusals"])
    xs, ys = [], []
    best = np.inf
    for _, r in p.iterrows():
        if r["kl"] > xmax:
            break
        if r["refusals"] < best:
            best = r["refusals"]
            xs.append(r["kl"]); ys.append(best)
    xs.append(xmax); ys.append(ys[-1])
    return np.array(xs), np.array(ys)


def spread(vals, gap, lo=-1e9):
    """Label positions >= gap apart, order kept, as close to vals as possible (simple forward/backward pass)."""
    order = np.argsort(vals, kind="stable")
    v = np.array(vals, dtype=float)[order]
    pos = v.copy()
    pos[0] = max(pos[0], lo)
    for i in range(1, len(pos)):
        pos[i] = max(pos[i], pos[i - 1] + gap)
    out = np.empty_like(pos)
    out[order] = pos
    return out


FIG4_METHODS = ["1x_general", "dim", "mpoa", "som", "jspace", "ara"]
SHORT4 = {"1x_general": "1.x general", "dim": "DiM", "mpoa": "MPOA", "som": "SOM", "jspace": "J-space",
          "ara": "ARA"}


def fig4(dp, cells):
    """Dev-set keyword refusals reachable under a first-token KL cap (search records; no held-out data).
    The re-exported 1.x trial (cells.csv qwen2.5-32b__heretic1x-general-kl0.1: dev_trial 96) is marked on the 32B
    1.x Pareto front at its journal values (dev_points.csv)."""
    rx = cells[cells["method"] == "1x_general_kl01"]
    rx = rx.iloc[0] if len(rx) else None
    xmax = 0.5
    fig, axs = plt.subplots(1, 2, figsize=(S.TEXT_W, 3.5), sharey=True)
    for a, m, title in [(axs[0], "qwen2.5-32b", "(a) Qwen2.5-32B"), (axs[1], "qwen2.5-7b", "(b) Qwen2.5-7B")]:
        g = dp[dp["model"] == m]
        ends = []
        for cap, txt in [(0.01, "1.x cap"), (0.1, "2.0 cap")]:
            a.axvline(cap, color=S.TEXT_SECONDARY, linewidth=0.7, linestyle=(0, (1, 1.5)), zorder=1)
            a.text(cap, 104, txt, fontsize=S.FONT_SMALL, color=S.TEXT_SECONDARY, va="bottom", ha="center")
        for meth in FIG4_METHODS:
            q = g[g["method"] == meth]
            if q.empty:
                continue
            xs, ys = step_curve(q, xmax)
            is1x = meth == "1x_general"
            a.step(xs, ys, where="post", color=COLOR[meth], linewidth=2.0 if is1x else 1.25,
                   linestyle="-" if is1x else DASH[meth], zorder=4 if is1x else 2)
            ends.append((meth, ys[-1]))
            sel = q[q["is_selected"]]
            if len(sel):
                s = sel.iloc[0]
                a.plot([s["kl"]], [s["refusals"]], zorder=5, **mkw(meth, size=S.MS + 0.5))
            if is1x and rx is not None and m == rx["model"]:
                t = q[q["trial"] == int(rx["dev_trial"])]
                assert len(t) == 1 and bool(t.iloc[0]["on_pareto"]), "re-exported trial not on the 1.x Pareto front"
                t = t.iloc[0]
                a.plot([t["kl"]], [t["refusals"]], zorder=6, **mkw("1x_general_kl01", size=S.MS + 0.5))
                a.annotate(f"trial #{int(t['trial'])}, re-exported", (t["kl"], t["refusals"]), xytext=(-26, -20),
                           textcoords="offset points", ha="right", va="center", fontsize=S.FONT_SMALL,
                           color=COLOR["1x_general"], zorder=6,
                           arrowprops=dict(arrowstyle="-", color=COLOR["1x_general"], linewidth=0.6,
                                           shrinkA=0, shrinkB=3.5),
                           bbox=dict(boxstyle="square,pad=0.1", facecolor=S.SURFACE, edgecolor="none"))
        # direct end labels at the right edge, de-collided
        ly = spread([e[1] for e in ends], 7.0, lo=0)
        for (meth, yv), yl in zip(ends, ly):
            a.annotate(SHORT4[meth], (xmax, yv), xytext=(xmax * 1.1, yl), textcoords="data", ha="left",
                       va="center", fontsize=S.FONT_SMALL,
                       color=darker(COLOR[meth], 0.8) if meth in LIGHT else COLOR[meth], annotation_clip=False,
                       arrowprops=dict(arrowstyle="-", color=S.AXIS, linewidth=0.5, shrinkA=0, shrinkB=0)
                       if abs(yl - yv) > 1.5 else None)
        a.set_xscale("log")
        a.set_xlim(0.0015, xmax)
        a.set_ylim(-3, 103)
        a.set_yticks(range(0, 101, 20))
        a.set_xticks([0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5])
        a.set_xticklabels(["0.002", "0.005", "0.01", "0.02", "0.05", "0.1", "0.2", "0.5"])
        a.minorticks_off()
        a.set_title(title, loc="left", fontsize=S.FONT, pad=11)
        a.set_xlabel("First-token KL cap (dev; log scale)")
    axs[0].set_ylabel("Fewest dev keyword refusals (of 100)")
    handles = [Line2D([], [], color=COLOR["1x_general"], linestyle="-", linewidth=2.0, marker=MARKER["1x_general"],
                      markersize=S.MS * MSCALE["h"], label="Pareto front; filled = released trial"),
               Line2D([], [], label="trial #96, re-exported (KL ≤ 0.1 rule)", **mkw("1x_general_kl01"))]

    def h2(k):
        return Line2D([], [], color=COLOR[k], linestyle=DASH[k], linewidth=1.25, marker=MARKER[k],
                      markersize=S.MS * MSCALE.get(MARKER[k], 1.0), markerfacecolor=COLOR[k],
                      markeredgecolor=edge(k), label=LABEL[k].replace("2.0 ", ""))
    family_legend(fig, [("Heretic 1.x general (solid)", handles),
                        ("Heretic 2.0 (dashed)", [h2("dim"), h2("mpoa"), h2("ara")]),
                        ("Fork plug-ins (dashed)", [h2("som"), h2("jspace")])],
                  loc="outside lower center", handlelength=2.8, columnspacing=1.6)
    fig.get_layout_engine().set(w_pad=0.02, wspace=0.08)
    save(fig, "fig4_dev_kl_budget")
    plt.close(fig)


# ================================================================================================ Figure 5
def fig5(cells, tests):
    """Headline: Qwen2.5-32B, 1.x general released (dev KL <= 0.01) and re-exported (dev KL <= 0.1) vs the 2.0
    and fork configurations selected at dev KL <= 0.1. (a) WildGuard residual refusal (cells.csv, Wilson CI);
    (b) paired differences (tests_refusal.csv: kl_cap_1x, h1_1x_vs_2.0, h1_kl_matched; WildGuard, harmful_all)."""
    m = "qwen2.5-32b"
    c = cells[cells["model"] == m].set_index("method")
    rows = [("original", "Original"),
            ("1x_general", "1.x general, released (dev KL ≤ 0.01)"),
            ("1x_general_kl01", "1.x general, re-exported (dev KL ≤ 0.1)"),
            ("dim", "2.0 DiM (dev KL ≤ 0.1)"),
            ("mpoa", "2.0 MPOA (dev KL ≤ 0.1)"),
            ("ara", "2.0 ARA (dev KL ≤ 0.1)"),
            ("som", "SOM, fork (dev KL ≤ 0.1)")]
    t = tests[(tests["judge"] == "wg") & (tests["group"] == "harmful_all") & (tests["model"] == m)]

    def get(fam, a, b):
        r = t[(t["family"] == fam) & (t["a"] == f"{m}__{a}") & (t["b"] == f"{m}__{b}")]
        assert len(r) == 1, (fam, a, b)
        return r.iloc[0]
    K01, REL = "heretic1x-general-kl0.1", "heretic1x-general"
    diffs = [  # (label, test row)
        ("Re-exported − released (same search)", get("kl_cap_1x", K01, REL)),
        ("Released 1.x − 2.0 DiM (as pre-stated)", get("h1_1x_vs_2.0", REL, "dim")),
        ("Re-exported 1.x − 2.0 DiM (same rule)", get("h1_kl_matched", K01, "dim")),
        ("Re-exported 1.x − 2.0 MPOA", get("h1_kl_matched", K01, "mpoa")),
        ("Re-exported 1.x − 2.0 ARA", get("h1_kl_matched", K01, "ara")),
        ("Re-exported 1.x − SOM (fork)", get("h1_kl_matched", K01, "som")),
    ]
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(S.TEXT_W, 3.7),
                                 gridspec_kw={"height_ratios": [len(rows) + 0.7, len(diffs) + 0.7]})
    # (a) rates
    ys = np.arange(len(rows), dtype=float)
    ys[1:] += 0.35
    ys[3:] += 0.35
    for y, (meth, lab) in zip(ys, rows):
        r = c.loc[meth]
        rate, lo, hi = 100 * r["wg_rate"], 100 * r["wg_lo"], 100 * r["wg_hi"]
        ci_line(ax, [lo, hi], [y, y], meth, lw=1.5)
        ax.plot([rate], [y], zorder=3, **mkw(meth, size=S.MS + 1.0))
        ax.annotate(f"{rate:.1f}", (hi, y), xytext=(5, 0), textcoords="offset points", va="center", ha="left",
                    fontsize=S.FONT_SMALL, color=S.TEXT_SECONDARY)
    ax.axvline(100 * c.loc["original", "wg_rate"], color=COLOR["original"], linestyle=(0, (2.5, 2)),
               linewidth=0.8, zorder=1)
    ax.set_yticks(ys)
    ax.set_yticklabels([lab for _, lab in rows])
    ax.tick_params(axis="y", length=0, pad=3)
    ax.set_ylim(ys[-1] + 0.6, -0.6)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, 100)
    ax.set_xticks(range(0, 101, 20))
    ax.set_xlabel("WildGuard residual refusal (%; Wilson 95% CI)")
    ax.set_title("(a) Qwen2.5-32B: residual refusal on the 972 harmful held-out prompts", loc="left", fontsize=S.FONT)
    # (b) paired differences
    yb = np.arange(len(diffs), dtype=float)
    yb[1:] += 0.35
    yb[2:] += 0.35
    for y, (lab, r) in zip(yb, diffs):
        dd, lo, hi = 100 * r["diff"], 100 * r["lo"], 100 * r["hi"]
        bx.plot([lo, hi], [y, y], color=S.TEXT, linewidth=1.5, solid_capstyle="butt", zorder=2)
        bx.plot([dd], [y], marker="o", markersize=S.MS, color=S.TEXT, linestyle="none", zorder=3)
        txt = f"{dd:+.1f} [{lo:+.1f}, {hi:+.1f}]".replace("-", "−")
        bx.annotate(txt, (hi, y), xytext=(5, 0), textcoords="offset points", va="center", ha="left",
                    fontsize=S.FONT_SMALL, color=S.TEXT_SECONDARY)
    bx.axvline(0, color=S.TEXT_SECONDARY, linewidth=S.LW_THIN, zorder=1)
    bx.set_yticks(yb)
    bx.set_yticklabels([d[0] for d in diffs])
    bx.tick_params(axis="y", length=0, pad=3)
    bx.set_ylim(yb[-1] + 0.6, -0.6)
    bx.grid(axis="y", visible=False)
    bx.set_xlim(-70, 55)
    bx.set_xticks([-60, -40, -20, 0, 20, 40])
    bx.set_xlabel("Paired difference (pp; Newcombe 95% CI)")
    bx.set_title("(b) Paired differences in residual refusal (negative = first model refuses less)", loc="left",
                 fontsize=S.FONT)
    fig.get_layout_engine().set(w_pad=0.02)
    save(fig, "fig5_kl_matched_32b")
    plt.close(fig)


def main():
    cells, tests, dp = load()
    fig1(cells, tests)
    fig2(cells, tests)
    fig3(cells)
    fig4(dp, cells)
    fig5(cells, tests)


if __name__ == "__main__":
    main()
