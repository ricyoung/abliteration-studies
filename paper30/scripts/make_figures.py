#!/usr/bin/env python3
"""Paper 30 v3 figures (reads only analysis/*.csv written by analysis/assemble_p30.py).

Run:
    /data/heretic_2/33_2611_does_refusal_removal_get_harder/scripts/figures/.venv/bin/python \
        /data/heretic_2/paper30/v3_2611/scripts/make_figures.py

Uses paper 33's style module (colours, fonts, sizes, pending encodings) but writes into v3_2611/figures/
(PDF for the paper, preview/*.png for visual review). Pending cells are drawn explicitly, never dropped.

Revision 2026-11-06 (v3, after the added runs):
  * the 7B Heretic 1.x cells are judged and drawn (no pending markers remain);
  * cells added after the pre-stated design: external tools (OBLITERATUS, jwest33/abliterator; one neutral dark ink,
    one marker shape per tool preset) and the re-exported 1.x general cell at KL <= 0.1 (1.x teal, half-filled
    hexagon); tick labels of added cells carry a section sign;
  * fig1 and fig2 panels (a, b) show the Qwen2.5 cells; fig2 panels (c, d) show Qwen3.5-4B (Heretic 2.0 cells of the
    companion study for context, jwest33 as released and with the hybrid fix; OBLITERATUS not supported);
  * fig4 marks the re-exported trial #96 on the 32B 1.x Pareto front (dev_points.csv).
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
from matplotlib.lines import Line2D  # noqa: E402

V3 = Path("/data/heretic_2/paper30/v3_2611")
A = V3 / "analysis"
FIG = V3 / "figures"

S.apply()

# ------------------------------------------------------------------------------------------------ encodings
# Heretic 2.0 methods, original and community keep paper 33's colours/markers; the two Heretic 1.x variants get
# two hues not used by paper 33 (deep teal, sienna) and their own marker shapes. External tools share one neutral
# dark ink (distinct in lightness from every method hue and from the grey original under all three CVD
# simulations; CIEDE2000 >= 19.7 to the nearest hue, deutan, vs 1.x teal) and differ by marker shape. Half-filled
# markers = a variant of the same tool: 1.x general re-exported at KL <= 0.1; jwest33 with the hybrid fix.
EXT_INK = "#262626"
COLOR = {"original": S.METHOD_COLORS["base"], "1x_general": "#0b6e69", "1x_code": "#a0521d",
         "1x_general_kl01": "#0b6e69",
         "dim": S.METHOD_COLORS["dim"], "mpoa": S.METHOD_COLORS["mpoa"], "som": S.METHOD_COLORS["som"],
         "jspace": S.METHOD_COLORS["jspace"], "ara": S.METHOD_COLORS["ara"],
         "huihui": S.METHOD_COLORS["community"], "zetasepic": S.METHOD_COLORS["community"],
         "obliteratus_basic": EXT_INK, "obliteratus_advanced": EXT_INK, "obliteratus_spectral": EXT_INK,
         "jwest33": EXT_INK, "jwest33_hybridfix": EXT_INK}
MARKER = {"original": "*", "1x_general": "h", "1x_code": "p", "1x_general_kl01": "h", "dim": "o", "mpoa": "s",
          "som": "D", "jspace": "v", "ara": "^", "huihui": "P", "zetasepic": "X",
          "obliteratus_basic": "<", "obliteratus_advanced": ">", "obliteratus_spectral": "d",
          "jwest33": "8", "jwest33_hybridfix": "8"}
HALF = {"1x_general_kl01", "jwest33_hybridfix"}
MSCALE = {"*": 1.35, "D": 0.85, "X": 1.05, "P": 1.05, "h": 1.05, "p": 1.05, "<": 1.0, ">": 1.0, "d": 1.1,
          "8": 1.0}
# SOM and J-space are plug-ins of the project fork (paper33-plugins, commit 8fd4d76), not part of Heretic 2.0
# releases; labelled "(fork)" in every figure (revision 2026-11-06).
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
ADDED_SIGN = " §"


def is_added(c) -> bool:
    return c["tool"] == "external" or c["method"] == "1x_general_kl01" or c["model"] == "qwen3.5-4b"


def mkw(method, size=S.MS, filled=True, **extra):
    m = MARKER[method]
    c = COLOR[method]
    kw = dict(marker=m, markersize=size * MSCALE.get(m, 1.0), markeredgecolor=c,
              markerfacecolor=c if filled else S.SURFACE, markeredgewidth=0.6 if filled else S.HOLLOW_MEW,
              linestyle="none", color=c)
    if filled and method in HALF:
        kw.update(fillstyle="left", markerfacecoloralt=S.SURFACE, markeredgewidth=S.HOLLOW_MEW * 0.85)
    kw.update(extra)
    return kw


def handle(method, label=None):
    return Line2D([], [], label=label or LABEL[method], **mkw(method))


def blank():
    return Line2D([], [], linestyle="none", marker=None, label=" ")


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


# ================================================================================================ Figure 1
def fig1(cells, tests):
    """Residual refusal (WildGuard) and keyword-minus-WildGuard gap per Qwen2.5 cell, grouped by model.
    Gap tests: family kw_vs_judge (pre-stated cells) and kw_vs_judge_added (added cells; separate Holm family)."""
    gap = tests[tests["family"].isin(["kw_vs_judge", "kw_vs_judge_added"]) & (tests["judge"] == "kw_minus_wg")
                & (tests["group"] == "harmful_all")].set_index("a")
    rows = []  # (y, kind, payload)
    y = 0.0
    for m in MODEL_ORDER:
        rows.append((y, "header", m))
        y += 0.85
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
        y += 0.35
    ymax = y - 0.35
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(S.TEXT_W, 0.152 * ymax + 0.75), sharey=True,
                                 gridspec_kw={"width_ratios": [2.0, 1.0], "wspace": 0.04})
    for a in (ax, bx):
        a.set_ylim(ymax + 0.2, -0.75)
        a.grid(axis="y", visible=False)
    ticks, ticklabels = [], []
    for yy, kind, p in rows:
        if kind == "header":
            ax.text(-0.01, yy, MODEL_LABEL[p], transform=ax.get_yaxis_transform(), ha="right", va="center",
                    fontsize=S.FONT, fontweight="bold", color=S.TEXT, clip_on=False)
            for a in (ax, bx):
                a.axhline(yy + 0.42, color=S.AXIS, linewidth=S.LW_THIN, zorder=0)
            continue
        if kind == "sep":
            for a in (ax, bx):
                a.axhline(yy, color=S.AXIS, linewidth=S.LW_THIN, linestyle=(0, (1.5, 2)), zorder=0)
            ax.text(1.0, yy + 0.06, "external tools", ha="left", va="top",
                    fontsize=S.FONT_SMALL, color=S.TEXT_SECONDARY, style="italic", zorder=4)
            continue
        c = p
        ticks.append(yy)
        tag = " †" if c["label"] in COMPANION else (ADDED_SIGN if is_added(c) else "")
        ticklabels.append(LABEL[c["method"]] + tag)
        meth = c["method"]
        if c["status"] != "judged":
            S.pending_marker(ax, 2.0, yy, text="pending")
            bx.text(0.0, yy, "pending", ha="center", va="center", fontsize=S.FONT_SMALL, color=S.TEXT_SECONDARY)
            continue
        r, lo, hi = 100 * c["wg_rate"], 100 * c["wg_lo"], 100 * c["wg_hi"]
        ax.plot([lo, hi], [yy, yy], color=COLOR[meth], linewidth=1.3, solid_capstyle="butt", zorder=2)
        ax.plot([r], [yy], zorder=3, **mkw(meth))
        ax.text(hi + 1.4, yy, f"{r:.1f}", va="center", ha="left", fontsize=S.FONT_SMALL, color=S.TEXT_SECONDARY,
                zorder=4, bbox=dict(boxstyle="square,pad=0.08", facecolor=S.SURFACE, edgecolor="none"))
        if c["label"] in gap.index:
            g = gap.loc[c["label"]]
            d, glo, ghi = 100 * g["diff"], 100 * g["lo"], 100 * g["hi"]
            sig = g["p_holm"] < 0.05
            bx.plot([glo, ghi], [yy, yy], color=COLOR[meth], linewidth=1.3, solid_capstyle="butt", zorder=2)
            kw = mkw(meth, filled=bool(sig))
            kw.pop("fillstyle", None)          # (b): fill encodes significance only
            kw.pop("markerfacecoloralt", None)
            if sig:
                kw["markeredgewidth"] = 0.6
            bx.plot([d], [yy], zorder=3, **kw)
    # original-rate reference line per model block in (a)
    for m in MODEL_ORDER:
        b = cells[(cells["model"] == m) & (cells["method"] == "original")]
        ys = [yy for yy, k, p in rows if k == "cell" and p["model"] == m]
        if b.empty or b.iloc[0]["status"] != "judged" or not ys:
            continue
        ax.plot([100 * b.iloc[0]["wg_rate"]] * 2, [min(ys) - 0.45, max(ys) + 0.45], color=S.METHOD_COLORS["base"],
                linestyle=(0, (3, 2)), linewidth=0.8, zorder=1)
    ax.set_yticks(ticks)
    ax.set_yticklabels(ticklabels)
    ax.tick_params(axis="y", length=0, pad=2)
    ax.set_xlim(0, 100)
    ax.set_xticks(range(0, 101, 20))
    ax.set_xlabel("WildGuard residual refusal on 972 held-out harmful prompts\n(%; Wilson 95% CI)")
    ax.set_title("(a) Residual refusal; dashed line = original", loc="left", fontsize=S.FONT)
    bx.axvline(0, color=S.AXIS, linewidth=S.LW_THIN, zorder=0)
    bx.set_xlim(-6, 25.5)
    bx.set_xticks([-5, 0, 5, 10, 15, 20])
    bx.set_xlabel("Keyword minus WildGuard\n(pp; Newcombe 95% CI)")
    bx.set_title("(b) Keyword minus judge", loc="left", fontsize=S.FONT)
    bx.text(0.98, 0.005, "filled = Holm p < 0.05", transform=bx.transAxes, ha="right", va="bottom",
            fontsize=S.FONT_SMALL, color=S.TEXT_SECONDARY)
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02)
    save(fig, "fig1_residual_refusal")
    plt.close(fig)


# ================================================================================================ Figure 2
# direct-label placement for panel (a): label -> (dx pt, dy pt, ha)
LAB2 = {
    "qwen2.5-14b__heretic1x-code": (-5, -4, "right"), "qwen2.5-32b__heretic1x-general": (5, -3, "left"),
    "qwen2.5-coder-32b__heretic1x-general": (0, -8, "center"), "qwen2.5-coder-32b__heretic1x-code": (5, 4, "left"),
    "qwen2.5-32b__heretic1x-code": (5, -3, "left"), "qwen2.5-14b__heretic1x-general": (5, 1, "left"),
    "qwen2.5-7b__jspace": (5, -2, "left"), "qwen2.5-coder-14b__heretic1x-code": (5, 3, "left"),
    "qwen2.5-32b__dim": (5, 0, "left"), "qwen2.5-32b__som": (5, 0, "left"),
    "qwen2.5-7b__dim": (-5, 0, "right"), "qwen2.5-7b__mpoa": (-5, 0, "right"), "qwen2.5-7b__som": (5, 0, "left"),
    "qwen2.5-7b__ara": (5, -3, "left"), "qwen2.5-32b__mpoa": (0, 8, "center"),
    "qwen2.5-coder-14b__heretic1x-general": (5, 0, "left"), "qwen2.5-32b__ara": (5, -4, "left"),
    "qwen2.5-32b__community-zetasepic-v2": (5, 4, "left"), "qwen2.5-32b__community-huihui": (5, 0, "left"),
    "qwen2.5-7b__heretic1x-general": (5, 0, "left"), "qwen2.5-7b__heretic1x-code": (5, 3, "left"),
    "qwen2.5-coder-7b__heretic1x-general": (-5, 0, "right"), "qwen2.5-coder-7b__heretic1x-code": (-5, 2, "right"),
    "qwen2.5-32b__heretic1x-general-kl0.1": (-5, 3, "right"),
    "qwen2.5-7b__obliteratus-basic": (-5, 0, "right"), "qwen2.5-7b__obliteratus-advanced": (5, 0, "left"),
    "qwen2.5-7b__obliteratus-spectral_cascade": (-5, 0, "right"), "qwen2.5-7b__jwest33-recommended": (5, 0, "left"),
}
# panels (c), (d): Qwen3.5-4B; short method labels next to each marker in (c)
LAB2C = {"dim": (5, 0, "left"), "mpoa": (-5, 2, "right"), "som": (5, -3, "left"), "jspace": (5, 0, "left"),
         "ara": (5, 0, "left"), "jwest33": (-5, 0, "right"), "jwest33_hybridfix": (5, 0, "left")}
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
    fig, axs = plt.subplots(2, 2, figsize=(S.TEXT_W, 5.55), sharey=True, sharex="col",
                            gridspec_kw={"width_ratios": [1.3, 1.0], "height_ratios": [1.55, 1.0]})
    (ax, bx), (cx, dx_) = axs
    for row_axes, sel in [((ax, bx), d[d["model"] != "qwen3.5-4b"]), ((cx, dx_), d[d["model"] == "qwen3.5-4b"])]:
        a1, a2 = row_axes
        for _, c in sel.iterrows():
            meth = c["method"]
            for a, x in [(a1, c["tf_divergence"]), (a2, c["capability_mean_delta_pp"])]:
                a.plot([x, x], [c["red_lo"], c["red_hi"]], color=COLOR[meth], linewidth=0.9, alpha=0.7, zorder=2)
                a.plot([x], [c["red"]], zorder=3, **mkw(meth))
            a2.plot([c["capability_mean_delta_lo"], c["capability_mean_delta_hi"]], [c["red"]] * 2, color=COLOR[meth],
                    linewidth=0.9, alpha=0.45, zorder=1)
            if c["model"] == "qwen3.5-4b":
                dx, dy, ha = LAB2C.get(meth, (5, 0, "left"))
                txt = SHORT4B[meth]
            else:
                dx, dy, ha = LAB2.get(c["label"], (5, 0, "left"))
                txt = MODEL_SHORT[c["model"]]
            a1.annotate(txt, (c["tf_divergence"], c["red"]), xytext=(dx, dy), textcoords="offset points",
                        fontsize=S.FONT_SMALL, color=S.TEXT_SECONDARY, ha=ha, va="center", zorder=5)
    for a in (ax, cx):
        a.set_xscale("log")
        a.set_xlim(0.003, 0.65)
        a.set_xticks([0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5])
        a.set_xticklabels(["0.005", "0.01", "0.02", "0.05", "0.1", "0.2", "0.5"])
        a.minorticks_off()
    for a in (bx, dx_):
        a.axvline(0, color=S.AXIS, linewidth=S.LW_THIN, zorder=0)
        a.set_xlim(-7.3, 2.2)
        a.set_xticks([-7, -6, -5, -4, -3, -2, -1, 0, 1, 2])
    ax.set_ylim(-4, 93)
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
    ax.text(0.0034, 90, "labels: Qwen2.5 model (C = Coder)", fontsize=S.FONT_SMALL, color=S.TEXT_SECONDARY,
            va="top")
    cx.text(0.6, 2, "OBLITERATUS: Qwen3.5-4B not supported (no model)", fontsize=S.FONT_SMALL,
            color=S.TEXT_SECONDARY, va="bottom", ha="right")
    # legend: columns = tool families (Heretic 1.x | Heretic 2.0 | fork | community | OBLITERATUS | jwest33)
    H = handle
    cols = [[H("1x_general"), H("1x_general_kl01", "1.x general, KL ≤ 0.1 §"), H("1x_code")],
            [H("dim"), H("mpoa"), H("ara")],
            [H("som"), H("jspace"), blank()],
            [H("huihui"), H("zetasepic"), blank()],
            [H("obliteratus_basic", "OBLITERATUS basic §"),
             H("obliteratus_advanced", "OBLITERATUS advanced §"),
             H("obliteratus_spectral", "OBLITERATUS spectral cascade §")],
            [H("jwest33", "jwest33, as released §"), H("jwest33_hybridfix", "jwest33 + hybrid fix §"),
             blank()]]
    handles = [h for col in cols for h in col]
    fig.legend(handles=handles, loc="outside lower center", ncol=6, fontsize=S.FONT_SMALL, handletextpad=0.25,
               columnspacing=0.9, handlelength=1.2)
    save(fig, "fig2_tradeoff")
    plt.close(fig)


# ================================================================================================ Figure 3
def fig3(cells):
    """Heretic 1.x general vs code variant on non-code and code-as-weapon prompts."""
    fig, axs = plt.subplots(1, 2, figsize=(S.TEXT_W, 2.75), sharey=False)
    specs = [("harmful_noncode", "wg_noncode", "(a) Non-code harmful prompts (n = 772)", (-2, 104)),
             ("code_weapon", "wg_code", "(b) Code-as-weapon prompts (n = 200)", (-1, 45))]
    meths = ["original", "1x_general", "1x_code"]
    off = {"original": -0.24, "1x_general": 0.0, "1x_code": 0.24}
    for a, (grp, col, title, ylim) in zip(axs, specs):
        for i, m in enumerate(MODEL_ORDER):
            g = cells[cells["model"] == m].set_index("method")
            if any(g.loc[k, "status"] != "judged" for k in meths if k in g.index):
                S.draw_pending(a, i - 0.4, 0.5, 0.8, 0.5 * ylim[1],
                               text="pending", rotation=90)
                continue
            for k in meths:
                c = g.loc[k]
                r, lo, hi = 100 * c[f"{col}_rate"], 100 * c[f"{col}_lo"], 100 * c[f"{col}_hi"]
                a.plot([i + off[k]] * 2, [lo, hi], color=COLOR[k], linewidth=1.3, solid_capstyle="butt", zorder=2)
                a.plot([i + off[k]], [r], zorder=3, **mkw(k))
        a.set_xticks(range(len(MODEL_ORDER)))
        a.set_xticklabels([MODEL_LABEL[m].replace("Qwen2.5-", "") for m in MODEL_ORDER])
        a.set_xlim(-0.6, len(MODEL_ORDER) - 0.4)
        a.set_ylim(*ylim)
        a.set_yticks(range(0, 101, 20) if ylim[1] > 50 else range(0, 41, 10))
        a.grid(axis="x", visible=False)
        a.set_title(title, loc="left", fontsize=S.FONT)
        a.tick_params(axis="x", length=0)
    axs[0].set_ylabel("WildGuard residual refusal (%)")
    handles = [Line2D([], [], label=LABEL[k], **mkw(k)) for k in meths]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, fontsize=S.FONT_SMALL,
               title="Qwen2.5 Instruct models; WildGuard refusal with 95% Wilson intervals", title_fontsize=S.FONT_SMALL)
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


def fig4(dp, cells):
    """Dev-set keyword refusals reachable under a first-token KL cap (search records; no held-out data).
    The re-exported 1.x trial (cells.csv qwen2.5-32b__heretic1x-general-kl0.1: dev_trial 96) is marked on the 32B
    1.x Pareto front at its journal values (dev_points.csv)."""
    rx = cells[cells["method"] == "1x_general_kl01"]
    rx = rx.iloc[0] if len(rx) else None
    fig, axs = plt.subplots(1, 2, figsize=(S.TEXT_W, 3.1), sharey=True)
    for a, m, title in [(axs[0], "qwen2.5-32b", "(a) Qwen2.5-32B"), (axs[1], "qwen2.5-7b", "(b) Qwen2.5-7B")]:
        g = dp[dp["model"] == m]
        for meth in ["1x_general", "dim", "mpoa", "som", "jspace", "ara"]:
            q = g[g["method"] == meth]
            if q.empty:
                continue
            xs, ys = step_curve(q)
            ls = "-" if meth == "1x_general" else (0, (4, 1.6))
            a.step(xs, ys, where="post", color=COLOR[meth], linewidth=1.9 if meth == "1x_general" else 1.3,
                   linestyle=ls, zorder=3 if meth == "1x_general" else 2)
            sel = q[q["is_selected"]]
            if len(sel):
                s = sel.iloc[0]
                a.plot([s["kl"]], [s["refusals"]], zorder=4, **mkw(meth, size=S.MS + 0.5))
            if meth == "1x_general" and rx is not None and m == rx["model"]:
                t = q[q["trial"] == int(rx["dev_trial"])]
                assert len(t) == 1 and bool(t.iloc[0]["on_pareto"]), "re-exported trial not on the 1.x Pareto front"
                t = t.iloc[0]
                a.plot([t["kl"]], [t["refusals"]], zorder=5, **mkw("1x_general_kl01", size=S.MS + 0.5))
                a.annotate(f"#{int(t['trial'])} re-exported", (t["kl"], t["refusals"]), xytext=(-6, -13),
                           textcoords="offset points", ha="right", va="center", fontsize=S.FONT_SMALL,
                           color=COLOR["1x_general"], zorder=6,
                           arrowprops=dict(arrowstyle="-", color=COLOR["1x_general"], linewidth=0.6,
                                           shrinkA=0, shrinkB=3.5),
                           bbox=dict(boxstyle="square,pad=0.1", facecolor=S.SURFACE, edgecolor="none"))
        for cap, txt in [(0.01, "1.x cap"), (0.1, "2.0 cap")]:
            a.axvline(cap, color=S.TEXT_SECONDARY, linewidth=0.7, linestyle=(0, (2, 2)), zorder=1)
            a.text(cap, -7, txt, fontsize=S.FONT_SMALL, color=S.TEXT_SECONDARY, va="center", ha="center", zorder=5,
                   bbox=dict(boxstyle="square,pad=0.12", facecolor=S.SURFACE, edgecolor="none"))
        a.set_xscale("log")
        a.set_xlim(0.0015, 0.5)
        a.set_ylim(-11, 103)
        a.set_yticks(range(0, 101, 20))
        a.set_xticks([0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5])
        a.set_xticklabels(["0.002", "0.005", "0.01", "0.02", "0.05", "0.1", "0.2", "0.5"])
        a.minorticks_off()
        a.set_title(title, loc="left", fontsize=S.FONT)
        a.set_xlabel("First-token KL cap (dev; log scale)")
    axs[0].set_ylabel("Fewest dev keyword refusals (of 100)")
    present = [k for k in ["1x_general", "dim", "mpoa", "som", "jspace", "ara"] if k in set(dp["method"])]
    handles = []
    for k in present:
        ls = "-" if k == "1x_general" else (0, (4, 1.6))
        handles.append(Line2D([], [], color=COLOR[k], linestyle=ls, linewidth=1.6, marker=MARKER[k],
                              markersize=S.MS * MSCALE.get(MARKER[k], 1.0), markerfacecolor=COLOR[k],
                              markeredgecolor=COLOR[k], label=LABEL[k].replace("1.x general", "1.x (general)")))
    if rx is not None:
        handles.append(Line2D([], [], label="1.x trial #96, re-exported (KL ≤ 0.1 rule)", **mkw("1x_general_kl01")))
    fig.legend(handles=handles, loc="outside lower center", ncol=4, fontsize=S.FONT_SMALL, handletextpad=0.4,
               columnspacing=0.9, title="filled marker = released trial (1.x: archived runs' rule, KL ≤ 0.01; "
               "2.0 and fork: companion study's rule, KL ≤ 0.1)", title_fontsize=S.FONT_SMALL)
    save(fig, "fig4_dev_kl_budget")
    plt.close(fig)


def main():
    cells, tests, dp = load()
    fig1(cells, tests)
    fig2(cells, tests)
    fig3(cells)
    fig4(dp, cells)


if __name__ == "__main__":
    main()
