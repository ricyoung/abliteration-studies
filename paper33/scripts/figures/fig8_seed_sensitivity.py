#!/usr/bin/env python3
"""fig8_seed_sensitivity.py -- how much the dev-set result of one optimisation run depends on its seed (paper 33).

Small pair only (Qwen2.5-7B -> Qwen3.5-4B, RTX 4090), the one pair with a second seed for every method.
Each of the 10 model x method cells was optimised twice with the same configuration, Heretic commit and GPU,
differing only in the seed (runs <model>__<method>__core-s1 and __core-s2).

Left  : dumbbell chart, one row per cell. The two markers are the dev keyword refusals of the trial picked by the
        selection rule (fewest dev refusals at first-token KL <= 0.1) in the seed-1 and the seed-2 run, joined by a
        line; |s1 - s2| is printed in a column on the right. Seed 1 = full method colour, seed 2 = the same marker
        lighter (as the s2 markers of fig3). Generation keeps the house encoding (older model: hollow marker, dashed
        line; newer: filled, solid). A seed-2 run that has not finished (runs.csv status not 'present' or
        run_finished False) is drawn as pending with its trial count, never as a value.
Right : the Pareto fronts (trials.csv on_pareto_front) of both seeds for the cell with the LARGEST and the cell with
        the SMALLEST |s1 - s2| among cells whose two runs have both finished (ties: first in row order), with the
        selected trial of each seed ringed and the KL = 0.1 threshold dotted.

Inputs : paper/figures/data/runs.csv (selected_refusals, status, run_finished), trials.csv (fronts, cross-check)
Outputs: paper/figures/fig8_seed_sensitivity.pdf, preview/fig8_seed_sensitivity.png, captions.md section
Run    : cd scripts/figures && .venv/bin/python fig8_seed_sensitivity.py
"""
from __future__ import annotations

import math
import statistics

import numpy as np
from matplotlib.colors import to_hex, to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import FixedLocator, NullFormatter
from matplotlib.transforms import offset_copy

import style as S

NAME = "fig8_seed_sensitivity"
MODELS = [("qwen2.5-7b", "small-old"), ("qwen3.5-4b", "small-new")]   # small pair: older, newer
METHODS = ["dim", "mpoa", "som", "jspace", "ara"]
SEEDS = (1, 2)
KL_MAX = 0.1
S2_STRENGTH = 0.45          # seed 2 = method colour mixed with white at this strength (= fig3's S2_ALPHA on white)
LABEL_DX = 6.0              # pt, gap between a dumbbell marker centre and its value label
XLIM_D = (-15, 106)         # dumbbell x range (labels need room left of 0; the spine is drawn 0..100 only)
FRONT_MS = 5.0              # pt, Pareto-front vertex markers (as fig2)
RING_MS = 11.0              # pt, ring around a selected trial (as fig2)
RING_MEW = 1.1
SWATCH_PT = 7.0             # pt, width of the hatched pending glyph in a dumbbell row
Y_LO, Y_HI = -4, 104


# ================================================================================================ helpers
def tint(hex_, strength=S2_STRENGTH):
    """Opaque equivalent of drawing `hex_` at alpha=strength on the white surface (no alpha build-up)."""
    c = np.array(to_rgb(hex_))
    return to_hex(strength * c + (1 - strength) * np.array(to_rgb(S.SURFACE)))


def seed_kw(method, gen, seed, size=S.MS, line=False):
    """House marker/line kwargs for a method and generation; seed 2 in the lighter tint."""
    kw = S.line_kw(method, gen, size=size) if line else S.marker_kw(method, gen, size=size)
    if seed != 1:
        c = tint(S.color(method))
        kw["color"] = c
        kw["markeredgecolor"] = c
        if kw.get("markerfacecolor") != S.SURFACE:
            kw["markerfacecolor"] = c
    return kw


def pt_per_unit(ax):
    """Printed points per x data unit of the (linear) dumbbell axis, from the current axes position."""
    w_in = ax.get_position().width * ax.figure.get_figwidth()
    return w_in * 72.0 / (XLIM_D[1] - XLIM_D[0])


def truthy(v):
    return str(v).strip().lower() == "true"


def pending_text(row):
    """Short reason for a run that has no final value yet (from runs.csv)."""
    if row is None:
        return "pending (no run yet)"
    st = str(row["status"])
    if st.startswith("partial") or (st.startswith("present") and not truthy(row["run_finished"])):
        n, N = row.get("n_trials_complete"), row.get("n_trials_planned")
        if n == n and N == N:   # not NaN
            return f"pending (running, {int(n)}/{int(N)} trials)"
        return "pending (running)"
    if "queued" in st:
        return "pending (queued)"
    return "pending (not started)"


# ================================================================================================ data
def load(runs, trials):
    """cells: list of dicts (model, gen, method, s1, s2, why1, why2, run1, run2) in row order."""
    problems = []
    cells = []
    for model, gen in MODELS:
        for m in METHODS:
            c = {"model": model, "gen": gen, "method": m}
            for seed in SEEDS:
                r = runs[(runs.model == model) & (runs.method == m) & (runs.seed == seed)]
                row = r.iloc[0] if len(r) == 1 else None
                if len(r) > 1:
                    problems.append(f"{model} {m} s{seed}: {len(r)} rows in runs.csv")
                ok = (row is not None and str(row["status"]).startswith("present") and truthy(row["run_finished"])
                      and row["selected_refusals"] == row["selected_refusals"])
                c[f"s{seed}"] = float(row["selected_refusals"]) if ok else math.nan
                c[f"kl{seed}"] = float(row["selected_kl"]) if ok else math.nan
                c[f"run{seed}"] = str(row["run"]) if row is not None and row["run"] == row["run"] else ""
                c[f"why{seed}"] = "" if ok else pending_text(row)
                if ok:
                    if not c[f"kl{seed}"] <= KL_MAX:
                        problems.append(f"{model} {m} s{seed}: selected KL {c[f'kl{seed}']} > {KL_MAX}")
                    t = trials[(trials.run == row["run"]) & trials["is_selected_kl0.1"]]
                    if len(t) != 1 or float(t.refusals.iloc[0]) != c[f"s{seed}"] or int(t.seed.iloc[0]) != seed:
                        problems.append(f"{model} {m} s{seed}: trials.csv selected trial disagrees with runs.csv")
                    elif not bool(t.on_pareto_front.iloc[0]):
                        problems.append(f"{model} {m} s{seed}: selected trial not on the Pareto front")
            c["gap"] = abs(c["s1"] - c["s2"]) if not (math.isnan(c["s1"]) or math.isnan(c["s2"])) else math.nan
            cells.append(c)
    for p in problems:
        print("  [fig8] CHECK FAILED:", p)
    if not problems:
        print("  [fig8] cross-checks OK (runs.csv selected_refusals = trials.csv selected trial, on front, KL <= 0.1)")
    return cells


def extremes(cells):
    done = [c for c in cells if not math.isnan(c["gap"])]
    if not done:
        return None, None
    big = max(done, key=lambda c: c["gap"])          # max/min keep the first of equal gaps (row order)
    small = min(done, key=lambda c: c["gap"])
    return big, (small if small is not big else None)


# ================================================================================================ dumbbell
def value_label(ax, v, yy, seed, sgn):
    txt = f"s{seed} {v:.0f}" if sgn < 0 else f"{v:.0f} s{seed}"
    tr = offset_copy(ax.transData, fig=ax.figure, x=sgn * LABEL_DX, y=0, units="points")
    ax.text(v, yy, txt, transform=tr, ha="right" if sgn < 0 else "left", va="center", fontsize=S.FONT_SMALL,
            color=S.TEXT if seed == 1 else S.TEXT_SECONDARY, zorder=5)


def text_w(txt, size=S.FONT_SMALL):
    """Printed width (pt) of a label in the house font."""
    from matplotlib.font_manager import FontProperties
    from matplotlib.textpath import TextPath
    return TextPath((0, 0), txt, size=size, prop=FontProperties(family=S.mpl.rcParams["font.sans-serif"])) \
        .get_extents().width


def place_pending(ax, pending):
    """Rows with a pending seed: the finished seed's value label on one side of its marker, a hatched swatch (the
    house pending glyph) and the reason on the other, whichever side has room for the reason. Called after a
    layout pass so the points-per-unit scale of the axis is final."""
    fig, ppu = ax.figure, pt_per_unit(ax)
    pad = LABEL_DX + SWATCH_PT + 2.5
    for c, yy, s_have in pending:
        if s_have is None:          # both seeds pending: swatch + reason from the left end of the row
            sgn, anchor, opts = +1, 0.0, [c["why1"] and f"s1 {c['why1']}", c["why2"] and f"s2 {c['why2']}"]
            txts = ["; ".join(x for x in opts if x)]
        else:
            s_miss = 2 if s_have == 1 else 1
            anchor = c[f"s{s_have}"]
            why = c[f"why{s_miss}"]
            short = why.replace(" trials)", ")")
            txts = [f"s{s_miss} {why}", f"s{s_miss} {short}", f"s{s_miss} {short.replace('running, ', '')}",
                    f"s{s_miss} pending"]
            room = {+1: (XLIM_D[1] - anchor) * ppu - pad, -1: (anchor - XLIM_D[0]) * ppu - pad}
            sgn = None
            for t in txts:              # longest text that fits; prefer the right-hand side
                fits = [d for d in (+1, -1) if text_w(t) <= room[d]]
                if fits:
                    sgn, txts = fits[0], [t]
                    break
            if sgn is None:
                sgn, txts = max(room, key=room.get), [txts[-1]]
            value_label(ax, anchor, yy, s_have, -sgn)
        sw_w = SWATCH_PT / ppu
        x_sw = sgn * LABEL_DX - (SWATCH_PT if sgn < 0 else 0.0)
        ax.add_patch(Rectangle((anchor, yy - 0.22), sw_w, 0.44, facecolor=S.PENDING_FACE, edgecolor=S.PENDING_EDGE,
                               hatch=S.PENDING_HATCH, linewidth=S.LW_THIN, zorder=5, clip_on=False,
                               transform=offset_copy(ax.transData, fig=fig, x=x_sw, y=0, units="points")))
        ax.text(anchor, yy, txts[0], va="center", ha="left" if sgn > 0 else "right", fontsize=S.FONT_SMALL,
                color=S.TEXT_SECONDARY, zorder=5,
                transform=offset_copy(ax.transData, fig=fig, x=sgn * pad, y=0, units="points"))


def dumbbell(ax, cells, big, small):
    """Draw the dumbbell rows; returns the rows with a pending seed (labelled later by place_pending)."""
    fig = ax.figure
    pending = []
    ys, yticks, ylabels, headers = {}, [], [], []
    y = 0.0
    for model, gen in MODELS:
        headers.append((y, model, gen))
        y += 1.0
        for m in METHODS:
            ys[(model, m)] = y
            yticks.append(y)
            ylabels.append(S.label(m))
            y += 1.0
        y += 0.35
    y_end = y - 0.35

    # highlight the two rows shown on the right
    for c in (big, small):
        if c is not None:
            yy = ys[(c["model"], c["method"])]
            ax.axhspan(yy - 0.44, yy + 0.44, xmin=0, xmax=1, color=S.NEUTRAL_MID, lw=0, zorder=0)

    for c in cells:
        yy, m, gen = ys[(c["model"], c["method"])], c["method"], c["gen"]
        g = S.GEN_STYLE[gen]
        vals = {s: c[f"s{s}"] for s in SEEDS}
        have = [s for s in SEEDS if not math.isnan(vals[s])]
        if len(have) == 2:
            lo_s, hi_s = sorted(SEEDS, key=lambda s: (vals[s], s))
            ax.plot([vals[lo_s], vals[hi_s]], [yy, yy], color=S.color(m), linestyle=g["linestyle"], lw=1.2,
                    zorder=2, solid_capstyle="butt")
            sides = {lo_s: -1, hi_s: +1}
        else:
            sides = {}
        # markers: seed 2 first so seed 1 stays on top where they overlap
        for s in sorted(have, reverse=True):
            ax.plot([vals[s]], [yy], zorder=3 + (s == 1), clip_on=False, **seed_kw(m, gen, s))
        if len(have) == 1:          # one seed pending: its label side is chosen after layout (place_pending)
            pending.append((c, yy, have[0]))
            continue
        if not have:
            pending.append((c, yy, None))
        # value labels on the outer side of each marker: "s2 20 o----o 55 s1" (as fig3: tag outside, value next
        # to the marker; seed-2 labels in secondary ink)
        for s, sgn in sides.items():
            value_label(ax, vals[s], yy, s, sgn)

    # |s1 - s2| column, right of the plot area
    col = offset_copy(ax.get_yaxis_transform(), fig=fig, x=8, y=0, units="points")
    for c in cells:
        yy = ys[(c["model"], c["method"])]
        if math.isnan(c["gap"]):
            ax.text(1.0, yy, "pending", transform=col, ha="left", va="center", fontsize=S.FONT_SMALL,
                    color=S.TEXT_SECONDARY)
            continue
        tag = "  largest" if c is big else ("  smallest" if c is small else "")
        ax.text(1.0, yy, f"{c['gap']:.0f}{tag}", transform=col, ha="left", va="center", fontsize=S.FONT_SMALL,
                color=S.TEXT)

    for yh, model, gen in headers:
        rel = "older" if gen.endswith("old") else "newer"
        ax.text(0.0, yh, f"{S.MODEL_DISPLAY[model]}  ({rel} release)", transform=ax.get_yaxis_transform(),
                ha="left", va="center", fontsize=S.FONT, color=S.TEXT)
    ax.text(1.0, headers[0][0], "|s1 − s2|", transform=col, ha="left", va="center", fontsize=S.FONT_SMALL,
            color=S.TEXT_SECONDARY)

    ax.set_ylim(y_end - 0.45, -0.6)
    ax.set_yticks(yticks, ylabels)
    ax.tick_params(axis="y", length=0, pad=2, labelcolor=S.TEXT)
    ax.set_xlim(*XLIM_D)
    ax.set_xticks(range(0, 101, 20))
    ax.spines["bottom"].set_bounds(0, 100)
    ax.spines["left"].set_visible(False)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Dev keyword refusals of the selected trial\n(of 100, harmful_behaviors test[:100]; KL ≤ 0.1)")
    return pending


# ================================================================================================ fronts
def front_panel(ax, trials, cell, title):
    m, gen, model = cell["method"], cell["gen"], cell["model"]
    g = S.GEN_STYLE[gen]
    t = trials[(trials.model == model) & (trials.method == m)]
    for s in sorted(SEEDS, reverse=True):          # seed 1 drawn last, on top
        ts = t[t.seed == s]
        f = ts[ts.on_pareto_front].sort_values(["kl", "refusals"])
        if f.empty:
            continue
        col = S.color(m) if s == 1 else tint(S.color(m))
        ax.step(f.kl, f.refusals, where="post", color=col, linestyle=g["linestyle"], lw=S.LW, zorder=3 + (s == 1))
        ax.plot(f.kl, f.refusals, zorder=3.5 + (s == 1), **seed_kw(m, gen, s, size=FRONT_MS))
        sel = ts[ts["is_selected_kl0.1"]]
        ax.plot(sel.kl, sel.refusals, linestyle="none", marker="o", markersize=RING_MS, markerfacecolor="none",
                markeredgecolor=col, markeredgewidth=RING_MEW, zorder=4.5)
    ax.axvline(KL_MAX, color=S.TEXT_SECONDARY, lw=0.9, linestyle=(0, (1, 1.5)), zorder=2)
    ax.set_title(title, loc="left", pad=3, fontsize=S.FONT)

    # selected values, stacked in the lower-left corner (below and left of every front: free of data), as fig2
    fig = ax.figure
    rows = [(s, f"seed {s}: {cell[f's{s}']:.0f}") for s in reversed(SEEDS)] + [(None, "selected:")]
    x0, y0, lh = 0.022, 0.035, 8.2
    for i, (s, txt) in enumerate(rows):
        if s is None:
            ax.text(x0, y0, txt, transform=offset_copy(ax.transAxes, fig=fig, x=0, y=i * lh, units="points"),
                    fontsize=S.FONT_SMALL, color=S.TEXT_SECONDARY, ha="left", va="bottom", zorder=7)
            continue
        glyph = offset_copy(ax.transAxes, fig=fig, x=3.2, y=i * lh + 3.1, units="points")
        ax.plot([x0], [y0], transform=glyph, zorder=7, clip_on=False, **seed_kw(m, gen, s))
        ax.text(x0, y0, txt, transform=offset_copy(ax.transAxes, fig=fig, x=9.0, y=i * lh, units="points"),
                fontsize=S.FONT_SMALL, color=S.TEXT if s == 1 else S.TEXT_SECONDARY, ha="left", va="bottom",
                zorder=7)


def style_fronts(axes, trials, cells_shown):
    kl = trials[trials.model.isin([c["model"] for c in cells_shown])
                & trials.method.isin([c["method"] for c in cells_shown]) & trials.on_pareto_front].kl
    kl = kl[kl > 0]
    lo = kl.min() / 1.6 if len(kl) else 1e-4
    hi = 10.0 ** math.ceil(math.log10(max(kl.max(), KL_MAX * 1.5))) if len(kl) else 1.0
    decades = [10.0 ** e for e in range(math.ceil(math.log10(lo)), int(round(math.log10(hi))) + 1)]
    fmt = {1e-4: "0.0001", 1e-3: "0.001", 1e-2: "0.01", 1e-1: "0.1", 1.0: "1", 10.0: "10"}
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlim(lo, hi)
        ax.set_ylim(Y_LO, Y_HI)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.xaxis.set_major_locator(FixedLocator(decades))
        ax.xaxis.set_minor_locator(FixedLocator([k * d for d in [decades[0] / 10] + decades for k in range(2, 10) if lo < k * d < hi]))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xticklabels([fmt.get(d, f"{d:g}") for d in decades])
        ax.tick_params(axis="x", which="minor", length=1.5, width=S.LW_THIN)
        ax.set_ylabel("Dev keyword refusals\n(of 100)")
    for ax in axes[:-1]:
        ax.tick_params(axis="x", labelbottom=False)
    # threshold label at the top of the line, on its right: above every front there (fronts fall to low refusals
    # before KL reaches 0.1), so it hides no data
    for ax in axes:
        ax.annotate(f"KL = {KL_MAX:g}", (KL_MAX, 1.0), xycoords=ax.get_xaxis_transform(), xytext=(2.5, -1),
                    textcoords="offset points", fontsize=S.FONT_SMALL, color=S.TEXT_SECONDARY, ha="left", va="top",
                    zorder=7)
    axes[-1].set_xlabel("First-token KL divergence from the original\n(harmless_alpaca test[:100]; log scale)")


# ================================================================================================ figure
def build(trials, runs, out_dir=None, caption_path=None):
    S.apply()
    for c in ("on_pareto_front", "is_selected_kl0.1", "run_finished"):
        trials[c] = trials[c].astype(str).str.lower().eq("true")
    cells = load(runs, trials)
    big, small = extremes(cells)

    fig = S.plt.figure(figsize=(S.TEXT_W, 3.9))
    fig.get_layout_engine().set(w_pad=0.05, h_pad=0.03, wspace=0.06, hspace=0.04)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.12, 1.0])
    ax_d = fig.add_subplot(gs[:, 0])
    ax_f = [fig.add_subplot(gs[0, 1])]
    ax_f.append(fig.add_subplot(gs[1, 1], sharex=ax_f[0], sharey=ax_f[0]))

    pending = dumbbell(ax_d, cells, big, small)
    shown = []
    for ax, c, what in ((ax_f[0], big, "Largest"), (ax_f[1], small, "Smallest")):
        if c is None:
            S.pending_axes(ax, "pending: needs two finished\nseed pairs")
            ax.set_title(f"{what} seed gap", loc="left", pad=3, fontsize=S.FONT)
            continue
        front_panel(ax, trials, c, f"{what} seed gap: {S.MODEL_DISPLAY[c['model']]} {S.label(c['method'])}")
        shown.append(c)
    style_fronts(ax_f, trials, shown or cells)

    # legend: encoding key in neutral ink (methods are named on the rows and in the panel titles)
    gray = S.TEXT_SECONDARY
    gtint = tint(S.METHOD_COLORS["base"])
    handles = [
        Line2D([], [], linestyle="none", marker="o", markersize=S.MS, markerfacecolor=gray, markeredgecolor=gray,
               markeredgewidth=0.6, label="seed 1"),
        Line2D([], [], linestyle="none", marker="o", markersize=S.MS, markerfacecolor=gtint, markeredgecolor=gtint,
               markeredgewidth=0.6, label="seed 2 (lighter)"),
        Line2D([], [], label="older release (hollow, dashed)",
               **S.line_kw("base", "old", marker="o", color=gray, markeredgecolor=gray)),
        Line2D([], [], label="newer release (filled, solid)",
               **S.line_kw("base", "new", marker="o", color=gray, markeredgecolor=gray, markerfacecolor=gray)),
        Line2D([], [], linestyle="none", marker="o", markersize=RING_MS, markerfacecolor="none",
               markeredgecolor=gray, markeredgewidth=RING_MEW, label="selected trial (KL ≤ 0.1 rule)"),
        Line2D([], [], color=gray, lw=0.9, linestyle=(0, (1, 1.5)), label="KL = 0.1"),
    ]
    if any(math.isnan(c["gap"]) for c in cells):
        handles.append(Patch(facecolor=S.PENDING_FACE, edgecolor=S.PENDING_EDGE, hatch=S.PENDING_HATCH,
                             linewidth=S.LW_THIN, label="pending"))
    fig.legend(handles=handles, loc="outside upper center", ncols=4, handlelength=2.0, columnspacing=1.4,
               handletextpad=0.4, borderaxespad=0.1)
    if pending:
        fig.canvas.draw()           # layout pass, so place_pending knows the final axis scale
        place_pending(ax_d, pending)
    S.save(fig, NAME, out_dir=out_dir, preview_dpi=150)
    return cells, big, small


def signed(x):
    return "0" if x == 0 else f"{x:+.0f}".replace("-", "$-$")


def budget_label(t, x, mins):
    """Print the budget x (a trial's KL) ROUNDED UP to the fewest significant figures at which both seeds' minima
    are unchanged, so 'KL <= printed value' is true as written (rounding 0.006546 down to 0.0065 would drop the
    trial that sets the minimum)."""
    for sig in range(2, 8):
        e = math.floor(math.log10(x)) - (sig - 1)
        thr = round(math.ceil(round(x / 10 ** e, 9)) * 10 ** e, 12)
        if tuple(t[(t.seed == s) & (t.kl <= thr)].refusals.min() for s in SEEDS) == tuple(mins):
            return f"{thr:.{max(0, -e)}f}"
    return repr(float(x))


def front_note(trials, cell, which):
    """One sentence on the two seeds' fronts of a highlighted cell (budget use, threshold, front gap)."""
    t = trials[(trials.model == cell["model"]) & (trials.method == cell["method"])]
    name = f"{S.MODEL_DISPLAY[cell['model']]} {S.label(cell['method'])}"
    n_in = {s: int(((t.seed == s) & (t.kl <= KL_MAX)).sum()) for s in SEEDS}
    n_all = {s: int((t.seed == s).sum()) for s in SEEDS}
    txt = (f"{name} ({which} difference): {n_in[1]} of {n_all[1]} seed-1 and {n_in[2]} of {n_all[2]} seed-2 trials "
           f"lie within KL $\\le$ {KL_MAX:g}")
    hi = max(SEEDS, key=lambda s: cell[f"s{s}"])
    beyond = t[(t.seed == hi) & t.on_pareto_front & (t.kl > KL_MAX)].sort_values("kl")
    if which == "largest" and len(beyond):
        b = beyond.iloc[0]
        txt += (f"; the first seed-{hi} front point beyond the threshold has {int(b.refusals)} refusals at "
                f"KL {b.kl:.3f}")
        return txt + "."
    # equal end points can hide different fronts: largest difference between the two fronts at any budget <= KL_MAX
    grid = np.unique(t[(t.kl <= KL_MAX) & t.on_pareto_front].kl)
    best = {s: [t[(t.seed == s) & (t.kl <= g)].refusals.min() for g in grid] for s in SEEDS}
    diffs = [abs(a - b) if a == a and b == b else -1 for a, b in zip(best[1], best[2])]
    if len(diffs) and max(diffs) > 0:
        i = int(np.argmax(diffs))
        thr = budget_label(t, grid[i], (best[1][i], best[2][i]))
        txt += (f"; at smaller budgets the two fronts differ by up to {diffs[i]:.0f} refusals (KL $\\le$ "
                f"{thr}: {best[1][i]:.0f} with seed 1, {best[2][i]:.0f} with seed 2)")
    return txt + "."


def caption(cells, big, small, trials, runs):
    done = [c for c in cells if not math.isnan(c["gap"])]
    pend = [c for c in cells if math.isnan(c["gap"])]
    cell_name = lambda c: f"{S.MODEL_DISPLAY[c['model']]} {S.label(c['method'])}"   # noqa: E731
    gaps = [c["gap"] for c in done]
    n_ge10 = sum(g >= 10 for g in gaps)

    pending_txt = ""
    if pend:
        pending_txt = (f"{len(pend)} of {len(cells)} cells are pending (seed-2 run not finished; hatched label "
                       "gives the trial count) and are excluded from the summary. ")
    ext_txt = ""
    if big is not None:
        ext_txt = (f"The right panels show both seeds' Pareto fronts for the cell with the largest difference "
                   f"({cell_name(big)}, {big['s1']:.0f} vs {big['s2']:.0f})"
                   + (f" and the cell with the smallest ({cell_name(small)}, {small['s1']:.0f} vs "
                      f"{small['s2']:.0f})" if small is not None else "") + ". ")

    cap = (
        "**Seed sensitivity of the selected trial (dev set, small pair).** Each row is one model $\\times$ method "
        "cell of the small Qwen pair, optimised twice with 100 TPE trials that differ only in the seed (same "
        "configuration, Heretic commit and RTX 4090). The two markers give the dev keyword refusals (of 100) of "
        "the trial chosen by the selection rule (fewest dev keyword refusals at first-token KL $\\le$ 0.1) in the "
        "seed-1 run (full colour) and the seed-2 run (lighter); the column on the right is their absolute "
        "difference. Hollow markers and dashed lines mark the older model (Qwen2.5-7B), filled markers and solid "
        "lines the newer one (Qwen3.5-4B). " + pending_txt + ext_txt +
        "In the fronts, the rings mark each seed's selected trial and the dotted line the KL = 0.1 threshold. "
        "All counts are keyword matches on the 100 dev prompts used for optimisation and selection, not held-out "
        "refusal rates; seed 1 is the run evaluated elsewhere in the paper."
    )

    notes = []
    if done:
        notes.append(f"{len(done)} of {len(cells)} cells have both seeds finished (snapshot {S.snapshot_utc()}); "
                     f"|s1 $-$ s2| ranges from {min(gaps):.0f} to {max(gaps):.0f} refusals (median "
                     f"{statistics.median(gaps):g}, mean {statistics.fmean(gaps):.1f}); {n_ge10} of {len(done)} "
                     "cells differ by 10 or more.")
        notes.append("Per cell (seed 1 vs seed 2): " + "; ".join(
            f"{cell_name(c)} {c['s1']:.0f} vs {c['s2']:.0f}" for c in done) + ".")
        base = {mo: sorted(set(runs[(runs.model == mo) & runs.method.isin(METHODS)].baseline_refusals.dropna()))
                for mo, _ in MODELS}
        vals = [v[0] for v in base.values() if len(v) == 1]
        if len(vals) == len(base) and len(set(vals)) == 1:
            notes.append(f"Both unedited models refuse {vals[0]:.0f} of the 100 dev prompts.")
        elif len(vals) == len(base):
            notes.append("The unedited models refuse " + " and ".join(
                f"{v:.0f} ({S.MODEL_DISPLAY[mo]})" for mo, v in zip(base, vals)) + " of the 100 dev prompts.")
        # what the fronts show for the two highlighted cells
        if big is not None:
            notes.append(front_note(trials, big, "largest"))
        if small is not None:
            notes.append(front_note(trials, small, "smallest"))
        # does the old -> new change keep its direction across seeds?
        parts, rev, gone = [], [], []
        for m in METHODS:
            o = next(c for c in cells if c["method"] == m and c["gen"].endswith("old"))
            n = next(c for c in cells if c["method"] == m and c["gen"].endswith("new"))
            if any(math.isnan(x) for x in (o["s1"], o["s2"], n["s1"], n["s2"])):
                continue
            d1, d2 = n["s1"] - o["s1"], n["s2"] - o["s2"]
            parts.append(f"{S.label(m)} {signed(d1)} (seed 1) vs {signed(d2)} (seed 2)")
            if d1 * d2 < 0:
                rev.append(S.label(m))
            elif (d1 == 0) != (d2 == 0):
                gone.append(S.label(m))
        if parts:
            notes.append("Change from Qwen2.5-7B to Qwen3.5-4B in dev refusals (newer minus older): "
                         + "; ".join(parts) + ". "
                         + (f"The direction reverses between seeds for {' and '.join(rev)}" if rev else
                            "The direction is the same with both seeds for every method")
                         + (f", and for {' and '.join(gone)} the change is zero with one seed" if gone else "")
                         + ".")
    if pend:
        notes.append("Pending: " + "; ".join(f"{cell_name(c)} seed {s} {c[f'why{s}']}" for c in pend
                                             for s in SEEDS if c[f"why{s}"]) + ".")
    if notes:
        cap += "\n\n*For the text:* " + " ".join(notes)
    src = (S.SOURCE_DEV.replace("(100 trials, seed 1)", "(100 trials per run; seeds 1 and 2, small pair, RTX 4090)")
           + " " + S.SOURCE_SELECTION + " Values from paper/figures/data/runs.csv (`selected_refusals`), fronts from "
           "trials.csv (`on_pareto_front`).")
    return cap, src


def main():
    trials, runs = S.load_trials(), S.load_runs()
    cells, big, small = build(trials, runs)
    cap, src = caption(cells, big, small, trials, runs)
    S.write_caption(NAME, cap, source=src)
    for c in cells:
        print(f"  [fig8] {c['model']:<11} {c['method']:<6} s1 {c['s1']:>4.0f}  s2 {c['s2']:>4.0f}  |d| "
              f"{c['gap']:>4.0f}  {c['why2'] or c['why1']}")


if __name__ == "__main__":
    main()
