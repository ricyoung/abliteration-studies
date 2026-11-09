"""fig3_generation_effect -- dev-set keyword refusals of the selected trials, older vs newer release.

Slope charts in two rows.
  Top row (main study, four 22-32B pairs): one panel per method (DiM, MPOA, ARA, SOM). The gamma-MPOA control
    (Gemma and OLMo pairs only) used to be a fifth, narrow panel here; it was moved out to give the four core panels
    more width (its selected counts, old and new, are the KL <= 0.1 column of the tables in fig9_gamma_mpoa, and the
    caption's for-the-text part keeps its old -> new changes; SHOW_GAMMA = True draws the panel again). In each panel one line per family joins the older model
    (x = old, hollow marker) to its successor (x = new, filled marker). Family tags (Q, G, O, M) are printed at BOTH
    ends; markers at the same end that would overlap are nudged sideways towards the middle.
  Bottom row (small pair, Qwen2.5-7B -> Qwen3.5-4B): the same for DiM, MPOA, ARA, SOM and J-space, one line per panel.
    Seed-2 runs are NOT drawn here (they crowded the small panels once all ten finished); fig8 compares the seeds and
    the caption summarises the seed-2 differences. Set SHOW_SEED2 = True to draw them again (lighter markers, "s2").
y = number of dev keyword refusals (of 100) left by the selected trial (fewest refusals at first-token KL <= 0.1,
seed 1). The grey band spans the ORIGINAL (unedited) models' dev refusals of that row; exact values under the panels.
These are dev-set (search and selection) numbers: they are optimistically biased and single-seed; the hypothesis tests
use held-out judged refusal.

Data: paper/figures/data/runs.csv (selected_refusals / baseline_refusals; seed 2 rows for the small pair),
cross-checked against results.csv (dev_refusals) and trials.csv (is_selected_kl0.1).

Run:  cd scripts/figures && .venv/bin/python fig3_generation_effect.py
"""
from __future__ import annotations

import math
from decimal import ROUND_HALF_UP, Decimal

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.transforms import ScaledTranslation

import style as S

NAME = "fig3_generation_effect"
METHODS = ["dim", "mpoa", "ara", "som"]          # top row (main pairs)
SMALL_METHODS = ["dim", "mpoa", "ara", "som", "jspace"]
GAMMA_FAMILIES = ["Gemma", "OLMo"]               # gamma-MPOA is planned for the post-norm Gemma and OLMo pairs
SEED = 1
KL_MAX = 0.1
TAG = {"Qwen": "Q", "Gemma": "G", "OLMo": "O", "Mistral Small": "M"}
SMALL_FAMILY = "Small pair"
MAIN_FAMILIES = [p[0] for p in S.PAIRS if p[0] != SMALL_FAMILY]
X_OLD, X_NEW = 0.0, 1.0
NUDGE = 0.085                    # x shift (old-new span = 1) for a marker that would overlap a neighbour
LABEL_DX_PT = 6.5                # gap between marker centre and its end label (points)
XLIM = (-0.5, 1.5)               # provisional; fit_xlim() sets the final limits from the label widths
YLIM = (-4, 104)
S2_ALPHA = 0.45                  # seed-2 markers: same shape/colour, lighter
SHOW_SEED2 = False               # seed 2 is compared in fig8; drawing it here crowded the small-pair panels
SHOW_GAMMA = False               # gamma-MPOA is compared in fig9; a fifth top panel made the four core panels narrow


def half_up(x, nd=1):
    """Round half away from zero (so -2.75 -> -2.8 and 20.25 -> 20.3, as a reader would round)."""
    q = Decimal(1).scaleb(-nd)
    return float(Decimal(repr(x)).quantize(q, rounding=ROUND_HALF_UP))


def signed(x, nd=1):
    return f"{half_up(x, nd):+.{nd}f}".replace("-", "\u2212")


# ================================================================================================ data
def load():
    runs_all = S.load_runs()
    res = S.load_results()
    trials = S.load_trials()
    runs = runs_all[(runs_all["seed"] == SEED) & runs_all["method"].isin(SMALL_METHODS + ["gamma_mpoa"])]

    cells = {}      # (family, method) -> dict(old=, new=, old_status=, new_status=)
    base = {}       # family -> (old original refusals, new original refusals)
    seed2 = {}      # (family, method) -> dict(old=, new=)  finished seed-2 runs only
    problems = []
    for fam, old, new in S.PAIRS:
        b = []
        for model in (old, new):
            vals = runs.loc[runs["model"] == model, "baseline_refusals"].dropna().unique()
            if len(vals) > 1:
                problems.append(f"{model}: baseline_refusals differ across runs {vals}")
            r = res[(res["model"] == model) & (res["method"] == "base")]
            rb = r["dev_refusals"].dropna().unique()
            v = float(vals[0]) if len(vals) else (float(rb[0]) if len(rb) else math.nan)
            if len(vals) and len(rb) and float(rb[0]) != v:
                problems.append(f"{model}: original refusals runs.csv {v} != results.csv {rb[0]}")
            b.append(v)
        base[fam] = tuple(b)
        for m in SMALL_METHODS + (["gamma_mpoa"] if fam in GAMMA_FAMILIES else []):
            cell, c2 = {}, {}
            for gen, model in (("old", old), ("new", new)):
                r2 = runs_all[(runs_all["seed"] == 2) & (runs_all["model"] == model) & (runs_all["method"] == m)
                              & (runs_all["status"].astype(str) == "present")]
                c2[gen] = float(r2["selected_refusals"].iloc[0]) if len(r2) else math.nan
                row = runs[(runs["model"] == model) & (runs["method"] == m)]
                if len(row) != 1:
                    cell[gen], cell[gen + "_status"] = math.nan, "missing: no run row"
                    continue
                row = row.iloc[0]
                v = row["selected_refusals"]
                ok = str(row["status"]) == "present" and str(row["run_finished"]) == "True" and not math.isnan(v)
                cell[gen] = float(v) if ok else math.nan
                cell[gen + "_status"] = str(row["status"]) if not ok else "present"
                if ok:
                    if not float(row["selected_kl"]) <= KL_MAX:
                        problems.append(f"{model} {m}: selected KL {row['selected_kl']} > {KL_MAX}")
                    rr = res[(res["model"] == model) & (res["method"] == m)]["dev_refusals"].dropna()
                    if len(rr) and float(rr.iloc[0]) != v:
                        problems.append(f"{model} {m}: runs.csv {v} != results.csv {rr.iloc[0]}")
                    t = trials[(trials["run"] == row["run"]) & (trials["is_selected_kl0.1"].astype(str) == "True")]
                    if len(t) != 1 or float(t["refusals"].iloc[0]) != v:
                        problems.append(f"{model} {m}: trials.csv selected trial disagrees ({len(t)} rows)")
            cells[(fam, m)] = cell
            seed2[(fam, m)] = c2
    gm = runs_all[(runs_all["method"] == "gamma_mpoa") & (runs_all["seed"] == SEED)]
    gamma = {r.model: str(r.status) for r in gm.itertuples()}     # model -> runs.csv status
    for p in problems:
        print("  [fig3] CHECK FAILED:", p)
    if not problems:
        print("  [fig3] cross-checks OK (runs.csv = results.csv = trials.csv selected trial; all KL <= 0.1)")
    return cells, base, seed2, gamma


# ================================================================================================ labels
def spread(targets, gap, lo, hi):
    """Place labels at y close to `targets` (already sorted ascending) with >= `gap` between neighbours,
    inside [lo, hi]. Clusters of overlapping labels are merged and centred on their mean target."""
    n = len(targets)
    if n == 0:
        return []
    gap = min(gap, (hi - lo) / max(n - 1, 1))
    clusters = [[i] for i in range(n)]

    def place(cl):
        c = float(np.mean([targets[i] for i in cl]))
        start = c - gap * (len(cl) - 1) / 2
        start = min(max(start, lo), hi - gap * (len(cl) - 1))
        return start

    starts = [place(c) for c in clusters]
    changed = True
    while changed:
        changed = False
        for k in range(len(clusters) - 1):
            top_k = starts[k] + gap * (len(clusters[k]) - 1)
            if starts[k + 1] - top_k < gap - 1e-9:
                clusters[k] = clusters[k] + clusters.pop(k + 1)
                starts.pop(k + 1)
                starts[k] = place(clusters[k])
                changed = True
                break
    out = [0.0] * n
    for cl, s in zip(clusters, starts):
        for j, i in enumerate(cl):
            out[i] = s + gap * j
    return out


def end_labels(ax, items, side, gap):
    """items: list of (y_marker, tie_key, text, kind, x_marker); kind 'main' or 's2' (grey).
    side: 'old' (labels left of X_OLD) or 'new' (labels right of X_NEW)."""
    items = sorted(items, key=lambda t: (t[0], t[1]))
    ys = spread([t[0] for t in items], gap, YLIM[0] + 2.5, YLIM[1] - 2.5)
    x_l = X_OLD if side == "old" else X_NEW
    sgn = -1 if side == "old" else 1
    # label anchor = (end x, label y) shifted horizontally by a fixed distance in POINTS, so the gap to the
    # marker does not depend on the x-limits (which fit_xlim() may widen)
    shift = ax.transData + ScaledTranslation(sgn * LABEL_DX_PT / 72, 0, ax.figure.dpi_scale_trans)
    for (y, _, text, kind, x_m), yl in zip(items, ys):
        leader = abs(yl - y) > 1.2 or abs(x_m - x_l) > 1e-9   # thin leader line to a displaced label / marker
        ax.annotate(text, xy=(x_m, y), xycoords="data", xytext=(x_l, yl), textcoords=shift,
                    ha="right" if side == "old" else "left", va="center", gid="endlabel",
                    fontsize=S.FONT_SMALL, color=S.TEXT_SECONDARY if kind == "s2" else S.TEXT, zorder=5,
                    annotation_clip=False,
                    arrowprops=({"arrowstyle": "-", "color": S.AXIS, "linewidth": S.LW_THIN, "shrinkA": 0.8,
                                 "shrinkB": S.MS * 0.6, "relpos": (1, 0.5) if side == "old" else (0, 0.5)}
                                if leader else None))


def nudge(points, thresh):
    """points: list of (key, y). Return key -> x offset (towards the middle of the panel) so that markers at the
    same end whose y values are closer than `thresh` do not sit on top of each other."""
    pts = sorted(points, key=lambda t: t[1])
    off, cluster = {}, []
    for key, y in pts + [(None, math.inf)]:
        if cluster and y - cluster[-1][1] >= thresh:
            # seed-1 markers keep their place first; seed-2 markers are the ones moved
            order = sorted(cluster, key=lambda t: t[0][1] if isinstance(t[0], tuple) else 1)
            for k, (kk, _) in enumerate(order):
                off[kk] = k * NUDGE
            cluster = []
        if key is not None:
            cluster.append((key, y))
    return off


def fit_xlim(fig, axes, pad_pt=3.0):
    """Set the common x-limits so that the end labels just fit inside every panel and the old-new span
    gets all remaining width. The labels sit a fixed number of points from their marker, which constrained
    layout does not see; without this they either run off the figure edge or waste panel width."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    pad = pad_pt * fig.dpi / 72
    need_l = need_r = 0.0       # pixels needed left of the old marker / right of the new marker
    width = min(ax.get_window_extent(r).width for ax in axes)
    for ax in axes:
        px_old = ax.transData.transform((X_OLD, 0))[0]
        px_new = ax.transData.transform((X_NEW, 0))[0]
        for t in ax.texts:
            if t.get_gid() != "endlabel":
                continue
            bb = t.get_window_extent(r)
            need_l = max(need_l, px_old - bb.x0)
            need_r = max(need_r, bb.x1 - px_new)
    unit = width - need_l - need_r - 2 * pad      # pixels per data unit (old -> new span)
    if unit <= 0:
        raise RuntimeError(f"end labels do not fit in the panel width (width {width * 72 / fig.dpi:.0f} pt, need {need_l * 72 / fig.dpi:.0f} + {need_r * 72 / fig.dpi:.0f} pt)")
    lo, hi = X_OLD - (need_l + pad) / unit, X_NEW + (need_r + pad) / unit
    for ax in axes:
        ax.set_xlim(lo, hi)
    print(f"  [fig3] old-new span {unit * 72 / fig.dpi:.0f} pt of {width * 72 / fig.dpi:.0f} pt panel width")


# ================================================================================================ plot
def status_short(status):
    """runs.csv status -> short pending reason, e.g. 'partial: run running (48/100 trials)' -> 'running, 48/100'."""
    st = str(status)
    m = __import__("re").search(r"\((\d+)/(\d+) trials\)", st)
    if "running" in st:
        return f"running, {m.group(1)}/{m.group(2)}" if m else "running"
    if st.startswith("planned") or "not started" in st or "queued" in st:
        return "not started"
    return st.split(":")[0]


def slope_panel(ax, cells, seed2, base_rng, method, families, gap, thresh, show_tags):
    """One method panel. Returns (list of new-minus-old deltas, list of pending entries)."""
    c = S.color(method)
    lo, hi = base_rng
    if not math.isnan(lo):
        ax.axhspan(lo - 0.6, hi + 0.6, facecolor=S.METHOD_COLORS["base"], alpha=0.16, edgecolor="none", zorder=0.5)
    pts = {"old": [], "new": []}          # every marker at each end (seed 1 and seed 2), for the nudge
    for fam in families:
        cell = cells[(fam, method)]
        for gen in ("old", "new"):
            if not math.isnan(cell[gen]):
                pts[gen].append(((fam, 1), cell[gen]))
            v2 = seed2.get((fam, method), {}).get(gen, math.nan)
            if not math.isnan(v2):
                pts[gen].append(((fam, 2), v2))
    off = {gen: nudge(pts[gen], thresh) for gen in pts}
    xs = {}
    for gen, x0, sgn in (("old", X_OLD, 1), ("new", X_NEW, -1)):
        for (fam, seed), _ in pts[gen]:
            xs[(fam, gen, seed)] = x0 + sgn * off[gen].get((fam, seed), 0.0)
    left, right, deltas, pend = [], [], [], []
    for fam in families:
        cell = cells[(fam, method)]
        yo, yn = cell["old"], cell["new"]
        tag = TAG.get(fam, "")
        if not math.isnan(yo) and not math.isnan(yn):
            ax.plot([xs[(fam, "old", 1)], xs[(fam, "new", 1)]], [yo, yn], color=c, linewidth=S.LW, linestyle="-",
                    solid_capstyle="round", zorder=2)
            deltas.append(yn - yo)
        for gen, y, other in (("old", yo, yn), ("new", yn, yo)):
            if math.isnan(y):
                pend.append((X_OLD if gen == "old" else X_NEW, fam, gen, cell[gen + "_status"]))
                continue
            ring = {"markeredgecolor": S.SURFACE, "markeredgewidth": 0.7} if gen == "new" else {}
            ax.plot([xs[(fam, gen, 1)]], [y], zorder=3 if gen == "new" else 3.1,
                    **S.marker_kw(method, gen, size=S.MS + (1.2 if ring else 0), **ring))
            if show_tags:
                txt = f"{tag} {y:.0f}" if gen == "old" else f"{y:.0f} {tag}"
            else:
                txt = f"{y:.0f}"
            tie = other if not math.isnan(other) else y
            (left if gen == "old" else right).append((y, tie, txt, "main", xs[(fam, gen, 1)]))
        # finished seed-2 runs: lighter marker at the same end, label 's2'
        for gen in ("old", "new"):
            v2 = seed2.get((fam, method), {}).get(gen, math.nan)
            if math.isnan(v2):
                continue
            x2 = xs[(fam, gen, 2)]
            ax.plot([x2], [v2], zorder=2.9, alpha=S2_ALPHA, **S.marker_kw(method, gen, size=S.MS))
            txt = f"s2 {v2:.0f}" if gen == "old" else f"{v2:.0f} s2"
            (left if gen == "old" else right).append((v2, v2, txt, "s2", x2))
    end_labels(ax, left, "old", gap)
    end_labels(ax, right, "new", gap)
    for i, (x, fam, gen, status) in enumerate(pend):   # never silently drop a missing cell
        if method == "gamma_mpoa":
            # narrow panel: the status text goes UNDER the pending marker, centred on its end, so it needs no
            # more width than an end label
            S.pending_marker(ax, x, 50 - 30 * i, text=None)
            ax.annotate(f"{TAG.get(fam, fam)}: pending\n(" + status_short(status).replace(", ", ",\n") + ")",
                        (x, 50 - 30 * i), xytext=(0, -5), textcoords="offset points", ha="center", va="top",
                        fontsize=S.FONT_SMALL, color=S.TEXT_SECONDARY, linespacing=1.1, gid="endlabel")
        else:
            S.pending_marker(ax, x, 50 - 8 * i, text=f"{TAG.get(fam, fam)}: pending")
            if x == X_NEW:
                ax.texts[-1].set_gid("endlabel")
        print(f"  [fig3] pending: {fam} {gen} {method}: {status}")
    return deltas, pend


def style_axes(ax, first, ylabel):
    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    ax.set_xticks([X_OLD, X_NEW], ["old", "new"])
    ax.tick_params(axis="x", length=0, pad=3, labelcolor=S.TEXT)
    ax.grid(axis="x", visible=False)
    ax.spines["left"].set_bounds(0, 100)
    ax.spines["bottom"].set_visible(False)
    ax.set_yticks(range(0, 101, 20))
    if first:
        ax.set_ylabel(ylabel)
    else:
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0, labelleft=False)


def main():
    S.apply()
    cells, base, seed2_all, gamma = load()
    seed2 = seed2_all if SHOW_SEED2 else {}
    b_main = [v for f in MAIN_FAMILIES for v in base[f] if not math.isnan(v)]
    rng_main = (min(b_main), max(b_main)) if b_main else (math.nan, math.nan)
    b_small = [v for v in base[SMALL_FAMILY] if not math.isnan(v)]
    rng_small = (min(b_small), max(b_small)) if b_small else (math.nan, math.nan)

    fig = S.plt.figure(figsize=(S.TEXT_W, 4.75 if SHOW_GAMMA else 4.95))
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.03, wspace=0.03, hspace=0.02)
    sf_top, sf_bot = fig.subfigures(2, 1, height_ratios=[1.0, 0.70] if SHOW_GAMMA else [1.2, 0.68], hspace=0.0)
    wr = [1, 1, 1, 1, 0.8]
    wr_top = [1, 1, 1, 1, 1.05] if SHOW_GAMMA else [1, 1, 1, 1]   # gamma panel: two families + a pending note
    axes_top = sf_top.subplots(1, len(wr_top), sharey=True, gridspec_kw={"width_ratios": wr_top})
    axes_bot = sf_bot.subplots(1, 5, sharey=True, gridspec_kw={"width_ratios": wr})
    sf_top.suptitle("Main pairs (22–32B), optimised on an H200", x=0.005, ha="left", fontsize=S.FONT,
                    fontweight="bold")
    sf_top.text(0.995, 0.985, "family tags:  Q Qwen,  G Gemma,  O OLMo,  M Mistral Small", ha="right", va="top",
                fontsize=S.FONT_SMALL, color=S.TEXT_SECONDARY)
    sf_bot.suptitle("Small pair: Qwen2.5-7B → Qwen3.5-4B, optimised on an RTX 4090", x=0.005, ha="left",
                    fontsize=S.FONT, fontweight="bold")
    fig.canvas.draw()

    def gap_and_thresh(ax):   # label spacing (7 pt text + 1.6 pt air) and marker size, in data units
        h_pt = ax.get_window_extent().height * 72 / fig.dpi
        return (S.FONT_SMALL + 1.6) * (YLIM[1] - YLIM[0]) / h_pt, (S.MS + 0.8) * (YLIM[1] - YLIM[0]) / h_pt

    summary = {}
    gap, thresh = gap_and_thresh(axes_top[0])
    for i, (ax, m) in enumerate(zip(axes_top[:4], METHODS)):
        deltas, _ = slope_panel(ax, cells, seed2, rng_main, m, MAIN_FAMILIES, gap, thresh, show_tags=True)
        mean_d = float(np.mean(deltas)) if len(deltas) == len(MAIN_FAMILIES) else math.nan
        summary[m] = (deltas, mean_d)
        ax.set_title(S.label(m), loc="center", fontsize=S.FONT, fontweight="bold", pad=12)
        sub = f"mean change: {signed(mean_d)}" if not math.isnan(mean_d) else "mean change: pending"
        ax.text(0.5, 1.012, sub, transform=ax.transAxes, ha="center", va="bottom", fontsize=S.FONT_SMALL,
                color=S.TEXT_SECONDARY)
        style_axes(ax, i == 0, "Dev keyword refusals (of 100)\nhigher = removal harder")
    # gamma-MPOA: Gemma and OLMo pairs only; unfinished ends are pending markers (status from runs.csv)
    if SHOW_GAMMA:
        gax = axes_top[4]
        slope_panel(gax, cells, seed2, rng_main, "gamma_mpoa", GAMMA_FAMILIES, gap, thresh, show_tags=True)
        gax.set_title(S.label("gamma_mpoa"), loc="center", fontsize=S.FONT, fontweight="bold", pad=12)
        gsub = []
        for fam in GAMMA_FAMILIES:
            c = cells[(fam, "gamma_mpoa")]
            gsub.append(f"{TAG[fam]} " + ("pending" if math.isnan(c["old"]) or math.isnan(c["new"])
                                          else signed(c["new"] - c["old"], 0)))
        gax.text(0.5, 1.012, "change: " + ", ".join(gsub), transform=gax.transAxes, ha="center", va="bottom",
                 fontsize=S.FONT_SMALL, color=S.TEXT_SECONDARY)
        style_axes(gax, False, "")
    print(f"  [fig3] gamma-MPOA: " + "; ".join(f"{fam} {cells[(fam, 'gamma_mpoa')]['old']} -> "
                                               f"{cells[(fam, 'gamma_mpoa')]['new']}" for fam in GAMMA_FAMILIES)
          + f"  (runs.csv: {gamma})")

    gap_s, thresh_s = gap_and_thresh(axes_bot[0])
    small = {}
    for i, (ax, m) in enumerate(zip(axes_bot, SMALL_METHODS)):
        deltas, _ = slope_panel(ax, cells, seed2, rng_small, m, [SMALL_FAMILY], gap_s, thresh_s, show_tags=False)
        small[m] = deltas[0] if deltas else math.nan
        ax.set_title(S.label(m), loc="center", fontsize=S.FONT, fontweight="bold", pad=12)
        ax.text(0.5, 1.012, f"change: {signed(small[m], 0)}" if not math.isnan(small[m]) else "change: pending",
                transform=ax.transAxes, ha="center", va="bottom", fontsize=S.FONT_SMALL, color=S.TEXT_SECONDARY)
        style_axes(ax, i == 0, "Dev keyword refusals\n(of 100)")

    # ---- legend (encoding key, neutral ink) on top
    drawn = ([(f, m) for f in MAIN_FAMILIES for m in METHODS]
             + ([(f, "gamma_mpoa") for f in GAMMA_FAMILIES] if SHOW_GAMMA else [])
             + [(SMALL_FAMILY, m) for m in SMALL_METHODS])
    any_pending = any(math.isnan(cells[k][gen]) for k in drawn for gen in ("old", "new"))
    g = S.TEXT_SECONDARY
    handles = [
        Line2D([], [], linestyle="none", marker="o", markersize=S.MS, markerfacecolor=S.SURFACE,
               markeredgecolor=g, markeredgewidth=S.HOLLOW_MEW, label="older release"),
        Line2D([], [], linestyle="none", marker="o", markersize=S.MS, markerfacecolor=g, markeredgecolor=g,
               markeredgewidth=0.6, label="newer release"),
    ] + ([
        Line2D([], [], linestyle="none", marker="o", markersize=S.MS, markerfacecolor=g, markeredgecolor=g,
               alpha=S2_ALPHA, label="seed 2 (s2, finished runs)")] if SHOW_SEED2 else []) + [
    ] + ([
        Line2D([], [], linestyle="none", marker="o", markersize=S.MS, markerfacecolor=S.SURFACE,
               markeredgecolor=S.PENDING_EDGE, markeredgewidth=S.LW_THIN, label="pending (run not finished)")]
         if any_pending else []) + [
        Patch(facecolor=S.METHOD_COLORS["base"], alpha=0.16, edgecolor="none", label="original models (no edit)"),
    ]
    fig.legend(handles=handles, loc="outside upper center", ncols=4, handlelength=1.2, columnspacing=1.6,
               handletextpad=0.4, borderaxespad=0.1)

    # ---- exact original-model refusals under the panels
    parts = []
    for fam, _, _ in S.PAIRS:
        o, n = base[fam]
        fo = "pending" if math.isnan(o) else f"{o:.0f}"
        fn = "pending" if math.isnan(n) else f"{n:.0f}"
        parts.append(f"{TAG.get(fam, 'small pair')} {fo}→{fn}")
    sf_bot.supxlabel("Original models, dev keyword refusals (old→new):  " + ",  ".join(parts),
                     fontsize=S.FONT_SMALL, color=S.TEXT_SECONDARY)

    for _ in range(3):   # the layout engine re-sizes the axes once the labels fit, so iterate to a fixed point
        fit_xlim(fig, list(axes_top[:4]))                         # the four main-pair panels (wider)
        fit_xlim(fig, list(axes_bot[:4]))                         # equal old-new span in the small-pair panels
        fit_xlim(fig, [axes_bot[4]])                              # the narrow J-space panel on its own
        if SHOW_GAMMA:
            fit_xlim(fig, [axes_top[4]])                          # the narrow gamma-MPOA panel on its own
    S.save(fig, NAME, preview_dpi=150)

    # ---- console summary + caption
    for m, (d, md) in summary.items():
        print(f"  [fig3] {S.label(m):5s} deltas (Qwen, Gemma, OLMo, Mistral) = {d}  mean {md:+.2f}")
    print("  [fig3] small pair:", {m: (cells[(SMALL_FAMILY, m)]['old'], cells[(SMALL_FAMILY, m)]['new'])
                                    for m in SMALL_METHODS})

    def tex(x, nd=1):
        return f"{half_up(x, nd):+.{nd}f}".replace("-", "$-$")

    mixed = [S.label(m) for m in METHODS if len(summary[m][0]) == 4
             and min(summary[m][0]) < 0 < max(summary[m][0])]
    s2_diff = [abs(c2[gen] - cells[(fam, m)][gen]) for (fam, m), c2 in seed2_all.items() for gen in ("old", "new")
               if not math.isnan(c2.get(gen, math.nan)) and not math.isnan(cells[(fam, m)][gen])]
    n_s2 = len(S.load_runs().query("seed == 2"))
    if s2_diff:
        s2_txt = (f"on the small pair a second seed ({len(s2_diff)} of {n_s2} seed-2 runs finished) changed the "
                  f"selected count by a median of {np.median(s2_diff):g} and up to {max(s2_diff):.0f} refusals "
                  "(Fig. 8; seed 2 is not drawn here)")
    else:
        s2_txt = "no seed-2 run has finished yet"
    gpend_txt = [f"{S.MODEL_DISPLAY[S.PAIRS[[p[0] for p in S.PAIRS].index(fam)][1 if gen == 'old' else 2]]} "
                 f"pending ({status_short(cells[(fam, 'gamma_mpoa')][gen + '_status'])} trials)"
                 for fam in GAMMA_FAMILIES for gen in ("old", "new") if math.isnan(cells[(fam, "gamma_mpoa")][gen])]
    if SHOW_GAMMA:
        gamma_txt = ("The top-right panel is $\\gamma$-MPOA, a corrected MPOA run on the Gemma and OLMo pairs only "
                     "(compared with MPOA in Fig. 9" + ("; " + ", ".join(gpend_txt) if gpend_txt else "") + ")")
    else:
        gamma_txt = ("$\\gamma$-MPOA, a corrected MPOA run on the Gemma and OLMo pairs only, is not drawn here; its "
                     "selected counts for both releases are the KL $\\le$ 0.1 column of the tables in Fig. 9"
                     + (" (" + ", ".join(gpend_txt) + ")" if gpend_txt else ""))

    caption = (
        "**Dev-set keyword refusals of the selected trials, older vs newer release.** Each line joins an older "
        "model (left, hollow marker) to its successor (right, filled marker) for one refusal-removal method; top: "
        "the four main pairs (Q Qwen, G Gemma, O OLMo, M Mistral Small), bottom: the "
        f"small Qwen pair. {gamma_txt}. The value is the number of the 100 dev prompts that still trigger a refusal keyword for "
        "the trial picked by the selection rule (fewest dev refusals at first-token KL $\\le$ 0.1, seed 1); higher "
        "means that removal was harder; the grey band spans the unedited originals. The same prompts were used "
        "for search and selection, so the counts are optimistically biased, and run-to-run variation can reach "
        f"tens of refusals: {s2_txt}; a repeated Qwen3.8-27B ARA run gave 59 vs 63; "
        "Heretic issue #499 reports one trial at 52 vs 11 refusals on two machines. The differences shown are "
        "descriptive; H1 is tested on held-out judged refusal."
    )
    detail = (
        "*For the text:* mean change (new minus old) over the four main pairs: "
        + ", ".join(f"{S.label(m)} {tex(summary[m][1])}" for m in METHODS) + " refusals"
        + (f"; the per-pair changes differ in sign for {' and '.join(mixed)}" if mixed else "")
        + "; small pair: " + ", ".join(f"{S.label(m)} {tex(small[m], 0)}" for m in SMALL_METHODS
                                        if not math.isnan(small[m]))
        + "; $\\gamma$-MPOA: " + ", ".join(
            f"{fam} " + ("pending" if math.isnan(cells[(fam, 'gamma_mpoa')]['old'])
                         or math.isnan(cells[(fam, 'gamma_mpoa')]['new'])
                         else f"{tex(cells[(fam, 'gamma_mpoa')]['new'] - cells[(fam, 'gamma_mpoa')]['old'], 0)} "
                              f"({cells[(fam, 'gamma_mpoa')]['old']:.0f} $\\to$ {cells[(fam, 'gamma_mpoa')]['new']:.0f})")
            for fam in GAMMA_FAMILIES)
        + ". Main pairs optimised on an H200, the small pair on an RTX 4090."
    )
    runs = S.load_runs()
    r1 = runs[(runs["seed"] == SEED) & runs["method"].isin(SMALL_METHODS)]
    short_runs = [f"{S.MODEL_DISPLAY[r.model]} {S.label(r.method)} ({int(r.n_trials_complete)} completed trials, "
                  f"{int(r.n_trials_failed)} failed)" for r in r1.itertuples()
                  if r.n_trials_complete == r.n_trials_complete and int(r.n_trials_complete) != 100]
    if short_runs:
        detail += " Selection used fewer than 100 trials for " + "; ".join(short_runs) + "."
    print(f"  [fig3] caption {len(caption.split())} words")
    S.write_caption(NAME, caption + "\n\n" + detail,
                    source=S.SOURCE_DEV + " " + S.SOURCE_SELECTION + f" Seed-2 status: snapshot {S.snapshot_utc()}.")


if __name__ == "__main__":
    main()
