#!/usr/bin/env python3
"""fig9_gamma_mpoa.py -- gamma-corrected MPOA vs MPOA on the four post-norm models (paper 33, dev set).

Run with the figures venv from scripts/figures (normally via make_figures.sh):
    scripts/figures/.venv/bin/python scripts/figures/fig9_gamma_mpoa.py

Small multiples: rows = Gemma, OLMo; columns = old / new release (same arrangement as fig2). Per panel and method
(MPOA, gamma-MPOA; seed-1 runs only):
  * y(t) = fewest dev keyword refusals among the run's COMPLETE trials with first-token KL <= t, drawn as a step
    curve over the budget t in [0.01, 1] (log). The curve drops exactly at the run's Pareto-front trials, which
    carry the method marker (old generation: dashed line, hollow markers; new: solid line, filled markers).
  * beyond the largest KL the run ever sampled the value is only carried forward (no trial there): that stretch is
    drawn as a thin, light line after a short end tick, so a flat tail is not read as "no further gain possible".
  * every complete trial in the plotted range as a faint dot (context), and the KL = 0.1 selection budget dotted.
  * a small read-off table (each curve at KL <= 0.1, the selection rule, and KL <= 0.5). Its corner is chosen per
    panel from the data (choose_corner): lower left, upper right, lower right, upper left, the first in which the
    table touches no curve, marker or end tick (and, among those, the fewest faint trial dots). The lower left used
    to be assumed free "by construction", which fails once a curve drops early (OLMo-2-32B gamma-MPOA reaches 10
    refusals at KL <= 0.05).
gamma-MPOA runs that have not finished (runs.csv run_finished False / status partial or planned) are NOT drawn:
the panel says "pending" with the run state. Set SHOW_PARTIAL = True to draw partial runs (thin, for checking only).

Inputs : paper/figures/data/trials.csv, runs.csv
Outputs: paper/figures/fig9_gamma_mpoa.pdf, preview/fig9_gamma_mpoa.png (150 dpi), captions.md section
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import FixedLocator, NullFormatter
from matplotlib.transforms import offset_copy

import style as S

NAME = "fig9_gamma_mpoa"
SEED = 1
METHODS = ["mpoa", "gamma_mpoa"]
PANELS = [("Gemma", "gemma-2-27b", "gemma-4-31b"), ("OLMo", "olmo-2-32b", "olmo-3.1-32b")]
KL_SEL = 0.1                  # selection budget (select_trial.select)
READ_AT = [0.1, 0.5]          # budgets read off in the panel tables
X_LO, X_HI = 0.01, 1.0
Y_LO, Y_HI = -4, 104
CLOUD_SIZE = 2.8              # pt, faint all-trials dots (as fig2)
CLOUD_ALPHA = {"mpoa": 0.30, "gamma_mpoa": 0.30}
STEP_MS = 5.0                 # pt, Pareto-front (step) markers, house minimum 5
TAIL_LW = 0.8                 # pt, carried-forward tail beyond the largest sampled KL
TAIL_ALPHA = 0.55
CORNER_PAD = 0.022            # axes fraction between the read-off table and the axes edge (as fig2's list)
SHOW_PARTIAL = False          # draw unfinished gamma-MPOA runs (debug only; the paper figure marks them pending)


def budget_curve(g, x_lo=X_LO, x_hi=X_HI):
    """Step curve of min refusals within KL budget t on [x_lo, x_hi].

    Returns (x, y, xv, yv, kl_max): post-step polyline up to min(kl_max, x_hi), the drop vertices inside
    (x_lo, x_hi] and the largest KL sampled by the run.
    """
    g = g.sort_values(["kl", "refusals"])
    kl = g.kl.to_numpy(float)
    cm = np.minimum.accumulate(g.refusals.to_numpy(float))
    kl_max = float(kl.max())
    drop = np.r_[True, cm[1:] < cm[:-1]]
    inside = drop & (kl > x_lo) & (kl <= x_hi)
    xv, yv = kl[inside], cm[inside]
    before = kl <= x_lo
    if before.any():
        x0, y0 = [x_lo], [cm[before][-1]]
    else:                                  # no trial within the left edge: the curve starts at the first trial
        x0, y0 = [xv[0]], [yv[0]]
        xv, yv = xv[1:], yv[1:]
    x_end = min(kl_max, x_hi)
    x = np.r_[x0, xv, x_end]
    y = np.r_[y0, yv, (yv[-1] if len(yv) else y0[-1])]
    return x, y, np.asarray(xv), np.asarray(yv), kl_max


def at_budget(g, t):
    s = g[g.kl <= t]
    return int(s.refusals.min()) if len(s) else None


def gamma_state(runs, model):
    """-> ('done' | 'partial' | 'planned', run row or None)."""
    r = runs[(runs.model == model) & (runs.method == "gamma_mpoa") & (runs.seed == SEED)]
    if r.empty:
        return "planned", None
    r = r.iloc[0]
    st = str(r.status)
    if st.startswith("present") and str(r.run_finished).lower() == "true":
        return "done", r
    if st.startswith("partial"):
        return "partial", r
    return "planned", r


def pending_text(state, r):
    if state == "partial":
        return (f"$\\gamma$-MPOA: {S.PENDING_LABEL} (run in progress, "
                f"{int(r.n_trials_complete)}/{int(r.n_trials_planned)} trials)")
    return f"$\\gamma$-MPOA: {S.PENDING_LABEL} (run not started)"


def panel(ax, trials, runs, model, gen):
    t = trials[trials.model == model]
    state, grow = gamma_state(runs, model)
    shown = [m for m in METHODS if m == "mpoa" or state == "done" or (SHOW_PARTIAL and state == "partial")]
    g_style = S.GEN_STYLE[gen]

    # 1. faint clouds first (all trials in range), so curves sit on top of every cloud
    for m in shown:
        tm = t[(t.method == m) & (t.kl >= X_LO) & (t.kl <= X_HI)]
        ax.scatter(tm.kl, tm.refusals, s=CLOUD_SIZE ** 2, marker="o", facecolors=S.color(m), edgecolors="none",
                   alpha=CLOUD_ALPHA[m], zorder=1)

    # 2. budget curves (post-steps), drop vertices with method markers, carried-forward tail
    info = {}
    obst = {"segs": [], "pts": [], "dots": []}          # data coords, for choose_corner()
    for m in shown:
        tm = t[(t.method == m) & (t.kl >= X_LO) & (t.kl <= X_HI)]
        obst["dots"] += list(zip(tm.kl, tm.refusals))
    for m in shown:
        tm = t[t.method == m]
        x, y, xv, yv, kl_max = budget_curve(tm)
        partial = m == "gamma_mpoa" and state != "done"
        lw = S.LW * (0.6 if partial else 1.0)
        ax.step(x, y, where="post", color=S.color(m), linestyle=g_style["linestyle"], linewidth=lw, zorder=3)
        ax.plot(xv, yv, zorder=3.5, **S.marker_kw(m, gen, size=STEP_MS))
        for i in range(len(x) - 1):                   # post-step: horizontal to x[i+1], then vertical
            obst["segs"] += [((x[i], y[i]), (x[i + 1], y[i])), ((x[i + 1], y[i]), (x[i + 1], y[i + 1]))]
        obst["pts"] += list(zip(xv, yv))
        if kl_max < X_HI:
            obst["segs"].append(((kl_max, y[-1]), (X_HI, y[-1])))
            obst["pts"].append((kl_max, y[-1]))
            ax.plot([kl_max, X_HI], [y[-1], y[-1]], color=S.color(m), lw=TAIL_LW, alpha=TAIL_ALPHA,
                    solid_capstyle="butt", zorder=2.5)
            ax.plot([kl_max], [y[-1]], marker="|", markersize=5.5, markeredgewidth=1.2, color=S.color(m),
                    linestyle="none", zorder=3.6)
        info[m] = {"kl_max": kl_max, **{tb: at_budget(tm, tb) for tb in READ_AT}}

    # 3. selection budget
    ax.axvline(KL_SEL, color=S.TEXT_SECONDARY, lw=0.9, linestyle=(0, (1, 1.5)), zorder=2)

    # 4. read-off table: drawn lower left here; main() moves it to a data-free corner once the layout is final
    rows = []
    if "gamma_mpoa" not in shown or state != "done":
        rows.append(("pending", "gamma_mpoa", pending_text(state, grow)))
    for m in reversed(METHODS):
        if m in shown and not (m == "gamma_mpoa" and state != "done"):
            rows.append(("value", m, S.label(m)))
    rows.append(("header", None, "KL $\\leq$"))
    table = {"rows": rows, "info": info, "gen": gen}
    table["artists"] = draw_table(ax, table, CORNER_PAD, CORNER_PAD)

    ax.set_title(S.MODEL_DISPLAY[model], loc="left", pad=3, fontsize=S.FONT)
    return state, grow, info, table, obst


def draw_table(ax, table, ox, oy):
    """Read-off table with its lower-left corner at axes fraction (ox, oy); rows stack upwards in points.
    Returns the artists (so the table can be moved)."""
    fig = ax.figure
    rows, info, gen = table["rows"], table["info"], table["gen"]
    lh = 8.2                                # line height (pt)
    col_lab, col_v = 9.0, [64.0, 84.0]      # pt from the anchor: label start, right edges of the value columns
    arts = []

    def tr(dx, i, dy=0.0):
        return offset_copy(ax.transAxes, fig=fig, x=dx, y=i * lh + dy, units="points")

    def txt(i, dx, s_, color, ha):
        arts.append(ax.text(ox, oy, s_, transform=tr(dx, i), fontsize=S.FONT_SMALL, color=color, ha=ha,
                            va="bottom", zorder=7))

    for i, (kind, m, s_) in enumerate(rows):
        if kind == "header":
            txt(i, col_lab, s_, S.TEXT_SECONDARY, "left")
            for c, tb in zip(col_v, READ_AT):
                txt(i, c, f"{tb:g}", S.TEXT_SECONDARY, "right")
        elif kind == "value":
            arts += ax.plot([ox], [oy], transform=tr(3.2, i, 3.1), zorder=7, clip_on=False,
                            **S.marker_kw(m, gen, size=S.MS))
            txt(i, col_lab, s_, S.TEXT, "left")
            for c, tb in zip(col_v, READ_AT):
                v = info[m][tb]
                txt(i, c, "n/a" if v is None else f"{v}", S.TEXT, "right")
        else:
            arts.append(ax.add_patch(Rectangle((ox, oy), 0.026, 0.06, transform=tr(0, i, 0.4),
                                               facecolor=S.PENDING_FACE, edgecolor=S.PENDING_EDGE,
                                               hatch=S.PENDING_HATCH, linewidth=S.LW_THIN, zorder=7,
                                               clip_on=False)))
            txt(i, col_lab, s_, S.TEXT_SECONDARY, "left")
    for a in arts:
        a.set_in_layout(False)
    return arts


def choose_corner(ax, table, obst, pad_pt=2.0):
    """Move the read-off table to the first corner (lower left, upper right, lower right, upper left) where it
    touches no curve, drop marker or end tick; among such corners the fewest faint trial dots wins only if the
    first one covers dots. Returns the corner name."""
    fig = ax.figure
    r = fig.canvas.get_renderer()
    px = fig.dpi / 72.0
    bb = None
    for a in table["artists"]:
        e = a.get_window_extent(r)
        bb = e if bb is None else bb.union([bb, e])
    axb = ax.get_window_extent(r)
    w, h = bb.width, bb.height
    off_x, off_y = bb.x0 - axb.x0, bb.y0 - axb.y0          # table bbox relative to its anchor (lower left)
    to = ax.transData.transform
    segs = [(to(p), to(q)) for p, q in obst["segs"]]
    pts = [to(p) for p in obst["pts"]]
    dots = [to(p) for p in obst["dots"]]
    pad, mk, dt = pad_pt * px, (STEP_MS / 2 + 0.5) * px, (CLOUD_SIZE / 2) * px
    corners = {"lower left": (off_x, off_y), "upper right": (axb.width - off_x - w, axb.height - off_y - h),
               "lower right": (axb.width - off_x - w, off_y), "upper left": (off_x, axb.height - off_y - h)}
    score = {}
    for name, (bx, by) in corners.items():
        x0, y0, x1, y1 = axb.x0 + bx - pad, axb.y0 + by - pad, axb.x0 + bx + w + pad, axb.y0 + by + h + pad
        hit = sum(1 for p, q in segs if not (max(p[0], q[0]) < x0 or min(p[0], q[0]) > x1
                                             or max(p[1], q[1]) < y0 or min(p[1], q[1]) > y1))
        hit += sum(1 for p in pts if x0 - mk < p[0] < x1 + mk and y0 - mk < p[1] < y1 + mk)
        nd = sum(1 for p in dots if x0 - dt < p[0] < x1 + dt and y0 - dt < p[1] < y1 + dt)
        score[name] = (hit, nd)
    free = [n for n in corners if score[n][0] == 0]
    if free:
        best = free[0] if score[free[0]][1] == 0 else min(free, key=lambda n: score[n][1])
    else:
        best = min(corners, key=lambda n: score[n])
        print(f"  [fig9] WARNING: read-off table overlaps data in every corner of {ax.get_title(loc='left')}: {score}")
    if best != "lower left":
        bx, by = corners[best]
        for a in table["artists"]:
            a.remove()
        table["artists"] = draw_table(ax, table, (bx - off_x + CORNER_PAD * axb.width) / axb.width,
                                      (by - off_y + CORNER_PAD * axb.height) / axb.height)
    print(f"  [fig9] {ax.get_title(loc='left')}: read-off table {best} (curve/marker hits, dots per corner: {score})")
    return best


def main():
    S.apply()
    trials = S.load_trials()
    runs = S.load_runs()
    models = [m for _, o, n in PANELS for m in (o, n)]
    trials = trials[(trials.seed == SEED) & trials.method.isin(METHODS) & trials.model.isin(models)].copy()
    trials["run_finished"] = trials.run_finished.astype(str).str.lower().eq("true")
    assert trials[trials.method == "mpoa"].run_finished.all(), "an MPOA seed-1 run is not finished"
    assert (trials.n_prompts == 100).all()

    fig, axes = S.figure("double", height=3.9, nrows=2, ncols=2, sharex=True, sharey=True)
    fig.get_layout_engine().set(h_pad=0.03, w_pad=0.04, hspace=0.05, wspace=0.03)

    out, ax_of = {}, {}
    for r, (fam, old, new) in enumerate(PANELS):
        for c, (model, gen) in enumerate([(old, "old"), (new, "new")]):
            out[model] = panel(axes[r, c], trials, runs, model, gen)
            ax_of[model] = axes[r, c]

    ax0 = axes[0, 0]
    ax0.set_xscale("log")
    ax0.set_xlim(X_LO, X_HI)
    ax0.set_ylim(Y_LO, Y_HI)
    major = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
    ax0.xaxis.set_major_locator(FixedLocator(major))
    ax0.xaxis.set_minor_locator(FixedLocator([k * d for d in (0.01, 0.1) for k in (3, 4, 6, 7, 8, 9)]))
    ax0.xaxis.set_minor_formatter(NullFormatter())
    ax0.set_xticklabels([f"{v:g}" for v in major])
    ax0.set_yticks([0, 25, 50, 75, 100])
    for ax in axes.flat:
        ax.tick_params(axis="x", which="minor", length=1.5, width=S.LW_THIN)
        ax.grid(True, which="major")
    for ax in axes[0]:
        ax.text(KL_SEL, 1.0, f"KL = {KL_SEL:g}", transform=ax.get_xaxis_transform(), fontsize=S.FONT_SMALL,
                color=S.TEXT_SECONDARY, ha="center", va="bottom", zorder=7)
    for c, head in enumerate(["Older release", "Newer release"]):
        axes[0, c].annotate(head, xy=(0.5, 1), xycoords="axes fraction", xytext=(0, 13),
                            textcoords="offset points", ha="center", va="bottom", fontsize=S.FONT,
                            fontweight="bold")

    fig.supxlabel("KL budget t: first-token KL divergence from the original model (100 harmless prompts; log scale)",
                  fontsize=S.FONT)
    fig.supylabel("Fewest dev keyword refusals within budget\n(of 100)", fontsize=S.FONT)

    gray = S.TEXT_SECONDARY
    h = [Line2D([], [], label=S.label(m), **S.line_kw(m, "new", size=STEP_MS)) for m in METHODS]
    h += [
        Line2D([], [], label="older release", **S.line_kw("base", "old", marker="o", size=STEP_MS, color=gray,
                                                           markeredgecolor=gray)),
        Line2D([], [], label="newer release", **S.line_kw("base", "new", marker="o", size=STEP_MS, color=gray,
                                                           markeredgecolor=gray, markerfacecolor=gray)),
        tuple(Line2D([], [], linestyle="none", marker="o", markersize=CLOUD_SIZE + 0.8, markerfacecolor=S.color(m),
                     markeredgecolor="none", alpha=CLOUD_ALPHA[m] + 0.2) for m in METHODS),
        Line2D([], [], color=gray, lw=0.9, linestyle=(0, (1, 1.5)), label="KL = 0.1"),
        Line2D([], [], color=gray, lw=TAIL_LW, alpha=TAIL_ALPHA + 0.15, marker="|", markersize=5.5,
               markeredgewidth=1.2, markevery=[0], label="beyond the largest KL sampled (value carried forward)"),
    ]
    if any(o[0] != "done" for o in out.values()):
        h.append(Patch(facecolor=S.PENDING_FACE, edgecolor=S.PENDING_EDGE, hatch=S.PENDING_HATCH,
                       linewidth=S.LW_THIN, label="pending"))
    labels = [x.get_label() if not isinstance(x, tuple) else "each trial (context)" for x in h]
    fig.legend(handles=h, labels=labels, handler_map={tuple: HandlerTuple(ndivide=None, pad=0.3)},
               loc="outside upper center", ncol=4, handlelength=2.4, columnspacing=1.0,
               handletextpad=0.5, borderaxespad=0.2, frameon=False)

    # cross-check: the KL <= 0.1 read-off of every drawn finished run equals runs.csv selected_refusals
    for model, (state, _, info, _, _) in out.items():
        for m, d in info.items():
            r = runs[(runs.model == model) & (runs.method == m) & (runs.seed == SEED)]
            if len(r) and str(r.iloc[0].run_finished).lower() == "true":
                assert d[0.1] == int(r.iloc[0].selected_refusals), (model, m, d[0.1], r.iloc[0].selected_refusals)

    fig.canvas.draw()                      # final layout: place each read-off table in a data-free corner
    corners = {model: choose_corner(ax_of[model], out[model][3], out[model][4]) for model in out}
    S.save(fig, NAME, preview_dpi=150)
    write_caption(trials, runs, out, corners)


# ------------------------------------------------------------------------------------------------ caption
def write_caption(trials, runs, out, corners):
    D = S.MODEL_DISPLAY
    done = [m for m in out if out[m][0] == "done"]
    pend = [m for m in out if out[m][0] != "done"]

    def v(model, method, tb):
        return out[model][2][method][tb]

    def cov(model, method):
        g = trials[(trials.model == model) & (trials.method == method)]
        return {"n": len(g), "le1": int((g.kl <= 0.1).sum()), "gt1": int((g.kl > 0.1).sum()),
                "gt3": int((g.kl > 0.3).sum()), "max": float(g.kl.max())}

    def plural(n, word):
        return f"{n} {word}" + ("" if n == 1 else "s")

    # one finding per finished pair (MPOA vs gamma-MPOA at KL <= 0.1 and <= 0.5, with search coverage), from data
    findings = []
    for m in done:
        a1, b1, a5, b5 = v(m, "mpoa", 0.1), v(m, "gamma_mpoa", 0.1), v(m, "mpoa", 0.5), v(m, "gamma_mpoa", 0.5)
        if None in (a1, b1, a5, b5):
            continue
        cm, cg = cov(m, "mpoa"), cov(m, "gamma_mpoa")
        d1, d5 = a1 - b1, a5 - b5
        pair = f"{a1} vs {b1} at KL $\\le$ 0.1, {a5} vs {b5} at KL $\\le$ 0.5"
        if max(cm["max"], cg["max"]) <= 0.1:      # nothing sampled beyond the selection budget: 0.5 is carried forward
            pair = (f"{a1} vs {b1} at KL $\\le$ 0.1; neither search sampled a larger KL, so the KL $\\le$ 0.5 "
                    "values only repeat these")
        if abs(d1) <= 3 and abs(d5) <= 5:
            findings.append(f"on {D[m]} the correction changes little ({pair})")
        elif abs(d1) <= 3 and abs(d5) >= 10:
            f = (f"on {D[m]} the two are close at KL $\\le$ 0.1 ({a1} vs {b1}) but separate at larger budgets "
                 f"({a5} vs {b5} at KL $\\le$ 0.5)")
            lo, hi = (cm, cg) if d5 > 0 else (cg, cm)
            if lo["gt1"] * 2 < hi["gt1"]:
                lo_n, hi_n = ("MPOA", "$\\gamma$-MPOA") if d5 > 0 else ("$\\gamma$-MPOA", "MPOA")
                f += (f", a gap confounded with search coverage ({lo_n} placed {lo['gt1']} of {lo['n']} trials "
                      f"above KL 0.1, {hi_n} {hi['gt1']}): the {lo_n} curve is flat there because that region was "
                      "barely explored, not because lower counts were shown to be unreachable")
            findings.append(f)
        elif abs(d1) >= 10:
            who = "the correction lowers" if d1 > 0 else "the correction raises"
            f = f"on {D[m]} {who} the count at the selection budget itself ({pair})"
            if min(cm["le1"], cg["le1"]) >= 50:
                none3 = ", none above 0.3" if cm["gt3"] == cg["gt3"] == 0 else ""
                f += (f", with similar search coverage ({cm['le1']} and {cg['le1']} of 100 trials at KL $\\le$ 0.1"
                      f"{none3}), so this gap is not a coverage artefact")
            findings.append(f)
        else:
            findings.append(f"on {D[m]}: {pair}")
    find_txt = ""
    if findings:
        find_txt = "MPOA vs $\\gamma$-MPOA: " + ". ".join(x[:1].upper() + x[1:] if i else x
                                                                 for i, x in enumerate(findings)) + ". "

    pend_parts = []
    for m in pend:
        state, r = out[m][0], out[m][1]
        pend_parts.append(f"{D[m]} (run in progress, {int(r.n_trials_complete)}/{int(r.n_trials_planned)} trials)"
                          if state == "partial" else f"{D[m]} (run not started)")
    pend_txt = ("$\\gamma$-MPOA is pending on " + " and ".join(x[:-1] + "; only MPOA drawn)" for x in pend_parts)
                + ". ") if pend_parts else ""

    moved = [m for m in out if corners.get(m) != "lower left"]
    where = ("lower-left table" if not moved else
             "small table (" + "; ".join(f"{corners[m]} for {D[m]}" for m in moved)
             + ", lower left elsewhere)")
    res = S.load_results()
    rg = res[res.method == "gamma_mpoa"]
    heldout_pending = rg.empty or not rg.status.astype(str).str.startswith("present").all()

    caption = (
        "**Gamma-corrected MPOA versus MPOA on the post-norm models (dev set).** One panel per model (columns: "
        "older and newer release; rows: Gemma, OLMo). A curve gives, for every KL budget $t$, the fewest dev keyword "
        "refusals (of 100) among the completed TPE trials of one run (100 trials, seed 1) with first-token KL "
        "divergence from the original at most $t$; it steps down at the run's Pareto-front trials (method markers; "
        "older release dashed and hollow, newer solid and filled). Past the largest KL sampled (end tick) the value "
        "is only carried forward (thin line). Faint dots: all completed trials; dotted line: the selection budget "
        f"KL = 0.1; the {where} reads each curve at KL $\\le$ 0.1 and $\\le$ 0.5. "
        + find_txt + pend_txt +
        "All counts are dev keyword refusals on the 100 prompts also used for optimisation and selection, not "
        "held-out refusal rates; one search per run (seed 1), budgets other than 0.1 read post hoc"
        + ("; the held-out judged evaluation of $\\gamma$-MPOA is pending." if heldout_pending else ".")
    )

    notes = []
    for m in done:
        parts = []
        for meth in METHODS:
            c = cov(m, meth)
            parts.append(f"{S.label(meth)} {plural(c['gt1'], 'trial')} above KL 0.1, {c['gt3']} above 0.3, "
                         f"max {c['max']:.2f}")
        notes.append(f"{D[m]}: " + "; ".join(parts))
    drawn = trials[(trials.method == "mpoa") | trials.model.isin(done)]
    n_lo = drawn[drawn.kl < X_LO].groupby(["model", "method"]).size()
    txt = ("Search coverage (where the trials lie): " + ". ".join(notes) + ". " if notes else "")
    txt += (f"Trials with KL < {X_LO:g} lie left of the plotted range ({int(n_lo.min())}--{int(n_lo.max())} per "
            "run); they set the curve's starting value but are not drawn. ")
    txt += ("$\\gamma$-MPOA = MPOA with the edit corrected for the post-norm output gains $\\Gamma$ (ridge-"
            "regularised $\\Gamma^{-1}$; configs/core_gmpoa.toml differs from core_mpoa.toml only in the modifier), "
            "same search space, trial budget and seed; all runs optimised on an H200 NVL; data snapshot "
            f"{S.snapshot_utc()}.")
    caption += "\n\n*For the text:* " + txt
    print(f"  [fig9] caption {len(caption.split('*For the text:*')[0].split())} words")
    S.write_caption(NAME, caption, source=S.SOURCE_DEV)


if __name__ == "__main__":
    main()
