"""fig7_search_convergence -- how quickly each 100-trial TPE search reached the value that was finally selected.

Writes two figures from the same per-run best-so-far curves (seed 1; dev set):

  fig7_search_convergence        main text, ONE summary series per panel. Top row: one panel per core method with
                                 10 runs (DiM, MPOA, ARA, SOM): the median over the runs of the best (lowest) dev
                                 keyword refusal count among the trials so far whose first-token KL divergence is
                                 <= 0.1 (the selection budget), with the interquartile range (25th-75th percentile,
                                 linear interpolation) and the full range (min-max) of the runs as bands, drawn from
                                 the first trial at which every run has a value. Bottom: dot histogram (one marker
                                 per run, 5-trial bins) of the trial at which the final selected count was first
                                 reached, for the 42 seed-1 runs of the five core methods (gamma-MPOA not counted).
  figA4_search_convergence_runs  appendix, every run as its own step curve, split by method (rows) and by release
                                 (older / newer, columns), so that a panel holds at most five runs, each labelled at
                                 its right end. Last row: J-space (small pair) and gamma-MPOA (Gemma, OLMo).
                                 The earlier fig7 drew all ten runs of a method plus the median in one panel.

The marker on a run's curve sits at the trial where the run FIRST reached the refusal count it finally selected
(finished runs only). A curve starts at the run's first trial inside the KL budget (before that no trial qualifies).
The grey band marks the 30 random-sampling startup trials of Optuna's TPE sampler (n_startup_trials = 30, checked in
every run's config.toml); the dotted line marks trial 75.

The selected count itself is cross-checked against trials.csv `is_selected_kl0.1` (selection rule: fewest dev
refusals at KL <= 0.1; ties -> lower KL, then lower trial index). The rule's tie-break can pick a LATER trial with the
same count and lower KL; this figure uses the first trial that reached the count (the convergence question).

Data: paper/figures/data/trials.csv (seed 1; dev set), runs.csv (gamma-MPOA status). Dev-set numbers only.

Run:  cd scripts/figures && .venv/bin/python fig7_search_convergence.py
"""
from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
from matplotlib import patheffects as pe
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

import style as S

NAME = "fig7_search_convergence"
NAME_A = "figA4_search_convergence_runs"
CORE = ["dim", "mpoa", "ara", "som", "jspace"]
PANELS = CORE + ["gamma_mpoa"]
SEED = 1
KL_MAX = 0.1
N_TRIALS = 100
N_STARTUP = 30          # Heretic config.default.toml n_startup_trials; verified per run below
LATE = 75               # "late" = first reached after this trial
BIN = 5                 # dot-histogram bin width (trials); 30 and 75 are bin edges
MEDIAN_METHODS = ["dim", "mpoa", "ara", "som"]   # median curve only for these (10 runs each); never J-space/gamma
LW_RUN = 0.85           # thin per-run step curves
LW_MED = 2.0            # bold median
LW_RUN_A = 1.1          # per-run curves in the appendix figure (at most five per panel)
BAND_IQR, BAND_RANGE = 0.30, 0.12   # alpha of the interquartile / min-max bands
MS_RUN = 3.6            # first-reach marker on the step curves
MS_DOT = 4.6            # dot-histogram markers
YLIM = (-3, 103)
XLIM = (0, 101.5)
GEN_OLD = ("old", "small-old")
DASH_THIN = (0, (3.2, 1.6))   # old generation, thin line (pattern scales with linewidth)


def gen_key(g):
    return "old" if g in GEN_OLD else "new"


# ================================================================================================ data
def startup_check(runs):
    """n_startup_trials of every plotted run, from <run dir>/config.toml (config.used.toml when present)."""
    seen = {}
    for r in runs.itertuples():
        if not isinstance(r.journal, str):
            continue
        d = S.ROOT / Path(r.journal).parent.parent
        for f in ("config.used.toml", "config.toml"):
            p = d / f
            if p.exists():
                m = re.search(r"^\s*n_startup_trials\s*=\s*(\d+)", p.read_text(), re.M)
                if m:
                    seen[r.run] = int(m.group(1))
                    break
    bad = {k: v for k, v in seen.items() if v != N_STARTUP}
    if bad:
        print(f"  [fig7] WARNING: n_startup_trials != {N_STARTUP}: {bad}")
    missing = sorted(set(runs.run.dropna()) - set(seen))
    if missing:
        print(f"  [fig7] WARNING: no n_startup_trials found for {missing}")
    return len(seen), bad


def curves(trials):
    """One record per run: best-so-far array over trial index 1..N_TRIALS (NaN before the first feasible trial or
    after the last completed trial), final count, first-reach trial, selected trial."""
    out = []
    for run, g in trials.groupby("run", sort=False):
        g = g.sort_values("trial")
        last = int(g.trial.max())
        best = np.full(N_TRIALS + 1, np.nan)        # index = trial number; [0] unused
        cur = math.inf
        by_t = dict(zip(g.trial.astype(int), zip(g.refusals.astype(float), g.kl.astype(float))))
        for t in range(1, last + 1):
            if t in by_t and by_t[t][1] <= KL_MAX:
                cur = min(cur, by_t[t][0])
            if cur < math.inf:
                best[t] = cur
        feas = g[g.kl <= KL_MAX]
        final = float(feas.refusals.min())
        first = int(feas.loc[feas.refusals == final, "trial"].min())
        sel = g[g["is_selected_kl0.1"]]
        assert len(sel) == 1 and float(sel.refusals.iloc[0]) == final, f"{run}: selection mismatch"
        r0 = g.iloc[0]
        out.append(dict(run=run, model=r0.model, method=r0.method, gen=gen_key(r0.generation),
                        generation=r0.generation, finished=bool(r0.run_finished), n=len(g), last=last, best=best,
                        final=final, first=first, selected=int(sel.trial.iloc[0]),
                        first_feasible=int(feas.trial.min()),
                        at30=best[N_STARTUP], at50=best[50], at75=best[LATE]))
    return out


def median_curve(recs):
    """Median best-so-far over the runs, from the first trial at which every run has a value."""
    m = np.vstack([r["best"] for r in recs])
    ok = ~np.isnan(m).any(axis=0)
    med = np.full(m.shape[1], np.nan)
    med[ok] = np.median(m[:, ok], axis=0)
    return med


# ================================================================================================ drawing
def step(ax, best, **kw):
    t = np.arange(len(best))
    ok = ~np.isnan(best)
    x, y = t[ok], best[ok]
    if len(x) == 0:
        return
    # steps-post: the value of trial t holds until the next trial; finished runs extend to the right edge
    x_end = kw.pop("x_end", x[-1])
    xs = np.append(x, x_end)
    ys = np.append(y, y[-1])
    ax.plot(xs, ys, drawstyle="steps-post", solid_joinstyle="miter", **kw)


def run_line(ax, r, alpha=0.9, zorder=3):
    c = S.color(r["method"])
    ls = "-" if r["gen"] == "new" else DASH_THIN
    step(ax, r["best"], color=c, lw=LW_RUN, linestyle=ls, alpha=alpha, zorder=zorder,
         x_end=N_TRIALS if r["finished"] else r["last"])
    if not r["finished"]:       # nothing selected yet: the caller draws the neutral 'running' marker instead
        return
    ax.plot([r["first"]], [r["final"]], zorder=zorder + 2, alpha=1.0,
            **S.marker_kw(r["method"], r["generation"], size=MS_RUN, markeredgewidth=0.5 if r["gen"] == "new"
                          else 0.9))


def base_axes(ax, left, bottom):
    ax.axvspan(0.5, N_STARTUP + 0.5, color=S.NEUTRAL_MID, lw=0, zorder=0)
    # dotted reference at trial LATE: markers to its right = selected count first reached in the last quarter.
    # (The 10-run panels carry no run names: every free spot tried hid another run's line or marker.)
    ax.axvline(LATE + 0.5, color=S.TEXT_SECONDARY, lw=0.9, linestyle=(0, (1, 1.5)), zorder=1)
    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    ax.set_xticks([1, N_STARTUP, 50, LATE, N_TRIALS])
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.tick_params(labelleft=left, labelbottom=bottom)
    ax.grid(True, axis="y")
    ax.grid(False, axis="x")


def title(ax, head, sub, right=""):
    """Bold method title (left), optional secondary note (right) and a secondary subtitle line under both.
    The extra texts are kept out of the layout engine (they are sized to fit the panel width)."""
    ax.set_title(head, loc="left", fontsize=S.FONT, fontweight="bold", pad=10)
    if right:
        ax.set_title(right, loc="right", fontsize=S.FONT_SMALL, color=S.TEXT_SECONDARY, pad=10.6)
    t = ax.text(0.0, 1.015, sub, transform=ax.transAxes, ha="left", va="bottom", fontsize=S.FONT_SMALL,
                color=S.TEXT_SECONDARY)
    t.set_in_layout(False)


def direct_label(ax, x, y, text, color=S.TEXT_SECONDARY, dx=-3, dy=2.5, ha="right", va="bottom"):
    ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va=va, fontsize=S.FONT_SMALL,
                color=color, zorder=8,
                path_effects=[pe.withStroke(linewidth=2.2, foreground=S.SURFACE)])


def method_panel(ax, recs, method):
    rs = [r for r in recs if r["method"] == method]
    fin = [r for r in rs if r["finished"]]
    for r in sorted(rs, key=lambda r: (r["gen"] == "new", -r["final"])):
        run_line(ax, r)
    med = None
    if method in MEDIAN_METHODS:
        med = median_curve(fin)
        step(ax, med, color=S.TEXT, lw=LW_MED, zorder=6, x_end=N_TRIALS,
             path_effects=[pe.withStroke(linewidth=LW_MED + 1.6, foreground=S.SURFACE)])
    return rs, fin, med


def status_text(status):
    """runs.csv status -> short run state, e.g. 'partial: run running (48/100 trials)' -> 'running (48/100)'."""
    st = str(status)
    m = re.search(r"\((\d+)/(\d+) trials\)", st)
    if "running" in st:
        return f"running ({m.group(1)}/{m.group(2)})" if m else "running"
    if st.startswith("planned") or "not started" in st:
        return f"{S.PENDING_LABEL} (not started)"
    if "queued" in st:
        return f"{S.PENDING_LABEL} (queued)"
    return f"{S.PENDING_LABEL} ({st.split(':')[0]})"


def spread(targets, gap, lo, hi):
    """Label y positions close to `targets` (sorted ascending) with >= gap between neighbours, inside [lo, hi]."""
    n = len(targets)
    if n == 0:
        return []
    gap = min(gap, (hi - lo) / max(n - 1, 1))
    clusters, starts = [[i] for i in range(n)], list(targets)

    def place(cl):
        c = float(np.mean([targets[i] for i in cl]))
        return min(max(c - gap * (len(cl) - 1) / 2, lo), hi - gap * (len(cl) - 1))

    starts = [place(c) for c in clusters]
    changed = True
    while changed:
        changed = False
        for k in range(len(clusters) - 1):
            if starts[k + 1] - (starts[k] + gap * (len(clusters[k]) - 1)) < gap - 1e-9:
                clusters[k] += clusters.pop(k + 1)
                starts.pop(k + 1)
                starts[k] = place(clusters[k])
                changed = True
                break
    out = [0.0] * n
    for cl, s0 in zip(clusters, starts):
        for j, i in enumerate(cl):
            out[i] = s0 + gap * j
    return out


def end_labels(ax, rs, name):
    """Model name at the right end of every run, spread vertically, thin leader when displaced."""
    fig = ax.figure
    h_pt = ax.get_window_extent().height * 72 / fig.dpi
    gap = (S.FONT_SMALL + 1.2) * (YLIM[1] - YLIM[0]) / h_pt
    items = sorted(((r["best"][r["last"]] if not r["finished"] else r["final"], name(r)) for r in rs))
    ys = spread([y for y, _ in items], gap, YLIM[0] + gap / 2, YLIM[1] - gap / 2)
    for (y, txt), yl in zip(items, ys):
        ax.annotate(txt, xy=(N_TRIALS, y), xytext=(N_TRIALS + 9, yl), textcoords="data", ha="left", va="center",
                    fontsize=S.FONT_SMALL, color=S.TEXT_SECONDARY, annotation_clip=False,
                    arrowprops={"arrowstyle": "-", "color": S.AXIS, "lw": S.LW_THIN, "shrinkA": 0.5, "shrinkB": 0.5,
                                "relpos": (0, 0.5)})


def summary_panel(ax, recs, method):
    """Median + interquartile band + min-max band of the finished runs of one method."""
    fin = [r for r in recs if r["method"] == method and r["finished"]]
    m = np.vstack([r["best"] for r in fin])
    ok = ~np.isnan(m).any(axis=0)
    t = np.arange(m.shape[1])[ok]
    q = {k: np.percentile(m[:, ok], k, axis=0) for k in (0, 25, 50, 75, 100)}
    c = S.color(method)
    # bands as step functions (post), extended to the right edge
    tx = np.append(t, N_TRIALS)
    for lo_k, hi_k, a in ((0, 100, BAND_RANGE), (25, 75, BAND_IQR)):
        ax.fill_between(tx, np.append(q[lo_k], q[lo_k][-1]), np.append(q[hi_k], q[hi_k][-1]), step="post",
                        color=c, alpha=a, linewidth=0, zorder=2)
    med = np.full(m.shape[1], np.nan)
    med[ok] = q[50]
    step(ax, med, color=c, lw=LW_MED, zorder=4, x_end=N_TRIALS)
    return fin, med


def histogram(hax, core, med_first, n_late, share_late, n_in_startup):
    hax.axvspan(0.5, N_STARTUP + 0.5, color=S.NEUTRAL_MID, lw=0, zorder=0)
    morder = {m: i for i, m in enumerate(CORE)}
    bins = {}
    for r in sorted(core, key=lambda r: (morder[r["method"]], r["gen"] == "new", r["first"])):
        bins.setdefault((r["first"] - 1) // BIN, []).append(r)
    ymax = max(len(v) for v in bins.values())
    for b, rs_b in bins.items():
        xc = b * BIN + (BIN + 1) / 2
        for k, r in enumerate(rs_b):
            hax.plot([xc], [k + 1], zorder=4, clip_on=False, **S.marker_kw(r["method"], r["generation"], size=MS_DOT))
    hax.axvline(med_first, color=S.TEXT, lw=1.1, zorder=3)
    hax.text(med_first + 1.0, ymax + 0.55, f"median: trial {med_first:.0f}", ha="left", va="bottom",
             fontsize=S.FONT_SMALL, color=S.TEXT)
    yb = ymax + 0.25
    hax.plot([LATE + 0.5, LATE + 0.5], [0, yb], color=S.TEXT_SECONDARY, lw=0.9, linestyle=(0, (1, 1.5)), zorder=1)
    hax.annotate("", xy=(N_TRIALS + 0.5, yb), xytext=(LATE + 0.5, yb),
                 arrowprops={"arrowstyle": "|-|,widthA=0.25,widthB=0.25", "color": S.TEXT_SECONDARY,
                             "lw": S.LW_THIN, "shrinkA": 0, "shrinkB": 0})
    hax.text((LATE + N_TRIALS + 1) / 2, yb + 0.3, f"after trial {LATE}: {n_late} of {len(core)} runs "
             f"({100 * share_late:.0f}%)", ha="center", va="bottom", fontsize=S.FONT_SMALL, color=S.TEXT)
    hax.text(N_STARTUP / 2 + 0.5, ymax + 0.55, f"random startup: {n_in_startup} runs", ha="center", va="bottom",
             fontsize=S.FONT_SMALL, color=S.TEXT_SECONDARY)
    hax.set_xlim(*XLIM)
    hax.set_ylim(0, ymax + 1.7)
    hax.set_xticks([1, N_STARTUP, 50, LATE, N_TRIALS])
    hax.set_yticks(range(0, ymax + 1))
    hax.grid(True, axis="y")
    hax.grid(False, axis="x")
    hax.set_xlabel(f"Trial at which the selected count was first reached ({BIN}-trial bins)")
    hax.set_ylabel("Runs")


# ================================================================================================ main
def main():
    S.apply()
    trials = S.load_trials()
    runs = S.load_runs()
    trials = trials[(trials.seed == SEED) & trials.method.isin(PANELS)].copy()
    for c in ("is_selected_kl0.1", "run_finished"):
        trials[c] = trials[c].astype(str).str.lower().eq("true")
    runs1 = runs[(runs.seed == SEED) & runs.method.isin(PANELS)]
    n_cfg, bad_cfg = startup_check(runs1[runs1.run.isin(set(trials.run))])

    recs = curves(trials)
    core = [r for r in recs if r["method"] in CORE]
    assert all(r["finished"] for r in core), "a seed-1 core run is not finished"
    assert len(core) == 42, f"expected 42 seed-1 core runs, got {len(core)}"
    gam = [r for r in recs if r["method"] == "gamma_mpoa"]
    gam_pending = runs1[(runs1.method == "gamma_mpoa") & ~runs1.run.isin([r["run"] for r in gam])]
    status = {r.run: str(r.status) for r in runs1[runs1.method == "gamma_mpoa"].itertuples()}

    firsts = np.array([r["first"] for r in core])
    med_first = float(np.median(firsts))
    late = [r for r in core if r["first"] > LATE]
    n_late, share_late = len(late), len(late) / len(core)
    n_in_startup = int((firsts <= N_STARTUP).sum())

    # ======================================================================== fig7: summary (main text)
    fig = S.plt.figure(figsize=(S.TEXT_W, 4.05))
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02, wspace=0.0, hspace=0.0)
    sf_top, sf_bot = fig.subfigures(2, 1, height_ratios=[1.15, 1.0], hspace=0.0)
    top = sf_top.subplots(1, 4, sharey=True, gridspec_kw={"wspace": 0.06})
    hax = sf_bot.subplots(1, 1)
    stats = {}
    for i, (ax, m) in enumerate(zip(top, MEDIAN_METHODS)):
        base_axes(ax, left=(i == 0), bottom=True)
        fin, med = summary_panel(ax, recs, m)
        stats[m] = ([r for r in recs if r["method"] == m], fin, med)
        mf = np.median([r["first"] for r in fin])
        title(ax, S.label(m), f"median first reach: trial {mf:g}", right=f"{len(fin)} runs")
        ax.set_xlabel("Trial")
        if i == 0:
            ax.set_ylabel("Best dev keyword refusals\nso far (of 100; KL ≤ 0.1)")
    histogram(hax, core, med_first, n_late, share_late, n_in_startup)
    title(hax, "When the selected count was first reached",
          f"one marker per run; {len(core)} seed-1 runs of the five core methods "
          "(older release hollow, newer filled; colour and shape = method)")

    g = S.TEXT_SECONDARY
    handles = [
        Line2D([], [], color=g, lw=LW_MED, label="median of the 10 runs"),
        Patch(facecolor=g, alpha=BAND_IQR + 0.05, edgecolor="none", label="middle 50% of runs"),
        Patch(facecolor=g, alpha=BAND_RANGE + 0.05, edgecolor="none", label="all runs (min–max)"),
        Patch(facecolor=S.NEUTRAL_MID, edgecolor="none", label=f"random startup (trials 1–{N_STARTUP})"),
        Line2D([], [], color=g, lw=0.9, linestyle=(0, (1, 1.5)), label=f"trial {LATE}"),
    ]
    hl = S.method_handles(CORE, "new")
    hl[-1].set_label("J-space (small pair)")
    # two rows (legend fills column by column): summary encodings on top, method colour + shape below
    inter = [h for pair in zip(handles, hl) for h in pair]
    fig.legend(handles=inter, loc="outside upper center", ncols=len(handles), handlelength=1.8,
               columnspacing=1.4, handletextpad=0.5, borderaxespad=0.1, labelspacing=0.3)
    S.save(fig, NAME, preview_dpi=150)
    S.plt.close(fig)

    # ======================================================================== figA4: every run (appendix)
    figa = S.plt.figure(figsize=(S.TEXT_W, 7.0))
    figa.get_layout_engine().set(w_pad=0.02, h_pad=0.02, wspace=0.0, hspace=0.0)
    sfs = figa.subfigures(5, 1, height_ratios=[1, 1, 1, 1, 1.08], hspace=0.0)
    gen_title = {"old": "older release", "new": "newer release"}

    def name_main(r):
        return S.MODEL_DISPLAY[r["model"]]

    def name_short(r):      # 'Gemma-4-31B' -> 'Gemma-4' (unique within a gamma panel)
        return re.sub(r"-\d+B$", "", S.MODEL_DISPLAY[r["model"]])

    for k, m in enumerate(MEDIAN_METHODS):
        axs = sfs[k].subplots(1, 2, sharey=True, gridspec_kw={"wspace": 0.10})
        for j, (ax, gen) in enumerate(zip(axs, ("old", "new"))):
            base_axes(ax, left=(j == 0), bottom=(k == len(MEDIAN_METHODS) - 1))
            rs = [r for r in recs if r["method"] == m and r["gen"] == gen]
            for r in sorted(rs, key=lambda r: -r["final"]):
                run_line(ax, r, alpha=1.0)
            for ln in ax.lines:
                if ln.get_drawstyle() == "steps-post":
                    ln.set_linewidth(LW_RUN_A)
            end_labels(ax, rs, name_main)
            title(ax, f"{S.label(m)}, {gen_title[gen]}",
                  f"{len(rs)} runs; median first reach: trial {np.median([r['first'] for r in rs]):g}")
        axs[0].set_ylabel("Best dev refusals\nso far (KL ≤ 0.1)")
    axs = sfs[4].subplots(1, 4, sharey=True, gridspec_kw={"wspace": 0.10})
    for j, (m, gen) in enumerate((("jspace", "old"), ("jspace", "new"), ("gamma_mpoa", "old"),
                                  ("gamma_mpoa", "new"))):
        ax = axs[j]
        base_axes(ax, left=(j == 0), bottom=True)
        rs = [r for r in recs if r["method"] == m and r["gen"] == gen]
        for r in sorted(rs, key=lambda r: -r["final"]):
            run_line(ax, r, alpha=1.0)
            if not r["finished"]:
                ax.plot([r["last"]], [r["best"][r["last"]]], marker="o", markersize=MS_RUN + 0.6,
                        markerfacecolor=S.SURFACE, markeredgecolor=S.PENDING_EDGE, markeredgewidth=S.LW_THIN,
                        linestyle="none", zorder=7)
        for ln in ax.lines:
            if ln.get_drawstyle() == "steps-post":
                ln.set_linewidth(LW_RUN_A)
        if m == "gamma_mpoa":
            end_labels(ax, rs, name_short)
            pend = gam_pending[gam_pending.generation.astype(str).map(gen_key) == gen]
            for i_, p_ in enumerate(pend.itertuples()):
                ax.text(0.97, 0.08 + 0.12 * i_, f"{S.MODEL_DISPLAY[p_.model]}: {status_text(p_.status)}",
                        transform=ax.transAxes, ha="right", va="bottom", fontsize=S.FONT_SMALL,
                        color=S.TEXT_SECONDARY, style="italic")
            sub = ", ".join(sorted({name_short(r).split("-")[0] for r in rs})) or "no run"
        else:
            sub = ", ".join(S.MODEL_DISPLAY[r["model"]] for r in rs)
        title(ax, f"{S.label(m)}, {gen_title[gen].split()[0]}", sub)
        ax.set_xlabel("Trial")
        if j == 0:
            ax.set_ylabel("Best dev refusals\nso far (KL ≤ 0.1)")
    ha = [
        Line2D([], [], color=g, lw=LW_RUN_A, linestyle=DASH_THIN, marker="o", markersize=MS_RUN + 0.6,
               markerfacecolor=S.SURFACE, markeredgecolor=g, markeredgewidth=0.9, label="older release"),
        Line2D([], [], color=g, lw=LW_RUN_A, linestyle="-", marker="o", markersize=MS_RUN + 0.6,
               markerfacecolor=g, markeredgecolor=g, markeredgewidth=0.5, label="newer release"),
        Line2D([], [], linestyle="none", marker="o", markersize=MS_RUN + 0.6, markerfacecolor=g, markeredgecolor=g,
               label="selected count first reached"),
        Patch(facecolor=S.NEUTRAL_MID, edgecolor="none", label=f"random startup (trials 1–{N_STARTUP})"),
        Line2D([], [], color=g, lw=0.9, linestyle=(0, (1, 1.5)), label=f"trial {LATE}"),
    ]
    if any(not r["finished"] for r in gam):
        ha.append(Line2D([], [], linestyle="none", marker="o", markersize=MS_RUN + 0.6, markerfacecolor=S.SURFACE,
                         markeredgecolor=S.PENDING_EDGE, markeredgewidth=S.LW_THIN, label="running: last trial"))
    figa.legend(handles=ha, loc="outside upper center", ncols=len(ha), handlelength=2.2, columnspacing=1.3,
                handletextpad=0.5, borderaxespad=0.1)
    S.save(figa, NAME_A, preview_dpi=150)
    S.plt.close(figa)

    # ================================================================================== numbers for caption/text
    def fmt(x):   # medians of 10 runs can end in .5: print exactly, never round half to even
        return f"{x:g}"

    print(f"  [fig7] n_startup_trials = {N_STARTUP} in {n_cfg} run configs; mismatches: {bad_cfg or 'none'}")
    print(f"  [fig7] core runs {len(core)}: median first reach {med_first}, <= {N_STARTUP}: {n_in_startup}, "
          f"> {LATE}: {n_late} ({100 * share_late:.1f}%)")
    gains = [(r, r["at75"] - r["final"]) for r in late]
    for r, d in sorted(gains, key=lambda x: -x[1]):
        print(f"  [fig7]   late: {r['model']:<20} {r['method']:<5} first {r['first']:>3}  best@75 {r['at75']:.0f} -> "
              f"final {r['final']:.0f}  (gain {d:.0f}; selected trial {r['selected']})")
    gap50 = np.array([r["at50"] - r["final"] for r in core])
    gap30 = np.array([r["at30"] - r["final"] for r in core])
    print(f"  [fig7] gap to final at trial 30: median {np.median(gap30):.1f}, mean {gap30.mean():.1f}; "
          f"at 50: median {np.median(gap50):.1f}, mean {gap50.mean():.1f}, runs with gap>0 {int((gap50 > 0).sum())}")
    med_pts = {}
    for m in ["dim", "mpoa", "ara", "som"]:
        med = stats[m][2]
        med_pts[m] = {t: med[t] for t in (N_STARTUP, 50, LATE, N_TRIALS)}
        print(f"  [fig7] median curve {m}: " + ", ".join(f"t{t} {v:.1f}" for t, v in med_pts[m].items()))
    differ = [r for r in core if r["selected"] != r["first"]]
    print(f"  [fig7] selected trial later than first reach (tie at lower KL): {len(differ)} runs")
    feas_late = [r for r in recs if r["first_feasible"] > 1]
    print("  [fig7] first trial within KL budget > 1: "
          + ", ".join(f"{r['model']} {r['method']} t{r['first_feasible']}" for r in feas_late))
    for r in gam:
        print(f"  [fig7] gamma-MPOA {r['model']}: finished={r['finished']} n={r['n']} final {r['final']:.0f} "
              f"first {r['first']}")
    # seed-2 (small pair) for the text
    t2 = S.load_trials()
    t2 = t2[t2.seed == 2].copy()
    for c in ("is_selected_kl0.1", "run_finished"):
        t2[c] = t2[c].astype(str).str.lower().eq("true")
    rec2 = curves(t2) if len(t2) else []
    f2 = [r for r in rec2 if r["finished"]]
    s2_txt = ""
    if f2:
        fs2 = np.array([r["first"] for r in f2])
        s1_small = np.array([r["first"] for r in core if r["model"] in ("qwen2.5-7b", "qwen3.5-4b")])
        s2_txt = (f"Seed 2 (small pair, {len(f2)} of {len(rec2)} runs finished): median first reach at trial "
                  f"{np.median(fs2):g} (seed 1, same 10 cells: {np.median(s1_small):g}); {int((fs2 > LATE).sum())} of "
                  f"{len(f2)} after trial {LATE}.")
        print("  [fig7] " + s2_txt)

    def late_desc(r, d):
        return (f"{S.MODEL_DISPLAY[r['model']]} {S.label(r['method'])} (trial {r['first']}, "
                f"{r['at75']:.0f}$\\to${r['final']:.0f})")

    def gamma_desc(r):
        if r["finished"]:
            return f"{S.MODEL_DISPLAY[r['model']]} selected {r['final']:.0f}, first reached at trial {r['first']}"
        return (f"{S.MODEL_DISPLAY[r['model']]} running ({r['n']}/{N_TRIALS} trials, best so far {r['final']:.0f} "
                f"at trial {r['first']})")

    late_sorted = sorted(gains, key=lambda x: -x[1])
    gmin, gmax = min(d for _, d in gains), max(d for _, d in gains)
    n_big = sum(1 for _, d in gains if d >= 5)

    ara_ff = [r["first_feasible"] for r in core if r["method"] == "ara"]
    oth_ff = [r["first_feasible"] for r in core if r["method"] != "ara"]

    def rng(v):
        return f"trial {min(v)}" if min(v) == max(v) else f"trials {min(v)}--{max(v)}"

    n_by75 = sum(1 for r in core if r["first"] <= LATE)
    gap75 = np.array([r["at75"] - r["final"] for r in core])
    short = [r for r in recs if r["finished"] and r["n"] < N_TRIALS]
    g_state = []
    for r in gam:
        if not r["finished"]:
            st = status_text(status.get(r["run"], "running"))            # e.g. 'running (48/100)'
            m_ = re.fullmatch(r"running \((\d+/\d+)\)", st)
            g_state.append(f"{S.MODEL_DISPLAY[r['model']]} is still "
                           + (f"running ({m_.group(1)} trials; the open circle marks its last completed trial)"
                              if m_ else st))
    g_state += [f"{S.MODEL_DISPLAY[p.model]} is {status_text(p.status)}" for p in gam_pending.itertuples()]
    best_def = ("after trial $t$, the fewest dev keyword refusals (of 100) among trials 1 to $t$ with first-token KL "
                "divergence $\\le$ 0.1 from the original, i.e. the count the selection rule would have returned had "
                "the search stopped at trial $t$")
    start_txt = (f"Curves start at the run's first trial within the KL budget ({rng(ara_ff)} for ARA, "
                 f"{rng(oth_ff).replace('trials ', '').replace('trial ', '')} otherwise)")
    late_txt = (f"The {n_late} runs ({100 * share_late:.0f}%) that still improved after trial {LATE}, by "
                f"{gmin:.0f}--{gmax:.0f} refusals, show that the selected counts are not converged minima; for the "
                f"other {n_by75} runs these data cannot tell whether a longer search would have helped.")
    bias = ("All counts are keyword matches on the 100 dev prompts that were also used for search and selection, "
            "not held-out refusal rates.")
    caption = (
        "**Search-budget convergence of the 100-trial TPE searches (dev set).** Top: for each core method with ten "
        f"runs (seed 1; five pairs, older and newer release), the median over the runs of the best count so far: "
        f"{best_def}. Dark band: middle 50% of the runs (25th--75th percentile); light band: all runs (minimum to "
        f"maximum). {start_txt}; the median and bands start once every run has a value. The grey band marks the "
        f"{N_STARTUP} random startup trials of the TPE sampler, the dotted line trial {LATE}. Bottom: the trial at "
        f"which each of the {len(core)} core runs (DiM, MPOA, ARA, SOM and, on the small pair, J-space) first "
        f"reached the count it finally selected (median {fmt(med_first)}; {n_in_startup} runs within the random "
        "startup); hollow markers are older, filled markers newer releases, colour and shape the method. "
        + late_txt + " Every run is drawn as its own curve, by method and release, in Fig. A4, together with the "
        f"two J-space runs and the {len(gam)} $\\gamma$-MPOA runs (Gemma and OLMo pairs only; not counted in the "
        "histogram" + ("; " + "; ".join(g_state) if g_state else "") + "). " + bias
    )
    caption_a = (
        "**Search-budget convergence of every optimisation run (dev set).** One step curve per run (seed 1): "
        f"{best_def}. Rows are methods and columns the older and newer release of each pair, so that a panel holds at "
        "most five runs, each named at its right end; bottom row: J-space (small pair only) and $\\gamma$-MPOA "
        "(Gemma and OLMo pairs only). Dashed lines with hollow markers are older, solid lines with filled markers "
        "newer releases; the marker sits where the run first reached the count it finally selected. "
        f"{start_txt}. The grey band marks the {N_STARTUP} random startup trials of the TPE sampler, the dotted line "
        f"trial {LATE}. The medians, spreads and first-reach histogram of the core runs are in Fig. 7. " + bias
    )
    detail = (
        f"*For the text:* {n_by75} of the {len(core)} core runs had found their selected count by trial {LATE} "
        f"and did not improve over the last {N_TRIALS - LATE} trials, so most searches had stopped improving well "
        "within the budget, but not all: the selected counts are the best found within this budget, not converged "
        f"minima, and the late gains ({gmin:.0f}--{gmax:.0f} refusals, {n_big} of {n_late} runs by 5 or more) are as "
        "large as several of the older-vs-newer differences between selected models. "
        f"Runs whose selected count was first reached after trial {LATE} (best count at trial "
        f"{LATE} $\\to$ selected count): "
        + "; ".join(late_desc(r, d) for r, d in late_sorted) + ". "
        f"Remaining gap between the best count so far and the selected count, over the {len(core)} core runs: "
        f"median {fmt(np.median(gap30))} (mean {gap30.mean():.1f}) refusals at trial {N_STARTUP}, median "
        f"{fmt(np.median(gap50))} (mean {gap50.mean():.1f}) at trial 50, median {fmt(np.median(gap75))} "
        f"(mean {gap75.mean():.1f}) at trial {LATE}. Median curves at trials {N_STARTUP} / 50 / {LATE} / "
        f"{N_TRIALS}: "
        + "; ".join(f"{S.label(m)} " + " / ".join(fmt(v) for v in med_pts[m].values()) for m in med_pts) + ". "
        f"The selection rule's tie-break (lower KL) picked a later trial with the same count in {len(differ)} of "
        f"{len(core)} runs; the figure uses the first trial with that count. "
        + s2_txt
        + " $\\gamma$-MPOA (not in the histogram): " + "; ".join(gamma_desc(r) for r in gam) + "."
        + ("".join(f" {S.MODEL_DISPLAY[r['model']]} {S.label(r['method'])} has {r['n']} completed trials "
                   f"({N_TRIALS - r['n']} failed)." for r in short))
        + f" The ARA median starts at trial {max(ara_ff)}, the first trial at which all ten ARA runs have a value."
        + f" TPE: {N_TRIALS} trials, n_startup_trials = {N_STARTUP} in every run's config.toml."
    )
    print(f"  [fig7] caption {len(caption.split())} words")
    S.write_caption(NAME_A, caption_a, source=S.SOURCE_DEV + " " + S.SOURCE_SELECTION
                    + f" Data snapshot {S.snapshot_utc()}.")
    S.write_caption(NAME, caption + "\n\n" + detail,
                    source=S.SOURCE_DEV + " " + S.SOURCE_SELECTION + f" Data snapshot {S.snapshot_utc()}.")


if __name__ == "__main__":
    main()
