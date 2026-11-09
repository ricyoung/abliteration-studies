#!/usr/bin/env python3
"""fig2_pareto_fronts.py -- dev-set refusal/KL trade-off of every optimisation run (paper 33).

Run with the figures venv from scripts/figures (normally via make_figures.sh):
    scripts/figures/.venv/bin/python scripts/figures/fig2_pareto_fronts.py

Writes three figures with the same grid (rows = families, columns = methods; the four main pairs share a 4 x 4
block, the small pair has its own row with J-space as a fifth column):

  fig2_pareto_fronts           main text. One panel per family x method with only TWO series: the Pareto front of
                               the older release (dashed line, hollow markers) and of the newer release (solid,
                               filled), so the generations are compared inside one panel. The trial picked by the
                               selection rule (is_selected_kl0.1) is ringed; its dev refusal count is printed in the
                               panel title, right ("89 → 47" = older → newer), outside the plot area.
                               Grey stars = the two original models (KL = 0, runs.csv baseline_refusals).
  figA1_trials_old_generation  appendix. Same grid, ONE run per panel (older release): every completed TPE trial as a
  figA2_trials_new_generation  dot, plus the run's front, selected trial and original model. These dots were the
                               faint background cloud of the earlier fig2 (several methods per panel); they moved
                               here so that the main figure carries fewer series per panel. No trial is dropped.

Trials with KL exactly 0 are drawn on the left axis edge (log axis; the edge is labelled "0" behind an axis
break). Only the five core methods are drawn: gamma-MPOA (Gemma/OLMo only) is compared with MPOA in fig9_gamma_mpoa.
Seed-2 runs (small pair) are excluded here (fig8_seed_sensitivity).

Inputs : paper/figures/data/trials.csv, runs.csv
Outputs: paper/figures/{fig2_pareto_fronts,figA1_trials_old_generation,figA2_trials_new_generation}.pdf,
         preview/*.png, captions.md sections
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, NullFormatter

import style as S

NAME = "fig2_pareto_fronts"
NAME_A = {"old": "figA1_trials_old_generation", "new": "figA2_trials_new_generation"}
SEED = 1
KL_MAX = 0.1                 # selection threshold (select_trial.select)
X_ZERO = 4e-5                # where KL == 0 is drawn (left axis edge, behind an axis break; smallest KL > 0 is 1.5e-4)
X_LO, X_HI = X_ZERO, 60.0
Y_LO, Y_HI = -4, 104
CORE = ["dim", "mpoa", "som", "ara"]
SMALL = ["dim", "mpoa", "som", "jspace", "ara"]
SMALL_FAMILY = "Small pair"
CLOUD_SIZE = 3.0             # pt, diameter of the all-trials dots (appendix figures)
CLOUD_ALPHA = {"dim": 0.40, "mpoa": 0.42, "ara": 0.50, "som": 0.60, "jspace": 0.60}   # light hues need more
FRONT_MS = 4.6               # pt, Pareto-front vertex markers
RING_MS = 9.5                # pt, ring around the selected trial (method colour)
RING_MEW = 1.0
PANEL_H = 1.16               # in, height of one panel row (plot area + its title)


def kl_x(kl):
    kl = np.asarray(kl, dtype=float)
    return np.where(kl <= 0, X_ZERO, kl)


def gen_of(study, gen):
    return gen if study == "main" else f"small-{gen}"


def draw_break(ax):
    """Two short slanted strokes on the x spine between the '0' edge and the first log decade."""
    xb = np.sqrt(X_ZERO * 1e-4)
    for dx in (0.93, 1.07):
        ax.plot([xb * dx / 1.06, xb * dx * 1.06], [-0.035, 0.035], transform=ax.get_xaxis_transform(),
                color=S.TEXT_SECONDARY, lw=S.LW_THIN, clip_on=False, zorder=6)
    ax.plot([xb * 0.9, xb * 1.1], [0, 0], transform=ax.get_xaxis_transform(), color=S.SURFACE, lw=1.6,
            clip_on=False, zorder=5, solid_capstyle="butt")


def style_axes(ax, show_x, show_y):
    ax.set_xscale("log")
    ax.set_xlim(X_LO, X_HI)
    ax.set_ylim(Y_LO, Y_HI)
    decades = [1e-4, 1e-3, 1e-2, 1e-1, 1, 10]
    ax.xaxis.set_major_locator(FixedLocator([X_ZERO] + decades))
    ax.xaxis.set_minor_locator(FixedLocator([k * d for d in decades for k in range(2, 10) if k * d < X_HI]))
    ax.xaxis.set_minor_formatter(NullFormatter())
    # narrow panels: label every other decade, always including the 0.1 selection threshold
    ax.set_xticklabels(["0", "", "0.001", "", "0.1", "", "10"])
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.tick_params(axis="x", which="minor", length=1.3, width=S.LW_THIN)
    ax.tick_params(labelbottom=show_x, labelleft=show_y)
    ax.grid(True, which="major")
    ax.axvline(KL_MAX, color=S.TEXT_SECONDARY, lw=0.9, linestyle=(0, (1, 1.5)), zorder=2)
    draw_break(ax)


def front(ax, t, method, gen, size=FRONT_MS):
    f = t[(t.method == method) & t.on_pareto_front].sort_values(["kl", "refusals"])
    if f.empty:
        return
    g = S.GEN_STYLE[gen]
    x, y = kl_x(f.kl), f.refusals.to_numpy()
    ax.step(x, y, where="post", color=S.color(method), linestyle=g["linestyle"], linewidth=S.LW * 0.9, zorder=3,
            clip_on=False)
    ax.plot(x, y, zorder=3.5 if gen.endswith("new") else 3.6, clip_on=False, **S.marker_kw(method, gen, size=size))


def ring(ax, t, method):
    sm = t[(t.method == method) & t["is_selected_kl0.1"]]
    ax.plot(kl_x(sm.kl), sm.refusals, linestyle="none", marker="o", markersize=RING_MS, markerfacecolor="none",
            markeredgecolor=S.color(method), markeredgewidth=RING_MEW, zorder=4)
    return int(sm.refusals.iloc[0]) if len(sm) else None


def original(ax, runs, model, gen):
    base = runs[(runs.model == model) & (runs.seed == SEED)].baseline_refusals.dropna().unique()
    if len(base):
        ax.plot([X_ZERO], [base[0]], clip_on=False, zorder=4.5 if gen.endswith("new") else 4.4,
                **S.marker_kw("base", gen, size=S.MS))


def selected_title(ax, entries):
    """Right-aligned panel title with the selected trial's dev refusal count, older -> newer release.
    entries: [(gen, value or None)]. Kept out of the plot area so that it can never hide a front."""
    vals = ["n/a" if v is None else f"{v}" for _, v in entries]
    ax.set_title(" → ".join(vals), loc="right", fontsize=S.FONT_SMALL, color=S.TEXT_SECONDARY,
                 pad=2)


def no_run(ax, text):
    ax.text(0.5, 0.5, text, transform=ax.transAxes, ha="center", va="center", fontsize=S.FONT_SMALL,
            color=S.TEXT_SECONDARY, style="italic")


# ================================================================================================ panels
def panel_main(ax, trials, runs, fam, old, new, method, study):
    entries = []
    for model, gen in ((old, gen_of(study, "old")), (new, gen_of(study, "new"))):
        t = trials[(trials.model == model) & (trials.method == method)]
        if t.empty:
            entries.append((gen, None))
            continue
        front(ax, t, method, gen)
        entries.append((gen, ring(ax, t, method)))
        original(ax, runs, model, gen)
    if all(v is None for _, v in entries):
        no_run(ax, "no run")
    selected_title(ax, entries)


def panel_trials(ax, trials, runs, model, gen, method):
    t = trials[(trials.model == model) & (trials.method == method)]
    if t.empty:
        no_run(ax, "no run")
        return
    g = S.GEN_STYLE[gen]
    c = S.color(method)
    ax.scatter(kl_x(t.kl), t.refusals, s=CLOUD_SIZE ** 2, marker="o",
               facecolors=c if g["filled"] else "none", edgecolors="none" if g["filled"] else c,
               linewidths=0 if g["filled"] else 0.6, alpha=CLOUD_ALPHA[method], zorder=1, clip_on=False)
    front(ax, t, method, gen)
    v = ring(ax, t, method)
    original(ax, runs, model, gen)
    selected_title(ax, [(gen, v)])


# ================================================================================================ figure
def build(trials, runs, kind):
    """kind: 'fronts' (fig2) or 'old' / 'new' (appendix trial figures)."""
    main_pairs = [p for p in S.PAIRS if p[0] != SMALL_FAMILY]
    small = [p for p in S.PAIRS if p[0] == SMALL_FAMILY][0]
    n_main = len(main_pairs)
    h = 0.42 + n_main * PANEL_H + 0.20 + (PANEL_H + 0.12) + 0.30
    fig = S.plt.figure(figsize=(S.TEXT_W, h))
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.015, wspace=0.0, hspace=0.0)
    sf_main, sf_small = fig.subfigures(2, 1, height_ratios=[n_main * PANEL_H + 0.20, PANEL_H + 0.12], hspace=0.0)
    rows = sf_main.subfigures(n_main, 1, hspace=0.0)
    all_axes = []

    def header(sf, fam, old, new):
        if kind == "fronts":
            txt = f"{S.MODEL_DISPLAY[old]} → {S.MODEL_DISPLAY[new]}"
        else:
            txt = S.MODEL_DISPLAY[old if kind == "old" else new]
        txt = f"{fam}:  {txt}"   # pair label first (Qwen, Gemma, OLMo, Mistral Small, Small pair)
        sf.suptitle(txt, x=0.0, ha="left", fontsize=S.FONT, fontweight="bold")

    for r, (fam, old, new) in enumerate(main_pairs):
        axs = rows[r].subplots(1, len(CORE), sharey=True, gridspec_kw={"wspace": 0.05})
        header(rows[r], fam, old, new)
        for c, (ax, m) in enumerate(zip(axs, CORE)):
            style_axes(ax, show_x=(r == n_main - 1), show_y=(c == 0))
            ax.set_title(S.label(m), loc="left", fontsize=S.FONT_SMALL, fontweight="bold", pad=2, color=S.TEXT)
            if kind == "fronts":
                panel_main(ax, trials, runs, fam, old, new, m, "main")
            else:
                panel_trials(ax, trials, runs, old if kind == "old" else new, kind, m)
            all_axes.append(ax)
    fam, old, new = small
    axs = sf_small.subplots(1, len(SMALL), sharey=True, gridspec_kw={"wspace": 0.05})
    header(sf_small, fam, old, new)
    for c, (ax, m) in enumerate(zip(axs, SMALL)):
        style_axes(ax, show_x=True, show_y=(c == 0))
        ax.set_title(S.label(m), loc="left", fontsize=S.FONT_SMALL, fontweight="bold", pad=2, color=S.TEXT)
        if kind == "fronts":
            panel_main(ax, trials, runs, fam, old, new, m, "small")
        else:
            panel_trials(ax, trials, runs, old if kind == "old" else new, f"small-{kind}", m)
        all_axes.append(ax)

    fig.supxlabel("First-token KL divergence from the original model (100 harmless prompts; log scale)",
                  fontsize=S.FONT)
    fig.supylabel("Dev keyword refusals (of 100)", fontsize=S.FONT)

    # legend: encodings only (the method is named in every panel title)
    gray = S.TEXT_SECONDARY
    gens = ("old", "new") if kind == "fronts" else (kind,)
    h_enc = []
    if kind != "fronts":
        g = S.GEN_STYLE[kind]
        h_enc.append(Line2D([], [], linestyle="none", marker="o", markersize=CLOUD_SIZE + 0.4,
                            markerfacecolor=gray if g["filled"] else "none", markeredgecolor=gray,
                            markeredgewidth=0 if g["filled"] else 0.6, alpha=0.6, label="each completed trial"))
    for gen in gens:
        h_enc.append(Line2D([], [], label=("Pareto front, older release" if gen == "old" else
                                           "Pareto front, newer release"),
                            **S.line_kw("base", gen, marker="o", size=FRONT_MS, color=gray, markeredgecolor=gray,
                                        **({"markerfacecolor": gray} if S.GEN_STYLE[gen]["filled"] else {}))))
    ring_h = Line2D([], [], linestyle="none", marker="o", markersize=RING_MS, markerfacecolor="none",
                    markeredgecolor=gray, markeredgewidth=RING_MEW, label="selected trial (KL ≤ 0.1 rule)")
    note_h = Line2D([], [], linestyle="none", label=("top right: selected count, older → newer" if kind == "fronts"
                                                     else "top right: selected count"))
    kl_h = Line2D([], [], color=gray, lw=0.9, linestyle=(0, (1, 1.5)), label="KL = 0.1")
    orig_h = Line2D([], [], label="original model" + ("s (KL = 0)" if kind == "fronts" else " (KL = 0)"),
                    **S.marker_kw("base", "new" if kind == "fronts" else kind, size=S.MS))
    # two rows, filled column by column: (front/trial, ring) (front, note) (KL, original)
    h_enc = h_enc[:2] + [ring_h, note_h, kl_h, orig_h]
    h_enc = [h_enc[0], h_enc[2], h_enc[1], h_enc[3], h_enc[4], h_enc[5]]
    fig.legend(handles=h_enc, loc="outside upper center", ncol=3, handlelength=2.2, columnspacing=2.0,
               handletextpad=0.5, borderaxespad=0.2, labelspacing=0.35, frameon=False)
    return fig


def main():
    S.apply()
    trials = S.load_trials()
    runs = S.load_runs()
    methods_all = SMALL
    # gamma-MPOA has its own figure (fig9); keep only the plotted methods BEFORE the sanity checks
    n_gamma_runs = trials[(trials.seed == SEED) & (trials.method == "gamma_mpoa")].groupby("model").ngroups
    trials = trials[(trials.seed == SEED) & trials.method.isin(methods_all)].copy()
    for c in ("on_pareto_front", "is_selected_kl0.1", "run_finished"):
        trials[c] = trials[c].astype(str).str.lower().eq("true")

    # sanity: every plotted run finished, every selection on its front and within the threshold
    assert trials.run_finished.all(), "a seed-1 run is not finished"
    sel = trials[trials["is_selected_kl0.1"]]
    assert sel.on_pareto_front.all() and (sel.kl <= KL_MAX).all()
    assert (sel.groupby(["model", "method"]).size() == 1).all()

    for kind, name in (("fronts", NAME), ("old", NAME_A["old"]), ("new", NAME_A["new"])):
        fig = build(trials, runs, kind)
        S.save(fig, name, preview_dpi=150)
        S.plt.close(fig)

    # ---------------------------------------------------------------- captions (numbers from the data)
    n_runs = trials.groupby(["model", "method"]).ngroups
    n_trials = len(trials)
    r1 = runs[(runs.seed == SEED) & runs.run.notna()].set_index(["model", "method"])
    short = trials.groupby(["model", "method"]).size()
    short = short[short < 100]
    short_txt = "; ".join(
        f"{S.MODEL_DISPLAY[m]} {S.label(me)} has {n} completed trials"
        + (f" ({int(r1.loc[(m, me), 'n_trials_failed'])} failed)" if (m, me) in r1.index
           and pd.notna(r1.loc[(m, me), "n_trials_failed"]) else "")
        for (m, me), n in short.items())
    zero = trials[trials.kl <= 0]
    zero_txt = "; ".join(f"{S.MODEL_DISPLAY[r.model]} {S.label(r.method)} trial {r.trial} ({r.refusals} refusals, "
                         f"equal to the original's)" if r.refusals == r1.loc[(r.model, r.method), "baseline_refusals"]
                         else f"{S.MODEL_DISPLAY[r.model]} {S.label(r.method)} trial {r.trial} ({r.refusals} refusals)"
                         for r in zero.itertuples())
    gamma_txt = ("$\\gamma$-MPOA (Gemma and OLMo pairs only) is not drawn here; its fronts are compared with "
                 "MPOA in Fig. 9. " if n_gamma_runs or (runs.method == "gamma_mpoa").any() else "")
    other = runs[runs.seed != SEED]
    seed_txt = ""
    if len(other):
        done = other[other.status.astype(str).str.startswith("present")]
        seed_txt = (f"Seed-2 runs (small pair, {len(done)} of {len(other)} finished at snapshot {S.snapshot_utc()}) "
                    "are not shown here; they are compared with seed 1 in Fig. 8.")
    bias = ("Refusal counts are keyword matches on the 100 dev prompts that were also used for optimisation and "
            "selection, not held-out refusal rates.")
    axis_txt = ("The x axis is logarithmic, with KL = 0 drawn at the left edge behind an axis break; the dotted "
                "line is the selection budget KL = 0.1.")

    caption = (
        "**Refusal--KL trade-off explored by each optimisation run (dev set).** Rows are model pairs (older "
        "$\\to$ newer release), columns refusal-removal methods; J-space was run on the small pair only. Each "
        "panel holds two Pareto fronts, one per release (one run each: 100 TPE trials, seed 1): the trials that no "
        "other trial of that run beats on both dev keyword refusals and first-token KL divergence; dashed with "
        "hollow markers for the older release, solid with filled markers for the newer one. A front that lies "
        "higher at a given KL means that more refusal survived at that divergence. The ring marks the trial "
        "chosen by the selection rule (fewest dev keyword refusals at KL $\\le$ 0.1); its count is given at the "
        "top right of the panel (older $\\to$ newer release). Grey stars are the original models (KL = 0). " + axis_txt + " "
        "Every completed trial, not only the fronts, is shown per run in Figs. A1 (older releases) and A2 (newer "
        "releases). " + gamma_txt + bias
    )
    notes = [x for x in (f"{n_trials} completed trials from {n_runs} runs; the small pair was optimised on an "
                         "RTX 4090, all other runs on an H200.",
                         short_txt + "." if short_txt else "",
                         f"KL exactly 0 (drawn on the left edge): {zero_txt}." if zero_txt else "",
                         seed_txt) if x]
    if notes:
        caption += "\n\n*Notes (for the text):* " + " ".join(notes)
    S.write_caption(NAME, caption, source=S.SOURCE_DEV + " " + S.SOURCE_SELECTION)

    for gen, word, other_fig in (("old", "older", "A2"), ("new", "newer", "A1")):
        models = [p[1] if gen == "old" else p[2] for p in S.PAIRS]
        tg = trials[trials.model.isin(models)]
        cap = (
            f"**All completed TPE trials of each optimisation run, {word} releases (dev set).** One panel per run "
            f"(model $\\times$ method, seed 1), laid out as in Fig. 2; the {word} model of each pair heads its row. "
            "Each dot is one completed trial; the line is the run's Pareto front ("
            + ("dashed, hollow markers" if gen == "old" else "solid, filled markers")
            + "), the ring the trial chosen by the selection rule (fewest dev keyword refusals at KL $\\le$ 0.1; "
            "count at the top right of the panel) and the grey star the original model (KL = 0). " + axis_txt + " "
            f"{len(tg)} trials from {tg.groupby(['model', 'method']).ngroups} runs; the {('newer' if gen == 'old' else 'older')} "
            f"releases are in Fig. {other_fig}. " + bias
        )
        S.write_caption(NAME_A[gen], cap, source=S.SOURCE_DEV + " " + S.SOURCE_SELECTION)


if __name__ == "__main__":
    main()
