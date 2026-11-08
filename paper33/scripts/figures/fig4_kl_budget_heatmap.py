"""fig4_kl_budget_heatmap -- how many dev refusals are left at a given KL budget?

For every model x method optimisation run (seed 1) and every KL budget t in {0.05, 0.1, 0.2, 0.5}: the fewest dev
keyword refusals (count of 100) among the run's COMPLETED trials whose dev first-token KL divergence is <= t.
This is the selection rule of scripts/select_trial.py evaluated at other budgets; the t = 0.1 column is the rule the
paper uses, so it equals the selected model's dev refusals (checked below against runs.csv and, where present,
paper/tables/t1_dev_refusals.csv).

Layout: one panel per family, three per row (Qwen, Gemma, OLMo / Mistral Small, Qwen small pair, key), columns = KL
budget, rows = method, grouped into an upper block (older release) and a lower block (newer release). (Until
2026-11-05 all five families sat side by side, which left about 0.2 in per cell.) Each block starts with the
unedited original (KL 0, inside every budget) as one merged cell. Sequential single-hue scale (light = few
refusals left, dark = many), value printed in every cell with contrast-chosen ink. Pending conditions (planned
gamma-MPOA runs, or runs still running) are hatched and labelled "pending". Two honesty marks, both appended to the
printed value so they survive print size:
  * asterisk  "13*"  -> budget not binding: every completed trial of the run is already within it, so a larger
                        budget was never explored by the search (the value is the run's overall minimum);
  * dagger    "13†"  -> fewer than SPARSE_N completed trials fall within the budget (minimum over very few points).
Width: 6.5 in (template text width), include at width=\textwidth.

Run:  cd scripts/figures && .venv/bin/python fig4_kl_budget_heatmap.py
In:   paper/figures/data/trials.csv, runs.csv (+ paper/tables/t1_dev_refusals.csv for a cross-check, if present)
Out:  paper/figures/fig4_kl_budget_heatmap.pdf, paper/figures/preview/fig4_kl_budget_heatmap.png (150 dpi),
      '## fig4_kl_budget_heatmap' section of paper/figures/captions.md
"""
from __future__ import annotations

import math
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, to_hex
from matplotlib.patches import Rectangle

import style as S

NAME = "fig4_kl_budget_heatmap"
THRESHOLDS = [0.05, 0.1, 0.2, 0.5]
SELECTION_KL = 0.1
SEED = 1
SPARSE_N = 10                       # dagger when fewer completed trials than this are within the budget
METHODS = ["dim", "mpoa", "ara", "som", "jspace", "gamma_mpoa"]   # row order inside a block (after Original)
MAX_METHOD_ROWS = 5                 # most method rows any model has (4 core + J-space or gamma-MPOA)

CMAP = S.seq_cmap()
NORM = Normalize(vmin=0, vmax=100)

# ------------------------------------------------------------------------------------------------ geometry (inches)
# Three panels per row (row 1: Qwen, Gemma, OLMo; row 2: Mistral Small, Qwen small pair, key). The earlier layout put
# all five families side by side, which left ~0.2 in per cell; three across gives each cell about twice the width.
W = S.TEXT_W
MARGIN_L, MARGIN_R = 0.02, 0.02
NCOLS = 3
LABEL_W = 0.50                      # method-label column of each panel
PANEL_GAP = 0.20                    # horizontal gap between panels
RH = 0.150                          # row height
BLOCK_GAP = 0.08                    # between old and new block
CELL_PAD = 0.011                    # half of the 2-px surface gap between cells
NONBIND = "*"                       # appended to a value whose budget is not binding
TOP = 0.04                          # first panel row starts here
P_FAMILY = 0.09                     # family title (centre), relative to the panel top
P_COLHEAD = 0.25                    # KL-budget column heads (centre)
P_GRID = 0.34                       # first block starts here
ROW_GAP = 0.16                      # vertical gap between the two panel rows


def _flag(v):
    return v is True or str(v) == "True"


# ================================================================================================ data
def build_table():
    trials = S.load_trials()
    runs = S.load_runs()
    trials = trials[trials.seed == SEED]
    runs = runs[runs.seed == SEED]

    base = (runs.dropna(subset=["baseline_refusals"]).groupby("model")["baseline_refusals"]
            .agg(lambda s: sorted(set(s))))
    for m, vals in base.items():
        if len(vals) != 1:
            print(f"  WARNING: {m}: runs disagree on the original's dev refusals {vals}; using the first")

    blocks, rows = [], []
    for family, old, new in S.PAIRS:
        for gen, model in (("old", old), ("new", new)):
            b = {"family": family, "model": model, "gen": gen,
                 "baseline": int(base[model][0]) if model in base else None, "rows": []}
            for method in METHODS:
                rr = runs[(runs.model == model) & (runs.method == method)]
                if rr.empty:
                    continue                      # not part of the design (e.g. J-space on the main study)
                r = rr.iloc[0]
                t = trials[(trials.model == model) & (trials.method == method)]
                status = str(r.status)
                row = {"family": family, "model": model, "gen": gen, "method": method, "run": r.run,
                       "status": status, "n_complete": len(t)}
                if not status.startswith("present") or not _flag(r.run_finished) or t.empty:
                    if status.startswith("partial"):
                        row["pending"] = f"pending ({int(r.n_trials_complete)}/{int(r.n_trials_planned)})"
                    else:
                        row["pending"] = S.PENDING_LABEL
                else:
                    row["pending"] = None
                    for th in THRESHOLDS:
                        within = t[t.kl <= th + 1e-12]
                        n = len(within)
                        row[f"min_{th}"] = int(within.refusals.min()) if n else None
                        row[f"n_{th}"] = n
                    sel = row[f"min_{SELECTION_KL}"]
                    if pd.notna(r.selected_refusals) and sel != int(r.selected_refusals):
                        print(f"  WARNING: {model} {method}: KL<={SELECTION_KL} minimum {sel} != runs.csv "
                              f"selected_refusals {int(r.selected_refusals)}")
                b["rows"].append(row)
                rows.append(row)
            blocks.append(b)
    return blocks, pd.DataFrame(rows)


def cross_check_t1(df):
    """Compare with paper/tables/t1_dev_refusals.csv (present cells only); the table may be older."""
    p = S.ROOT / "paper" / "tables" / "t1_dev_refusals.csv"
    if not p.exists():
        return
    t1 = pd.read_csv(p)
    t1 = t1[(t1.status == "present") & (t1.method != "base") & t1.kl_max.notna()]
    n_ok, bad = 0, []
    for _, r in t1.iterrows():
        hit = df[(df.model == r.model) & (df.method == r.method)]
        if hit.empty or pd.notna(hit.iloc[0]["pending"]):
            continue
        mine = hit.iloc[0].get(f"min_{float(r.kl_max)}")
        if mine is None or (isinstance(mine, float) and math.isnan(mine)):
            continue
        if int(mine) == int(r.refusals_count):
            n_ok += 1
        else:
            bad.append(f"{r.model} {r.method} KL<={r.kl_max}: table {r.refusals_count}, here {mine}")
    print(f"  cross-check vs paper/tables/t1_dev_refusals.csv: {n_ok} equal, {len(bad)} differ")
    for b in bad:
        print("    DIFFERS:", b)


# ================================================================================================ drawing
def panel_geometry():
    """Panel width, cell width and the (x0, y0) of every grid slot (row-major, NCOLS per row)."""
    pw = (W - MARGIN_L - MARGIN_R - (NCOLS - 1) * PANEL_GAP) / NCOLS
    cw = (pw - LABEL_W) / len(THRESHOLDS)
    block_rows = 2 + MAX_METHOD_ROWS                       # header + Original + methods
    ph = P_GRID + 2 * block_rows * RH + BLOCK_GAP           # panel height
    slots = []
    for k in range(2 * NCOLS):
        r, c = divmod(k, NCOLS)
        slots.append((MARGIN_L + c * (pw + PANEL_GAP), TOP + r * (ph + ROW_GAP)))
    return pw, cw, ph, slots


def cell(ax, x, y, w, h, face, **kw):
    ax.add_patch(Rectangle((x + CELL_PAD, y + CELL_PAD), w - 2 * CELL_PAD, h - 2 * CELL_PAD, facecolor=face,
                           edgecolor="none", linewidth=0, **kw))


def draw(blocks):
    block_rows = 2 + MAX_METHOD_ROWS                       # header + Original + methods
    pw, cw, ph, slots = panel_geometry()
    H = TOP + 2 * ph + ROW_GAP + 0.06

    fig = plt.figure(figsize=(W, H))
    fig.set_layout_engine("none")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)                                       # inches, y downwards
    ax.axis("off")

    fam_index = {p[0]: i for i, p in enumerate(S.PAIRS)}
    fam_titles = {}
    sparse_used = nonbinding_used = pending_used = empty_used = False
    texts_to_fit = []                                       # (text artist, max right edge in inches)

    for family, _, _ in S.PAIRS:
        x0, y0 = slots[fam_index[family]]
        xc = x0 + LABEL_W
        right = xc + len(THRESHOLDS) * cw
        t = ax.text(x0, y0 + P_FAMILY, fam_titles.get(family, family), fontsize=S.FONT, fontweight="bold",
                    color=S.TEXT, ha="left", va="center")
        texts_to_fit.append((t, right))
        ax.text(xc - 0.04, y0 + P_COLHEAD, "KL ≤", fontsize=S.FONT_SMALL, ha="right", va="center",
                color=S.TEXT_SECONDARY)
        for j, th in enumerate(THRESHOLDS):
            sel = math.isclose(th, SELECTION_KL)
            ax.text(xc + (j + 0.5) * cw, y0 + P_COLHEAD, f"{th:g}", fontsize=S.FONT_SMALL, ha="center", va="center",
                    color=S.TEXT if sel else S.TEXT_SECONDARY, fontweight="bold" if sel else "normal")

    for b in blocks:
        x0, y0 = slots[fam_index[b["family"]]]
        xc = x0 + LABEL_W
        right = xc + len(THRESHOLDS) * cw
        y = y0 + P_GRID + (0 if b["gen"] == "old" else block_rows * RH + BLOCK_GAP)

        # header: model name + generation
        t = ax.text(x0, y + RH / 2, S.MODEL_DISPLAY[b["model"]], fontsize=S.FONT_SMALL, color=S.TEXT,
                    ha="left", va="center", fontweight="bold")
        fig.canvas.draw_idle()
        bb = t.get_window_extent(renderer=fig.canvas.get_renderer())
        x_after = bb.x1 / fig.dpi + 0.05
        t2 = ax.text(x_after, y + RH / 2, "(older release)" if b["gen"] == "old" else "(newer release)",
                     fontsize=S.FONT_SMALL, color=S.TEXT_SECONDARY, ha="left", va="center")
        texts_to_fit.append((t2, right))
        y += RH

        # Original: one merged cell (KL 0 is inside every budget)
        ax.text(xc - 0.04, y + RH / 2, S.METHOD_LABELS["base"], fontsize=S.FONT_SMALL, color=S.TEXT,
                ha="right", va="center")
        if b["baseline"] is not None:
            face = to_hex(CMAP(NORM(b["baseline"])))
            cell(ax, xc, y, len(THRESHOLDS) * cw, RH, face)
            ax.text(xc + len(THRESHOLDS) * cw / 2, y + RH / 2, f"{b['baseline']}  (KL 0)", fontsize=S.FONT_SMALL,
                    color=S.ink_on(face), ha="center", va="center")
        else:
            S.draw_pending(ax, xc + CELL_PAD, y + CELL_PAD, len(THRESHOLDS) * cw - 2 * CELL_PAD, RH - 2 * CELL_PAD)
            pending_used = True
        y += RH

        sel_col = THRESHOLDS.index(SELECTION_KL)
        sel_rows = []
        for r in b["rows"]:
            ax.text(xc - 0.04, y + RH / 2, S.METHOD_LABELS[r["method"]], fontsize=S.FONT_SMALL, color=S.TEXT,
                    ha="right", va="center")
            if r["pending"]:
                S.draw_pending(ax, xc + CELL_PAD, y + CELL_PAD, len(THRESHOLDS) * cw - 2 * CELL_PAD,
                               RH - 2 * CELL_PAD, text=r["pending"])
                pending_used = True
            else:
                sel_rows.append(y)
                for j, th in enumerate(THRESHOLDS):
                    v, n = r[f"min_{th}"], r[f"n_{th}"]
                    x = xc + j * cw
                    if v is None:                        # no completed trial within this budget
                        cell(ax, x, y, cw, RH, S.NEUTRAL_MID)
                        ax.text(x + cw / 2, y + RH / 2, "–", fontsize=S.FONT_SMALL, color=S.TEXT_SECONDARY,
                                ha="center", va="center")
                        empty_used = True
                        continue
                    face = to_hex(CMAP(NORM(v)))
                    ink = S.ink_on(face)
                    cell(ax, x, y, cw, RH, face)
                    sparse = n < SPARSE_N
                    nonbinding = n == r["n_complete"]
                    label = f"{v}" + ("†" if sparse else "") + (NONBIND if nonbinding else "")
                    tt = ax.text(x + cw / 2, y + RH / 2, label, fontsize=S.FONT_SMALL, color=ink, ha="center",
                                 va="center")
                    texts_to_fit.append((tt, x + cw - CELL_PAD))
                    sparse_used |= sparse
                    nonbinding_used |= nonbinding
            y += RH

        # outline the selection budget column over the rows that have values
        if sel_rows:
            top, bot = min(sel_rows), max(sel_rows) + RH
            ax.add_patch(Rectangle((xc + sel_col * cw, top), cw, bot - top, facecolor="none", edgecolor=S.TEXT,
                                   linewidth=1.0, zorder=4))

    # ---------------------------------------------------------------------------------------- key (sixth slot)
    rend = fig.canvas.get_renderer()
    kx, ky = slots[len(S.PAIRS)]
    kx += 0.15
    cb_w, cb_h = pw - 0.35, 0.10
    ax.text(kx, ky + P_FAMILY, "Key", fontsize=S.FONT, fontweight="bold", color=S.TEXT, ha="left", va="center")
    ax.text(kx, ky + P_COLHEAD + 0.02, "Dev keyword refusals left (of 100)", fontsize=S.FONT_SMALL, color=S.TEXT,
            ha="left", va="center")
    y_cb = ky + P_COLHEAD + 0.12
    cax = fig.add_axes([kx / W, 1 - (y_cb + cb_h) / H, cb_w / W, cb_h / H])
    cb = fig.colorbar(ScalarMappable(norm=NORM, cmap=CMAP), cax=cax, orientation="horizontal")
    cb.set_ticks([0, 20, 40, 60, 80, 100])
    cb.outline.set_linewidth(S.LW_THIN)
    cb.outline.set_edgecolor(S.AXIS)
    cax.grid(False)
    cax.tick_params(labelsize=S.FONT_SMALL, length=2, pad=1.5)

    any_partial = any(str(r["pending"]).startswith("pending (") for b in blocks for r in b["rows"] if r["pending"])
    items = []
    if pending_used:
        items.append(("pending", "pending: planned or still running" if any_partial
                      else "pending: planned (γ-MPOA), not run yet"))
    items.append(("outline", f"KL ≤ {SELECTION_KL:g}: selection budget\n(= dev count of the selected trial)"))
    if nonbinding_used:
        items.append(("nonbind", "budget not binding: every trial\nof the run lies within it"))
    if sparse_used:
        items.append(("dagger", f"fewer than {SPARSE_N} trials\nwithin the budget"))
    if empty_used:
        items.append(("empty", "no completed trial within budget"))
    sw_w, sw_h, sw_gap = 0.28, RH - 2 * CELL_PAD, 0.07
    yy = y_cb + cb_h + 0.42
    face = to_hex(CMAP(NORM(40)))
    for kind, text in items:
        top = yy - sw_h / 2
        t = ax.text(kx + sw_w + sw_gap, yy, text, fontsize=S.FONT_SMALL, color=S.TEXT, ha="left", va="center",
                    linespacing=1.15)
        texts_to_fit.append((t, W - MARGIN_R))
        if kind == "pending":
            S.draw_pending(ax, kx, top, sw_w, sw_h, text=None)
        elif kind == "outline":
            ax.add_patch(Rectangle((kx, top), sw_w, sw_h, facecolor=face, edgecolor=S.TEXT, linewidth=1.0))
        elif kind == "nonbind":
            ax.add_patch(Rectangle((kx, top), sw_w, sw_h, facecolor=face, edgecolor="none"))
            ax.text(kx + sw_w / 2, yy, "13" + NONBIND, fontsize=S.FONT_SMALL, color=S.ink_on(face), ha="center",
                    va="center")
        elif kind == "dagger":
            ax.add_patch(Rectangle((kx, top), sw_w, sw_h, facecolor=face, edgecolor="none"))
            ax.text(kx + sw_w / 2, yy, "13†", fontsize=S.FONT_SMALL, color=S.ink_on(face), ha="center",
                    va="center")
        elif kind == "empty":
            ax.add_patch(Rectangle((kx, top), sw_w, sw_h, facecolor=S.NEUTRAL_MID, edgecolor="none"))
            ax.text(kx + sw_w / 2, yy, "–", fontsize=S.FONT_SMALL, color=S.TEXT_SECONDARY, ha="center",
                    va="center")
        yy += 0.36

    # ---------------------------------------------------------------------------------------- fit check
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    for t, limit in texts_to_fit:
        x1 = t.get_window_extent(renderer=rend).x1 / fig.dpi
        if x1 > limit + 1e-3:
            print(f"  WARNING: text {t.get_text()!r} ends at {x1:.3f} in, beyond {limit:.3f} in")
    if yy - 0.36 + 0.15 > H:
        print(f"  WARNING: key runs below the figure ({yy:.2f} in > {H:.2f} in)")
    return fig


# ================================================================================================ caption
def caption(df):
    done = df[df.pending.isna()]
    n_runs = len(done)
    odd = done[done.n_complete != 100]
    odd_txt = "; ".join(f"{S.MODEL_DISPLAY[r.model]} {S.METHOD_LABELS[r.method]}: {r.n_complete}"
                        for r in odd.itertuples())
    pend = df[df.pending.notna()]
    pend_txt = ", ".join(f"{S.MODEL_DISPLAY[r.model]}" for r in pend.itertuples())
    # pending rows, described from their state (run still running: 'pending (k/n)'; no run yet: 'pending')
    pp = []
    for r in pend.itertuples():
        st = str(r.pending)
        m_ = re.search(r"\((\d+/\d+)\)", st)
        pp.append(f"{S.METHOD_LABELS[r.method]} on {S.MODEL_DISPLAY[r.model]}, "
                  + (f"run in progress, {m_.group(1)} trials" if m_ else "not run yet"))
    pend_sentence = (("Hatched rows are pending (" + "; ".join(pp) + "). ") if len(pp) > 1 else
                     ("The hatched row is pending (" + pp[0] + "). ") if pp else "")
    nb = sum(int(r[f"n_{th}"] == r["n_complete"]) for _, r in done.iterrows() for th in THRESHOLDS)
    sp = sum(int(r[f"n_{th}"] < SPARSE_N) for _, r in done.iterrows() for th in THRESHOLDS)
    caption_ = (
        "**How much refusal is left at a given KL budget (dev set).** Each cell is the fewest dev keyword refusals "
        "(of 100) among the completed TPE trials of one optimisation run whose first-token KL "
        "divergence from the original is at most the column budget $t$. Within each family the upper block is the "
        "older and the lower block the newer release; each block starts with the unedited original (KL 0). Darker "
        "cells mean more refusal survives. The outlined $t = 0.1$ column is the selection budget, so it equals the "
        f"dev refusals of the selected models. An asterisk marks a budget that is not binding ({nb} cells: every "
        "completed trial already lies within it, so larger budgets were never explored and the value is the run's "
        f"overall minimum); a dagger marks a minimum over fewer than {SPARSE_N} trials ({sp} cells). "
        + pend_sentence +
        "The columns are post-hoc budgets on one 100-trial "
        "search per run (seed 1), not separate optimisations, and the counts are dev-set numbers, not held-out "
        "refusal rates."
    )
    detail = (f"*For the text:* {n_runs} runs; J-space on the small pair only; $\\gamma$-MPOA on the Gemma and OLMo "
              "pairs only; " + (f"pending rows: {pend_txt}; " if pend_txt else "")
              + (f"completed trials: 100 each except {odd_txt}." if odd_txt else "100 completed trials each."))
    return caption_ + "\n\n" + detail


def main():
    S.apply()
    blocks, df = build_table()
    pd.set_option("display.width", 200)
    cols = ["model", "method", "n_complete"] + [f"min_{t}" for t in THRESHOLDS] + [f"n_{t}" for t in THRESHOLDS]
    print(df[[c for c in cols if c in df.columns] + ["pending"]].to_string(index=False))
    done = df[df.pending.isna()]
    for th_a, th_b in zip(THRESHOLDS, THRESHOLDS[1:]):      # minimum over a superset can only fall
        assert (done[f"min_{th_b}"].fillna(101) <= done[f"min_{th_a}"].fillna(101)).all(), (th_a, th_b)
    cross_check_t1(df)
    fig = draw(blocks)
    S.save(fig, NAME, preview_dpi=150)
    S.write_caption(NAME, caption(df), source=S.SOURCE_DEV + " " + S.SOURCE_SELECTION)
    plt.close(fig)


if __name__ == "__main__":
    main()
