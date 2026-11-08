#!/usr/bin/env python3
"""fig6_divergence.py -- how far each refusal-removed model drifts from its original on benign text.

Cleveland dot plot of the held-out teacher-forced divergence (results.csv `tf_divergence`), defined as in the
methods (paper/sections/04_methods.tex) and scripts/bench/bench.py: the token-averaged
    log p_orig(y) - log p_mod(y)
over the ORIGINAL model's own greedy answers y to the 350 benign held-out prompts. 0 = the original model; > 0 = the
derived model finds the original's answers less likely (it has moved away from the original). `tf_divergence` is
-`tf_shift` (results.csv keeps both).

Layout: one panel per model family (S.PAIRS order) plus a legend cell; inside a panel one row per method, each row
with two lanes: old generation on top (hollow marker), new generation below (filled marker), joined by a thin grey
connector when both are present. Colour + marker shape = method. Same x range in every panel. Community reference
models are one row per public release (taken from results.csv, i.e. from the evaluation queues), labelled by
Hugging Face uploader; releases by the same uploader share a row across the two generations. Cells without a
finished, paired teacher-forcing value are written into their lane ("pending (...)", "planned", "none queued"),
never dropped. Width: 6.5 in (template text width).
Since 2026-11-05 the community rows are OFF here (SHOW_COMMUNITY = False): their uploader labels crowded the panels;
the eight public releases are drawn in figA3_community_releases (with the range of this figure's values for the same
original as context). The legend cell now stacks the Method and Generation keys (side by side they ran off the page).

Reads only paper/figures/data/results.csv. Run with scripts/figures/.venv/bin/python (cwd scripts/figures).
"""
from __future__ import annotations

import math

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
from matplotlib.transforms import ScaledTranslation, blended_transform_factory

import style as S

NAME = "fig6_divergence"
LANE = 0.22           # lane offset from the row centre (row pitch = 1)
ROW_PITCH = 0.25      # in per row (was 0.215 with the community rows)
SHOW_COMMUNITY = False   # community releases: figA3_community_releases
OLD_GENS = ("old", "small-old")


def owner(row):
    """Hugging Face uploader of a community release (results.csv `repo`), else its variant token."""
    repo = str(row.get("repo") or "")
    return repo.split("/")[0] if "/" in repo else str(row.get("variant") or "community")


def short_status(row):
    """Short lane text for a cell without a plotted value (None if it has one)."""
    if row is None:
        return "n/a"
    st = str(row["status"])
    if S.is_pending(st):
        if st.startswith("planned"):
            return "planned"
        if "started" in st or "in progress" in st:
            return "pending (running)"
        if "not queued" in st:
            return "pending (not queued)"
        if "queued" in st:
            return "pending (queued)"
        return "pending"
    if not bool(row["finished"]) or not math.isfinite(row["tf_divergence"]):
        return "pending (no teacher forcing)"
    if str(row["tf_paired"]) != "True":
        return "unpaired (not shown)"
    return None


def main():
    S.apply()
    res = S.load_results()
    der = res[(res["method"] != "base") & ((res["method"] != "community") | SHOW_COMMUNITY)].copy()
    der["tf_divergence"] = der["tf_divergence"].astype(float)

    def cells(model, method, partner):
        sub = der[(der["model"] == model) & (der["method"] == method)].sort_values(["variant", "label"],
                                                                                    na_position="last")
        rows = [r for _, r in sub.iterrows()]
        if method == "community":   # uploaders shared with the other generation first, so they line up
            other = {owner(r) for _, r in der[(der["model"] == partner) & (der["method"] == method)].iterrows()}
            rows.sort(key=lambda r: (owner(r) not in other, owner(r).lower()))
        return rows

    # ---------------------------------------------------------------- panel content
    panels = []
    for fam, old, new in S.PAIRS:
        rows = []
        for m in S.METHOD_ORDER:
            if m == "base":
                continue
            lo_, hi_ = cells(old, m, new), cells(new, m, old)
            for k in range(max(len(lo_), len(hi_))):
                lanes = []
                for model, lst, lane_y in ((old, lo_, -LANE), (new, hi_, +LANE)):
                    r = lst[k] if k < len(lst) else None
                    text = short_status(r)
                    if m == "community":    # short form: uploader + bare status ("queued", "running", ...)
                        bare = (text or "").replace("pending (", "").rstrip(")") or None
                        text = (f"{owner(r)}, {bare}" if text is not None else None) if r is not None \
                            else "none queued"
                    lanes.append({"model": model, "row": r, "dy": lane_y,
                                  "gen": None if r is None else r["generation"], "text": text})
                rows.append((m, lanes))
        panels.append((fam, old, new, rows))

    plotted = der[[short_status(r) is None for _, r in der.iterrows()]]
    vals = plotted["tf_divergence"].to_numpy(float)
    hi = max(0.10, math.ceil((vals.max() + 0.004) / 0.05) * 0.05) + 0.012 if len(vals) else 0.10
    lo = -0.012 if not len(vals) or vals.min() >= 0 else math.floor(vals.min() / 0.05) * 0.05 - 0.012

    # ---------------------------------------------------------------- figure
    top_n = max(len(p[3]) for p in panels[:3])
    bot_n = max(len(p[3]) for p in panels[3:])
    fig = plt.figure(figsize=(S.TEXT_W, 0.62 + ROW_PITCH * (top_n + bot_n) + 0.95))
    gs = fig.add_gridspec(2, 3, height_ratios=[top_n, bot_n])
    axes = [fig.add_subplot(gs[i // 3, i % 3]) for i in range(6)]

    for ax, (fam, old, new, rows) in zip(axes, panels):
        n_rows = top_n if ax in axes[:3] else bot_n
        trans = blended_transform_factory(ax.transAxes, ax.transData)
        ax.set_xlim(lo, hi)
        ax.set_ylim(n_rows - 0.5, -0.5)
        ax.xaxis.set_major_locator(MultipleLocator(0.1))
        ax.xaxis.set_minor_locator(MultipleLocator(0.05))
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", which="both", color=S.GRID, linewidth=S.LW_THIN)
        ax.tick_params(axis="x", which="minor", length=1.5, color=S.AXIS)
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
        ax.axvline(0, color=S.TEXT_SECONDARY, linewidth=0.9, zorder=1.6)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([S.label(m) for m, _ in rows], color=S.TEXT, fontsize=S.FONT_SMALL)
        for tl, (m, _) in zip(ax.get_yticklabels(), rows):
            if m == "community":
                tl.set_color(S.METHOD_COLORS["community"])
        ax.set_title(f"{S.MODEL_DISPLAY[old]} → {S.MODEL_DISPLAY[new]}", loc="left",
                     fontsize=S.FONT_SMALL, color=S.TEXT_SECONDARY, pad=2)
        ax.annotate(fam, xy=(0, 1), xycoords="axes fraction", xytext=(0, 11.5), textcoords="offset points",
                    ha="left", va="bottom", fontsize=S.FONT, fontweight="bold", color=S.TEXT)

        for i, (m, lanes) in enumerate(rows):
            for ln in lanes:   # lane guide lines (Cleveland dot-plot rules)
                ax.axhline(i + ln["dy"], color=S.GRID, linewidth=S.LW_THIN, zorder=0.5)
            xs = {}
            for ln in lanes:
                if ln["text"] is None:
                    x = float(ln["row"]["tf_divergence"])
                    xs[ln["dy"]] = x
                    ax.plot([x], [i + ln["dy"]], zorder=3, **S.marker_kw(m, ln["gen"]))
                    if m == "community":    # name the release next to its marker
                        ax.annotate(owner(ln["row"]), (x, i + ln["dy"]), xytext=(5, 0), textcoords="offset points",
                                    va="center", fontsize=S.FONT_SMALL, color=S.TEXT_SECONDARY)
            if len(xs) == 2:  # thin neutral connector old -> new
                ax.plot([xs[-LANE], xs[LANE]], [i - LANE, i + LANE], color=S.AXIS, linewidth=0.8, zorder=2)
            texts = [ln["text"] for ln in lanes]
            # status text sits at the right end of the lane (values cluster near 0, on the left)
            txt_kw = dict(transform=trans, ha="right", va="center", fontsize=S.FONT_SMALL, color=S.TEXT_SECONDARY,
                          style="italic", zorder=4,
                          bbox={"boxstyle": "square,pad=0.05", "facecolor": S.SURFACE, "edgecolor": "none"})
            if all(t is not None for t in texts) and texts[0] == texts[1]:   # same status in both lanes: one line
                ax.text(0.98, i, f"old + new: {texts[0]}", **txt_kw)
                continue
            for ln in lanes:
                if ln["text"] is not None:
                    tag = "old: " if ln["dy"] < 0 else "new: "
                    ax.text(0.98, i + ln["dy"], tag + ln["text"], **txt_kw)

    # ---------------------------------------------------------------- legend cell (Method above Generation)
    lax = axes[5]
    lax.axis("off")
    methods_shown = [m for m in S.METHOD_ORDER if m in set(plotted["method"])]
    top = 1.22   # legends may use the title band above the cell (the other panels' titles sit there)
    # ... and the empty y-label margin to its left (the cell has no tick labels): start 0.45 in left of the cell
    cell = lax.transAxes + ScaledTranslation(-0.45, 0, fig.dpi_scale_trans)
    leg1 = lax.legend(handles=S.method_handles(methods_shown, "new"), title="Method", loc="upper left",
                      bbox_to_anchor=(0.0, top), bbox_transform=cell, handlelength=1.2, handletextpad=0.5,
                      labelspacing=0.35, columnspacing=1.2, ncol=2, alignment="left")
    leg1.set_in_layout(False)   # placed by hand inside the cell; must not resize the panel grid
    lax.add_artist(leg1)
    gen_h = S.generation_handles(marker="o")
    for h in gen_h:
        h.set_linestyle("none")
    gen_h[0].set_label("older release (upper lane)")
    gen_h[1].set_label("newer release (lower lane)")
    zero_h = Line2D([], [], color=S.TEXT_SECONDARY, linewidth=0.9, label="original model (0)")
    # the Generation legend starts below the Method legend's MEASURED lower edge
    fig.canvas.draw()
    y_low = cell.inverted().transform(leg1.get_window_extent(fig.canvas.get_renderer()).p0)[1]
    leg2 = lax.legend(handles=gen_h + [zero_h], title="Generation", loc="upper left",
                      bbox_to_anchor=(0.0, y_low - 0.04), bbox_transform=cell,
                      handlelength=1.2, handletextpad=0.5, labelspacing=0.35, alignment="left")
    leg2.set_in_layout(False)

    n_all = len(der)
    n_plot = len(plotted)
    stat = der["status"].astype(str)
    n_planned = int(stat.str.startswith("planned").sum())
    n_pending = n_all - n_plot - n_planned
    if n_pending or n_planned:   # only while some cells have no value yet
        note = lax.text(1.0, -0.04, f"{n_plot} of {n_all} cells shown;\n{n_pending} pending, {n_planned} planned\n"
                        f"(snapshot {S.snapshot_utc()}).", transform=lax.transAxes, ha="right", va="bottom",
                        fontsize=S.FONT_SMALL, color=S.TEXT_SECONDARY, linespacing=1.3)
        note.set_in_layout(False)

    fig.supxlabel("Teacher-forced divergence from the original model (nats per token, held-out benign prompts; "
                  "higher = further away)", fontsize=S.FONT, x=0.5)
    fig.get_layout_engine().set(h_pad=0.04, w_pad=0.04, hspace=0.06, wspace=0.04)
    S.save(fig, NAME, preview_dpi=150)

    # ---------------------------------------------------------------- caption
    groups = {}   # lane text -> cells, in figure order
    for fam, old, new, rows in panels:
        for m, lanes in rows:
            for ln in lanes:
                if ln["text"] not in (None, "n/a", "none queued") and ln["row"] is not None:
                    key = short_status(ln["row"])
                    lab = S.label(m) + (f" ({owner(ln['row'])})" if m == "community" else "")
                    groups.setdefault(key, []).append(f"{S.MODEL_DISPLAY[ln['model']]} {lab}")
    by_gen = plotted.assign(old=plotted["generation"].isin(OLD_GENS))

    def num(v):
        return f"{v:.3f}"

    if len(vals):
        worst = plotted.loc[plotted["tf_divergence"].idxmax()]
        least = plotted.loc[plotted["tf_divergence"].idxmin()]
        shown = (f"{n_plot} of {n_all} cells are shown at the data snapshot ({S.snapshot_utc()}); divergences range "
                 f"from {num(least['tf_divergence'])} ({S.MODEL_DISPLAY[least['model']]} {S.label(least['method'])}) "
                 f"to {num(worst['tf_divergence'])} ({S.MODEL_DISPLAY[worst['model']]} {S.label(worst['method'])}) "
                 "nats per token. ")
    else:
        shown = f"0 of {n_all} cells are shown at the data snapshot ({S.snapshot_utc()}). "
    small_shown = plotted[plotted["model"] == "qwen3.5-4b"]
    caveat = ("The Qwen3.5-4B values also contain an export difference: its evaluated checkpoints lack the 15 "
              "mtp.* tensors and store 48 F32 linear-attention tensors (A\\_log, norm.weight) as BF16, which by "
              "itself changes the log-probabilities; an export-only control (zero edit) would size this. "
              if len(small_shown) else "")
    caption = (
        "**Divergence of the refusal-removed models from their originals on benign held-out prompts.** Each dot is "
        "the teacher-forced divergence of one selected model: the token-averaged "
        "$\\log p_{\\mathrm{orig}}(y)-\\log p_{\\mathrm{mod}}(y)$ over the original model's own greedy answers $y$ to "
        "the 350 benign held-out prompts; 0 is the original model and larger values mean the model has moved "
        "further away from it on harmless inputs. Panels are families; in each row the older release is in the "
        "upper lane (hollow marker) and the newer one in the lower lane (filled), joined by a grey line. "
        + ("Community rows are public releases, named by uploader. " if SHOW_COMMUNITY else
           f"The {int((res['method'] == 'community').sum())} public community releases are shown in Fig. A3. ")
        + ("Cells without a result are labelled in their lane. " if n_pending or n_planned else "")
        + shown + caveat
        + "Single seed; no uncertainty intervals are drawn."
    )
    order = sorted(groups, key=lambda t: (t.startswith("planned"), t))
    notes = ("*Not yet shown (for the text):* "
             + " ".join(f"{t[0].upper() + t[1:]}: {', '.join(groups[t])}." for t in order) if groups else "")
    detail = ("*Scoring (for the text):* answers of at most 512 tokens, system prompt \"You are a helpful "
              "assistant.\", thinking disabled; vLLM prompt log-probabilities (natural log, pooled over all answer "
              "tokens) on each family's evaluation GPU (Qwen and Gemma: RTX PRO 6000; OLMo and Mistral: H200 NVL; "
              "small pair: RTX 4090). Each derived model is the dev-set "
              "selection of its cell (fewest dev keyword refusals among 100 TPE trials with first-token KL $\\le$ 0.1); "
              "the held-out prompts were not used for optimisation or selection.")
    print(f"  [fig6] caption {len(caption.split())} words")
    S.write_caption(NAME, caption + "\n\n" + detail + ("\n\n" + notes if notes else ""), source=S.SOURCE_HELDOUT_TF)
    print(f"  {n_plot} cells plotted, {n_pending} pending, {n_planned} planned; x range [{lo:.3f}, {hi:.3f}]")
    print(f"  old-gen cells {int(by_gen['old'].sum())}, new-gen cells {int((~by_gen['old']).sum())}")


if __name__ == "__main__":
    main()
