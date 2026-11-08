"""fig10_heldout_generation_effect -- held-out judged refusal, older vs newer release, per method and family (H1).

One panel per pair (Qwen, Gemma, OLMo, Mistral Small, small Qwen pair). Within a panel one slot per condition, in
H2-class order: the unedited originals, the direction-based methods (DiM, MPOA, SOM; J-space on the small pair only)
and, right of a divider, the optimisation-based ARA. In each slot the older release (left, hollow marker) is joined
to the newer one (right, filled marker); vertical bars are Wilson 95% intervals of the cell's WildGuard refusal rate
on the 972 harmful held-out prompts. The number above a slot is the paired generation effect (newer minus older, pp);
grey with "ns" when it is not significant: Holm over the 23-test H1 family for the methods, the unadjusted exact
McNemar test for the originals (they are the baseline, not an H1 test). A ring marks a failed edit: a modified cell
whose refusal is at or above its own original's (h1h2_cells.csv diff_vs_orig >= 0).
Seed 1 only (the small pair's seed 2 is compared in fig13); gamma-MPOA is compared with MPOA in fig12.

Inputs : results/analysis/h1h2_cells.csv (judge wg, subset all: rate, lo, hi, diff_vs_orig),
         results/analysis/h1h2_generation_effects.csv (judge wg, subset all: delta, p_holm_H1, p_mcnemar)
Outputs: paper/figures/fig10_heldout_generation_effect.pdf, preview PNG, captions.md section
Run    : cd scripts/figures && .venv/bin/python fig10_heldout_generation_effect.py
"""
from __future__ import annotations

from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory

import judged as J
import style as S

NAME = "fig10_heldout_generation_effect"
D = 0.2                 # x offset of the older (-D) and newer (+D) release within a slot
ALPHA = 0.05
CI_LW = 1.0
RING_MS = 12.0


def load():
    cells = J.load("h1h2_cells.csv")
    cells = cells[(cells["judge"] == "wg") & (cells["subset"] == "all")]
    eff = J.load("h1h2_generation_effects.csv")
    eff = eff[(eff["judge"] == "wg") & (eff["subset"] == "all")]
    out = {}
    for fam, old, new in S.PAIRS:
        pk = J.PAIR_KEY[fam]
        slots = []
        for m in ["original"] + J.METHODS_HELDOUT:
            cs = cells[(cells["pair"] == pk) & (cells["method"] == m)]
            if cs.empty:
                continue                      # J-space exists on the small pair only
            o = J.one(cs[cs["label"].str.startswith(old + "__")], f"{old} {m}")
            n = J.one(cs[cs["label"].str.startswith(new + "__")], f"{new} {m}")
            e = J.one(eff[(eff["pair"] == pk) & (eff["method"] == m)], f"{pk} {m} effect")
            p = e["p_mcnemar"] if m == "original" else e["p_holm_H1"]
            slots.append(dict(fam=fam, method=m, old=o, new=n, delta=e["delta"], lo=e["delta_lo"], hi=e["delta_hi"],
                              p=float(p), sig=float(p) < ALPHA, did=e.get("did_baseline_adj")))
        out[fam] = slots
    return out


def failed(row):
    return row["method"] != "original" and float(row["diff_vs_orig"]) >= 0


def panel(ax, fam, old, new, slots):
    tr_top = blended_transform_factory(ax.transData, ax.transAxes)
    gen_of = {"old": "old", "new": "new"}
    n_dir = sum(s["method"] in J.DIRECTION for s in slots)
    for i, s in enumerate(slots):
        m = s["method"] if s["method"] != "original" else "base"
        c = S.color(m)
        xs = {"old": i - D, "new": i + D}
        rows = {"old": s["old"], "new": s["new"]}
        ax.plot([xs["old"], xs["new"]], [100 * rows["old"]["rate"], 100 * rows["new"]["rate"]], color=c,
                lw=1.1, zorder=2, solid_capstyle="round")
        for g in ("old", "new"):
            r = rows[g]
            ax.vlines(xs[g], 100 * r["lo"], 100 * r["hi"], color=c, lw=CI_LW, zorder=2.5)
            ax.plot([xs[g]], [100 * r["rate"]], **S.marker_kw(m, gen_of[g]), zorder=3)
            if failed(r):
                ax.plot([xs[g]], [100 * r["rate"]], marker="o", ms=RING_MS, mfc="none", mec=S.TEXT, mew=0.8,
                        ls="none", zorder=4)
                ax.annotate("failed edit", (xs[g], 100 * r["rate"]), xytext=(i + 0.62, 99.0),
                            textcoords="data", ha="left", va="center", fontsize=S.FONT_SMALL, color=S.TEXT,
                            arrowprops=dict(arrowstyle="-", lw=S.LW_THIN, color=S.TEXT_SECONDARY,
                                            shrinkA=1.0, shrinkB=RING_MS / 2), zorder=5)
        txt = J.pp(s["delta"]) + ("" if s["sig"] else " ns")
        ax.text(i, 1.035, txt, transform=tr_top, ha="center", va="bottom", fontsize=S.FONT_SMALL,
                color=S.TEXT if s["sig"] else S.TEXT_SECONDARY, clip_on=False)
    # dividers: originals | direction-based | ARA
    for xv in (0.5, n_dir + 0.5):
        ax.axvline(xv, color=S.AXIS, lw=S.LW_THIN, ls=(0, (2, 2)), zorder=1)
    ax.set_xticks(range(len(slots)))
    ax.set_xticklabels([S.label(s["method"] if s["method"] != "original" else "base") for s in slots])
    ax.set_xlim(-0.55, len(slots) - 0.45)
    ax.grid(axis="x", visible=False)
    ax.tick_params(axis="x", length=0)
    n_meth = sum(s["method"] != "original" for s in slots)
    n_hi = sum(s["method"] != "original" and s["sig"] and s["delta"] > 0 for s in slots)
    n_lo = sum(s["method"] != "original" and s["sig"] and s["delta"] < 0 for s in slots)
    ax.set_title(f"{fam}: {S.MODEL_DISPLAY[old]} → {S.MODEL_DISPLAY[new]}", loc="left", fontsize=S.FONT,
                 fontweight="bold", pad=13)
    return n_hi, n_lo, n_meth


def legend_panel(ax):
    ax.axis("off")
    neutral = S.TEXT_SECONDARY
    h = [Line2D([], [], color=neutral, marker="o", ms=S.MS, mfc=S.SURFACE, mec=neutral, mew=S.HOLLOW_MEW,
                ls="none", label="older release (left in each slot)"),
         Line2D([], [], color=neutral, marker="o", ms=S.MS, mfc=neutral, mec=neutral, ls="none",
                label="newer release (right)"),
         Line2D([], [], color=neutral, marker="|", ms=9, mew=CI_LW, ls="none", label="95% Wilson interval"),
         Line2D([], [], color=S.TEXT, marker="o", ms=RING_MS * 0.8, mfc="none", mew=0.8, ls="none",
                label="failed edit (refusal ≥ own original)")]
    leg1 = ax.legend(handles=h, loc="upper left", bbox_to_anchor=(0.0, 1.02), frameon=False, handlelength=1.4,
                     borderaxespad=0.0, labelspacing=0.55)
    ax.add_artist(leg1)
    mh = S.method_handles(["base", "dim", "mpoa", "som", "jspace", "ara"], generation="new")
    ax.legend(handles=mh, loc="upper left", bbox_to_anchor=(0.0, 0.53), ncols=3, frameon=False,
              handlelength=1.0, columnspacing=1.0, borderaxespad=0.0, labelspacing=0.55)
    ax.text(0.0, 0.17, "Above each slot: newer − older (pp), paired;\n"
                       "grey “ns”: not significant (methods: Holm, 23 H1 tests;\n"
                       "originals: exact McNemar, unadjusted).\n"
                       "Dashed dividers: originals | direction-based | ARA.\n"
                       "Most intervals are shorter than the markers (n = 972).",
            transform=ax.transAxes, ha="left", va="top", fontsize=S.FONT_SMALL, color=S.TEXT_SECONDARY,
            linespacing=1.35)


def main():
    S.apply()
    data = load()
    fig, axes = S.figure("double", height=5.5, nrows=3, ncols=2, sharey=True)
    fig.get_layout_engine().set(w_pad=0.03, h_pad=0.06, wspace=0.04, hspace=0.06)
    tallies = {}
    for ax, (fam, old, new) in zip(axes.flat, S.PAIRS):
        tallies[fam] = panel(ax, fam, old, new, data[fam])
        ax.set_ylim(-4, 103)                        # room for markers at 0 % / 100 %
        ax.spines["left"].set_bounds(0, 100)
        ax.set_yticks([0, 25, 50, 75, 100])
    for ax in axes[:, 0]:
        ax.set_ylabel("Held-out refusal (%, WildGuard)")
    legend_panel(axes.flat[-1])
    S.save(fig, NAME)
    write_caption(data, tallies)


def write_caption(data, tallies):
    main_fams = [f for f, _, _ in S.PAIRS if f != "Qwen (small pair)"]
    main_slots = [s for f in main_fams for s in data[f] if s["method"] != "original"]
    n_hi = sum(s["sig"] and s["delta"] > 0 for s in main_slots)
    n_lo = sum(s["sig"] and s["delta"] < 0 for s in main_slots)
    n_ns = len(main_slots) - n_hi - n_lo
    fails = [(f, g, s) for f, *_ in S.PAIRS for s in data[f] for g in ("old", "new") if failed(s[g])]
    fail_txt = "; ".join(
        f"{S.MODEL_DISPLAY[s[g]['label'].split('__')[0]]} {S.label(s['method'])}, "
        f"{J.pct(s[g]['rate'])}% vs its original's {J.pct(s[g]['orig_rate'])}%" for f, g, s in fails)
    big = max(main_slots, key=lambda s: s["delta"])
    fail_hi = [s for f, g, s in fails if g == "new" and s["sig"] and s["delta"] > 0]
    gem_txt = ("; " + ", ".join(f"the {s['fam']} {S.label(s['method'])} increase is the failed edit"
                                for s in fail_hi)) if fail_hi else ""
    big_f = big["fam"]
    orig = {f: [s for s in data[f] if s["method"] == "original"][0] for f, *_ in S.PAIRS}
    orig_txt = ", ".join(f"{f} {J.pp(orig[f]['delta'])}" + ("" if orig[f]["sig"] else " (ns)")
                         for f, *_ in S.PAIRS)
    per_pair = "; ".join(f"{f} {tallies[f][0]} higher, {tallies[f][1]} lower, "
                         f"{tallies[f][2] - tallies[f][0] - tallies[f][1]} ns" for f, *_ in S.PAIRS)
    cap = (
        "**Held-out, judged refusal of the original and refusal-removed models, older vs newer release.** "
        "Each panel is one pair; each slot is one condition: the unedited originals (grey star), the direction-based "
        "methods (DiM, MPOA, SOM; J-space on the small pair only) and, right of the second dashed divider, the "
        "optimisation-based ARA. Within a slot the older release (left, hollow marker) is joined to the newer one "
        "(right, filled marker). The value is the share of the 972 harmful held-out prompts whose response WildGuard "
        "labels a refusal; bars are Wilson 95% intervals. The number above a slot is the paired generation effect "
        "(newer minus older, percentage points); grey with “ns” where it is not significant (methods: exact "
        "McNemar, Holm over the 23 H1 tests; originals: exact McNemar, unadjusted, as they are the baseline rather "
        "than an H1 test). The ring marks a failed edit, a selected model whose held-out refusal is at or above its "
        f"own original's ({fail_txt}). "
        f"Across the 16 main-study pair $\\times$ method comparisons the newer model refuses significantly more in "
        f"{n_hi}, less in {n_lo} and not significantly differently in {n_ns}; no main-study pair is higher for "
        f"every method (H1 not supported){gem_txt}. The unedited originals themselves differ by {orig_txt} pp "
        f"(baseline-adjusted effects in the text). The largest effect is {big_f} {S.label(big['method'])} "
        f"({J.pct(big['old']['rate'])}% $\\to$ {J.pct(big['new']['rate'])}%). Each modified model is the dev-set "
        "selection of its cell (fewest dev keyword refusals at first-token KL $\\le$ 0.1, seed 1); the held-out "
        "prompts were not used for optimisation or selection. Seed 2 of the small pair is compared in Fig. 13 and "
        "$\\gamma$-MPOA in Fig. 12.")
    detail = (
        f"*For the text:* per pair (newer higher / lower / ns after Holm): {per_pair}. "
        "Generation effects with 95% Newcombe intervals and baseline-adjusted values are in "
        "results/analysis/h1h2_generation_effects.csv; per-cell rates in h1h2_cells.csv.")
    S.write_caption(NAME, J.tex(cap) + "\n\n" + J.tex(detail), source=J.source_text())


if __name__ == "__main__":
    main()
