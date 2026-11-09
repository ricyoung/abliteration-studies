"""Figure 3v: generation effect on the dev set vs on held-out prompts, one vertical block per model pair.

Replaces the horizontal fig3 in the paper (user request 2026-11-05: "for the main pairs and the small pairs ...
go vertical"). One block per pair, stacked; one row per method. Left column: what the search saw (dev keyword
refusals of the selected trial, of 100; seed 1). Right column: the H1 measure (held-out refusal judged by WildGuard,
paired over the 972 harmful prompts). Hollow = older model, filled = newer model, joined by a line in the method
colour. The change (newer - older) is printed right of each panel.

Held-out values are the paired rates and deltas of h1h2_generation_effects.csv (judge wg, subset all), i.e. the same
numbers as Table tab:h1-heldout and fig10; gamma-MPOA from final_gmpoa_generation.csv. Significance: methods
Holm-adjusted McNemar within the 23-test H1 family (p_holm_H1); originals and gamma-MPOA exact McNemar, unadjusted.
Dev values: paper/figures/data/runs.csv (seed 1; baseline_refusals for originals, selected_refusals otherwise).
"""
import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnchoredOffsetbox, DrawingArea, HPacker, TextArea, VPacker

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import style as S  # noqa: E402

NAME = "fig3v_generation_dev_heldout"
ROOT = S.ROOT
A = ROOT / "results" / "analysis"

PAIRS = [("qwen", "Qwen", "qwen2.5-32b", "qwen3.8-27b"),
         ("gemma", "Gemma", "gemma-2-27b", "gemma-4-31b"),
         ("olmo", "OLMo", "olmo-2-32b", "olmo-3.1-32b"),
         ("mistral", "Mistral Small", "mistral-small-2409", "mistral-small-3.2"),
         ("qwen-small", "Small pair", "qwen2.5-7b", "qwen3.5-4b")]   # family labels as in S.PAIRS
ORDER = ["original", "dim", "mpoa", "som", "jspace", "gamma_mpoa", "ara"]
LABEL = {"original": "Original", "dim": "DiM", "mpoa": "MPOA", "gamma_mpoa": "γ-MPOA", "som": "SOM", "ara": "ARA",
         "jspace": "J-space"}
STYLE_KEY = {"original": "base"}
GREY = "#a9a8a2"


def load():
    runs = pd.read_csv(ROOT / "paper/figures/data/runs.csv")
    runs = runs[runs.seed == 1]
    eff = pd.read_csv(A / "h1h2_generation_effects.csv").query("judge == 'wg' and subset == 'all'")
    gm = pd.read_csv(A / "final_gmpoa_generation.csv")
    # cross-check: delta is p_new - p_old in both tables
    for t, d in ((eff, "delta"), (gm, "diff")):
        bad = (t[d] - (t.p_new - t.p_old)).abs() > 1e-5
        if bad.any():
            raise SystemExit(f"{NAME}: delta != p_new - p_old in {len(bad[bad])} rows")
    return runs, eff, gm


def dev_value(runs, model, method):
    if method == "original":
        v = runs[runs.model == model].baseline_refusals.dropna()
    else:
        v = runs[(runs.model == model) & (runs.method == method)].selected_refusals.dropna()
    return float(v.iloc[0]) if len(v) else None


def held(eff, gm, pair, method):
    """(old %, new %, delta pp, significant, failed edit) or None."""
    if method == "gamma_mpoa":
        r = gm[(gm.pair == pair) & (gm.method == method)]
        if r.empty:
            return None
        r = r.iloc[0]
        return 100 * r.p_old, 100 * r.p_new, 100 * r["diff"], r.p_mcnemar < 0.05, False
    r = eff[(eff.pair == pair) & (eff.method == method)]
    if r.empty:
        return None
    r = r.iloc[0]
    p = r.p_mcnemar if method == "original" else r.p_holm_H1
    failed = False
    if method != "original":   # failed edit: newer model's refusal not below its own original
        o = eff[(eff.pair == pair) & (eff.method == "original")].iloc[0]
        failed = r.p_new >= o.p_new
    return 100 * r.p_old, 100 * r.p_new, 100 * r.delta, p < 0.05, failed


def main():
    runs, eff, gm = load()
    S.apply()
    # explicit GridSpec margins and x-only grid; house default is constrained layout with a full grid
    plt.rcParams.update({"figure.constrained_layout.use": False, "axes.grid": False})
    rows_per = []
    for key, fam, old, new in PAIRS:
        rows_per.append([m for m in ORDER
                         if dev_value(runs, old, m) is not None or held(eff, gm, key, m) is not None])

    fig = plt.figure(figsize=(S.TEXT_W, 7.1))   # 7.1 in: figure + caption fit on one 9 in text page
    gs = GridSpec(len(PAIRS), 2, figure=fig, height_ratios=[len(m) + 1.3 for m in rows_per],
                  left=0.115, right=0.91, top=0.925, bottom=0.095, wspace=0.30, hspace=0.20)
    tally = {"dev": [], "held": []}
    for b, ((key, fam, old, new), ms) in enumerate(zip(PAIRS, rows_per)):
        for col in (0, 1):
            ax = fig.add_subplot(gs[b, col])
            n = len(ms)
            for i, m in enumerate(ms):
                y = n - 1 - i
                if i % 2 == 0:
                    ax.axhspan(y - 0.5, y + 0.5, color="#f6f5f2", zorder=0, lw=0)
                if col == 0:
                    vo, vn = dev_value(runs, old, m), dev_value(runs, new, m)
                    if vo is None or vn is None:
                        continue
                    d, sig, failed = vn - vo, True, False
                    txt = f"{d:+.0f}"
                    tally["dev"].append((key, m, d))
                else:
                    h = held(eff, gm, key, m)
                    if h is None:
                        continue
                    vo, vn, d, sig, failed = h
                    txt = f"{d:+.1f}" + ("†" if failed else "")
                    tally["held"].append((key, m, d, sig))
                sk = STYLE_KEY.get(m, m)
                c, mk = S.METHOD_COLORS[sk], S.METHOD_MARKERS[sk]
                ms_ = S.MS * S.MARKER_SCALE.get(mk, 1.0)
                ax.plot([vo, vn], [y, y], color=c, lw=1.4, alpha=0.55, zorder=2, solid_capstyle="round")
                ax.plot(vn, y, marker=mk, ms=ms_, mfc=c, mec=c, ls="none", zorder=3)
                ax.plot(vo, y, marker=mk, ms=ms_, mfc="white", mec=c, mew=S.HOLLOW_MEW, ls="none", zorder=4)
                txt = txt.replace("-", "\u2212")   # typographic minus
                ax.text(1.02, y, txt, transform=ax.get_yaxis_transform(), va="center", ha="left",
                        fontsize=S.FONT_SMALL, color=S.TEXT if sig else GREY,
                        fontweight="bold" if (sig and col == 1) else "normal")
            ax.set_ylim(-0.6, n - 0.4)
            ax.set_xlim(-3, 103)
            ax.set_yticks(range(n))
            ax.set_yticklabels([LABEL[m] for m in reversed(ms)] if col == 0 else [])
            ax.tick_params(axis="y", length=0, labelsize=S.FONT_SMALL, pad=3)
            ax.tick_params(axis="x", labelsize=S.FONT_SMALL, length=2.5, color=S.AXIS)
            ax.set_xticks([0, 25, 50, 75, 100])
            ax.grid(axis="x", color=S.GRID, lw=S.LW_THIN, zorder=1)
            for sp in ("top", "right", "left"):
                ax.spines[sp].set_visible(False)
            ax.spines["bottom"].set_color(S.AXIS)
            if b < len(PAIRS) - 1:
                ax.set_xticklabels([])
            if col == 0:
                t = ax.text(-0.155, 1.0, fam, transform=ax.transAxes, fontsize=S.FONT, fontweight="bold",
                            va="bottom", ha="left", color=S.TEXT)
                ax.annotate(f"{S.MODEL_DISPLAY[old]} → {S.MODEL_DISPLAY[new]}",
                            xy=(1, 0), xycoords=t, xytext=(5, 0), textcoords="offset points",
                            fontsize=S.FONT_SMALL, color=S.TEXT_SECONDARY, va="bottom", ha="left")
            if b == 0:
                title = ("a   Development set: keyword refusals\nof the selected trial (of 100)" if col == 0 else
                         "b   Held-out: refusal judged by WildGuard\n(% of 972 harmful prompts; tests H1)")
                ax.set_title(title, fontsize=S.FONT, loc="left", pad=18, color=S.TEXT, fontweight="bold")
            if b == len(PAIRS) - 1:
                ax.set_xlabel("Dev keyword refusals (of 100)" if col == 0 else "Held-out refusal (%, WildGuard)",
                              fontsize=S.FONT_SMALL, color=S.TEXT_SECONDARY)

    # key + note in one block at the bottom (the marker key used to sit in a legend line above the titles)
    def _key(filled):
        da = DrawingArea(8, 8, 0, 0)
        da.add_artist(Line2D([4], [4], marker="o", ms=S.MS, ls="none", mec=S.TEXT_SECONDARY,
                             mfc=S.TEXT_SECONDARY if filled else "white", mew=1.0 if filled else S.HOLLOW_MEW))
        return da

    def _txt(s):
        return TextArea(s, textprops=dict(fontsize=S.FONT_SMALL, color=S.TEXT_SECONDARY))
    line1 = HPacker(children=[_key(False), _txt("older model"), _txt("   "), _key(True), _txt("newer model"),
                              _txt("      Numbers right of each panel: change, newer − older.")],
                    align="center", pad=0, sep=3)
    line2 = HPacker(children=[_txt("Held-out: bold = significant (methods Holm-adjusted, 23 H1 tests; originals unadjusted); grey = n.s.;  † failed edit.")],
                    align="center", pad=0, sep=0)
    note = VPacker(children=[line1, line2], align="center", pad=0, sep=3)
    fig.add_artist(AnchoredOffsetbox(loc="lower center", child=note, bbox_to_anchor=(0.52, 0.008),
                                     bbox_transform=fig.transFigure, frameon=False, pad=0, borderpad=0))
    S.save(fig, NAME)
    write_caption(tally)


def write_caption(tally):
    main_dev = [(k, m, d) for k, m, d in tally["dev"] if m not in ("original", "gamma_mpoa") and k != "qwen-small"]
    main_held = [(k, m, d, s) for k, m, d, s in tally["held"]
                 if m not in ("original", "gamma_mpoa") and k != "qwen-small"]
    opp = sum(1 for (k, m, d) in main_dev for (k2, m2, d2, s) in main_held
              if (k, m) == (k2, m2) and s and d * d2 < 0)
    cap = (
        "**Generation effect on the development set (a) and on held-out prompts (b).** One block per model pair "
        "(older → newer release), one row per condition; hollow markers are the older, filled markers the newer "
        "model, and the number right of each panel is the change, newer − older. (a) Dev keyword refusals (of 100) "
        "of the trial chosen by the selection rule (fewest dev refusals at first-token KL ≤ 0.1, seed 1); the "
        "originals' row gives the unedited models. These prompts were used for search and selection, so the counts "
        "are optimistically biased and no test is attached. (b) Held-out refusal judged by WildGuard, paired over "
        "the 972 harmful prompts (the H1 measure; the same values as Figure fig10 and Table t4). Bold: significant "
        "(methods: Holm-adjusted exact McNemar within the 23 H1 tests; originals and γ-MPOA: exact McNemar, "
        "unadjusted); grey: not significant; † failed edit (held-out refusal not below the model's own original). "
        f"In {opp} of {len(main_dev)} main-pair method cells the dev change and a significant held-out change point "
        "in opposite directions. γ-MPOA was run on the Gemma and OLMo pairs only, J-space on the small pair only. "
        "Single seed; 95% Wilson intervals for (b) are in Figure fig10.")
    S.write_caption(NAME, cap, source="paper/figures/data/runs.csv (seed 1); results/analysis/"
                    "h1h2_generation_effects.csv (judge wg, subset all); results/analysis/final_gmpoa_generation.csv")
    print(f"  {NAME}: dev/held-out opposite-direction cells (main pairs, significant held-out): {opp}/{len(main_dev)}")


if __name__ == "__main__":
    main()
