"""fig11_judge_agreement -- keyword counts vs WildGuard vs Qwen3Guard on the same held-out responses (H3).

(a) Per-cell gap = keyword refusal rate - WildGuard refusal rate (pp) on the 972 harmful held-out responses of each
    of the 74 judged cells, with its 95% interval (Newcombe, paired). Slots: originals, the ablated (seed-1) cells by
    method (direction-based, then ARA), then gamma-MPOA, community releases and the small pair's seed-2 replicates.
    Within a slot the cells run in pair order (Qwen, Gemma, OLMo, Mistral, small pair), older before newer release.
    The horizontal bar and band are the pooled gap of the slot group with its interval (h3_gap_pooled.csv; the five
    ablated slots share one pooled value). Below 0 (shaded) is the direction H3 predicted (keyword < judge).
(b) Pooled gap by comparison and by response completeness: keyword - WildGuard and keyword - Qwen3Guard on all
    responses; keyword - WildGuard on responses that ended before the 512-token cap and on capped responses.
(c) Cohen's kappa (prompt-bootstrap 95% interval) for keyword vs WildGuard, keyword vs Qwen3Guard and WildGuard vs
    Qwen3Guard, before (originals) and after ablation.

Inputs : results/analysis/h3_gap_cells.csv (comparison kw_vs_wg, set_group harmful_all),
         results/analysis/h3_gap_pooled.csv (subset harmful_all), results/analysis/h3_agreement_pairs.csv (harmful,
         pooled), results/analysis/h3_models.csv (class-effect terms, caption only)
Outputs: paper/figures/fig11_judge_agreement.pdf, preview PNG, captions.md section
Run    : cd scripts/figures && .venv/bin/python fig11_judge_agreement.py
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

import judged as J
import style as S

NAME = "fig11_judge_agreement"
FAM_ORDER = [J.PAIR_KEY[f] for f, *_ in S.PAIRS]
SLOTS = [  # (key, label, filter on h3_gap_cells, pooled group in h3_gap_pooled)
    ("original", "Originals", lambda d: d["role"] == "original", "original"),
    ("dim", "DiM", lambda d: (d["role"] == "primary") & (d["method"] == "dim"), "ablated_primary"),
    ("mpoa", "MPOA", lambda d: (d["role"] == "primary") & (d["method"] == "mpoa"), "ablated_primary"),
    ("som", "SOM", lambda d: (d["role"] == "primary") & (d["method"] == "som"), "ablated_primary"),
    ("jspace", "J-space", lambda d: (d["role"] == "primary") & (d["method"] == "jspace"), "ablated_primary"),
    ("ara", "ARA", lambda d: (d["role"] == "primary") & (d["method"] == "ara"), "ablated_primary"),
    ("gamma_mpoa", "$\\gamma$-MPOA", lambda d: d["role"] == "control", "role:control"),
    ("community", "Community", lambda d: d["role"] == "external", "role:external"),
    ("seed2", "Seed 2\n(small pair)", lambda d: d["role"] == "replicate", "role:replicate"),
]
GROUPS = [  # rows of (b) and (c): (pooled group, label)
    ("original", "Originals"),
    ("ablated_primary:main", "Ablated, main study"),
    ("ablated_primary:small", "Ablated, small pair"),
    ("role:control", "$\\gamma$-MPOA"),
    ("role:external", "Community"),
    ("role:replicate", "Seed 2 (small pair)"),
]
GAP_SERIES = [  # (variant in review_out_c4_gap_pooled.csv, label, marker kwargs): keyword window vs WildGuard
    ("kw100_vs_wg", "keyword, first 100 tokens (Heretic's window)",
     dict(marker="o", mfc=S.TEXT, mec=S.TEXT, color=S.TEXT)),
    ("kw256_vs_wg", "keyword, first 256 tokens",
     dict(marker="D", mfc=S.TEXT_SECONDARY, mec=S.TEXT_SECONDARY, ms=S.MS * 0.85, color=S.TEXT_SECONDARY)),
    ("kw_vs_wg", "keyword, full response (≤ 512 tokens)",
     dict(marker="o", mfc=S.SURFACE, mec=S.TEXT, mew=S.HOLLOW_MEW, color=S.TEXT)),
]
KAPPA_COLORS = "#5f5f5f"
KAPPA_SERIES = [
    (("kw", "wg"), "keyword vs WildGuard", dict(marker="s", mfc=S.TEXT, mec=S.TEXT, ms=S.MS * 0.9, color=S.TEXT)),
    (("kw", "q3g"), "keyword vs Qwen3Guard",
     dict(marker="s", mfc=S.SURFACE, mec=S.TEXT, mew=S.HOLLOW_MEW, ms=S.MS * 0.9, color=S.TEXT)),
    (("wg", "q3g"), "WildGuard vs Qwen3Guard",
     dict(marker="^", mfc=S.TEXT_SECONDARY, mec=S.TEXT_SECONDARY, ms=S.MS, color=S.TEXT_SECONDARY)),
]
SPREAD = 0.36           # half-width over which the cells of a slot are spread
SUB = 0.19              # vertical offset between sub-rows in (b)/(c)
BELOW_FILL = "#f4f3f0"


def gen_of(row):
    return "new" if int(row["gen_new"]) == 1 else "old"


def load():
    # (a) and (b) at the keyword windows of the review revision (100-token window = the one Heretic scores)
    cells = J.load("review_out_c4_gap_cells.csv")
    cells = cells[(cells["comparison"] == "kw100_vs_wg") & (cells["set_group"] == "harmful_all")].copy()
    if len(cells) != len(J.judged_cells()):
        raise ValueError(f"h3_gap_cells has {len(cells)} kw_vs_wg cells, dir_status {len(J.judged_cells())}")
    cells["fam_i"] = cells["pair"].map({p: i for i, p in enumerate(FAM_ORDER)})
    cells = cells.sort_values(["fam_i", "gen_new", "label"])
    pooled = J.load("review_out_c4_gap_pooled.csv")
    pooled = pooled[pooled["subset"] == "harmful_all"]
    agree = J.load("h3_agreement_pairs.csv")
    agree = agree[(agree["subset"] == "harmful") & (agree["group_type"] == "pooled")]
    models = J.load("h3_models.csv")
    return cells, pooled, agree, models


def pooled_row(pooled, group, variant="kw100_vs_wg"):
    return J.one(pooled[(pooled["group"] == group) & (pooled["variant"] == variant)], f"{group} {variant}")


def kappa_row(agree, group, r1, r2):
    a = agree[(agree["group"] == group) & (((agree["rater1"] == r1) & (agree["rater2"] == r2)) |
                                           ((agree["rater1"] == r2) & (agree["rater2"] == r1)))]
    return J.one(a, f"kappa {group} {r1}-{r2}")


def panel_a(ax, cells, pooled):
    counts = []
    ymin, ymax = 0.0, 0.0
    for i, (key, lab, filt, grp) in enumerate(SLOTS):
        d = cells[filt(cells)]
        n = len(d)
        xs = np.linspace(-SPREAD, SPREAD, n) if n > 1 else np.array([0.0])
        for x, (_, r) in zip(xs, d.iterrows()):
            m = {"original": "base", "community": "community", "gamma_mpoa": "gamma_mpoa"}.get(
                r["method"], r["method"])
            kw = S.marker_kw(m, gen_of(r), size=S.MS * 0.9)
            ci_c = S.color(m)
            if key == "seed2":                      # seed 2 in the lighter tint (as fig3/fig8)
                c2 = J.tint(S.color(m))
                kw.update(color=c2, markeredgecolor=c2)
                if kw["markerfacecolor"] != S.SURFACE:
                    kw["markerfacecolor"] = c2
                ci_c = c2
            ax.vlines(i + x, 100 * r["gap_lo"], 100 * r["gap_hi"], color=ci_c, lw=0.8, zorder=2)
            ax.plot([i + x], [100 * r["gap"]], **kw, zorder=3)
            ymin, ymax = min(ymin, 100 * r["gap_lo"]), max(ymax, 100 * r["gap_hi"])
        counts.append((int((d["gap"] > 0).sum()), n))
    # pooled bars: one per pooled group, spanning its slots
    spans = {}
    for i, (key, lab, filt, grp) in enumerate(SLOTS):
        spans.setdefault(grp, []).append(i)
    for grp, idx in spans.items():
        p = pooled_row(pooled, grp)
        x0, x1 = min(idx) - 0.46, max(idx) + 0.46
        ax.add_patch(Rectangle((x0, 100 * p["gap_lo"]), x1 - x0, 100 * (p["gap_hi"] - p["gap_lo"]),
                               facecolor="#dcdbd5", edgecolor="none", alpha=0.75, zorder=1))
        ax.plot([x0, x1], [100 * p["gap"]] * 2, color=S.TEXT, lw=1.0, zorder=1.5, solid_capstyle="butt")
    lo = 5 * np.floor((ymin - 2) / 5)
    hi = 5 * np.ceil((ymax + 2) / 5)
    ax.axhspan(lo, 0, color=BELOW_FILL, zorder=0, lw=0)
    ax.axhline(0, color=S.TEXT_SECONDARY, lw=0.8, zorder=1.2)
    ax.text(-0.5, lo + 1.0, "H3 predicted direction: keyword < WildGuard", ha="left", va="bottom",
            fontsize=S.FONT_SMALL, color=S.TEXT_SECONDARY, style="italic", zorder=4)
    for xv in (0.5, 5.5):
        ax.axvline(xv, color=S.AXIS, lw=S.LW_THIN, ls=(0, (2, 2)), zorder=1)
    for i, (k, n) in enumerate(counts):
        ax.text(i, hi + 0.5, f"{k}/{n} > 0", ha="center", va="bottom", fontsize=S.FONT_SMALL,
                color=S.TEXT_SECONDARY)
    ax.set_ylim(lo, hi)
    ax.set_xlim(-0.55, len(SLOTS) - 0.45)
    ax.set_xticks(range(len(SLOTS)))
    ax.set_xticklabels([s[1] for s in SLOTS])
    ax.tick_params(axis="x", length=0)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("Keyword (100 tokens) −\nWildGuard refusal (pp)")
    ax.set_title("a   Gap per model at Heretic's 100-token keyword window, held-out harmful responses", loc="left",
                 fontweight="bold", pad=14)
    return counts


def row_panel(ax, rows_y, series, getter, xlab, title, ref=None):
    for g, y0 in rows_y.items():
        for j, (key, lab, mk) in enumerate(series):
            r = getter(g, key)
            if r is None:
                continue
            est, lo, hi = r
            y = y0 - (j - (len(series) - 1) / 2) * SUB      # first series on top
            kw = dict(mk)
            c = kw.pop("color")
            ax.hlines(y, lo, hi, color=c, lw=0.9, zorder=2)
            ax.plot([est], [y], ls="none", ms=kw.pop("ms", S.MS), **kw, zorder=3)
    if ref is not None:
        ax.axvline(ref, color=S.TEXT_SECONDARY, lw=0.8, zorder=1)
    ax.set_xlabel(xlab)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)


def legend_handles(series):
    out = []
    for key, lab, mk in series:
        kw = dict(mk)
        c = kw.pop("color")
        out.append(Line2D([], [], color=c, lw=0.9, ms=kw.pop("ms", S.MS), label=lab, **kw))
    return out


def main():
    S.apply()
    cells, pooled, agree, models = load()
    fig = plt.figure(figsize=(S.DOUBLE_COL, 5.2))
    fig.get_layout_engine().set(w_pad=0.03, h_pad=0.05, wspace=0.05, hspace=0.08)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.12], width_ratios=[1.0, 0.82])
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1], sharey=ax_b)
    counts = panel_a(ax_a, cells, pooled)

    rows_y = {g: -i for i, (g, _) in enumerate(GROUPS)}

    def gap_get(g, variant):
        r = pooled_row(pooled, g, variant)
        return 100 * r["gap"], 100 * r["gap_lo"], 100 * r["gap_hi"]

    def kap_get(g, pair):
        r = kappa_row(agree, g, *pair)
        return r["kappa"], r["kappa_lo"], r["kappa_hi"]

    row_panel(ax_b, rows_y, GAP_SERIES, gap_get, "Keyword − WildGuard refusal rate (pp), pooled",
              "b   Pooled gap by keyword window", ref=0.0)
    row_panel(ax_c, rows_y, KAPPA_SERIES, kap_get, "Cohen's κ (agreement beyond chance)",
              "c   Agreement, full-response keyword")
    ax_b.set_yticks(list(rows_y.values()))
    ax_b.set_yticklabels([lab for _, lab in GROUPS])
    plt.setp(ax_c.get_yticklabels(), visible=False)
    ax_b.set_ylim(min(rows_y.values()) - 0.55, 0.55)
    ax_c.set_xlim(0, 1)
    for ax in (ax_b, ax_c):
        for y in list(rows_y.values())[:-1]:
            ax.axhline(y - 0.5, color=S.GRID, lw=S.LW_THIN, zorder=0)
    # separator between "before" (originals) and "after" rows
    for ax in (ax_b, ax_c):
        ax.axhline(-0.5, color=S.AXIS, lw=0.8, zorder=0)
    # one key for (b) and (c): the filled / hollow circles mean the same rater pair in both panels
    hb = legend_handles(GAP_SERIES)
    hc = legend_handles(KAPPA_SERIES)
    hc[0].set_label("keyword vs WildGuard (c)")
    hc[1].set_label("keyword vs Qwen3Guard (c)")
    hc[2].set_label("WildGuard vs Qwen3Guard (c)")
    # two columns (filled column-wise): keyword vs WildGuard variants | Qwen3Guard pairs. Three columns overflowed the
    # 6.5 in figure width and clipped the outer labels (fact-check 2026-11-06).
    for h in hb:
        h.set_label(h.get_label() + " (b)")
    fig.legend(handles=hb + hc, loc="outside lower center", ncols=2, frameon=False,
               handlelength=1.6, columnspacing=2.0)
    # method key for (a), above the panel
    mh = S.method_handles(S.METHOD_ORDER)
    gh = S.generation_handles()
    for h, lab in zip(gh, ("older release", "newer release")):
        h.set_linestyle("none")
        h.set_label(lab)
    h_pool = [Line2D([], [], color=S.TEXT, lw=1.0, label="pooled gap"),
              Patch(facecolor="#dcdbd5", edgecolor="none", label="its 95% CI")]
    fig.legend(handles=mh + gh + h_pool, loc="outside upper center", ncols=6, frameon=False, handlelength=1.3,
               columnspacing=1.1)
    S.save(fig, NAME)
    # 2026-11-07 figure pass: the paper caption (05_results.tex, captions.md) is maintained by hand for this figure;
    # the generated draft below still describes the full-response version and is not written.
    # write_caption(cells, pooled, agree, models, counts)


def write_caption(cells, pooled, agree, models, counts):
    def g(group, variant="kw_vs_wg"):
        r = pooled_row(pooled, group, variant)
        return f"{J.pp(r['gap'])} pp [{J.pp(r['gap_lo'])}, {J.pp(r['gap_hi'])}]"

    def k(group, a, b):
        r = kappa_row(agree, group, a, b)
        return f"{J.num(r['kappa'], 2)} [{J.num(r['kappa_lo'], 2)}, {J.num(r['kappa_hi'], 2)}]"

    prim = cells[cells["role"] == "primary"]
    n_pos = int((prim["gap"] > 0).sum())
    n_pos_h = int(((prim["gap"] > 0) & (prim["p_holm"] < 0.05)).sum())
    n_neg = int((prim["gap"] < 0).sum())
    neg_h = prim[(prim["gap"] < 0) & (prim["p_holm"] < 0.05)]
    neg_names = ", ".join(f"{S.MODEL_DISPLAY[r['model']]} {S.label(r['method'])}" for _, r in neg_h.iterrows())

    def term(model, t):
        r = J.one(models[(models["model"] == model) & (models["study"] == "main") &
                         (models["subset"] == "harmful_all") & (models["term"] == t)], f"{model} {t}")
        return r

    cls = "class effect on gap (ARA - direction), averaged over generations"
    gee = term("GEE_linear_paired_gap (primary)", cls)
    ols = term("OLS_cell_level_gap", cls)
    small = models[(models["study"] == "small") & (models["subset"] == "harmful_all") &
                   (models["model"] == "GEE_linear_paired_gap (primary)") & (models["term"] == cls)]
    small_txt = ""
    if len(small) == 1:
        s = small.iloc[0]
        small_txt = (f"; in the small pair it is {J.pp(s['estimate'])} pp [{J.pp(s['lo'])}, {J.pp(s['hi'])}], "
                     f"p$_\\mathrm{{Holm}}$ = {J.p_text(s['p_holm'])}")
    cap = (
        "**Keyword counts versus two LLM judges on the same held-out, judged responses (H3).** "
        "(a) Each marker is one evaluated model (cell): its keyword refusal rate (any of Heretic's 33 refusal "
        "markers in the full response of at most 512 tokens) minus its WildGuard refusal rate on the 972 harmful "
        "held-out prompts, in percentage points, with a paired 95% interval (Newcombe). Slots, left to right: the "
        "unedited originals, the seed-1 ablated models by method (direction-based, then ARA), $\\gamma$-MPOA, "
        "public community releases and the small pair's seed-2 replicates (lighter); within a slot the models run "
        "in pair order (Qwen, Gemma, OLMo, Mistral Small, small pair), hollow = older, filled = newer release. The "
        "black bar and grey band are the pooled gap of the slot group with its 95% interval (the five ablated slots "
        "share one pooled value); the count above a slot is the number of cells with a positive gap. H3 predicted "
        "the shaded region (keyword below the judge). "
        f"After ablation keyword matching counts more refusals than WildGuard, not fewer: pooled "
        f"{g('ablated_primary')} over the 42 ablated cells (originals {g('original')}); keyword > WildGuard in "
        f"{n_pos} of 42 ablated cells ({n_pos_h} Holm-significant over the 74 cells), keyword < WildGuard in "
        f"{n_neg} ({len(neg_h)} significant: {neg_names}). "
        "(b) Pooled gap against WildGuard and against Qwen3Guard, and against WildGuard separately for responses "
        "that ended before the 512-token cap and for capped responses. The positive gap is concentrated in capped "
        f"responses (all 42 ablated cells: {g('ablated_primary', 'kw_vs_wg_capped_only')}; main study "
        f"{g('ablated_primary:main', 'kw_vs_wg_capped_only')}); on complete responses it is "
        f"{g('ablated_primary', 'kw_vs_wg_complete_only')} (main study "
        f"{g('ablated_primary:main', 'kw_vs_wg_complete_only')}, small pair "
        f"{g('ablated_primary:small', 'kw_vs_wg_complete_only')}). "
        "(c) Cohen's $\\kappa$ with prompt-bootstrap 95% intervals. Keyword agreement with WildGuard falls from "
        f"{k('original', 'kw', 'wg')} in the originals to {k('ablated_primary:main', 'kw', 'wg')} (main study) and "
        f"{k('ablated_primary:small', 'kw', 'wg')} (small pair) after ablation, while the two judges agree with "
        f"each other at {k('original', 'wg', 'q3g')} (originals) and {k('ablated_primary:main', 'wg', 'q3g')} "
        "(main study, after ablation). "
        "H3 part 1 is not supported in the predicted direction. The method-class difference in the gap (ARA minus "
        f"direction-based, main study) is {J.pp(gee['estimate'])} pp [{J.pp(gee['lo'])}, {J.pp(gee['hi'])}] "
        f"with the prompt as unit (GEE, p$_\\mathrm{{Holm}}$ {J.p_text(gee['p_holm'])}) but p = "
        f"{J.p_text(ols['p'])} with the cell as unit (n = 32){small_txt}: part 2 is not robustly supported. "
        "No human-labelled audit exists, so the panels show where keyword matching and the judges diverge, not "
        "which is correct.")
    S.write_caption(NAME, J.tex(cap), source=J.source_text(
        "Gaps: h3_gap_cells.csv (comparison kw_vs_wg, harmful_all) and h3_gap_pooled.csv; kappa: "
        "h3_agreement_pairs.csv (harmful, pooled; prompt-bootstrap CI); class-effect tests: h3_models.csv."))


if __name__ == "__main__":
    main()
