"""fig15_at_a_glance -- the H1 answer on one panel (placed in the Introduction).

(a) Generation effect per pair and condition: newer minus older held-out refusal (WildGuard, 972 prompts, pp).
    Blue = the newer release kept more refusal (removal harder), orange = less (removal easier).
    Bold = significant (methods: Holm over the 23 H1 tests; originals: exact McNemar, unadjusted); "ns" otherwise.
(b) Per-pair pooled effect of the direction-based methods and the ARA effect, with 95% intervals that include
    run-to-run (seed) variation -- the per-pair verdict scale of the paper.
Sources: results/analysis/h1h2_generation_effects.csv (judge wg, subset all);
         results/analysis/review_sel_pair_effects.csv (scale pp, 'direction raw' and 'ARA raw', *_total columns).
No number is computed here beyond unit conversion (x100).
"""
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import style as S

S.apply()
A = S.ANALYSIS_DIR
g = pd.read_csv(A / "h1h2_generation_effects.csv")
g = g[(g.judge == "wg") & (g.subset == "all")]
e = pd.read_csv(A / "revision_sigma_pair_verdicts.csv")   # per-pair effects, both run-to-run variance models

PAIR_KEYS = ["qwen", "gemma", "olmo", "mistral", "qwen-small"]
ROWLAB = [f"{fam}\n{S.MODEL_DISPLAY[o]} → {S.MODEL_DISPLAY[n]}".replace("Mistral-Small-2409 → Mistral-Small-3.2", "Small-2409 → Small-3.2") for fam, o, n in S.PAIRS]
COLS = ["original", "dim", "mpoa", "som", "jspace", "ara"]
COLLAB = ["Original\n(unedited)", "DiM", "MPOA", "SOM", "J-space", "ARA"]
FAILED = {("gemma", "dim")}   # selected model at/above its own original's held-out refusal (fig10 ring)

cmap = S.diverging_cmap(); norm = S.diverging_norm(65)
fig = plt.figure(figsize=(S.TEXT_W, 3.0), layout="none")   # manual layout: aligned rows across (a) and (b)
gs = fig.add_gridspec(1, 2, width_ratios=[3.05, 1.35], wspace=0.06, left=0.282, right=0.985, top=0.775, bottom=0.16)
ax = fig.add_subplot(gs[0]); bx = fig.add_subplot(gs[1], sharey=ax)

# ---- (a) heatmap
ax.grid(False)
for i, pk in enumerate(PAIR_KEYS):
    for j, c in enumerate(COLS):
        x0 = j + (0.18 if j >= 1 else 0) + (0.18 if j >= 5 else 0)   # gaps: originals | direction methods | ARA
        r = g[(g.pair == pk) & (g.method == c)]
        if r.empty:
            ax.add_patch(plt.Rectangle((x0, i), 0.94, 0.9, fc=S.SURFACE, ec=S.GRID, lw=S.LW_THIN))
            ax.text(x0 + 0.47, i + 0.45, "not run", ha="center", va="center", fontsize=S.FONT_SMALL, color=S.TEXT_SECONDARY)
            continue
        r = r.iloc[0]; d = 100 * r.delta
        p = r.p_mcnemar if c == "original" else r.p_holm_H1
        sig = p < 0.05
        fill = mpl_hex = plt.matplotlib.colors.to_hex(cmap(norm(d)))
        ax.add_patch(plt.Rectangle((x0, i), 0.94, 0.9, fc=fill, ec="white", lw=0.8))
        ink = S.ink_on(fill)
        txt = f"{d:+.1f}" + ("†" if (pk, c) in FAILED else "")
        ax.text(x0 + 0.47, i + (0.36 if not sig else 0.45), txt, ha="center", va="center", fontsize=S.FONT,
                fontweight="bold" if sig else "normal", color=ink)
        if not sig:
            ax.text(x0 + 0.47, i + 0.70, "ns", ha="center", va="center", fontsize=S.FONT_SMALL, color=ink)
ax.set_xlim(-0.05, 6.4); ax.set_ylim(len(PAIR_KEYS) - 0.05, -0.05)
xc = [j + (0.18 if j >= 1 else 0) + (0.18 if j >= 5 else 0) + 0.47 for j in range(len(COLS))]
ax.set_xticks(xc); ax.set_xticklabels(COLLAB); ax.xaxis.tick_top(); ax.tick_params(axis="x", length=0, pad=2)
ax.set_yticks([i + 0.45 for i in range(len(PAIR_KEYS))]); ax.set_yticklabels(ROWLAB, fontsize=S.FONT_SMALL)
ax.tick_params(axis="y", length=0, pad=3)
for sp in ax.spines.values(): sp.set_visible(False)
from matplotlib.transforms import blended_transform_factory as _btf
_tr = _btf(ax.transData, ax.transAxes)
ax.plot([xc[1] - 0.47, xc[4] + 0.47], [1.105, 1.105], transform=_tr, color=S.AXIS, lw=S.LW_THIN, clip_on=False)
ax.text((xc[1] + xc[4]) / 2, 1.12, "direction-based", transform=_tr, ha="center", va="bottom", fontsize=S.FONT_SMALL,
        color=S.TEXT_SECONDARY)
fig.text(0.015, 0.975, "(a) Change in held-out refusal, newer − older release (percentage points)", ha="left", va="top", fontsize=S.FONT, fontweight="bold")

# colour key under (a)
cax = fig.add_axes([0.385, 0.06, 0.22, 0.032])
sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
cb = fig.colorbar(sm, cax=cax, orientation="horizontal", ticks=[-60, -30, 0, 30, 60])
cb.outline.set_visible(False); cb.ax.tick_params(labelsize=S.FONT_SMALL, length=2, pad=1)
fig.text(0.378, 0.076, "removal easier", ha="right", va="center", fontsize=S.FONT_SMALL, color=S.TEXT_SECONDARY)
fig.text(0.612, 0.076, "removal harder", ha="left", va="center", fontsize=S.FONT_SMALL, color=S.TEXT_SECONDARY)

# ---- (b) pooled per-pair estimates with run-to-run variation
for i, pk in enumerate(PAIR_KEYS):
    for est, dy, m in [("direction raw", 0.30, "dim"), ("ARA raw", 0.62, "ara")]:
        r = e[(e.pair == pk) & (e.estimand == est)].iloc[0]
        pt = r.estimate_pp
        both = bool(r.excl0_const_point) and bool(r.excl0_level_point)
        col = S.TEXT if m == "dim" else S.color("ara")
        bx.plot([r.lo_level_point, r.hi_level_point], [i + dy] * 2, color=col, lw=S.LW_THIN + 0.3, solid_capstyle="butt")
        bx.plot([r.lo_const_point, r.hi_const_point], [i + dy] * 2, color=col, lw=S.LW + 1.0, solid_capstyle="butt")
        bx.plot(pt, i + dy, marker="o" if m == "dim" else "^", ms=S.MS, color=col,
                mfc=col if both else S.SURFACE, mew=S.HOLLOW_MEW, zorder=3)
bx.axvline(0, color=S.AXIS, lw=S.LW_THIN, zorder=0)
bx.set_xlim(-62, 78); bx.set_xticks([-50, -25, 0, 25, 50])
bx.tick_params(axis="y", left=False, labelleft=False)
bx.grid(axis="y", visible=False)
bx.spines["left"].set_visible(False)
for i in range(1, len(PAIR_KEYS)):
    bx.axhline(i - 0.05, color=S.GRID, lw=S.LW_THIN)
bx.set_xlabel("pp, 95% intervals with\nrun-to-run variation", fontsize=S.FONT_SMALL, labelpad=2)
fig.text(0.735, 0.975, "(b) Per-pair effect", ha="left", va="top", fontsize=S.FONT, fontweight="bold")
h = [Line2D([], [], color=S.TEXT, marker="o", lw=S.LW, ms=S.MS, label="direction-based, pooled"),
     Line2D([], [], color=S.color("ara"), marker="^", lw=S.LW, ms=S.MS, label="ARA")]
bx.legend(handles=h, loc="lower left", bbox_to_anchor=(-0.02, 1.0), ncol=1, frameon=False, fontsize=S.FONT_SMALL,
          handlelength=1.6, borderaxespad=0.1, labelspacing=0.2)

S.save(fig, "fig15_at_a_glance")
S.write_caption("fig15_at_a_glance", (
    "**Refusal removal did not become uniformly harder.** (a) Change in held-out refusal from the older to the newer "
    "release of each pair (newer minus older, percentage points; WildGuard judge, 972 harmful prompts not used for "
    "search or selection), for the unedited originals and for each removal method's selected model (fewest dev "
    "refusals at first-token KL ≤ 0.1, seed 1). Blue: the newer release kept more refusal (removal harder); "
    "orange: less (removal easier). Bold: significant (methods: Holm-adjusted exact McNemar over the 23 H1 tests; "
    "originals: exact McNemar, unadjusted); ns: not significant. † failed edit (Gemma-4-31B DiM left more "
    "refusal than its own original). (b) Per-pair effect of the direction-based methods (pooled over DiM, MPOA and "
    "SOM; J-space added on the small pair) and of ARA, with 95% intervals that add run-to-run variation (seed noise at its point estimate) to prompt "
    "sampling: thick, constant run-to-run variance; thin, variance that scales with the refusal level. Filled markers: "
    "both intervals exclude zero (only the OLMo direction-based increase). Levels and Wilson intervals of every cell are in Fig. fig10."),
    source="results/analysis/h1h2_generation_effects.csv (judge wg, subset all); "
           "results/analysis/revision_sigma_pair_verdicts.csv ('direction raw', 'ARA raw'; *_const_point, *_level_point)")
