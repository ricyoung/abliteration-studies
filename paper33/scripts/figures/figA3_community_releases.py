"""figA3_community_releases -- capability change and teacher-forced divergence of the public community releases.

Appendix figure. The eight community releases (public abliterated / Heretic checkpoints of the study's originals,
results.csv method == "community") used to be extra rows in fig5_capability_deltas and fig6_divergence, where their
uploader labels crowded the panels; they are drawn here instead, with the same encodings:
  Left : capability deltas vs the release's own original (percentage points), MMLU-Pro, GSM8K, IFEval, HumanEval,
         MBPP and the unweighted mean; the diverging scale of fig5 (orange loss, grey 0, blue gain, +/- 8 pp), exact
         values printed.
  Right: teacher-forced divergence on the 350 benign held-out prompts (results.csv `tf_divergence`, as in fig6), one
         marker per release (community marker; hollow = older release of a pair, filled = newer). The grey bar behind
         it spans the values of this paper's selected models on the same original (all conditions of fig6), as context.
Rows are grouped by original model in S.PAIRS order; the row label is the Hugging Face uploader.

Input: paper/figures/data/results.csv. Run with scripts/figures/.venv/bin/python (cwd scripts/figures).
"""
from __future__ import annotations

import numpy as np
from matplotlib import patheffects as pe
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import MultipleLocator

import style as S
from fig5_capability_deltas import COL_LABELS, COLS, HEAD_LABELS, VMAX, fmt, owner

NAME = "figA3_community_releases"
OLD_GENS = ("old", "small-old")

# ---------------------------------------------------------------- geometry (inches; y grows downwards)
W = S.TEXT_W
X0 = 0.04
LAB_W = 0.95                    # uploader column
CW = 0.40                       # heatmap cell width
MEAN_GAP = 0.06
DX0 = X0 + LAB_W + 6 * CW + MEAN_GAP + 0.42     # left edge of the divergence axis
DW = W - DX0 - 0.10
ROW_H = 0.175
GROUP_HDR = 0.21
GROUP_GAP = 0.05
TOP = 0.62
BOTTOM = 0.80
INSET = 0.008
TF_MAX = 0.3


def main():
    S.apply()
    S.mpl.rcParams["figure.constrained_layout.use"] = False
    res = S.load_results()
    com = res[res.method == "community"].copy()
    ours = res[(res.method != "base") & (res.method != "community")]
    cmap, norm = S.diverging_cmap(), S.diverging_norm(VMAX)

    groups = []
    for _, old, new in S.PAIRS:
        for model in (old, new):
            rows = [r for _, r in com[com.model == model].iterrows()]
            rows.sort(key=lambda r: owner(r).lower())
            if rows:
                groups.append((model, rows))
    n_rows = sum(len(g[1]) for g in groups)
    body = n_rows * ROW_H + len(groups) * GROUP_HDR + (len(groups) - 1) * GROUP_GAP
    H = round(TOP + body + BOTTOM, 2)

    fig = S.plt.figure(figsize=(W, H), layout="none")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")

    def col_x(j):
        return X0 + LAB_W + j * CW + (MEAN_GAP if j == 5 else 0)

    # ---- titles + column headers
    ax.text(X0, 0.13, "Capability change vs the original (percentage points)", ha="left", va="center",
            fontsize=S.FONT, fontweight="bold", color=S.TEXT)
    ax.text(DX0, 0.13, "Teacher-forced divergence from the original", ha="left", va="center", fontsize=S.FONT,
            fontweight="bold", color=S.TEXT)
    for j, lab in enumerate(COL_LABELS):
        ax.text(col_x(j) + CW / 2, TOP - 0.06, HEAD_LABELS.get(lab, lab), ha="center", va="bottom",
                fontsize=S.FONT_SMALL, linespacing=1.0, color=S.TEXT if j == 5 else S.TEXT_SECONDARY,
                fontweight="bold" if j == 5 else "normal")
    ax.text(DX0, 0.30, "nats/token, 350 benign held-out prompts; 0 = original", ha="left", va="center",
            fontsize=S.FONT_SMALL, color=S.TEXT_SECONDARY)

    # ---- body (heatmap) and row positions for the divergence axis
    y = TOP
    marks, bars, beyond = [], [], []
    for gi, (model, rows) in enumerate(groups):
        if gi:
            y += GROUP_GAP
        gen = rows[0]["generation"]
        older = gen in OLD_GENS
        t = ax.text(X0, y + GROUP_HDR * 0.55, S.MODEL_DISPLAY[model], ha="left", va="center", fontsize=S.FONT,
                    fontweight="bold", color=S.TEXT)
        fig.canvas.draw_idle()
        bb = t.get_window_extent(renderer=fig.canvas.get_renderer())
        ax.text(bb.x1 / fig.dpi + 0.05, y + GROUP_HDR * 0.55, "(older release)" if older else "(newer release)",
                ha="left", va="center", fontsize=S.FONT_SMALL, color=S.TEXT_SECONDARY)
        y += GROUP_HDR
        o = ours[(ours.model == model) & np.isfinite(ours.tf_divergence.astype(float))]
        for r in rows:
            yc = y + ROW_H / 2
            ax.text(X0 + LAB_W - 0.06, yc, owner(r), ha="right", va="center", fontsize=S.FONT_SMALL, color=S.TEXT)
            for j, c in enumerate(COLS):
                v = float(r[c])
                x0 = col_x(j)
                if not np.isfinite(v):
                    S.draw_pending(ax, x0 + INSET, y + INSET, CW - 2 * INSET, ROW_H - 2 * INSET)
                    continue
                fill = cmap(norm(np.clip(v, -VMAX, VMAX)))
                hexfill = "#{:02x}{:02x}{:02x}".format(*(int(round(q * 255)) for q in fill[:3]))
                ax.add_patch(Rectangle((x0 + INSET, y + INSET), CW - 2 * INSET, ROW_H - 2 * INSET, facecolor=fill,
                                       edgecolor="none", zorder=1))
                ax.text(x0 + CW / 2, yc, fmt(v), ha="center", va="center", fontsize=S.FONT_SMALL,
                        color=S.ink_on(hexfill), fontweight="bold" if j == 5 else "normal", zorder=2)
                if abs(v) > VMAX:
                    beyond.append((model, owner(r), COL_LABELS[j], v))
            tf = float(r["tf_divergence"])
            marks.append((yc, tf, gen, r["tf_paired"]))
            if len(o):
                bars.append((yc, float(o.tf_divergence.min()), float(o.tf_divergence.max()), len(o)))
            y += ROW_H
    y_end = y

    # ---- divergence axis, rows aligned with the heatmap (y in inches)
    dax = fig.add_axes([DX0 / W, 1 - y_end / H, DW / W, (y_end - TOP) / H])
    dax.set_ylim(y_end, TOP)
    dax.set_xlim(-0.012, TF_MAX + 0.012)
    dax.xaxis.set_major_locator(MultipleLocator(0.1))
    dax.xaxis.set_minor_locator(MultipleLocator(0.05))
    dax.grid(axis="y", visible=False)
    dax.grid(axis="x", which="both", color=S.GRID, linewidth=S.LW_THIN)
    dax.tick_params(axis="x", which="minor", length=1.5, color=S.AXIS)
    dax.set_yticks([])
    dax.spines["left"].set_visible(False)
    dax.axvline(0, color=S.TEXT_SECONDARY, linewidth=0.9, zorder=1.6)
    for yc, lo, hi, _ in bars:
        dax.plot([lo, hi], [yc, yc], color=S.METHOD_COLORS["base"], alpha=0.35, linewidth=5.5,
                 solid_capstyle="butt", zorder=2)
    for yc, tf, gen, paired in marks:
        if not np.isfinite(tf) or str(paired) != "True":
            dax.text(0.98, yc, "pending (no teacher forcing)", transform=dax.get_yaxis_transform(), ha="right",
                     va="center", fontsize=S.FONT_SMALL, color=S.TEXT_SECONDARY, style="italic")
            continue
        dax.plot([tf], [yc], zorder=3, **S.marker_kw("community", gen))
        x_lab = max([tf] + [hi for yb, _, hi, _ in bars if yb == yc])     # right of the marker and of the grey bar
        dax.annotate(f"{tf:.3f}", (x_lab, yc), xytext=(6, 0), textcoords="offset points", va="center",
                     fontsize=S.FONT_SMALL, color=S.TEXT_SECONDARY, zorder=4,
                     path_effects=[pe.withStroke(linewidth=2.2, foreground=S.SURFACE)])

    # ---- colour bar (left) + key (right)
    cb_y, cb_h, cb_w = y_end + 0.30, 0.09, 1.90
    cb_x = X0 + LAB_W + (6 * CW + MEAN_GAP - cb_w) / 2
    cax = fig.add_axes([cb_x / W, 1 - (cb_y + cb_h) / H, cb_w / W, cb_h / H])
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal", extend="both",
                      extendfrac=0.06)
    ticks = [-VMAX, -VMAX / 2, 0, VMAX / 2, VMAX]
    cb.set_ticks(ticks)
    cb.set_ticklabels([("0" if t == 0 else f"{t:+.0f}".replace("-", "−")) for t in ticks])
    cb.ax.tick_params(labelsize=S.FONT_SMALL, length=2, width=S.LW_THIN, color=S.AXIS, labelcolor=S.TEXT_SECONDARY)
    cb.outline.set_linewidth(S.LW_THIN)
    cb.outline.set_edgecolor(S.AXIS)
    ax.text(cb_x - 0.12, cb_y + cb_h / 2, "loss", ha="right", va="center", fontsize=S.FONT_SMALL,
            color=S.TEXT_SECONDARY)
    ax.text(cb_x + cb_w + 0.12, cb_y + cb_h / 2, "gain", ha="left", va="center", fontsize=S.FONT_SMALL,
            color=S.TEXT_SECONDARY)
    handles = [Line2D([], [], label="release of an older model", **S.marker_kw("community", "old")),
               Line2D([], [], label="release of a newer model", **S.marker_kw("community", "new")),
               Line2D([], [], color=S.METHOD_COLORS["base"], alpha=0.35, linewidth=5.5, solid_capstyle="butt",
                      label="range of this paper's selected\nmodels on the same original")]
    leg = ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(DX0 / W, 1 - (y_end + 0.22) / H),
                    bbox_transform=fig.transFigure, handlelength=1.6, handletextpad=0.5, labelspacing=0.3,
                    borderaxespad=0.0)
    leg.set_in_layout(False)

    S.save(fig, NAME, preview_dpi=150)

    # ---- caption (numbers from the data)
    def gpu_short(g):
        return g.replace("NVIDIA ", "").replace(" Blackwell Server Edition", "").replace("GeForce ", "")

    base_box = res[res.method == "base"].set_index("model")["machine"].to_dict()
    cm_other = sorted({S.MODEL_DISPLAY[r.model] for r in com.itertuples() if r.machine != base_box.get(r.model)})
    gpus = " and the ".join(sorted({gpu_short(g) for g in com.gpu.dropna()}))
    n_old = int(com.generation.isin(OLD_GENS).sum())
    tf = com.tf_divergence.astype(float)
    above = sum(1 for yc, t_, _, _ in marks for b in bars if b[0] == yc and t_ > b[2])
    hi_r = com.loc[tf.idxmax()]
    lo_r = com.loc[tf.idxmin()]
    mean_d = com.capability_mean_delta_pp.astype(float)

    def tex(v):
        return ("0.0" if abs(v) < 0.05 else f"{v:+.1f}").replace("-", "$-$")

    caption = (
        "**Public community releases: capability change and divergence from the original.** The "
        f"{len(com)} community releases evaluated alongside the selected models ({n_old} of older, "
        f"{len(com) - n_old} of newer models; row label = Hugging Face uploader), grouped by the original they were "
        "derived from. Left: benchmark score minus that of the same original, in percentage points, with the "
        f"diverging scale of Fig. 5 (saturating at $\\pm${VMAX:.0f} pp; printed values exact); the last column is the "
        "unweighted mean of the five benchmarks. Right: teacher-forced divergence on the 350 benign held-out prompts, "
        "defined as in Fig. 6 (0 = the original); hollow markers are releases of the older, filled of the newer "
        "model of a pair. The grey bar spans the divergences of this paper's selected models on the same original "
        f"(all conditions of Fig. 6). {above} of {len(marks)} releases lie above that range; divergences range from "
        f"{float(lo_r.tf_divergence):.3f} ({owner(lo_r)}, {S.MODEL_DISPLAY[lo_r.model]}) to "
        f"{float(hi_r.tf_divergence):.3f} ({owner(hi_r)}, {S.MODEL_DISPLAY[hi_r.model]}) nats per token, and mean "
        f"capability changes from {tex(mean_d.min())} to {tex(mean_d.max())} pp. The releases were made by third parties with their own methods and "
        "selection; they were not selected with this paper's rule and are not part of the H1--H3 tests."
        + (f" Beyond the colour scale: " + "; ".join(f"{S.MODEL_DISPLAY[m]} {o_} {b}, {tex(v)} pp"
                                                       for m, o_, b, v in beyond) + "." if beyond else "")
    )
    detail = (f"*For the text:* evaluated on the {gpus}"
              + (f"; for {', '.join(cm_other)} this differs from the GPU that evaluated the original."
                 if cm_other else ", each on the same GPU as its original.")
              + " Release repositories: " + "; ".join(f"{S.MODEL_DISPLAY[r.model]}: {r.repo}"
                                                       for r in com.itertuples()) + ".")
    source = (f"{S.SOURCE_CAPABILITY} {S.SOURCE_HELDOUT_TF} Values from paper/figures/data/results.csv "
              f"(method == community; data snapshot {S.snapshot_utc()}). Caveat: the Qwen2.5-32B original was scored "
              "with an earlier harness revision and its MBPP score was rescored from its logged samples with the "
              "patched code extractor.")
    S.write_caption(NAME, caption + "\n\n" + detail, source=source)
    print(f"  [figA3] {len(com)} releases; {above} above the selected-model range; height {H} in")


if __name__ == "__main__":
    main()
