"""fig5_capability_deltas -- capability change of every derived model vs its own original (percentage points).

Heatmap at the template text width (6.5 in). Two blocks side by side: old generation (left) and new generation
(right), with rows aligned by family x method, so one row compares the same method across generations. Columns:
MMLU-Pro, GSM8K, IFEval, HumanEval, MBPP, then the unweighted mean of the five. Diverging scale (orange = loss,
grey = 0, blue = gain), symmetric limits +/- VMAX, exact values printed. Cells without a finished evaluation are drawn
hatched and labelled with the reason ("pending: ..."). Community reference models are one row per public release
(label = Hugging Face uploader, purple square = the fixed community colour), taken from results.csv, i.e. from the
evaluation queues; releases by the same uploader share a row across the two generations. A side with no such
release says "n/a: no community release queued". Since 2026-11-05 the community rows are OFF here (SHOW_COMMUNITY =
False): they made the figure tall and tight, and they are drawn in figA3_community_releases (capability deltas and
teacher-forced divergence of the eight public releases); this figure keeps the paper's own conditions, with taller
rows.

Input: paper/figures/data/results.csv (extract_data.py). Run with scripts/figures/.venv/bin/python.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Rectangle

import style as S

NAME = "fig5_capability_deltas"
VMAX = 8.0                      # pp; symmetric colour limits (ramp steps every 2 pp); larger |deltas| saturate
COLS = [f"{b}_delta_pp" for b in S.BENCHES] + ["capability_mean_delta_pp"]
COL_LABELS = [S.BENCH_LABELS[b] for b in S.BENCHES] + ["Mean"]
HEAD_LABELS = {"MMLU-Pro": "MMLU-\nPro", "HumanEval": "Human\nEval"}   # headers wider than a cell
ROW_LABELS = dict(S.METHOD_LABELS, gamma_mpoa="γ-MPOA")   # unicode gamma: same font as the rest

# ---------------------------------------------------------------- geometry (inches; y grows downwards)
W = S.TEXT_W
MARGIN = 0.06
LAB_W = 0.72                    # method-label column of each block (community rows: square + uploader)
MEAN_GAP = 0.06                 # gap between the five benchmarks and the mean column
BLOCK_GAP = 0.14
CW = round((W - 2 * MARGIN - BLOCK_GAP - 2 * (LAB_W + MEAN_GAP)) / 12, 4)   # cell width (~0.39 in)
ROW_H = 0.155                   # was 0.14 while the community rows were drawn here
SHOW_COMMUNITY = False          # community releases: figA3_community_releases
GROUP_HDR = 0.18                # model-name band above each group
GROUP_GAP = 0.05
STUDY_GAP = 0.10                # extra space before the small pair
INSET = 0.008                   # half of the surface gap between neighbouring cells
TOP = 0.58                      # block titles + column headers (two-line headers for the long names)
BOTTOM = 0.74                   # colour bar + legend


def fmt(v):
    if abs(v) < 0.05:
        return "0.0"
    return f"{v:+.1f}".replace("-", "−")


def pending_text(row, ncells):
    """Short reason for a run of pending cells (falls back to 'pending' when the run is short)."""
    st = str(row["status"])
    if ncells < 3:
        return S.PENDING_LABEL
    if st.startswith("pending: evaluation queued"):
        return "pending: evaluation queued"
    if st.startswith("pending: evaluation started") or st.startswith("pending: evaluation in progress"):
        return "pending: evaluation running"
    if st.startswith("pending: evaluation not queued"):
        return "pending: evaluation not queued yet"
    if st.startswith("planned"):
        if row["method"] == "community" and isinstance(row.get("variant"), str) and row["variant"]:
            return f"pending: planned ({row['variant']})"
        if row["method"] == "gamma_mpoa":
            return "pending: planned (not run yet)"
        return "pending: planned (not evaluated yet)"
    if st.startswith("present"):
        return "pending: original not evaluated"
    return "pending"


def owner(row):
    """Hugging Face uploader of a community release (results.csv `repo`), else its variant token."""
    repo = str(row.get("repo") or "")
    return repo.split("/")[0] if "/" in repo else str(row.get("variant") or "community")


def ordered(res, model, method, partner):
    """Rows of one model x method in display order. Community releases: uploaders that also released the partner
    model (other generation) first, so a shared uploader lines up across the two blocks; then by uploader."""
    sub = res[(res.model == model) & (res.method == method)].sort_values(["variant", "label"], na_position="last")
    rows = [r for _, r in sub.iterrows()]
    if method == "community":
        other = {owner(r) for _, r in res[(res.model == partner) & (res.method == method)].iterrows()}
        rows.sort(key=lambda r: (owner(r) not in other, owner(r).lower()))
    return rows


def build_groups(res):
    """[(family, old, new, [(method, k), ...])] with the union of both models' method rows, in METHOD_ORDER."""
    groups = []
    for fam, old, new in S.PAIRS:
        keys = []
        for meth in S.METHOD_ORDER:
            if meth == "base" or (meth == "community" and not SHOW_COMMUNITY):
                continue
            n = max(len(ordered(res, old, meth, new)), len(ordered(res, new, meth, old)))
            keys += [(meth, k) for k in range(n)]
        groups.append((fam, old, new, keys))
    return groups


def lookup(res, model, method, k, partner):
    rows = ordered(res, model, method, partner)
    return rows[k] if k < len(rows) else None


def main():
    S.apply()
    # hand-placed layout (inches): keep the constrained-layout engine of the house rcParams off, also at savefig
    S.mpl.rcParams["figure.constrained_layout.use"] = False
    res = S.load_results()
    cmap = S.diverging_cmap()
    norm = S.diverging_norm(VMAX)
    groups = build_groups(res)

    # ---- vertical layout
    n_rows = sum(len(g[3]) for g in groups)
    body = n_rows * ROW_H + len(groups) * GROUP_HDR + (len(groups) - 1) * GROUP_GAP + STUDY_GAP
    H = round(TOP + body + BOTTOM, 2)
    fig = S.plt.figure(figsize=(W, H), layout="none")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")

    block_w = LAB_W + 5 * CW + MEAN_GAP + CW
    used = 2 * block_w + BLOCK_GAP
    x_blocks = [(W - used) / 2, (W - used) / 2 + block_w + BLOCK_GAP]

    def col_x(xb, j):
        return xb + LAB_W + j * CW + (MEAN_GAP if j == 5 else 0)

    # ---- block titles (generation encoded as hollow / filled marker, as in every other figure) + column headers
    for gi, (xb, title) in enumerate(zip(x_blocks, ["Older release", "Newer release"])):
        gen = "old" if gi == 0 else "new"
        filled = S.GEN_STYLE[gen]["filled"]
        ax.plot([xb + 0.05], [0.13], marker="o", markersize=S.MS, linestyle="none",
                markeredgecolor=S.TEXT_SECONDARY, markerfacecolor=S.TEXT_SECONDARY if filled else S.SURFACE,
                markeredgewidth=0.6 if filled else S.HOLLOW_MEW, clip_on=False)
        ax.text(xb + 0.13, 0.13, title, ha="left", va="center", fontsize=S.FONT, fontweight="bold", color=S.TEXT)
        for j, lab in enumerate(COL_LABELS):
            ax.text(col_x(xb, j) + CW / 2, TOP - 0.06, HEAD_LABELS.get(lab, lab), ha="center", va="bottom",
                    fontsize=S.FONT_SMALL, linespacing=1.0,
                    color=S.TEXT if j == 5 else S.TEXT_SECONDARY, fontweight="bold" if j == 5 else "normal")

    # ---- body
    pend_rows, na_rows, beyond = [], [], []
    y = TOP
    for gidx, (fam, old, new, keys) in enumerate(groups):
        if gidx > 0:
            y += GROUP_GAP + (STUDY_GAP if fam.startswith("Small pair") else 0)
        for gi, (xb, model) in enumerate(zip(x_blocks, [old, new])):
            name = S.MODEL_DISPLAY[model] + (" (small pair)" if fam.startswith("Small pair") else "")
            ax.text(xb, y + GROUP_HDR * 0.55, name, ha="left", va="center", fontsize=S.FONT, fontweight="bold",
                    color=S.TEXT)
        y += GROUP_HDR
        for meth, k in keys:
            for xb, model, partner in zip(x_blocks, [old, new], [new, old]):
                row = lookup(res, model, meth, k, partner)
                if meth == "community":
                    lab = owner(row) if row is not None else "Community"
                    t_ = ax.text(xb + LAB_W - 0.06, y + ROW_H / 2, lab, ha="right", va="center",
                                 fontsize=S.FONT_SMALL, color=S.TEXT if row is not None else S.TEXT_SECONDARY)
                    if row is not None:   # community colour square left of the uploader name
                        fig.canvas.draw_idle()
                        bb = t_.get_window_extent(renderer=fig.canvas.get_renderer())
                        ax.text(bb.x0 / fig.dpi - 0.03, y + ROW_H / 2, "\u25a0", ha="right", va="center",
                                fontsize=S.FONT_SMALL, color=S.METHOD_COLORS["community"])
                else:
                    ax.text(xb + LAB_W - 0.06, y + ROW_H / 2, ROW_LABELS.get(meth, meth), ha="right", va="center",
                            fontsize=S.FONT_SMALL, color=S.TEXT if row is not None else S.TEXT_SECONDARY)
                if row is None:      # no such condition for this model
                    na_rows.append((model, meth))
                    ax.text(col_x(xb, 0) + 0.04, y + ROW_H / 2, "n/a: no community release queued"
                            if meth == "community" else "n/a: not in the design", ha="left", va="center",
                            fontsize=S.FONT_SMALL, color=S.TEXT_SECONDARY, style="italic")
                    continue
                vals = [row[c] for c in COLS]
                # filled cells
                for j, v in enumerate(vals):
                    if not np.isfinite(v):
                        continue
                    fill = cmap(norm(np.clip(v, -VMAX, VMAX)))
                    hexfill = "#{:02x}{:02x}{:02x}".format(*(int(round(c * 255)) for c in fill[:3]))
                    x0 = col_x(xb, j)
                    ax.add_patch(Rectangle((x0 + INSET, y + INSET), CW - 2 * INSET, ROW_H - 2 * INSET,
                                           facecolor=fill, edgecolor="none", zorder=1))
                    ax.text(x0 + CW / 2, y + ROW_H / 2, fmt(v), ha="center", va="center", fontsize=S.FONT_SMALL,
                            color=S.ink_on(hexfill), fontweight="bold" if j == 5 else "normal", zorder=2)
                    if abs(v) > VMAX:
                        beyond.append((model, meth, COL_LABELS[j], v))
                # pending runs (consecutive missing cells, the mean column counts as adjacent)
                j = 0
                while j < len(vals):
                    if np.isfinite(vals[j]):
                        j += 1
                        continue
                    j1 = j
                    while j1 + 1 < len(vals) and not np.isfinite(vals[j1 + 1]):
                        j1 += 1
                    x0, x1 = col_x(xb, j), col_x(xb, j1) + CW
                    S.draw_pending(ax, x0 + INSET, y + INSET, x1 - x0 - 2 * INSET, ROW_H - 2 * INSET,
                                   text=pending_text(row, j1 - j + 1), fontsize=S.FONT_SMALL)
                    j = j1 + 1
                if not all(np.isfinite(v) for v in vals):
                    pend_rows.append((model, meth, str(row["status"])))
            y += ROW_H

    # ---- colour bar (left half) + pending / n/a key (right half)
    cb_y = y + 0.20
    cb_h = 0.09
    cb_w = 2.10
    cb_x = x_blocks[0] + LAB_W + (5 * CW + MEAN_GAP + CW - cb_w) / 2
    cax = fig.add_axes([cb_x / W, 1 - (cb_y + cb_h) / H, cb_w / W, cb_h / H])
    sm = ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal", extend="both", extendfrac=0.06)
    ticks = [-VMAX, -VMAX / 2, 0, VMAX / 2, VMAX]
    cb.set_ticks(ticks)
    cb.set_ticklabels([("0" if t == 0 else f"{t:+.0f}".replace("-", "−")) for t in ticks])
    cb.ax.tick_params(labelsize=S.FONT_SMALL, length=2, width=S.LW_THIN, color=S.AXIS, labelcolor=S.TEXT_SECONDARY)
    cb.outline.set_linewidth(S.LW_THIN)
    cb.outline.set_edgecolor(S.AXIS)
    ax.text(cb_x - 0.14, cb_y + cb_h / 2, "loss", ha="right", va="center", fontsize=S.FONT_SMALL,
            color=S.TEXT_SECONDARY)
    ax.text(cb_x + cb_w + 0.14, cb_y + cb_h / 2, "gain", ha="left", va="center", fontsize=S.FONT_SMALL,
            color=S.TEXT_SECONDARY)
    ax.text(cb_x + cb_w / 2, cb_y + cb_h + 0.25, "change vs the original model (percentage points)", ha="center",
            va="center", fontsize=S.FONT_SMALL, color=S.TEXT)

    kx = x_blocks[1] + 0.25
    keys = []
    if pend_rows:
        keys.append(("pending", "pending: no finished evaluation yet (reason in cell)"))
    if SHOW_COMMUNITY:
        keys.append(("community", "community release (row label = uploader)"))
    if na_rows:
        keys.append(("na", "no such condition for this model"))
    for i, (kind, text) in enumerate(keys):
        yk = cb_y - 0.04 + 0.19 * i
        if kind == "pending":
            S.draw_pending(ax, kx + INSET, yk, CW - 2 * INSET, ROW_H - 2 * INSET, text=None)
        elif kind == "community":
            ax.text(kx + CW / 2, yk + ROW_H / 2, "\u25a0", ha="center", va="center", fontsize=S.FONT_SMALL,
                    color=S.METHOD_COLORS["community"])
        elif kind == "na":
            ax.text(kx + CW / 2, yk + ROW_H / 2, "n/a", ha="center", va="center", fontsize=S.FONT_SMALL,
                    color=S.TEXT_SECONDARY, style="italic")
        ax.text(kx + (CW + 0.06 if kind != "note" else 0), yk + ROW_H / 2, text, ha="left", va="center",
                fontsize=S.FONT_SMALL, color=S.TEXT_SECONDARY)

    S.save(fig, NAME, preview_dpi=150)

    # ---- caption (numbers computed from the data, so a refresh keeps it true)
    shown = res[(res.method != "base") & ((res.method != "community") | SHOW_COMMUNITY)]
    present = shown[shown[COLS].notna().all(axis=1)]
    n_present = len(present)
    n_pending = len(pend_rows)
    reasons = {}
    for model, meth, st in pend_rows:
        key = ("queued" if "queued" in st else "running" if ("started" in st or "in progress" in st)
               else "planned" if st.startswith("planned") else "other")
        reasons.setdefault(key, []).append(f"{S.MODEL_DISPLAY[model]} {ROW_LABELS.get(meth, meth)}")
    n_q, n_r, n_p = (len(reasons.get(k, [])) for k in ("queued", "running", "planned"))
    se_cols = [f"{b}_se_pct" for b in S.BENCHES]
    se = res.loc[res.finished == True, se_cols].to_numpy(dtype=float)  # noqa: E712
    se_lo, se_hi = np.nanmin(se), np.nanmax(se)
    def tex(v):          # caption number in LaTeX notation, like the other captions
        return ("0.0" if abs(v) < 0.05 else f"{v:+.1f}").replace("-", "$-$")

    beyond_txt = "; ".join(f"{S.MODEL_DISPLAY[m]} {ROW_LABELS.get(me, me).replace(chr(947), '$\\gamma$')} "
                           f"{b}, {tex(v)} pp" for m, me, b, v in beyond)
    snap = S.snapshot_utc()
    running = (", ".join(reasons.get("running", [])) or "none").replace(chr(947), "$\\gamma$")
    planned = (", ".join(reasons.get("planned", [])) or "none").replace(chr(947), "$\\gamma$")
    n_cells = n_present + n_pending
    # community releases evaluated on a different GPU from their own original (comparability note)
    base_box = res[res.method == "base"].set_index("model")["machine"].to_dict()
    cm = res[res.method == "community"]
    cm_other = sorted({S.MODEL_DISPLAY[r.model] for r in cm.itertuples() if r.machine != base_box.get(r.model)})
    def gpu_short(g):    # 'NVIDIA RTX PRO 6000 Blackwell Server Edition' -> 'RTX PRO 6000'
        return g.replace("NVIDIA ", "").replace(" Blackwell Server Edition", "").replace("GeForce ", "")

    cm_gpu = sorted({gpu_short(g) for g in cm.gpu.dropna()})
    cm_note = ""
    if len(cm):
        cst = cm.status.astype(str)
        n_done, n_queued = int(cst.str.startswith("present").sum()), int(cst.str.contains("queued").sum())
        n_run = int(cst.str.contains("in progress|started|running").sum())
        n_other = len(cm) - n_done - n_queued - n_run
        state = ", ".join(x for x in (f"{n_done} finished" if n_done else "", f"{n_run} running" if n_run else "",
                                      f"{n_queued} queued" if n_queued else "",
                                      f"{n_other} in another state" if n_other else "") if x)
        gpus = " and the ".join(cm_gpu)
        cm_note = (f"The {len(cm)} community releases ({int(cm.generation.isin(['old', 'small-old']).sum())} of "
                   f"older, {int((~cm.generation.isin(['old', 'small-old'])).sum())} of newer models; {state}) are "
                   f"evaluated on the {gpus}"
                   + (f"; for {', '.join(cm_other)} this differs from the GPU that evaluated the original."
                      if cm_other else ", each on the same GPU as its original."))

    caption = (
        "**Capability change after refusal removal.** Each cell is the benchmark score of the selected model minus "
        "that of its own original, in percentage points (pp); the last column is the unweighted mean of the five "
        "benchmarks. Orange marks a loss, blue a gain; the scale saturates at "
        f"$\\pm${VMAX:.0f} pp and the printed values are exact"
        + ((f" (beyond the scale: {beyond_txt})" if beyond else ""))
        + ". Rows align the same method across the older (left) and newer (right) release of a family"
        + ("; community rows are public releases, labelled by uploader" if SHOW_COMMUNITY else
           f"; the {len(res[res.method == 'community'])} public community releases are shown in Fig. A3")
        + (". Hatched cells have no finished evaluation yet and give the reason" if n_pending else "")
        + f" ({n_present} of {n_cells} model $\\times$ condition rows have results at the data snapshot {snap}). One item is worth "
        "0.1--0.61 pp depending on the benchmark and the lm-eval standard error of a single score is "
        f"{se_lo:.1f}--{se_hi:.1f} pp, so differences of 1--2 pp are within noise (single seed). Each derived model "
        "is the dev-set selection of its cell; no refusal prompts, dev or held-out, enter this figure."
    )
    detail = (
        "*For the text (moved out of the caption):* pending cells: "
        f"{n_q} queued, {n_r} running ({running}), {n_p} planned ({planned}). " + (cm_note + " " if SHOW_COMMUNITY else "") + "Benchmarks: MMLU-Pro (5-shot chain of thought, exact match; fixed "
        "subject-stratified subset of 1000 of the 12,032 test questions), GSM8K (5-shot, flexible-extract exact "
        "match; fixed random subset of 500 of the 1319 test problems), IFEval (prompt-level strict accuracy; all "
        "541 prompts), HumanEval (instruct, pass@1; all 164 problems) and MBPP (instruct, 3-shot, pass@1; all 500 "
        "test problems), greedy decoding with the model's chat template. One item is worth 0.1 pp on MMLU-Pro, "
        "0.2 pp on GSM8K and MBPP, 0.18 pp on IFEval and 0.61 pp on HumanEval. The small pair was optimised and "
        "evaluated on an RTX 4090. Selection: fewest dev keyword refusals among 100 TPE trials with first-token KL "
        "$\\le$ 0.1."
    )
    caption = caption + "\n\n" + detail
    source = (
        f"{S.SOURCE_CAPABILITY} {S.SOURCE_SELECTION} Values from paper/figures/data/results.csv "
        f"(`*_delta_pp`, `capability_mean_delta_pp`; data snapshot {snap}). "
        "Caveat: the Qwen2.5-32B original was scored with an earlier harness revision (no U+2581 post-processing) "
        "and its MBPP score was rescored from its logged samples with the patched code extractor; all derived models "
        "were scored with the patched extractor directly. Qwen3.5-4B exports drop the 15 mtp.* tensors and store 48 "
        "F32 linear-attention tensors as BF16, so its deltas include that export difference as well as the edit."
    )
    S.write_caption(NAME, caption, source=source)

    print(f"  rows: {n_rows} per block; present {n_present}; pending {n_pending} "
          f"(queued {n_q}, running {n_r}, planned {n_p}); n/a {len(na_rows)}; beyond +/-{VMAX:.0f}: {beyond_txt or '-'}")
    print(f"  height {H} in; caption {len(caption.split(chr(10))[0].split())} words")


if __name__ == "__main__":
    main()
