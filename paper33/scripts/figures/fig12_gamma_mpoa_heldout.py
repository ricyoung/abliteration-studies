"""fig12_gamma_mpoa_heldout -- gamma-corrected MPOA vs MPOA on the four post-norm models: held-out judged refusal
next to capability change.

One block of three rows per model (Gemma-2-27B, Gemma-4-31B, OLMo-2-32B, OLMo-3.1-32B): the unedited original,
MPOA and gamma-MPOA (each the dev-set selection of its cell, seed 1).
Left : WildGuard refusal on the 972 harmful held-out prompts with Wilson 95% intervals; the column on the right of the
       panel gives gamma-MPOA minus MPOA (pp, paired; * = Holm-significant over the 4 models).
Right: capability change vs the model's own original (pp) on the five benchmarks and their unweighted mean, in the
       diverging palette of fig5 (saturating at +-8 pp; printed values exact); the original's row gives its absolute
       scores (%). * = gamma-MPOA differs from the original on that benchmark (paired per-item McNemar, Holm over the
       20 tests per reference in final_gmpoa_benchmarks.csv). Then the selected trial's dev first-token KL and the
       teacher-forced divergence on the 350 benign held-out prompts.

Inputs : results/analysis/final_gmpoa_control.csv (refusal, gamma - MPOA test, dev KL, divergence),
         results/analysis/final_capability.csv (benchmark scores and deltas), final_gmpoa_benchmarks.csv (paired tests)
Outputs: paper/figures/fig12_gamma_mpoa_heldout.pdf, preview PNG, captions.md section
Run    : cd scripts/figures && .venv/bin/python fig12_gamma_mpoa_heldout.py
"""
from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

import judged as J
import style as S

NAME = "fig12_gamma_mpoa_heldout"
MODELS = [("gemma-2-27b", "old"), ("gemma-4-31b", "new"), ("olmo-2-32b", "old"), ("olmo-3.1-32b", "new")]
ROWS = ["original", "mpoa", "gamma_mpoa"]
COLS = S.BENCHES + ["mean"]
COL_LAB = {**S.BENCH_LABELS, "mean": "Mean", "humaneval": "Human\nEval", "mmlu_pro": "MMLU-\nPro"}
VMAX = 8.0                  # pp, colour saturation (as fig5)
BLOCK = 4.0                 # y units per model block (3 rows + header)
CW = 1.0                    # heat-map column width (data units)
TEXT_X = [6.6, 7.65]       # x of the two text columns (dev KL, TF divergence)
ALPHA = 0.05


def load():
    ctl = J.load("final_gmpoa_control.csv")
    cap = J.load("final_capability.csv")
    bm = J.load("final_gmpoa_benchmarks.csv")
    data = {}
    for model, gen in MODELS:
        rows = {}
        for m in ROWS:
            r = J.one(ctl[(ctl["model"] == model) & (ctl["method"] == m)], f"{model} {m} control")
            c = J.one(cap[(cap["model"] == model) & (cap["method"] == m) & (cap["role"] != "replicate")],
                      f"{model} {m} capability")
            if abs(float(c["capability_mean_delta_pp"]) - float(r["capability_mean_delta_pp"])) > 1e-6:
                raise ValueError(f"{model} {m}: capability mean delta differs between final_* tables")
            sig = {}
            if m == "gamma_mpoa":
                for b in S.BENCHES:
                    t = J.one(bm[(bm["model"] == model) & (bm["benchmark"] == b) & (bm["reference"] == "original")],
                              f"{model} {b}")
                    if abs(float(t["diff_pp"]) - float(c[f"{b}_delta_pp"])) > 1e-6:
                        raise ValueError(f"{model} {b}: paired-test delta != final_capability delta")
                    sig[b] = float(t["p_holm_within_reference"]) < ALPHA
            rows[m] = dict(ctl=r, cap=c, sig=sig)
        data[model] = dict(gen=gen, rows=rows)
    return data


def ypos(bi, ri):
    return -(bi * BLOCK + 1 + ri)


def main():
    S.apply()
    data = load()
    fig = plt.figure(figsize=(S.DOUBLE_COL, 4.55))
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.03, wspace=0.02)
    gs = fig.add_gridspec(1, 2, width_ratios=[0.8, 1.55])
    ax_r = fig.add_subplot(gs[0, 0])
    ax_h = fig.add_subplot(gs[0, 1], sharey=ax_r)
    cmap, norm = S.diverging_cmap(), S.diverging_norm(VMAX)
    yt, ytl = [], []
    for bi, (model, gen) in enumerate(MODELS):
        d = data[model]
        y_head = ypos(bi, 0) + 0.95
        ax_r.text(0.0, y_head, S.MODEL_DISPLAY[model] + (" (older release)" if gen == "old" else
                                                         " (newer release)"),
                  ha="left", va="center", fontsize=S.FONT, fontweight="bold", clip_on=False)
        for ri, m in enumerate(ROWS):
            y = ypos(bi, ri)
            r = d["rows"][m]["ctl"]
            mk = "base" if m == "original" else m
            c = S.color(mk)
            ax_r.hlines(y, 100 * r["wg_lo"], 100 * r["wg_hi"], color=c, lw=1.0, zorder=2)
            ax_r.plot([100 * r["wg_rate"]], [y], **S.marker_kw(mk, gen), zorder=3)
            ax_r.annotate(J.pct(r["wg_rate"]), (100 * r["wg_rate"], y), xytext=(7, 0), textcoords="offset points",
                          ha="left", va="center", fontsize=S.FONT_SMALL, color=S.TEXT_SECONDARY)
            yt.append(y)
            ytl.append(S.label(mk))
            if m == "gamma_mpoa":
                sig = float(r["vs_mpoa_p_holm"]) < ALPHA
                ax_r.text(1.03, y, J.pp(r["vs_mpoa_diff"]) + ("*" if sig else ""),
                          transform=ax_r.get_yaxis_transform(), ha="left", va="center", fontsize=S.FONT_SMALL,
                          color=S.TEXT if sig else S.TEXT_SECONDARY, clip_on=False)
            # heat map row
            cap = d["rows"][m]["cap"]
            for ci, col in enumerate(COLS):
                x0 = ci * CW
                if m == "original":
                    v = float(cap["capability_mean_pct"] if col == "mean" else cap[f"{col}_pct"])
                    ax_h.text(x0 + CW / 2, y, J.num(v), ha="center", va="center", fontsize=S.FONT_SMALL,
                              color=S.TEXT_SECONDARY, fontweight="bold" if col == "mean" else "normal")
                    continue
                v = float(cap["capability_mean_delta_pp"] if col == "mean" else cap[f"{col}_delta_pp"])
                fill = cmap(norm(max(-VMAX, min(VMAX, v))))
                fill = to_hex(fill)
                ax_h.add_patch(Rectangle((x0 + 0.04, y - 0.44), CW - 0.08, 0.88, facecolor=fill, edgecolor="none",
                                         zorder=1))
                star = "*" if d["rows"][m]["sig"].get(col) else ""
                vtxt = J.num(0.0) if J.half_up(v) == 0 else J.signed(v)     # 0.0 unsigned, as fig5
                ax_h.text(x0 + CW / 2, y, vtxt + star, ha="center", va="center", fontsize=S.FONT_SMALL,
                          color=S.ink_on(fill), fontweight="bold" if col == "mean" else "normal", zorder=2)
            for xi, (key, nd) in zip(TEXT_X, (("dev_kl", 3), ("tf_divergence", 3))):
                if m == "original":
                    continue
                ax_h.text(xi, y, J.num(float(r[key]), nd), ha="center", va="center", fontsize=S.FONT_SMALL,
                          color=S.TEXT)
    # refusal axes
    ax_r.set_yticks(yt)
    ax_r.set_yticklabels(ytl)
    ax_r.tick_params(axis="y", length=0)
    ax_r.set_xlim(-2, 102)
    ax_r.spines["bottom"].set_bounds(0, 100)
    ax_r.set_xticks([0, 25, 50, 75, 100])
    ax_r.set_xlabel("Held-out refusal (%, WildGuard)")
    ax_r.grid(axis="y", visible=False)
    y_lo = ypos(len(MODELS) - 1, 2) - 0.6
    y_hi = 1.05
    ax_r.set_ylim(y_lo, y_hi)
    ax_r.text(1.03, y_hi - 0.35, "γ-MPOA −\nMPOA (pp)", transform=ax_r.get_yaxis_transform(), ha="left",
              va="center", fontsize=S.FONT_SMALL, color=S.TEXT_SECONDARY, clip_on=False, linespacing=1.1)
    for bi in range(1, len(MODELS)):
        yb = ypos(bi, 0) + 1.45
        for ax in (ax_r, ax_h):
            ax.axhline(yb, color=S.GRID, lw=S.LW_THIN, zorder=0)
    # heat-map axes: no spines/ticks, column headers on top
    ax_h.set_xlim(-0.05, TEXT_X[-1] + 0.5)
    ax_h.axis("off")
    for ci, col in enumerate(COLS):
        ax_h.text(ci * CW + CW / 2, y_hi - 0.95, COL_LAB[col], ha="center", va="bottom", fontsize=S.FONT_SMALL,
                  color=S.TEXT if col == "mean" else S.TEXT_SECONDARY, fontweight="bold" if col == "mean" else "normal",
                  linespacing=1.0)
    for xi, lab in zip(TEXT_X, ("dev KL", "TF div.")):
        ax_h.text(xi, y_hi - 0.95, lab, ha="center", va="bottom", fontsize=S.FONT_SMALL, color=S.TEXT_SECONDARY)
    ax_r.set_title("a   Held-out refusal", loc="left", fontweight="bold", pad=4)
    ax_h.set_title("b   Capability change vs the original (pp)", loc="left",
                   fontweight="bold", pad=4)
    # key
    h = [Line2D([], [], label=S.label(m), **S.marker_kw(m, "new")) for m in ("base", "mpoa", "gamma_mpoa")]
    h += [Line2D([], [], color=S.TEXT_SECONDARY, marker="o", ms=S.MS, mfc=S.SURFACE, mec=S.TEXT_SECONDARY,
                 mew=S.HOLLOW_MEW, ls="none", label="older release (hollow)"),
          Line2D([], [], color=S.TEXT_SECONDARY, lw=1.0, label="95% Wilson interval")]
    fig.legend(handles=h, loc="outside lower left", ncols=5, frameon=False, handlelength=1.4,
               columnspacing=1.2)
    ax_h.text(0.0, y_lo + 0.15, "Original rows: score (%); other rows: change vs the original (pp).\n"
              "Orange = loss, blue = gain; colour saturates at \u00b18 pp.\n"
              "* p < 0.05, exact McNemar, Holm-adjusted (a: 4 models; b: 20 tests).\n"
              "dev KL: selected trial's first-token KL; TF div.: teacher-forced\n"
              "divergence on benign held-out prompts (nats per token).", ha="left", va="top", fontsize=S.FONT_SMALL, color=S.TEXT_SECONDARY,
              linespacing=1.3)
    S.save(fig, NAME)
    write_caption(data)


def write_caption(data):
    def row(model, m):
        return data[model]["rows"][m]

    parts = []
    for model, _ in MODELS:
        g = row(model, "gamma_mpoa")["ctl"]
        mp = row(model, "mpoa")["ctl"]
        parts.append(f"{S.MODEL_DISPLAY[model]} {J.pct(mp['wg_rate'])}% $\\to$ {J.pct(g['wg_rate'])}% "
                     f"({J.pp(g['vs_mpoa_diff'])} pp [{J.pp(g['vs_mpoa_diff_lo'])}, {J.pp(g['vs_mpoa_diff_hi'])}], "
                     f"p$_\\mathrm{{Holm}}$ {J.p_text(float(g['vs_mpoa_p_holm']))})")
    losses = sorted(((float(row(mo, "gamma_mpoa")["cap"]["capability_mean_delta_pp"]), mo) for mo, _ in MODELS))
    big = [mo for v, mo in losses if v < -5]
    big_txt = "; ".join(
        f"{S.MODEL_DISPLAY[mo]} {J.signed(row(mo, 'gamma_mpoa')['cap']['capability_mean_delta_pp'])} pp mean "
        f"(HumanEval {J.num(row(mo, 'gamma_mpoa')['cap']['humaneval_pct'])}% vs "
        f"{J.num(row(mo, 'original')['cap']['humaneval_pct'])}%) at dev KL "
        f"{J.num(float(row(mo, 'gamma_mpoa')['ctl']['dev_kl']), 3)}" for mo in big)
    allcap = J.load("final_capability.csv")
    mod = allcap[allcap["role"] != "original"].sort_values("capability_mean_delta_pp")
    two = mod.head(2)
    if set(two["method"]) != {"gamma_mpoa"}:
        raise ValueError("the two largest capability losses are no longer both gamma-MPOA cells; edit the caption")
    nxt = mod.iloc[2]
    gg = J.load("final_gmpoa_generation.csv")

    def ge(pair, m):
        r = J.one(gg[(gg["pair"] == pair) & (gg["method"] == m)], f"{pair} {m} generation")
        return f"{J.pp(r['diff'])} pp [{J.pp(r['diff_lo'])}, {J.pp(r['diff_hi'])}]"

    gen_txt = "; ".join(f"{fam} {ge(pk, 'gamma_mpoa')} vs MPOA {ge(pk, 'mpoa')}"
                        for fam, pk in (("Gemma", "gemma"), ("OLMo", "olmo")))
    nxt_txt = (f"next largest: {S.MODEL_DISPLAY[nxt['model']]} {S.label(nxt['method'])}, "
               f"{J.signed(nxt['capability_mean_delta_pp'])} pp")
    cap = (
        "**$\\gamma$-corrected MPOA versus MPOA on the four post-norm models: held-out, judged refusal and "
        "capability.** Each block is one model (older or newer release of the Gemma and OLMo pairs) with its unedited "
        "original (grey star), the selected MPOA model (orange square) and the selected $\\gamma$-MPOA model (red "
        "cross; MPOA with the edit corrected for the post-norm output gains); hollow markers are the older releases. "
        "(a) Share of the 972 harmful held-out prompts whose response WildGuard labels a refusal, with Wilson 95% "
        "intervals; the column on the right is $\\gamma$-MPOA minus MPOA (percentage points, paired; * = "
        "significant, exact McNemar, Holm over the four models). $\\gamma$-MPOA refuses less than MPOA on all four "
        f"models (Newcombe 95% intervals): {'; '.join(parts)}. "
        "(b) Change of each benchmark score from the model's own original, in percentage points (orange = loss, blue "
        "= gain; the colour saturates at $\\pm$8 pp, the printed values are exact; the original's row gives its "
        "scores in %); * marks a $\\gamma$-MPOA score that differs from the original's (paired per-item McNemar, "
        "Holm over 20 tests: 4 models $\\times$ 5 benchmarks). The last two columns are the selected trial's dev "
        "first-token KL divergence and the teacher-forced divergence on the 350 benign held-out prompts (nats per "
        f"token). The two largest capability losses of all {len(mod)} modified models are $\\gamma$-MPOA cells: "
        f"{big_txt}, both inside the KL $\\le$ 0.1 selection budget ({nxt_txt}). "
        f"Generation effect (newer minus older release) with $\\gamma$-MPOA instead of MPOA: {gen_txt}. "
        "Most Wilson intervals in (a) are shorter than the markers (n = 972). Single seed; one benchmark item is "
        "worth 0.1--0.61 pp.")
    S.write_caption(NAME, J.tex(cap), source=J.source_text(
        "Refusal and gamma - MPOA tests: final_gmpoa_control.csv; benchmark deltas: final_capability.csv "
        "(cross-checked against final_gmpoa_benchmarks.csv, which holds the paired tests). " + S.SOURCE_CAPABILITY))


if __name__ == "__main__":
    main()
