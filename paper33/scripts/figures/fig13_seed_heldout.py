"""fig13_seed_heldout -- seed 1 vs seed 2: held-out judged refusal on the small pair (Qwen2.5-7B -> Qwen3.5-4B).

Each of the 10 small-pair model x method cells was optimised twice (seeds 1 and 2; same configuration, Heretic commit
and RTX 4090) and the selected model of each run was evaluated on the held-out set.
(a) One row per cell: WildGuard refusal on the 972 harmful held-out prompts for the seed-1 model (full method colour)
    and the seed-2 model (lighter), Wilson 95% intervals, joined by a line (dashed for the older model, solid for the
    newer, as fig8). Columns on the right: seed 2 minus seed 1 (pp, paired; * = Holm-significant over the 10 cells)
    and the dev keyword refusals (of 100) of the two selected trials, the quantity fig8 compares.
(b) Generation effect (newer minus older model, pp, paired) per method with seed 1 and with seed 2 (same marker
    encoding), Newcombe 95% intervals; small grey ticks = the two cross-seed combinations (older s1 / newer s2,
    older s2 / newer s1). Bottom row: the H2 risk-difference contrast (ARA minus the mean of the direction-based
    methods) for each seed.

Inputs : results/analysis/final_seed_cells.csv, results/analysis/final_seed_generation.csv
Outputs: paper/figures/fig13_seed_heldout.pdf, preview PNG, captions.md section
Run    : cd scripts/figures && .venv/bin/python fig13_seed_heldout.py
"""
from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import judged as J
import style as S

NAME = "fig13_seed_heldout"
MODELS = [("qwen2.5-7b", "old"), ("qwen3.5-4b", "new")]
METHODS = J.METHODS_HELDOUT            # DiM, MPOA, SOM, J-space, ARA
ALPHA = 0.05
SUB = 0.17                             # vertical offset of the seed-2 sub-row in (b)
H2_KEY = "H2 RD contrast (ARA - mean direction)"


def seed_kw(method, gen, seed, **extra):
    kw = S.marker_kw(method, gen, **extra)
    if seed == 2:
        c = J.tint(S.color(method))
        kw.update(color=c, markeredgecolor=c)
        if kw["markerfacecolor"] != S.SURFACE:
            kw["markerfacecolor"] = c
    return kw


def load():
    cells = J.load("final_seed_cells.csv")
    gen = J.load("final_seed_generation.csv")
    if len(cells) != 10:
        raise ValueError(f"final_seed_cells.csv has {len(cells)} rows, expected 10")
    return cells, gen


def main():
    S.apply()
    cells, gen = load()
    fig = plt.figure(figsize=(S.DOUBLE_COL, 3.35))
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.03, wspace=0.06)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1.0])
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])

    # ---------------------------------------------------------------- (a) per-cell dumbbells
    yt, ytl, y = [], [], 0.0
    heads = []
    for model, g in MODELS:
        heads.append((y, f"{S.MODEL_DISPLAY[model]} ({'older' if g == 'old' else 'newer'})"))
        y -= 1.0
        for m in METHODS:
            r = J.one(cells[(cells["model"] == model) & (cells["method"] == m)], f"{model} {m}")
            v1, v2 = 100 * r["s1_wg_rate"], 100 * r["s2_wg_rate"]
            ax_a.plot([v1, v2], [y, y], color=J.tint(S.color(m), 0.7), lw=1.2,
                      ls=S.GEN_STYLE[g]["linestyle"], zorder=2)
            for seed, v, lo, hi in ((1, v1, r["s1_wg_lo"], r["s1_wg_hi"]), (2, v2, r["s2_wg_lo"], r["s2_wg_hi"])):
                kw = seed_kw(m, g, seed)
                ax_a.hlines(y, 100 * lo, 100 * hi, color=kw["color"], lw=1.0, zorder=2.5)
                ax_a.plot([v], [y], **kw, zorder=3 if seed == 1 else 2.8)
            sig = float(r["p_holm"]) < ALPHA
            ax_a.text(1.03, y, J.pp(r["diff_s2_minus_s1"]) + ("*" if sig else ""),
                      transform=ax_a.get_yaxis_transform(), ha="left", va="center", fontsize=S.FONT_SMALL,
                      color=S.TEXT if sig else S.TEXT_SECONDARY, clip_on=False)
            ax_a.text(1.23, y, f"{int(r['s1_dev_refusals'])} / {int(r['s2_dev_refusals'])}",
                      transform=ax_a.get_yaxis_transform(), ha="center", va="center", fontsize=S.FONT_SMALL,
                      color=S.TEXT_SECONDARY, clip_on=False)
            yt.append(y)
            ytl.append(S.label(m))
            y -= 1.0
        y -= 0.3
    for yh, lab in heads:
        ax_a.text(0.0, yh, lab, ha="left", va="center", fontsize=S.FONT, fontweight="bold")
    y_top = heads[0][0] + 0.9
    ax_a.text(1.03, y_top, "s2 − s1\n(pp)", transform=ax_a.get_yaxis_transform(), ha="left", va="center",
              fontsize=S.FONT_SMALL, color=S.TEXT_SECONDARY, linespacing=1.1, clip_on=False)
    ax_a.text(1.23, y_top, "dev\ns1 / s2", transform=ax_a.get_yaxis_transform(), ha="center", va="center",
              fontsize=S.FONT_SMALL, color=S.TEXT_SECONDARY, linespacing=1.1, clip_on=False)
    ax_a.set_yticks(yt)
    ax_a.set_yticklabels(ytl)
    ax_a.tick_params(axis="y", length=0)
    ax_a.set_ylim(y + 0.6, y_top + 0.5)
    ax_a.set_xlim(-2, 102)
    ax_a.spines["bottom"].set_bounds(0, 100)
    ax_a.set_xticks([0, 25, 50, 75, 100])
    ax_a.grid(axis="y", visible=False)
    ax_a.set_xlabel("Held-out refusal (%, WildGuard)")
    ax_a.set_title("a   Selected model of each seed", loc="left", fontweight="bold")

    # ---------------------------------------------------------------- (b) generation effects by seed
    rows = METHODS + ["h2"]
    yb = {m: -i for i, m in enumerate(rows)}
    yb["h2"] -= 0.35
    xs_all = []
    for m in rows:
        key = H2_KEY if m == "h2" else m
        sub = gen[gen["method"] == key]
        y0 = yb[m]
        mk = "ara" if m == "h2" else m
        for seed, dy in ((1, SUB), (2, -SUB)):
            r = J.one(sub[sub["seeds"] == f"seed{seed}"], f"{key} seed{seed}")
            d, lo, hi = 100 * r["delta"], 100 * r["delta_lo"], 100 * r["delta_hi"]
            xs_all += [lo, hi]
            if m == "h2":
                kw = dict(marker="o", ms=S.MS, ls="none", mec=S.TEXT, mfc=S.TEXT if seed == 1 else J.tint(S.TEXT),
                          color=S.TEXT if seed == 1 else J.tint(S.TEXT))
                if seed == 2:
                    kw["mec"] = J.tint(S.TEXT)
            else:
                kw = seed_kw(mk, "new", seed)
            ax_b.hlines(y0 + dy, lo, hi, color=kw["color"], lw=1.0, zorder=2)
            ax_b.plot([d], [y0 + dy], **kw, zorder=3)
        for combo in ("old s1 / new s2", "old s2 / new s1"):
            r = J.one(sub[sub["seeds"] == combo], f"{key} {combo}")
            ax_b.plot([100 * r["delta"]], [y0], marker="|", ms=7, mew=1.0, color=S.TEXT_SECONDARY, ls="none",
                      zorder=2.6)
            xs_all.append(100 * r["delta"])
    ax_b.axvline(0, color=S.TEXT_SECONDARY, lw=0.8, zorder=1)
    ax_b.axhline((yb["ara"] + yb["h2"]) / 2, color=S.AXIS, lw=0.8, zorder=0)
    ax_b.set_yticks([yb[m] for m in rows])
    ax_b.set_yticklabels([S.label(m) for m in METHODS] + ["H2 contrast:\nARA − direction"])
    ax_b.tick_params(axis="y", length=0)
    ax_b.grid(axis="y", visible=False)
    lo, hi = min(xs_all), max(xs_all)
    ax_b.set_xlim(5 * ((lo - 4) // 5), 5 * (-(-(hi + 4) // 5)))
    ax_b.set_ylim(yb["h2"] - 0.55, 0.55)
    ax_b.set_xlabel("Newer − older model (pp), paired")
    ax_b.set_title("b   Generation effect by seed", loc="left", fontweight="bold")
    ax_b.text(0.5, 0.35, " newer refuses more →", ha="left", va="bottom", fontsize=S.FONT_SMALL,
              color=S.TEXT_SECONDARY, transform=ax_b.get_xaxis_transform() if False else ax_b.transData)

    # ---------------------------------------------------------------- key
    n1, n2 = S.TEXT_SECONDARY, J.tint(S.TEXT_SECONDARY)
    h = [Line2D([], [], label="seed 1", marker="o", ms=S.MS, mfc=n1, mec=n1, ls="none"),
         Line2D([], [], label="seed 2 (lighter)", marker="o", ms=S.MS, mfc=n2, mec=n2, ls="none"),
         Line2D([], [], color=S.TEXT_SECONDARY, marker="o", ms=S.MS, mfc=S.SURFACE, mec=S.TEXT_SECONDARY,
                mew=S.HOLLOW_MEW, ls=(0, (4, 2)), lw=1.2, label="older model (hollow, dashed)"),
         Line2D([], [], color=S.TEXT_SECONDARY, lw=1.0, label="95% interval"),
         Line2D([], [], color=S.TEXT_SECONDARY, marker="|", ms=7, mew=1.0, ls="none",
                label="cross-seed combinations (b)")]
    fig.legend(handles=h, loc="outside lower center", ncols=5, frameon=False, handlelength=1.6, columnspacing=1.2)
    S.save(fig, NAME)
    write_caption(cells, gen)


def write_caption(cells, gen):
    d = (100 * cells["diff_s2_minus_s1"]).abs()
    n_sig = int((cells["p_holm"] < ALPHA).sum())
    sig_txt = f"All {n_sig} cells" if n_sig == len(cells) else f"{n_sig} of the {len(cells)} cells"
    kap = cells["kappa_items"]

    def ge(m, seeds):
        key = H2_KEY if m == "h2" else m
        r = J.one(gen[(gen["method"] == key) & (gen["seeds"] == seeds)], f"{key} {seeds}")
        return r

    def eff(m, seeds):
        r = ge(m, seeds)
        return f"{J.pp(r['delta'])} [{J.pp(r['delta_lo'])}, {J.pp(r['delta_hi'])}]"

    per = "; ".join(f"{S.label(m)} {J.pp(ge(m, 'seed1')['delta'])} vs {J.pp(ge(m, 'seed2')['delta'])}"
                    for m in METHODS)
    a2 = ge("ara", "seed2")
    big = cells.loc[d.idxmax()]
    cap = (
        "**Seed sensitivity of held-out, judged refusal on the small pair (Qwen2.5-7B $\\to$ Qwen3.5-4B).** Each of "
        "the ten model $\\times$ method cells was optimised twice with 100 TPE trials that differ only in the seed, "
        "and the model selected from each run (fewest dev keyword refusals at first-token KL $\\le$ 0.1) was "
        "evaluated on the held-out set. (a) Share of the 972 harmful held-out prompts whose response WildGuard "
        "labels a refusal for the seed-1 model (full method colour) and the seed-2 model (lighter), with Wilson 95% "
        "intervals; hollow markers and dashed lines mark the older model. Columns on the right: seed 2 minus seed 1 "
        "(percentage points, paired; * = significant, exact McNemar, Holm over the ten cells) and the dev keyword "
        f"refusals (of 100) of the two selected trials (Fig. 8). {sig_txt} differ between seeds after "
        f"Holm; the absolute difference has median {J.num(d.median())} pp and maximum {J.num(d.max())} pp "
        f"({S.MODEL_DISPLAY[big['model']]} {S.label(big['method'])}), and item-level agreement between the two "
        f"seeds' models is $\\kappa$ = {J.num(kap.min(), 2)}--{J.num(kap.max(), 2)}. "
        "(b) Generation effect (newer minus older model, paired, Newcombe 95% intervals; item-bootstrap interval for "
        "the H2 row) with seed 1 (upper, full colour) and seed 2 (lower, lighter); the grey ticks are the two cross-seed combinations (older seed 1 with "
        "newer seed 2, and the reverse). The bottom row is the H2 contrast on the risk-difference scale (ARA minus "
        "the mean of DiM, MPOA, SOM and J-space; negative = ARA loses less effectiveness, the direction H2 "
        "predicted). Seed 1 vs seed 2 (pp): "
        f"{per}. With seed 2 the ARA effect is {eff('ara', 'seed2')} pp (exact McNemar p = "
        f"{J.p_text(float(a2['p_mcnemar']))}), so H1 holds in the small pair with seed 1 but not with seed 2; the H2 "
        f"contrast is {eff('h2', 'seed1')} pp with seed 1 and {eff('h2', 'seed2')} pp with seed 2. The "
        "main-study pairs have one seed only.")
    S.write_caption(NAME, J.tex(cap), source=J.source_text(
        "Per-cell rates and seed differences: final_seed_cells.csv; generation effects and the H2 contrast by "
        "seed: final_seed_generation.csv (small pair, RTX 4090)."))


if __name__ == "__main__":
    main()
