"""fig14_community -- the models selected in this study vs public community abliterations of the same checkpoints:
held-out judged refusal next to capability change and distribution shift.

One block of rows per base model that has at least one community release (Qwen2.5-32B, Qwen3.8-27B, Gemma-4-31B,
OLMo-3.1-32B, Mistral-Small-3.2; family order as everywhere else): the unedited original, the four seed-1 models
selected here (DiM, MPOA, SOM, ARA; hollow markers for the older release) and each community release (purple plus;
row label = Hugging Face uploader, as fig5/fig6).
(a) WildGuard refusal on the 972 harmful held-out prompts, Wilson 95% intervals; the ring marks the best selected model (lowest
    held-out refusal of the four). Column right of the panel: community minus that model (pp, paired; * = Holm-
    significant over the four comparisons of that community release, final_community_vs_selected.csv).
(b) change of the unweighted mean of the five capability benchmarks from the model's own original (pp).
(c) teacher-forced divergence from the original on the 350 benign held-out prompts (nats per token).
A scatter version (refusal against capability and divergence per model) was tried first and dropped: at 6.5 in its
low-refusal points and uploader labels collided. gamma-MPOA (a control, fig12) is not drawn.

Inputs : results/analysis/final_community_vs_selected.csv, results/analysis/h1h2_cells.csv (originals),
         results/analysis/final_capability.csv (cross-check), paper/figures/data/results.csv (uploader = repo owner)
Outputs: paper/figures/fig14_community.pdf, preview PNG, captions.md section
Run    : cd scripts/figures && .venv/bin/python fig14_community.py
"""
from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator

import judged as J
import style as S

NAME = "fig14_community"
OURS = ["dim", "mpoa", "som", "ara"]
ALPHA = 0.05
RING_MS = 9.5
HEAD_GAP = 1.25         # y units for a block header row
BLOCK_GAP = 0.45        # extra space between blocks


def load():
    cvs = J.load("final_community_vs_selected.csv")
    cells = J.load("h1h2_cells.csv")
    cells = cells[(cells["judge"] == "wg") & (cells["subset"] == "all")]
    cap = J.load("final_capability.csv")
    res = S.load_results()
    models = [m for m in S.MODEL_ORDER if m in set(cvs["model"])]
    data = {}
    for model in models:
        d = cvs[cvs["model"] == model]
        gen = J.one(d[["generation"]].drop_duplicates(), f"{model} generation")["generation"]
        orig = J.one(cells[cells["label"] == f"{model}__base"], f"{model} original")
        ours = {}
        for m in OURS:
            rs = d[d["our_method"] == m]
            if rs[["ours_wg_rate", "ours_capability_delta_pp", "ours_tf_divergence"]].nunique().max() != 1:
                raise ValueError(f"{model} {m}: 'ours' values differ between community rows")
            r = rs.iloc[0]
            c = J.one(cap[cap["label"] == r["ours"]], r["ours"])
            if abs(float(c["capability_mean_delta_pp"]) - float(r["ours_capability_delta_pp"])) > 1e-6:
                raise ValueError(f"{r['ours']}: capability delta differs between final_* tables")
            ours[m] = dict(label=r["ours"], rate=r["ours_wg_rate"], lo=r["ours_wg_lo"], hi=r["ours_wg_hi"],
                           cap=r["ours_capability_delta_pp"], tf=r["ours_tf_divergence"],
                           best=str(r["ours_is_best_wg"]) == "True")
        if sum(o["best"] for o in ours.values()) != 1:
            raise ValueError(f"{model}: not exactly one best cell")
        comm = []
        for lab, g in d.groupby("community", sort=True):
            best = J.one(g[g["ours_is_best_wg"].astype(str) == "True"], f"{lab} best ours")
            rr = J.one(res[res["label"] == lab], f"{lab} results.csv")
            comm.append(dict(label=lab, who=J.uploader(rr["repo"], rr["variant"]), rate=best["community_wg_rate"],
                             lo=best["community_wg_lo"], hi=best["community_wg_hi"],
                             cap=best["community_capability_delta_pp"], tf=best["community_tf_divergence"],
                             diff=best["diff_community_minus_ours"], diff_lo=best["diff_lo"],
                             diff_hi=best["diff_hi"], p=float(best["p_holm_within_community"]),
                             best_method=best["our_method"], best_rate=best["ours_wg_rate"],
                             best_tf=best["ours_tf_divergence"], best_cap=best["ours_capability_delta_pp"]))
        comm.sort(key=lambda c: c["who"].lower())
        data[model] = dict(gen=gen, orig=orig, ours=ours, comm=comm)
    return data


def layout(data):
    """Row positions: list of (y, kind, model, key) with kind in head/orig/ours/comm."""
    rows, y = [], 0.0
    for model, d in data.items():
        rows.append((y, "head", model, None))
        y -= HEAD_GAP
        rows.append((y, "orig", model, None))
        y -= 1
        for m in OURS:
            rows.append((y, "ours", model, m))
            y -= 1
        for i, _ in enumerate(d["comm"]):
            rows.append((y, "comm", model, i))
            y -= 1
        y -= BLOCK_GAP
    return rows, y


def main():
    S.apply()
    data = load()
    rows, y_end = layout(data)
    fig = plt.figure(figsize=(S.DOUBLE_COL, 6.7))
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.03, wspace=0.03)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.55, 0.95, 0.95])
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1], sharey=ax_a)
    ax_c = fig.add_subplot(gs[0, 2], sharey=ax_a)
    yt, ytl, ycol = [], [], []
    for y, kind, model, key in rows:
        d = data[model]
        gen = d["gen"]
        if kind == "head":
            ax_a.text(0.0, y, f"{S.MODEL_DISPLAY[model]} ({'older' if gen == 'old' else 'newer'} release)",
                      ha="left", va="center", fontsize=S.FONT, fontweight="bold")
            continue
        if kind == "orig":
            o = d["orig"]
            ax_a.hlines(y, 100 * o["lo"], 100 * o["hi"], color=S.color("base"), lw=1.0, zorder=2)
            ax_a.plot([100 * o["rate"]], [y], **S.marker_kw("base", gen), zorder=3)
            for ax in (ax_b, ax_c):
                ax.plot([0], [y], **S.marker_kw("base", gen), zorder=3)
            yt.append(y), ytl.append("Original"), ycol.append(S.TEXT_SECONDARY)
            continue
        if kind == "ours":
            r, mk = d["ours"][key], key
            lab, col = S.label(key), S.TEXT_SECONDARY
        else:
            r, mk = d["comm"][key], "community"
            lab, col = r["who"], S.TEXT
        c = S.color(mk)
        ax_a.hlines(y, 100 * r["lo"], 100 * r["hi"], color=c, lw=1.0, zorder=2)
        ax_a.plot([100 * r["rate"]], [y], **S.marker_kw(mk, gen), zorder=3)
        if kind == "ours" and r["best"]:
            ax_a.plot([100 * r["rate"]], [y], marker="o", ms=RING_MS, mfc="none", mec=S.TEXT, mew=0.7, ls="none",
                      zorder=2.9)
        ax_b.plot([r["cap"]], [y], **S.marker_kw(mk, gen), zorder=3)
        ax_c.plot([r["tf"]], [y], **S.marker_kw(mk, gen), zorder=3)
        if kind == "comm":
            sig = r["p"] < ALPHA
            ax_a.text(1.025, y, J.pp(r["diff"]) + ("*" if sig else ""), transform=ax_a.get_yaxis_transform(),
                      ha="left", va="center", fontsize=S.FONT_SMALL, color=S.TEXT if sig else S.TEXT_SECONDARY,
                      clip_on=False)
        yt.append(y), ytl.append(lab), ycol.append(col)
    # axes cosmetics
    y_top = rows[0][0] + 0.8
    ax_a.set_ylim(y_end + BLOCK_GAP - 0.6, y_top)
    ax_a.set_yticks(yt)
    ax_a.set_yticklabels(ytl)
    for t, col in zip(ax_a.get_yticklabels(), ycol):
        t.set_color(col)
    ax_a.tick_params(axis="y", length=0)
    for ax in (ax_b, ax_c):
        plt.setp(ax.get_yticklabels(), visible=False)
        ax.tick_params(axis="y", length=0)
    for ax in (ax_a, ax_b, ax_c):
        ax.grid(axis="y", visible=False)
        for y, kind, *_ in rows[1:]:
            if kind == "head":
                ax.axhline(y + HEAD_GAP / 2 + BLOCK_GAP / 2 + 0.25, color=S.GRID, lw=S.LW_THIN, zorder=0)
    ax_a.set_xlim(-2, 102)
    ax_a.spines["bottom"].set_bounds(0, 100)
    ax_a.set_xticks([0, 25, 50, 75, 100])
    ax_a.set_xlabel("Held-out refusal (%, WildGuard)")
    ax_a.text(1.025, y_top - 0.2, "community −\nbest selected\n(pp; * p < 0.05,\nHolm-adjusted)", transform=ax_a.get_yaxis_transform(),
              ha="left", va="top", fontsize=S.FONT_SMALL, color=S.TEXT_SECONDARY, linespacing=1.05,
              clip_on=False)
    caps = [r["cap"] for d in data.values() for r in list(d["ours"].values()) + d["comm"]]
    tfs = [r["tf"] for d in data.values() for r in list(d["ours"].values()) + d["comm"]]
    ax_b.set_xlim(min(caps) - 0.35, max(caps) + 0.3)          # upper end below the next unit tick
    ax_b.axvline(0, color=S.AXIS, lw=0.8, zorder=0)
    ax_b.xaxis.set_major_locator(MultipleLocator(1.0))
    ax_b.set_xlabel("Capability change\n(pp, mean of 5)")
    ax_c.set_xlim(-0.012, max(tfs) * 1.08)
    ax_c.axvline(0, color=S.AXIS, lw=0.8, zorder=0)
    ax_c.set_xlabel("Teacher-forced\ndivergence (nats/token)")
    ax_a.set_title("a   Held-out refusal", loc="left", fontweight="bold")
    ax_b.set_title("b   Capability", loc="left", fontweight="bold")
    ax_c.set_title("c   Divergence", loc="left", fontweight="bold")
    h = S.method_handles(["base", "dim", "mpoa", "som", "ara", "community"])
    h[-1].set_label("community release (row label = uploader)")
    h += [Line2D([], [], marker="o", ms=RING_MS * 0.85, mfc="none", mec=S.TEXT, mew=0.7, ls="none",
                 label="best selected model (lowest refusal)"),
          Line2D([], [], color=S.TEXT_SECONDARY, marker="o", ms=S.MS, mfc=S.SURFACE, mec=S.TEXT_SECONDARY,
                 mew=S.HOLLOW_MEW, ls="none", label="older release (hollow)")]
    fig.legend(handles=h, loc="outside lower center", ncols=4, frameon=False, handlelength=1.2, columnspacing=1.2)
    S.save(fig, NAME)
    write_caption(data)


def write_caption(data):
    comm = [(m, c) for m, d in data.items() for c in d["comm"]]
    lower = [(m, c) for m, c in comm if c["p"] < ALPHA and c["diff"] < 0]
    higher = [(m, c) for m, c in comm if c["p"] < ALPHA and c["diff"] > 0]
    same = [(m, c) for m, c in comm if c["p"] >= ALPHA]

    def one(m, c):
        return (f"{S.MODEL_DISPLAY[m]} {c['who']} {J.pct(c['rate'])}% vs {S.label(c['best_method'])} "
                f"{J.pct(c['best_rate'])}%")

    tf_lo = [c["tf"] for _, c in lower]
    tf_best = [c["best_tf"] for _, c in lower]
    cap_lo = [c["cap"] for _, c in lower]
    caps = [c["cap"] for _, c in comm]
    n_ours = sum(len(d["ours"]) for d in data.values())
    low_m = [m for m in data if any(mm == m for mm, _ in lower)]
    n_low_models = len(low_m)
    low_models = ", ".join(S.MODEL_DISPLAY[m] for m in low_m)
    ours_caps = [r["cap"] for d in data.values() for r in d["ours"].values()]
    cap = (
        "**Models selected in this study versus public community abliterations of the same checkpoints (held-out, "
        "judged).** One block per base model with at least one community release: the unedited original (grey "
        "star), the four seed-1 models selected here by the dev-set rule (fewest dev keyword refusals at first-token "
        "KL $\\le$ 0.1; hollow markers = older release) and each community release (purple plus; row label = "
        "Hugging Face uploader). (a) Share of the 972 harmful held-out prompts whose response WildGuard labels a "
        "refusal, with Wilson 95% intervals (mostly shorter than the markers); the ring marks the best selected model "
        "(lowest held-out refusal of the four), and the column on the right gives community minus that model "
        "(percentage points, paired; * = significant, exact McNemar, Holm over the four comparisons per community "
        "release). (b) Change of the unweighted mean of the five capability benchmarks from the model's own original "
        "(percentage points). (c) Teacher-forced divergence from the original on the 350 benign held-out prompts "
        f"(nats per token). Of the {len(comm)} community releases, {len(lower)} refuse significantly less than the "
        f"best selected model ({'; '.join(one(m, c) for m, c in lower)}), {len(same)} do not differ "
        f"({'; '.join(one(m, c) for m, c in same)}) and {len(higher)} refuse more "
        f"({'; '.join(one(m, c) for m, c in higher)}). The {len(lower)} lower ones lie at a larger teacher-forced "
        f"divergence than the best selected model ({J.num(min(tf_lo), 3)}--{J.num(max(tf_lo), 3)} vs "
        f"{J.num(min(tf_best), 3)}--{J.num(max(tf_best), 3)}) with a mean capability change of "
        f"{J.signed(min(cap_lo), 2)} to {J.signed(max(cap_lo), 2)} pp (all {len(comm)} community releases: "
        f"{J.signed(min(caps), 2)} to {J.signed(max(caps), 2)} pp; the {n_ours} models selected here for these "
        f"checkpoints: "
        f"{J.signed(min(ours_caps), 2)} to {J.signed(max(ours_caps), 2)} pp). On {n_low_models} of the "
        f"{len(data)} checkpoints ({low_models}) a public release reaches lower held-out refusal than "
        "every model selected here under the KL $\\le$ 0.1 cap, at a larger divergence. $\\gamma$-MPOA (Fig. 12) "
        "is not drawn.")
    S.write_caption(NAME, J.tex(cap), source=J.source_text(
        "Community vs selected models: final_community_vs_selected.csv (paired McNemar per community x selected cell, "
        "Holm within community release); originals: h1h2_cells.csv; capability deltas and teacher-forced divergence "
        "as in Figs. 5 and 6 (final_capability.csv). " + S.SOURCE_CAPABILITY))


if __name__ == "__main__":
    main()
