#!/usr/bin/env python3
"""Paper 30 v3 LaTeX tables, generated from analysis/*.csv (assemble_p30.py). Writes paper/tables/*.tex.

Run:
    /data/heretic_2/33_2611_does_refusal_removal_get_harder/scripts/figures/.venv/bin/python \
        /data/heretic_2/paper30/v3_2611/scripts/make_tables.py

Revision 2026-11-06 (v3, after the added runs):
  * all 7B Heretic 1.x cells are judged: the "pending" rows are filled from the analysis outputs;
  * the re-exported 1.x general study at the 2.0 selection rule (cell qwen2.5-32b__heretic1x-general-kl0.1,
    trial #96) appears in tab_cells, tab_capability, tab_devkl (marker) and as panel (b) of tab_h1
    (families h1_kl_matched and kl_cap_1x; separate Holm families);
  * new tab_external (label tab:external): external tools (OBLITERATUS, jwest33/abliterator) with context rows
    (original, Heretic 1.x general, best Heretic 2.0) on Qwen2.5-7B and Qwen3.5-4B;
  * cells added after the pre-stated design are marked with a section sign and use the "added" Holm families.
Numbers are read from analysis/*.csv only, except the tool-reported refusal checks of the external cells, whose
prompt count (OBLITERATUS verify_sample_size = 30) is read from /data/heretic_2/models/external/<label>/PROVENANCE.json
because cells.csv records the size of the corpus the 30 prompts are sampled from (842).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

V3 = Path("/data/heretic_2/paper30/v3_2611")
A = V3 / "analysis"
TD = V3 / "paper" / "tables"
EXT = Path("/data/heretic_2/models/external")
TD.mkdir(parents=True, exist_ok=True)

MODEL_ORDER = ["qwen2.5-7b", "qwen2.5-coder-7b", "qwen2.5-14b", "qwen2.5-coder-14b", "qwen2.5-32b",
               "qwen2.5-coder-32b"]
MODEL_TEX = {"qwen2.5-7b": "Qwen2.5-7B", "qwen2.5-coder-7b": "Qwen2.5-Coder-7B", "qwen2.5-14b": "Qwen2.5-14B",
             "qwen2.5-coder-14b": "Qwen2.5-Coder-14B", "qwen2.5-32b": "Qwen2.5-32B",
             "qwen2.5-coder-32b": "Qwen2.5-Coder-32B", "qwen3.5-4b": "Qwen3.5-4B"}
METHOD_ROWS = ["original", "1x_general", "1x_general_kl01", "1x_code", "dim", "mpoa", "som", "jspace", "ara",
               "huihui", "zetasepic", "obliteratus_basic", "obliteratus_advanced", "obliteratus_spectral", "jwest33",
               "jwest33_hybridfix"]
# SOM and J-space are plug-ins of the project fork (paper33-plugins, commit 8fd4d76), not part of Heretic 2.0
# releases; they are labelled so in every table (revision 2026-11-06).
ADDED_MARK = r"$^{\S}$"
METHOD_TEX = {"original": "Original", "1x_general": "1.x general", "1x_code": "1.x code",
              "1x_general_kl01": r"1.x general, KL $\le$ 0.1" + ADDED_MARK, "dim": "2.0 DiM",
              "mpoa": "2.0 MPOA", "som": "SOM (fork plug-in)", "jspace": "J-space (fork plug-in)", "ara": "2.0 ARA",
              "huihui": "huihui-ai", "zetasepic": "zetasepic v2",
              "obliteratus_basic": "OBLITERATUS basic" + ADDED_MARK,
              "obliteratus_advanced": "OBLITERATUS advanced" + ADDED_MARK,
              "obliteratus_spectral": "OBLITERATUS spectral cascade" + ADDED_MARK,
              "jwest33": "jwest33 recommended" + ADDED_MARK,
              "jwest33_hybridfix": "jwest33 + hybrid fix" + ADDED_MARK}
METHOD_SHORT = {"dim": "2.0 DiM", "mpoa": "2.0 MPOA", "som": "SOM (fork)", "jspace": "J-space (fork)",
                "ara": "2.0 ARA"}
# Cells first reported in the companion study (paper 33): the Qwen2.5-7B and -32B originals, the nine Heretic 2.0 /
# fork-plug-in cells and the two community releases; the Qwen3.5-4B original and 2.0 cells are also companion-study
# cells (seed 1; context only). All other cells are new in this study.
COMPANION = {"qwen2.5-7b__base", "qwen2.5-7b__dim", "qwen2.5-7b__mpoa", "qwen2.5-7b__som", "qwen2.5-7b__jspace",
             "qwen2.5-7b__ara", "qwen2.5-32b__base", "qwen2.5-32b__dim", "qwen2.5-32b__mpoa", "qwen2.5-32b__som",
             "qwen2.5-32b__ara", "qwen2.5-32b__community-huihui", "qwen2.5-32b__community-zetasepic-v2",
             "qwen3.5-4b__base", "qwen3.5-4b__dim", "qwen3.5-4b__mpoa", "qwen3.5-4b__som", "qwen3.5-4b__jspace",
             "qwen3.5-4b__ara"}
SRC_COMPANION = r"\cite{young2026harder}"
SRC_NEW = "new"
BENCH = ["mmlu_pro", "gsm8k", "ifeval", "humaneval", "mbpp"]
PEND = r"\multicolumn{%d}{l}{\textit{pending: generation on the RTX 4090, judging on the RTX PRO 6000}}"


def mtex(c):
    """Condition label; jwest33 on Qwen3.5-4B is the as-released run (primary)."""
    if c["method"] == "jwest33" and c["model"] == "qwen3.5-4b":
        return "jwest33, as released" + ADDED_MARK
    return METHOD_TEX[c["method"]]


def pc(x, d=1):
    return "--" if pd.isna(x) else f"{100 * x:.{d}f}"


def sgn(x, d=1, scale=100):
    if pd.isna(x):
        return "--"
    v = scale * x
    s = f"{abs(v):.{d}f}"
    if float(s) == 0:
        return f"{s}"
    return ("+" if v > 0 else "$-$") + s


def ci(lo, hi, d=1, scale=100):
    return f"[{sgn(lo, d, scale)}, {sgn(hi, d, scale)}]"


def fp(p):
    if pd.isna(p):
        return "--"
    if p < 0.001:
        return "$<$0.001"
    return f"{p:.3f}"


def note(ncol, text):
    """A table note as a full-width row under \\bottomrule (kept short enough not to widen the table)."""
    return rf"\multicolumn{{{ncol}}}{{@{{}}l}}{{\footnotesize {text}}} \\"


def sorted_block(cells, mod, keep=None):
    sub = cells[cells["model"] == mod].copy()
    if keep is not None:
        sub = sub[sub.apply(keep, axis=1)]
    sub["o"] = sub["method"].map({k: i for i, k in enumerate(METHOD_ROWS)})
    return sub.sort_values("o")


# ================================================================================================ tab_cells
def tab_cells(cells, tests):
    """Per-cell held-out results, Qwen2.5 design cells plus the re-exported 1.x general cell (KL <= 0.1).
    External tools and Qwen3.5-4B are in tab_external."""
    vb = tests[(tests["family"] == "vs_base") & (tests["judge"] == "wg") & (tests["group"] == "harmful_all")]
    vba = tests[(tests["family"] == "vs_base_added") & (tests["judge"] == "wg") & (tests["group"] == "harmful_all")]
    vb = pd.concat([vb, vba]).set_index("a")
    L = [r"\begin{tabular}{@{}lllrrrrrrrr@{}}", r"\toprule",
         r"Model & Condition & Source & \multicolumn{2}{c}{WildGuard refusal, harmful (\%)} & Change vs.\ & "
         r"Keyword (\%) & HarmBench & Over-refusal & Capability & TF div. \\",
         r" & & & rate & 95\% CI & original (pp) & full / 256 tok. & harmful (\%) & benign (\%) & $\Delta$ (pp) & "
         r"(nats/tok.) \\",
         r"\midrule"]
    first = True
    for mod in MODEL_ORDER:
        sub = sorted_block(cells, mod, keep=lambda c: c["tool"] != "external")
        if not first:
            L.append(r"\addlinespace[2pt]")
        first = False
        for j, (_, c) in enumerate(sub.iterrows()):
            mt = MODEL_TEX[mod] if j == 0 else ""
            src = SRC_COMPANION if c["label"] in COMPANION else SRC_NEW
            if c["status"] != "judged":
                L.append(f"{mt} & {mtex(c)} & {src} & " + (PEND % 8) + r" \\")
                continue
            if c["method"] == "original":
                ch = "--"
            else:
                t = vb.loc[c["label"]]
                ch = sgn(t["diff"]) + ("$^{\\mathrm{ns}}$" if t["p_holm"] >= 0.05 else "")
            cap = "--" if c["method"] == "original" else sgn(c["capability_mean_delta_pp"], 2, 1)
            tf = "--" if c["method"] == "original" else f"{c['tf_divergence']:.3f}"
            L.append(f"{mt} & {mtex(c)} & {src} & {pc(c['wg_rate'])} & "
                     f"[{pc(c['wg_lo'])}, {pc(c['wg_hi'])}] & "
                     f"{ch} & {pc(c['kw_rate'])} / {pc(c['kw256_rate'])} & {pc(c['hb_harm_rate'])} & "
                     f"{pc(c['overrefusal_wg_rate'])} & {cap} & {tf} \\\\")
    L += [r"\bottomrule",
          note(11, r"$^{\S}$ Added after the pre-stated design: the original 1.x general study re-exported at the 2.0 "
                   r"selection rule (dev KL $\le$ 0.1; trial \#96); its test is in a separate Holm family."),
          note(11, r"$^{\mathrm{ns}}$ $p_\mathrm{Holm} \ge 0.05$; all other changes $p_\mathrm{Holm} < 0.001$. "
                   r"External tools and Qwen3.5-4B: Table~\ref{tab:external}."),
          r"\end{tabular}"]
    (TD / "tab_cells.tex").write_text("\n".join(L) + "\n")


# ================================================================================================ tab_h1
def tab_h1(cells, tests):
    """H1 table. Panel (a), pre-stated design: primary (like-for-like) contrast first, 1.x general vs 2.0 DiM (same
    edit), then the other 1.x-vs-2.0 / fork-plug-in contrasts (Holm over the whole h1_1x_vs_2.0 family as computed by
    assemble_p30.py; conservative for the primary contrast). Panel (b), added: the original 1.x general study
    re-exported at the 2.0 selection rule (KL <= 0.1, trial #96) vs each 2.0 method on Qwen2.5-32B (family
    h1_kl_matched, Holm over 4) and vs the original 1.x export at KL <= 0.01 (family kl_cap_1x, single test)."""
    hd = tests[tests["group"] == "harmful_all"]
    t = hd[hd["family"] == "h1_1x_vs_2.0"]
    nfam = int(t[t["judge"] == "wg"]["n_in_family"].iloc[0])
    L = [r"\begin{tabular}{@{}llllrrrrrr@{}}", r"\toprule",
         r"Model & Contrast & 1.x variant & Comparator & 1.x (\%) & Comparator (\%) & 1.x $-$ comparator (pp) [95\% CI] & "
         r"1.x-only / comp.-only & $p_\mathrm{Holm}$ & keyword $p_\mathrm{Holm}$ \\", r"\midrule",
         rf"\multicolumn{{10}}{{@{{}}l}}{{\textit{{(a) Pre-stated design: released trials (1.x selected at dev KL "
         rf"$\le$ 0.01, 2.0 and fork at dev KL $\le$ 0.1); Holm over the {nfam} contrasts}}}} \\",
         r"\addlinespace[1pt]"]
    meth_order = {"dim": 0, "mpoa": 1, "ara": 2, "som": 3, "jspace": 4}
    first = True
    for mod in ["qwen2.5-32b", "qwen2.5-7b"]:
        tw = t[(t["model"] == mod) & (t["judge"] == "wg")].copy()
        tk = t[(t["model"] == mod) & (t["judge"] == "kw")].set_index(["a", "b"])
        if not first:
            L.append(r"\addlinespace[2pt]")
        first = False
        if tw.empty:
            L.append(f"{MODEL_TEX[mod]} & \\multicolumn{{9}}{{l}}{{\\textit{{pending: Heretic 1.x cells not yet judged}}}} \\\\")
            continue
        tw["meth"] = tw["b"].str.split("__").str[1]
        tw["var"] = np.where(tw["a"].str.contains("general"), "general", "code")
        tw["primary"] = (tw["var"] == "general") & (tw["meth"] == "dim")
        tw["o"] = (~tw["primary"]).astype(int) * 100 + (tw["var"] == "code").astype(int) * 10 + tw["meth"].map(meth_order)
        for j, (_, r) in enumerate(tw.sort_values("o").iterrows()):
            kp = tk.loc[(r["a"], r["b"]), "p_holm"] if (r["a"], r["b"]) in tk.index else np.nan
            kind = "primary" if r["primary"] else "additional"
            L.append(f"{MODEL_TEX[mod] if j == 0 else ''} & {kind} & {r['var']} & {METHOD_SHORT[r['meth']]} & "
                     f"{pc(r['rate_a'])} & {pc(r['rate_b'])} & {sgn(r['diff'])} {ci(r['lo'], r['hi'])} & "
                     f"{int(r['a_only'])} / {int(r['b_only'])} & {fp(r['p_holm'])} & {fp(kp)} \\\\")
            if r["primary"]:
                L.append(r"\addlinespace[1pt]")
    # ---- panel (b): added, KL-matched (src: tests_refusal.csv families h1_kl_matched, kl_cap_1x; RESULTS.md
    #      sections "1.x general re-exported at KL <= 0.1 (trial #96) vs Heretic 2.0" and "... vs the original 1.x export")
    km = hd[hd["family"] == "h1_kl_matched"]
    kc = hd[hd["family"] == "kl_cap_1x"]
    if not km.empty:
        nkm = int(km[km["judge"] == "wg"]["n_in_family"].iloc[0])
        g84 = cells.set_index("label").loc[kc["b"].iloc[0]]
        g96 = cells.set_index("label").loc[kc["a"].iloc[0]]
        L += [r"\midrule",
              rf"\multicolumn{{10}}{{@{{}}l}}{{\textit{{(b) Added: the original 1.x general study re-exported at the "
              rf"2.0 selection rule (dev KL $\le$ 0.1; trial \#96); separate Holm families}}}} \\",
              r"\addlinespace[1pt]"]
        kw_ = km[km["judge"] == "wg"].copy()
        kk = km[km["judge"] == "kw"].set_index(["a", "b"])
        kw_["meth"] = kw_["b"].str.split("__").str[1]
        kw_["o"] = kw_["meth"].map(meth_order)
        for j, (_, r) in enumerate(kw_.sort_values("o").iterrows()):
            kp = kk.loc[(r["a"], r["b"]), "p_holm"]
            L.append(f"{MODEL_TEX[r['model']] if j == 0 else ''} & KL-matched & general, \\#96 & "
                     f"{METHOD_SHORT[r['meth']]} & {pc(r['rate_a'])} & {pc(r['rate_b'])} & "
                     f"{sgn(r['diff'])} {ci(r['lo'], r['hi'])} & {int(r['a_only'])} / {int(r['b_only'])} & "
                     f"{fp(r['p_holm'])} & {fp(kp)} \\\\")
        r = kc[kc["judge"] == "wg"].iloc[0]
        kp = kc[kc["judge"] == "kw"].iloc[0]["p_holm"]
        L.append(r"\addlinespace[1pt]")
        L.append(f" & cap only & general, \\#96 & general, \\#84 & {pc(r['rate_a'])} & "
                 f"{pc(r['rate_b'])} & {sgn(r['diff'])} {ci(r['lo'], r['hi'])} & "
                 f"{int(r['a_only'])} / {int(r['b_only'])} & {fp(r['p_holm'])} & {fp(kp)} \\\\")
        L += [r"\bottomrule",
              note(10, rf"Panel (b): KL-matched rows, Holm over {nkm} contrasts (family \texttt{{h1\_kl\_matched}}); "
                       r"cap-only row, a single test (family \texttt{kl\_cap\_1x})."),
              note(10, rf"\#{int(g84['dev_trial'])}: the released 1.x general model of panel (a) (dev KL "
                       rf"{g84['dev_kl']:.3f}); \#{int(g96['dev_trial'])}: fewest dev refusals at dev KL $\le$ 0.1 "
                       rf"(KL {g96['dev_kl']:.3f}); same study, journal and heretic-llm 1.4.0."),
              # checker 2026-11-06: "KL-matched" = same selection rule, not equal realised KL; single runs
              note(10, r"KL-matched: both sides selected by the same rule (fewest dev refusals at dev KL $\le$ 0.1); "
                       r"realised dev KL still differs (Table~\ref{tab:devkl}). One optimisation run per condition."),
              r"\end{tabular}"]
    else:
        L += [r"\bottomrule", r"\end{tabular}"]
    (TD / "tab_h1.tex").write_text("\n".join(L) + "\n")


# ================================================================================================ tab_h2
def tab_h2(cells, tests):
    t = tests[(tests["family"] == "h2_general_vs_code") & (tests["judge"] == "wg")].set_index(["model", "group"])
    groups = [("harmful_noncode", "Non-code harmful (772)"), ("code_weapon", "Code-as-weapon (200)"),
              ("code_weapon_excl_risk", "Code-as-weapon, excl.\\ RMCBench/MalwareBench (159)")]
    L = [r"\begin{tabular}{@{}llrrrr@{}}", r"\toprule",
         r"Model & Prompt subset ($n$) & General (\%) & Code (\%) & General $-$ code (pp) [95\% CI]; "
         r"$<0$: general lower & $p_\mathrm{Holm}$ \\",
         r"\midrule"]
    first = True
    for mod in MODEL_ORDER:
        if not first:
            L.append(r"\addlinespace[2pt]")
        first = False
        if (mod, "harmful_noncode") not in t.index:
            L.append(f"{MODEL_TEX[mod]} & \\multicolumn{{5}}{{l}}{{\\textit{{pending (1.x 7B evaluation)}}}} \\\\")
            continue
        for j, (g, gl) in enumerate(groups):
            r = t.loc[(mod, g)]
            L.append(f"{MODEL_TEX[mod] if j == 0 else ''} & {gl} & {pc(r['rate_a'])} & {pc(r['rate_b'])} & "
                     f"{sgn(r['diff'])} {ci(r['lo'], r['hi'])} & {fp(r['p_holm'])} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    (TD / "tab_h2.tex").write_text("\n".join(L) + "\n")


# ================================================================================================ tab_devkl
def tab_devkl(dm, cells):
    rows = dm[dm["model"].isin(MODEL_ORDER)]
    rows = rows[(rows["tool"] == "heretic2") | (rows["method"] == "1x_general")].copy()
    rows["mo"] = rows["model"].map({k: i for i, k in enumerate(MODEL_ORDER)})
    order = {"1x_general": 0, "dim": 1, "mpoa": 2, "som": 3, "jspace": 4, "ara": 5}
    rows["me"] = rows["method"].map(order)
    rows = rows.sort_values(["mo", "me"])
    # the re-exported trial (src: cells.csv qwen2.5-32b__heretic1x-general-kl0.1 dev_trial/dev_refusals/dev_kl;
    # it is the fewest-refusal Pareto point at KL <= 0.1 of the 32B 1.x general search, dev_points.csv trial 96)
    rx = cells[cells["method"] == "1x_general_kl01"]
    rx = rx.iloc[0] if len(rx) else None
    L = [r"\begin{tabular}{@{}llrrrrrr@{}}", r"\toprule",
         r"Model & Search & Selected: refusals (KL) & \multicolumn{5}{c}{Fewest dev refusals at first-token KL $\le$} \\",
         r"\cmidrule(l){4-8}", r" & & & 0.01 & 0.02 & 0.05 & 0.1 & 0.5 \\", r"\midrule"]
    prev = None
    for _, r in rows.iterrows():
        mt = MODEL_TEX[r["model"]] if r["model"] != prev else ""
        if prev is not None and r["model"] != prev:
            L.append(r"\addlinespace[2pt]")
        prev = r["model"]
        name = {"1x_general": "1.x general", "dim": "2.0 DiM", "mpoa": "2.0 MPOA", "som": "SOM (fork plug-in)",
                "jspace": "J-space (fork plug-in)", "ara": "2.0 ARA"}[r["method"]]
        cl = []
        for c in ["0.01", "0.02", "0.05", "0.1", "0.5"]:
            v = r.get(f"min_ref_kl{c}")
            s = "--" if pd.isna(v) else f"{int(v)}"
            if (rx is not None and c == "0.1" and r["model"] == rx["model"] and r["method"] == "1x_general"
                    and not pd.isna(v) and int(v) == int(rx["dev_refusals"])
                    and abs(r["kl_at_min_kl0.1"] - rx["dev_kl"]) < 1e-6):
                s += r"$^{a}$"
            cl.append(s)
        L.append(f"{mt} & {name} & {int(r['selected_refusals'])} ({r['selected_kl']:.3f}) & " + " & ".join(cl) + r" \\")
    L += [r"\bottomrule"]
    if rx is not None:
        L.append(note(8, rf"$^{{a}}$ Trial \#{int(rx['dev_trial'])} (KL {rx['dev_kl']:.3f}), re-exported and "
                         rf"evaluated held-out (Table~\ref{{tab:h1}}b);"))
        L.append(note(8, rf"\phantom{{$^{{a}}$}} dev re-evaluation after export: {int(rx['dev_refusals_reeval'])} of "
                         rf"100 at KL {rx['dev_kl_reeval']:.3f}."))
    L.append(r"\end{tabular}")
    (TD / "tab_devkl.tex").write_text("\n".join(L) + "\n")


# ================================================================================================ tab_capability
def tab_capability(cells, ct):
    fam_n = ct[ct["bench"] != "mean5"].groupby("family").size()
    L = [r"\begin{tabular}{@{}llrrrrrr@{}}", r"\toprule",
         r"Model & Condition & MMLU-Pro & GSM8K & IFEval & HumanEval & MBPP & Mean of five [95\% CI] \\", r"\midrule"]
    first = True
    for mod in MODEL_ORDER + ["qwen3.5-4b"]:
        sub = sorted_block(cells, mod)
        if not first:
            L.append(r"\addlinespace[2pt]")
        first = False
        for j, (_, c) in enumerate(sub.iterrows()):
            mt = (MODEL_TEX[mod] + (ADDED_MARK if mod == "qwen3.5-4b" else "")) if j == 0 else ""
            if not isinstance(c["cap_dir"], str) or not c["cap_dir"]:
                L.append(f"{mt} & {mtex(c)} & \\multicolumn{{6}}{{l}}{{\\textit{{pending}}}} \\\\")
                continue
            if c["method"] == "original":
                L.append(f"{mt} & Original (score, \\%) & " + " & ".join(f"{c[b + '_pct']:.1f}" for b in BENCH)
                         + f" & {c['capability_mean_pct']:.1f} \\\\")
                continue
            g = ct[ct["label"] == c["label"]].set_index("bench")
            vals = []
            for b in BENCH:
                d = g.loc[b, "delta_pp"]
                star = "$^{\\dagger}$" if str(g.loc[b, "test"]).startswith("unpaired") else ""
                v = sgn(d, 1, 1)
                if g.loc[b, "p_holm"] < 0.05:
                    v = r"\textbf{" + v + "}"
                vals.append(f"{v}{star}")
            mm = g.loc["mean5"]
            L.append(f"{mt} & {mtex(c)} & " + " & ".join(vals)
                     + f" & {sgn(mm['delta_pp'], 2, 1)} {ci(mm['lo_pp'], mm['hi_pp'], 2, 1)} \\\\")
    L += [r"\bottomrule",
          note(8, rf"Bold: per-benchmark change significant after Holm correction (pre-stated design: "
                  rf"{fam_n.get('original_design', 0)} tests; cells added$^{{\S}}$ after it: separate family of "
                  rf"{fam_n.get('added', 0)} tests)."),
          note(8, r"$^{\dagger}$ Unpaired test on summary scores (per-item scores of the original not reproducible)."),
          r"\end{tabular}"]
    (TD / "tab_capability.tex").write_text("\n".join(L) + "\n")


# ================================================================================================ tab_external
def tool_check(c):
    """Tool's own post-edit refusal check: (rate %, n prompts) or None.
    OBLITERATUS: quality-check refusal rate (cells.csv ext_self_refusal_rate) on verify_sample_size prompts sampled
    evenly from its built-in harmful corpus (PROVENANCE.json resolved_config.verify_sample_size);
    Heretic: dev keyword refusals of the released trial on its 100 dev prompts (cells.csv dev_refusals);
    jwest33: reports no post-edit refusal rate."""
    if c["tool"] == "external":
        if pd.isna(c["ext_self_refusal_rate"]):
            return None
        prov = json.loads((EXT / c["label"] / "PROVENANCE.json").read_text())
        return 100 * c["ext_self_refusal_rate"], int(prov["resolved_config"]["verify_sample_size"])
    if c["method"] != "original" and not pd.isna(c["dev_refusals"]):
        return float(c["dev_refusals"]), 100
    return None


def tab_external(cells, ct, resp):
    """External tools with context rows. Context: original; Heretic 1.x general (if the model has one); best Heretic 2.0
    = lowest WildGuard residual refusal among the Heretic 2.0 methods (DiM, MPOA, ARA; fork plug-ins excluded).
    Degenerate: count of the harmful responses flagged degenerate (repetition / character run) by the harness
    (responses_p30.csv.gz is_degenerate; equals cells.csv degenerate_rate x n_harmful)."""
    deg = resp[resp["kind"] == "harmful"].groupby("label")["is_degenerate"].apply(
        lambda s: int(s.astype(str).str.lower().isin(["true", "1", "1.0"]).sum()))
    # consistency check against cells.csv degenerate_rate
    for _, c in cells.iterrows():
        assert abs(deg.get(c["label"], 0) - c["degenerate_rate"] * c["n_harmful"]) < 0.5, c["label"]
    NC = 10
    L = [r"\begin{tabular}{@{}lrrrrrrrrr@{}}", r"\toprule",
         r"Condition & \multicolumn{2}{c}{WildGuard refusal,} & Keyword & HarmBench & "
         r"Over-refusal & Capability $\Delta$ (pp) & TF div. & Degenerate & Tool's own \\",
         r" & \multicolumn{2}{c}{harmful (\%) [95\% CI]} & (\%) & harmful (\%) & benign (\%) & [95\% CI] & "
         r"(nats/tok.) & (of 972)$^{e}$ & check (\%)$^{f}$ \\",
         r"\midrule"]

    def row(name, c):
        cap = "--"
        if c["method"] != "original":
            mm = ct[(ct["label"] == c["label"]) & (ct["bench"] == "mean5")].iloc[0]
            cap = f"{sgn(mm['delta_pp'], 2, 1)} {ci(mm['lo_pp'], mm['hi_pp'], 2, 1)}"
        tf = "--" if c["method"] == "original" else f"{c['tf_divergence']:.3f}"
        chk = tool_check(c)
        if chk is None:
            chk_s = "n.r." if c["tool"] == "external" else "--"
        else:
            chk_s = f"{chk[0]:.1f} ({chk[1]})"
        return (f"\\quad {name} & {pc(c['wg_rate'])} & [{pc(c['wg_lo'])}, {pc(c['wg_hi'])}] & {pc(c['kw_rate'])} & "
                f"{pc(c['hb_harm_rate'])} & {pc(c['overrefusal_wg_rate'])} & {cap} & {tf} & "
                f"{deg.get(c['label'], 0)} & {chk_s} \\\\")

    for k, mod in enumerate(["qwen2.5-7b", "qwen3.5-4b"]):
        if k:
            L.append(r"\addlinespace[3pt]")
        sub = cells[(cells["model"] == mod) & (cells["status"] == "judged")]
        base = sub[sub["method"] == "original"].iloc[0]
        h2 = sub[(sub["tool"] == "heretic2") & sub["method"].isin(["dim", "mpoa", "ara"])]
        best = h2.sort_values("wg_rate").iloc[0]
        g1 = sub[sub["method"] == "1x_general"]
        ext = sub[sub["tool"] == "external"].copy()
        ext["o"] = ext["method"].map({k_: i for i, k_ in enumerate(METHOD_ROWS)})
        L.append(rf"\multicolumn{{{NC}}}{{@{{}}l}}{{\textbf{{{MODEL_TEX[mod]}}}}} \\")
        L.append(row(r"\textit{Original}", base))
        if len(g1):
            L.append(row(r"\textit{Heretic 1.x general}", g1.iloc[0]))
        L.append(row(rf"\textit{{Heretic {METHOD_SHORT[best['method']]}}}$^{{a}}$", best))
        if mod == "qwen3.5-4b":
            L.append(rf"\quad OBLITERATUS, any preset & \multicolumn{{{NC - 1}}}{{l}}{{not supported: v0.1.4 does not "
                     r"support the Qwen3.5 hybrid-attention architecture; no model produced} \\")  # RESULTS.md wording
        for _, c in ext.sort_values("o").iterrows():
            if c["method"] == "jwest33" and mod == "qwen2.5-7b":
                name = r"jwest33 recommended$^{b}$"
            elif c["method"] == "jwest33":
                name = r"jwest33, as released$^{c}$"
            elif c["method"] == "jwest33_hybridfix":
                name = r"jwest33 + hybrid fix$^{d}$"
            else:
                name = METHOD_TEX[c["method"]].replace(ADDED_MARK, "")
            L.append(row(name, c))
    L += [r"\bottomrule",
          note(NC, r"Italic: context rows (original, Heretic). External tools: run once by the author with their "
                   r"documented default or recommended settings (OBLITERATUS v0.1.4,"),
          note(NC, r"\phantom{Italic: }commit 461ff086; jwest33/abliterator, commit 6ca3356e; seed 1); same "
                   r"evaluation harness as all other cells."),
          note(NC, r"$^{a}$ Best Heretic 2.0: lowest WildGuard refusal among DiM, MPOA and ARA (fork plug-ins excluded); "
                   r"companion study \cite{young2026harder}. No 1.x cell for Qwen3.5-4B."),
          note(NC, r"$^{b}$ Run with the patch of $^{d}$; Qwen2.5-7B is not hybrid (\texttt{is\_hybrid} = false), so the "
                   r"patch is a no-op and the model is effectively as released."),
          note(NC, r"$^{c}$ Primary. As released: the CLI path omits the hybrid-architecture detection, so the default "
                   r"\texttt{-{}-hybrid-strategy auto} has no effect."),
          note(NC, r"$^{d}$ Sensitivity analysis: runtime patch that sets the hybrid-architecture information before "
                   r"direction extraction, as the library path does."),
          note(NC, r"$^{e}$ Harmful responses flagged degenerate by the harness (repetition or character run)."),
          note(NC, r"$^{f}$ Refusal rate (\%) the tool reports for its own check (number of prompts). OBLITERATUS: "
                   r"post-edit quality check on 30 of its 842"),
          note(NC, r"\phantom{$^{f}$} built-in harmful prompts. Heretic: dev keyword refusals of the released trial "
                   r"(100 prompts). n.r.: not reported by the tool."),
          r"\end{tabular}"]
    (TD / "tab_external.tex").write_text("\n".join(L) + "\n")


def main():
    cells = pd.read_csv(A / "cells.csv")
    tests = pd.read_csv(A / "tests_refusal.csv")
    ct = pd.read_csv(A / "tests_capability.csv")
    dm = pd.read_csv(A / "dev_matched_kl.csv")
    resp = pd.read_csv(A / "responses_p30.csv.gz", low_memory=False)
    tab_cells(cells, tests)
    tab_h1(cells, tests)
    tab_h2(cells, tests)
    tab_devkl(dm, cells)
    tab_capability(cells, ct)
    tab_external(cells, ct, resp)
    print("tables written:", sorted(p.name for p in TD.glob("*.tex") if ".bak" not in p.name))


if __name__ == "__main__":
    main()
