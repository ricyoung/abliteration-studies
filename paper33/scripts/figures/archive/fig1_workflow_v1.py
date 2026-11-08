#!/usr/bin/env python3
"""fig1_workflow.py -- Figure 1: workflow of the whole study (Graphviz dot -> vector PDF).

Run with the figures venv (from scripts/figures, so `import style` resolves):
    scripts/figures/.venv/bin/python scripts/figures/fig1_workflow.py

Writes
    paper/figures/fig1_workflow.pdf             vector PDF, 6.5 in wide (template text width), TrueType fonts
    paper/figures/preview/fig1_workflow.png     150 dpi preview (pdftoppm), not for submission
    scripts/figures/fig1_workflow.dot           generated Graphviz source with fixed positions (neato -n2)
    paper/figures/captions.md                   "## fig1_workflow" section (replaced)

Everything that can change while runs are still going is read from paper/figures/data/*.csv (number of
studies, methods per study, GPU per stage, benchmark sizes, evaluation progress, status of the judges and of
the additional conditions), so re-running after extract_data.py refreshes the diagram. Two constants that are
not in the CSVs are read from their source files (read-only): the selection threshold from
scripts/select_trial.py and the held-out set composition from scripts/bench/refusal_sets.json (only the `kind`
field is read); the values in the paper are the fallback.

Encoding
  solid boxes / arrows       core pipeline, numbered stages 1-6
  dashed boxes / arrows      additional conditions (gamma-corrected MPOA, seed 2, small-pair reference variants,
                             community models)
  grey strip                 status of a stage or condition whose artefacts do not exist (yet); data-driven,
                             it disappears once the CSVs contain the results
  grey italic text           GPU that runs the stage (subtle annotation, not a data encoding)
  coloured squares           the fixed method colours used in every other figure
Fonts: 8 pt titles, 7.5 pt body, 7 pt notes at the final 6.5 in width (the script prints the effective sizes
after Graphviz scales the drawing to 6.5 in and warns if anything ends up below 7 pt). Include the PDF at
width=\textwidth (6.5 in) so it prints at scale 1.0.
"""
from __future__ import annotations

import datetime as dt
import json
import re
import shutil
import subprocess
import sys
from html import escape

import style as S

NAME = "fig1_workflow"
DOT_PATH = S.HERE / f"{NAME}.dot"
PDF_PATH = S.FIG_DIR / f"{NAME}.pdf"
PNG_PATH = S.PREVIEW_DIR / f"{NAME}.png"
PREVIEW_DPI = 150
TARGET_W = S.TEXT_W          # in, final printed width = template text width (Graphviz `size` includes the pad)
PAD = 0.03                   # in

FONT = "Liberation Sans"     # same family as the matplotlib figures (style.apply)
FS_HEAD = 8.0                # pt, stage titles
FS_BODY = 7.5                # pt, body text
FS_NOTE = 7.0                # pt, GPU notes, status strips, column heads (house minimum)
INK, INK2 = S.TEXT, S.TEXT_SECONDARY
BORDER = "#9d9c96"           # box outline: visible, quieter than text
CLUSTER_BORDER = S.AXIS
EDGE = "#6f6e69"
DASHED = S.PENDING_EDGE
LEQ, ARROW, DOWN, TIMES, SQUARE = "≤", "→", "↓", "×", "■"

# ======================================================================================== data (snapshot)
runs = S.load_runs()
res = S.load_results()
trials = S.load_trials()


def _gpu_short(name) -> str:
    s = str(name)
    for a in ("NVIDIA ", "GeForce ", " Blackwell Server Edition", " Blackwell"):
        s = s.replace(a, "")
    return s.strip()


GPU = {m: _gpu_short(g) for m, g in res.dropna(subset=["machine", "gpu"]).groupby("machine")["gpu"].first().items()}
for _m, _g in (("mc-h200-1", "H200 NVL"), ("mc-rtx-1", "RTX PRO 6000"), ("local-4090", "RTX 4090")):
    GPU.setdefault(_m, _g)

CORE = ["dim", "mpoa", "ara", "som", "jspace"]
seed1 = runs[(runs.seed == 1) & runs.method.isin(CORE)]
n_studies = int(seed1.run.notna().sum())
by_study = {st: seed1[seed1.study == st] for st in ("main", "small")}
n_models = {st: df.model.nunique() for st, df in by_study.items()}
methods = {st: [m for m in CORE if m in set(df.method)] for st, df in by_study.items()}
methods_small_only = [m for m in methods["small"] if m not in methods["main"]]
_planned = sorted(set(seed1.n_trials_planned.dropna().astype(int))) or [100]
n_tr = _planned[0] if len(_planned) == 1 else "/".join(map(str, _planned))
n_trials_complete = int((trials.seed == 1).sum())
n_selected = int(seed1.selected_trial.notna().sum())
opt_machine = {st: sorted(set(df.opt_machine.dropna())) for st, df in by_study.items()}

# evaluation box per family (results.csv `machine`; pending cells carry the box whose queue lists them)
ev = res[res.method.isin(["base"] + CORE)].dropna(subset=["machine"])
eval_by_machine: dict[str, list[str]] = {}
for fam in S.FAMILY_ORDER:
    for m in sorted(set(ev[ev.family == fam].machine)):
        eval_by_machine.setdefault(m, []).append(fam)

bench_n = {}
for b, fallback in zip(S.BENCHES, (1000, 500, 541, 164, 500)):
    vals = res[f"{b}_n"].dropna().astype(int) if f"{b}_n" in res else []
    bench_n[b] = int(vals.mode().iloc[0]) if len(vals) else fallback
_tf = res.tf_n_items.dropna().astype(int)
n_tf_items = int(_tf.mode().iloc[0]) if len(_tf) else 350

cells = res[res.method.isin(["base"] + CORE)]          # originals + core method cells
n_cells = len(cells)
n_cells_done = int(cells.status.astype(str).str.startswith("present").sum())

# held-out judge labels: a judged-refusal column with values means the judges have run
JUDGE_COL = re.compile(r"(judge|wildguard|heldout_refus|harmful_refus|refusal_rate)", re.I)
judges_done = any(res[c].notna().any() for c in res.columns if JUDGE_COL.search(c))

# additional conditions
gm = runs[runs.method == "gamma_mpoa"]
gm_started = int((~gm.status.astype(str).str.startswith("planned")).sum())
gm_done = int(res[res.method == "gamma_mpoa"].status.astype(str).str.startswith("present").sum())
gm_fams = [f for f in S.FAMILY_ORDER if f in set(gm.family)]

s2 = runs[runs.seed == 2]
SMALL_MODELS = {"qwen2.5-7b", "qwen3.5-4b"}
s2_small = s2[s2.model.isin(SMALL_MODELS)]
s2_running = s2_small[s2_small.status.astype(str).str.startswith("partial")]
s2_done = int(s2_small.status.astype(str).str.startswith("present").sum())
s2_queued = int(s2_small.status.astype(str).str.contains("queued").sum())
s2_methods = [m for m in CORE if m in set(s2_small.method)]
# Seed-2 design of the methods appendix / plan (paper/sections/08b_setup_details.tex "Planned conditions";
# notes/EXECUTION_PLAN_v2.md "Seeds"): DiM, MPOA, ARA on the Qwen and Gemma pairs, 12 runs. Status from runs.csv.
S2_MAIN_MODELS = ["qwen2.5-32b", "qwen3.8-27b", "gemma-2-27b", "gemma-4-31b"]
S2_MAIN_METHODS = ["dim", "mpoa", "ara"]
s2_main = s2[s2.model.isin(S2_MAIN_MODELS) & s2.method.isin(S2_MAIN_METHODS)]
s2_main_started = int(s2_main.status.astype(str).str.startswith(("present", "partial")).sum())
s2_main_n = len(S2_MAIN_MODELS) * len(S2_MAIN_METHODS)

# Small-pair reference variants (08b "Planned conditions"; plan "Small pair"): OBLITERATUS v0.1.4 presets basic,
# advanced, spectral_cascade, som; a paper-faithful SOM configuration; J-space as a forward hook (alpha = 1).
# Status: a run directory whose name carries one of these tokens counts as started (none existed at writing).
VAR_TOKENS = ("obliteratus", "faithful", "hook")
_var_dirs = [d.name for root in (S.ROOT / "runs" / "local-4090", S.ROOT / "runs" / "mc-h200-1" / "runs")
             if root.is_dir() for d in root.iterdir() if d.is_dir() and any(k in d.name.lower() for k in VAR_TOKENS)]
var_started = len(_var_dirs)

cm = res[res.method == "community"]
cm_done = int(cm.status.astype(str).str.startswith("present").sum())
cm_queued = int(cm.status.astype(str).str.contains("queued").sum())
cm_old = int(cm.generation.isin(["old", "small-old"]).sum())
cm_new = len(cm) - cm_old
cm_gpu = sorted({GPU.get(m, m) for m in cm.machine.dropna()})

# selection threshold (scripts/select_trial.py) and held-out set composition (refusal_sets.json)
KL_MAX, KL_SENS = 0.1, (0.05, 0.2, 0.5)
try:
    _src = (S.ROOT / "scripts" / "select_trial.py").read_text()
    KL_MAX = float(re.search(r"^KL_MAX\s*=\s*([0-9.]+)", _src, re.M).group(1))
    _thr = re.search(r"^REPORT_THRESHOLDS\s*=\s*\(([^)]*)\)", _src, re.M).group(1)
    KL_SENS = tuple(float(x) for x in _thr.split(",") if x.strip() and float(x) != KL_MAX)
except Exception as exc:  # noqa: BLE001
    print(f"  [fig1] could not read select_trial.py ({exc}); using KL <= {KL_MAX}")
assert (trials.loc[trials["is_selected_kl0.1"].astype(bool), "kl"] <= KL_MAX + 1e-12).all(), "selection > KL_MAX"

HELD = {"total": 1322, "harmful": 972, "benign": 350, "max_tokens": 512, "overlap": 44}
try:
    _rs = json.loads((S.ROOT / "scripts" / "bench" / "refusal_sets.json").read_text())
    _kinds = [it["kind"] for it in _rs["items"]]
    HELD = {"total": len(_kinds), "harmful": _kinds.count("harmful"), "benign": _kinds.count("benign"),
            "max_tokens": _rs["meta"].get("max_tokens", 512),
            # harmful items flagged as matching a mlabonne/harmful_behaviors (AdvBench) prompt, exact or fuzzy
            "overlap": sum(1 for it in _rs["items"] if it.get("advbench_overlap"))}
except Exception as exc:  # noqa: BLE001
    print(f"  [fig1] could not read refusal_sets.json ({exc}); using the paper's counts")
if HELD["benign"] != n_tf_items:
    print(f"  [fig1] WARNING: {HELD['benign']} benign held-out prompts but {n_tf_items} teacher-forced items")


# ======================================================================================== HTML helpers
# Every node label is a one-column table. Graphviz stretches a nested table to the width of its cell and
# spreads the extra width over the nested cells, so the numbered title row (chip + title) is a FIXEDSIZE
# nested table whose width is measured from the font (matplotlib TextPath, same TrueType file Graphviz uses).
NCOL = 1


def text_w(s, size, bold=False, italic=False) -> float:
    """Ink width of `s` in points for Liberation Sans at `size` pt."""
    from matplotlib.font_manager import FontProperties
    from matplotlib.textpath import TextPath
    fp = FontProperties(family=FONT, weight="bold" if bold else "normal", style="italic" if italic else "normal")
    return TextPath((0, 0), s, size=size, prop=fp).get_extents().width


def t(s) -> str:
    return escape(str(s), quote=False)


def font(text, size=FS_BODY, color=INK, bold=False, italic=False) -> str:
    if not str(text):
        return ""
    inner = text
    if bold:
        inner = f"<B>{inner}</B>"
    if italic:
        inner = f"<I>{inner}</I>"
    return f'<FONT POINT-SIZE="{size}" COLOR="{color}">{inner}</FONT>'


def row(content, bg=None, pad=None) -> str:
    b = f' BGCOLOR="{bg}"' if bg else ""
    p = f' CELLPADDING="{pad}"' if pad is not None else ""
    return f'<TR><TD COLSPAN="{NCOL}" ALIGN="LEFT" BALIGN="LEFT"{b}{p}>{content}</TD></TR>'


def hline(parts, sep=6, height=12) -> str:
    """One row made of several differently styled parts, left-aligned (FIXEDSIZE nested table, widths measured).
    parts: dicts with text, size, color, bold, italic, bg (strip fill) or chip (dark numbered square)."""
    cells, total = [], 0.0
    for i, q in enumerate(parts):
        if i:
            gap_w = q.get("gap", sep)
            cells.append(f'<TD WIDTH="{gap_w}"></TD>')
            total += gap_w
        if q.get("chip") is not None:
            cells.append(f'<TD BGCOLOR="{INK}" FIXEDSIZE="TRUE" WIDTH="11" HEIGHT="11" ALIGN="CENTER">'
                         f'{font(str(q["chip"]), FS_NOTE, "#ffffff", bold=True)}</TD>')
            total += 11
            continue
        size = q.get("size", FS_BODY)
        pad = 4 if q.get("bg") else 0
        # rendered (pango) width vs TextPath ink width, calibrated on the PDF: bold ~1.04x, regular/italic <= 1x
        f = 1.04 if q.get("bold") else 1.02
        w = f * text_w(q["text"], size, q.get("bold", False), q.get("italic", False)) + 3 + pad
        bg = f' BGCOLOR="{q["bg"]}"' if q.get("bg") else ""
        cells.append(f'<TD ALIGN="LEFT" WIDTH="{w:.0f}"{bg}>'
                     f'{font(t(q["text"]), size, q.get("color", INK), q.get("bold", False), q.get("italic", False))}'
                     "</TD>")
        total += w
    return (f'<TR><TD ALIGN="LEFT"><TABLE ALIGN="LEFT" FIXEDSIZE="TRUE" WIDTH="{total:.0f}" HEIGHT="{height}" '
            f'BORDER="0" CELLBORDER="0" CELLSPACING="0" CELLPADDING="0"><TR>{"".join(cells)}</TR></TABLE></TD></TR>')


def head(title, num=None, size=FS_HEAD, swatch=None, sub=None) -> str:
    """Title row: dark numbered chip (core stages) or a method-colour square (conditions), bold title, and an
    optional grey subtitle on the same line."""
    if num is not None:
        parts = [{"chip": num}, {"text": title, "size": size, "bold": True, "gap": 4}]
        if sub:
            parts.append({"text": sub, "size": FS_BODY, "color": INK2, "gap": 8})
        return hline(parts)
    pre = font(SQUARE, size, swatch) + " " if swatch else ""
    return row(pre + font(t(title), size, bold=True))


def lines(*items, size=FS_BODY, color=INK, italic=False) -> list[str]:
    return [row(font(x, size, color, italic=italic)) for x in items]


def note(*items) -> list[str]:
    """GPU annotation: grey italic."""
    return lines(*items, size=FS_NOTE, color=INK2, italic=True)


def status(text) -> str:
    """Explicit 'pending/planned' strip (grey fill + label)."""
    return row(font(text, FS_NOTE, INK2), bg=S.PENDING_FACE, pad=1.2)


def gap(h=2) -> str:
    return f'<TR><TD COLSPAN="{NCOL}" HEIGHT="{h}"></TD></TR>'


def table(rows, width=None, top=2.5) -> str:
    """Node label. `top`: blank space above the first row (labels are top-aligned in equal-height boxes).
    `width` (pt): minimum table width, used to spread the statistics band over the full figure width."""
    w = f' WIDTH="{width:.0f}"' if width else ""
    return (f'<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0" CELLPADDING="0.5"{w}>'
            + (gap(top) if top else "") + "".join(rows) + "</TABLE>>")


def mlabel(m) -> str:
    return S.METHOD_LABELS[m].replace("$\\gamma$", "γ")


def chips(ms) -> str:
    """Inline method colour squares + names (the colour is carried by the square; names stay in ink)."""
    return "  ".join(font(SQUARE, FS_BODY, S.METHOD_COLORS[m]) + " " + font(t(mlabel(m))) for m in ms)


def gpus(machines) -> str:
    return ", ".join(GPU.get(m, m) for m in machines)


# ======================================================================================== node labels
# 1  model pairs -------------------------------------------------------------------------------------------
def _ver(display, fam):
    for pre in ("Mistral-Small-", f"{fam}-", fam):
        if display.startswith(pre):
            return display[len(pre):]
    return display


grid = ['<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0" CELLPADDING="0.5">',
        "<TR>" + "".join(f'<TD ALIGN="{a}">{font(x, FS_NOTE, INK2)}</TD>' for a, x in
                         (("LEFT", "family"), ("RIGHT", "earlier"), ("CENTER", ""), ("LEFT", "later"))) + "</TR>"]
for fam, old, new in S.PAIRS:
    if "small" in fam:
        grid.append(f'<TR><TD COLSPAN="4" HEIGHT="2"></TD></TR><TR><TD COLSPAN="4" ALIGN="LEFT">'
                    f'{font("small pair", FS_NOTE, INK2, italic=True)}</TD></TR>')
    fam_short = "Mistral Small" if fam.startswith("Mistral") else fam.split(" (")[0]
    base = fam_short.split()[0]
    grid.append("<TR>"
                f'<TD ALIGN="LEFT">{font(t(fam_short))}</TD>'
                f'<TD ALIGN="RIGHT">{font(t(_ver(S.MODEL_DISPLAY[old], base)))}</TD>'
                f'<TD ALIGN="CENTER" CELLPADDING="1.5">{font(ARROW, FS_BODY, INK2)}</TD>'
                f'<TD ALIGN="LEFT">{font(t(_ver(S.MODEL_DISPLAY[new], base)))}</TD>'
                "</TR>")
grid.append("</TABLE>")
LBL_MODELS = table([head("Model pairs", 1), gap(2),
                    f'<TR><TD COLSPAN="{NCOL}" ALIGN="LEFT">{"".join(grid)}</TD></TR>', gap(3)]
                   + lines("instruction-tuned, bf16", size=FS_NOTE, color=INK2))

# 2  Heretic optimisation ---------------------------------------------------------------------------------
opt = [head("Heretic 2.0 optimisation", 2), gap(2),
       row(chips(methods["main"]))]
if methods_small_only:
    opt.append(row(chips(methods_small_only) + " " + font("(small pair only)", FS_NOTE, INK2)))
opt += [gap(3)] + lines(f"TPE, {n_tr} trials, seed 1;",
                        f"{n_studies} studies ({n_models['main']} {TIMES} {len(methods['main'])} + "
                        f"{n_models['small']} {TIMES} {len(methods['small'])})")
opt += [gap(3)] + lines(f"dev-set objectives, both {DOWN}:",
                        "keyword refusals (100 harmful)",
                        "first-token KL (100 harmless)")
opt += [gap(3)] + lines("directions, fits: train[:400]")
opt += [gap(3)] + note(f"{gpus(opt_machine['main'])}: main;", f"{gpus(opt_machine['small'])}: small pair")
LBL_OPT = table(opt)

# 3  selection --------------------------------------------------------------------------------------------
sel = [head("Trial selection", 3), gap(2)]
sel += lines("fewest dev keyword refusals", f"among trials with KL {LEQ} {KL_MAX:g}", "ties: lower KL, then index")
sel += [gap(3)] + lines("same rule for every study", "also at KL " + LEQ + " " + ", ".join(f"{x:g}" for x in KL_SENS),
                        font("(sensitivity)", FS_NOTE, INK2))
sel += [gap(3)] + lines(font(f"dev counts {ARROW} stage 6", FS_NOTE, INK2))
if n_selected < n_studies:
    sel += [gap(2), status(f"{n_studies - n_selected} of {n_studies} studies without a selection")]
LBL_SEL = table(sel)

# 4  export + release -------------------------------------------------------------------------------------
exp = [head("Export and release", 4), gap(2)]
exp += lines("LoRA merged into a", "standalone checkpoint;", "public Hugging Face repo")
exp += [gap(3)] + lines("reproduce manifest:", "DiM, MPOA, ARA", font("(not SOM, J-space)", FS_NOTE, INK2))
exp += [gap(3)] + note("re-exported on the", "evaluation GPU")
LBL_EXP = table(exp)

# 5  evaluation: four nodes in a cluster ------------------------------------------------------------------
LBL_HELD = table([head("Held-out refusal set"), gap(1)]
                 + lines(f"{HELD['total']:,} prompts: {HELD['harmful']} harmful,",
                         f"{HELD['benign']} benign; greedy responses,", f"{LEQ} {HELD['max_tokens']} new tokens",
                         font("JailbreakBench, XSTest, StrongREJECT,", FS_NOTE, INK2),
                         font("HarmBench, code-as-weapon", FS_NOTE, INK2))
                 + [gap(3)] + lines(font(f"{HELD['overlap']} items overlap harmful_behaviors:", FS_NOTE, INK2),
                                    font("flagged, results with and without", FS_NOTE, INK2)))

judge = [head("Judges"), gap(1)]
judge += lines("WildGuard (primary),", "HarmBench classifier,", "Qwen3Guard-Gen-8B,", "2 encoders, 33 keywords")
judge += [gap(3)] + lines(f"{ARROW} refusal, compliance,", "degenerate, incomplete")
if not judges_done:
    judge += [gap(2), status("pending: not run yet")]
LBL_JUDGE = table(judge)

LBL_TF = table([head("Teacher-forced"), head("divergence"), gap(1)]
               + lines("original's greedy answers y", f"to the {n_tf_items} benign prompts;",
                       "per token, log p(y | original)", f"{chr(0x2212)} log p(y | model)",
                       font(t("> 0: the model moved away"), FS_NOTE, INK2)))

bl = S.BENCH_LABELS
LBL_CAP = table([head("Capability"), gap(1)]
                + lines(f"{len(S.BENCHES)} benchmarks (items):",
                        f"{bl['mmlu_pro']} {bench_n['mmlu_pro']:,},",
                        f"{bl['gsm8k']} {bench_n['gsm8k']}, {bl['ifeval']} {bench_n['ifeval']},",
                        f"{bl['humaneval']} {bench_n['humaneval']}, {bl['mbpp']} {bench_n['mbpp']}")
                + [gap(3)] + lines("lm-eval on vLLM, greedy"))

fam_abbrev = {"Mistral Small": "Mistral", "Qwen (small pair)": "small pair"}
eval_gpu = []
for m in ("mc-rtx-1", "mc-h200-1", "local-4090"):
    fams = [fam_abbrev.get(f, f) for f in eval_by_machine.get(m, [])]
    if fams:
        eval_gpu.append(f"{GPU.get(m, m)}: {', '.join(fams)}")
ev_rows = [head("Evaluation of originals and selected models", 5,
                sub="held-out sets, not used for search or selection (overlap flagged)"), gap(1)]
ev_parts = [{"text": "; ".join(eval_gpu), "size": FS_NOTE, "color": INK2, "italic": True}]
if n_cells_done < n_cells:
    ev_parts.append({"text": f"capability + teacher forcing: {n_cells_done} of {n_cells} models done",
                     "size": FS_NOTE, "color": INK2, "bg": S.PENDING_FACE, "gap": 10})
ev_rows.append(hline(ev_parts, height=11))
LBL_EVAL = table(ev_rows)

# 6  statistics band ---------------------------------------------------------------------------------------
# columns follow paper/sections/04_methods.tex "Data analysis" (primary tests, H3, dev-vs-held-out Spearman,
# and the secondary outcomes in the order the methods list them)
STAT_COLS = [
    ("Refusal rates", ["Wilson intervals;", "McNemar vs original"]),
    ("H1, H2", ["logistic mixed model,", f"generation {TIMES} method class"]),
    ("H3", ["keyword vs judge labels;", "agreement κ"]),
    ("Dev vs held-out", ["Spearman: dev counts", "(stage 3) vs held-out", "judged refusal"]),
    ("Secondary", ["harmful compliance,", "over-refusal (benign),", "outcome classes,",
                   "capability Δ, TF divergence"]),
]
STATS_W = (TARGET_W - 2 * PAD - 0.14) * 72      # pt: band spans the full width (node margins 0.07 in)
COL_PAD = 6                                       # pt, left padding of columns 2..n (right of the rule)
_nrow = max(len(b) for _, b in STAT_COLS)          # pad every column to the same number of lines, otherwise
STAT_COLS = [(h, b + ["\u00a0"] * (_nrow - len(b))) for h, b in STAT_COLS]   # Graphviz spreads short ones
nat = [max(1.04 * text_w(h, FS_BODY, bold=True), *(1.02 * text_w(b, FS_BODY) if b.strip() else 0 for b in body)) + 3 + (COL_PAD if i else 0)
       for i, (h, body) in enumerate(STAT_COLS)]
extra = max(0.0, (STATS_W - 4 - sum(nat)) / len(nat))
band = []
for i, (h, body) in enumerate(STAT_COLS):
    inner = ('<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0" CELLPADDING="0.5">'
             + f'<TR><TD ALIGN="LEFT">{font(h, FS_BODY, bold=True)}</TD></TR>'
             + "".join(f'<TR><TD ALIGN="LEFT">{font(b)}</TD></TR>' for b in body) + "</TABLE>")
    rule = f' BORDER="1" SIDES="L" COLOR="{S.GRID}"' if i else ""
    lead = f'<TD WIDTH="{COL_PAD}"></TD>' if i else ""
    band.append(f'<TD ALIGN="LEFT" VALIGN="TOP" CELLPADDING="0" WIDTH="{nat[i] + extra:.0f}"{rule}>'
                f'<TABLE ALIGN="LEFT" BORDER="0" CELLBORDER="0" CELLSPACING="0" CELLPADDING="0"><TR>'
                f'{lead}<TD ALIGN="LEFT">{inner}</TD></TR></TABLE></TD>')
LBL_STATS = table(width=STATS_W, rows=[head("Statistics and tables", 6), gap(2),
                   f'<TR><TD COLSPAN="{NCOL}" ALIGN="LEFT"><TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0" '
                   f'CELLPADDING="0"><TR>{"".join(band)}</TR></TABLE></TD></TR>', gap(3),
                   row(font("Holm correction within test families; every reported number is labelled as dev set "
                            "or held-out", FS_NOTE, INK2))])


# additional conditions (dashed) ---------------------------------------------------------------------------
def cond(title, swatch, body, state):
    """body: list of strings or (text, state) groups; state: None, a string, or a list of strip lines."""
    rows_ = [head(title, size=FS_BODY, swatch=swatch), gap(1)] + lines(*body)
    for st_ in ([state] if isinstance(state, str) else (state or [])):
        rows_ += [gap(1.5), status(st_)]
    return table(rows_)


gm_state = None
if gm_done < len(gm):
    gm_state = (f"planned: 0/{len(gm)} started" if gm_started == 0
                else f"{gm_started}/{len(gm)} started, {gm_done} evaluated")
LBL_GM = cond("γ-corrected MPOA", S.METHOD_COLORS["gamma_mpoa"],
              ["post-norm models:", f"{', '.join(gm_fams)} pairs", f"({len(gm)} runs)"], gm_state)

# seed 2: two designs, each with its own status strip
run_txt = ", ".join(f"{int(r.n_trials_complete)}/{int(r.n_trials_planned)}" for _, r in s2_running.iterrows())
s2_small_state = (", ".join(x for x in ((f"{s2_done} done" if s2_done else ""),
                                        (f"{len(s2_running)} running" + (f" ({run_txt})" if run_txt else "")
                                         if len(s2_running) else ""),
                                        (f"{s2_queued} queued" if s2_queued else "")) if x)
                  if s2_done < len(s2_small) else "done")
s2_main_state = (f"planned: {s2_main_started}/{s2_main_n} started" if s2_main_started < s2_main_n else "done")
def strips(text, width=22):
    """Status text split at ', ' into strip lines of at most ~width characters."""
    out, cur = [], ""
    for part in text.split(", "):
        if cur and len(cur) + 2 + len(part) > width:
            out.append(cur + ",")
            cur = part
        else:
            cur = f"{cur}, {part}" if cur else part
    return out + [cur]


LBL_S2 = table([head("Seed 2", size=FS_BODY), gap(1)]
               + lines(f"small pair, {len(s2_methods)} methods ({len(s2_small)} runs)")
               + sum(([gap(1.5), status(x)] for x in strips(s2_small_state, 28)), []) + [gap(3)]
               + lines("Qwen + Gemma pairs,", f"DiM, MPOA, ARA ({s2_main_n} runs)")
               + [gap(1.5), status(s2_main_state)])

var_state = "planned: not started" if var_started == 0 else f"{var_started} run dirs started"
LBL_VAR = cond("Small-pair variants", None,
               ["OBLITERATUS v0.1.4", "presets (4), paper-", "faithful SOM, J-space", "forward hook (α = 1)"],
               var_state)

cm_state = None
if cm_done < len(cm):
    cm_state = (f"queued: {cm_done}/{len(cm)} evaluated" if cm_queued else f"planned: {cm_done}/{len(cm)} evaluated")
LBL_CM = cond("Community models", S.METHOD_COLORS["community"],
              [f"{len(cm)} public releases:", f"{cm_new} of later, {cm_old} of", "earlier models;",
               "same evaluation"] + ([f"({', '.join(cm_gpu)})"] if len(cm_gpu) == 1 else []), cm_state)

# key (one line under the statistics band) ----------------------------------------------------------------
KEY = ('<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="6" CELLPADDING="1.5"><TR>'
       f'<TD ALIGN="LEFT" CELLPADDING="0">{font("Key:", FS_NOTE, INK2, bold=True)}</TD>'
       f'<TD ALIGN="LEFT" BORDER="1" STYLE="rounded" COLOR="{BORDER}">{font("core stage", FS_NOTE)}</TD>'
       f'<TD ALIGN="LEFT" BORDER="1" STYLE="dashed" COLOR="{DASHED}">'
       f'{font("additional condition", FS_NOTE)}</TD>'
       f'<TD ALIGN="LEFT" BGCOLOR="{S.PENDING_FACE}">{font("pending / planned (status)", FS_NOTE, INK2)}</TD>'
       f'<TD ALIGN="LEFT">{font("GPU used", FS_NOTE, INK2, italic=True)}</TD>'
       "</TR></TABLE>>")

# ======================================================================================== graph
# Layout: Graphviz `dot` only MEASURES the nodes (HTML labels). Positions are computed here -- rows top-aligned,
# boxes of a row equal height, rows justified to the final width -- and the drawing is rendered with
# `neato -n2` (fixed positions), so the PDF is 6.5 in wide at scale 1.0 and fonts keep their nominal size.
# Edges are straight lines between invisible anchor points on the boxes' borders, so every arrow is horizontal or
# vertical. The evaluation "cluster" is a background box drawn first (outputorder=nodesfirst).
NODE = dict(shape="box", style="rounded", color=BORDER, penwidth="0.8", margin="0.06,0.045", labelloc="t")
DASHED_NODE = dict(style="rounded,dashed", color=DASHED)
DASH = dict(style="dashed", color=DASHED)
LBL_ORIG = table([row(font("originals (unedited)", FS_NOTE, INK2, italic=True))], top=0)
NODES = {   # name -> (label, extra attributes); declaration order = drawing order
    "evbox": ("", {"color": CLUSTER_BORDER, "fixedsize": "true"}),
    "models": (LBL_MODELS, {}), "opt": (LBL_OPT, {}), "sel": (LBL_SEL, {}), "exp": (LBL_EXP, {}),
    "gm": (LBL_GM, DASHED_NODE), "s2": (LBL_S2, DASHED_NODE), "var": (LBL_VAR, DASHED_NODE),
    "cm": (LBL_CM, DASHED_NODE),
    "evlabel": (LBL_EVAL, {"shape": "plain"}), "origlbl": (LBL_ORIG, {"shape": "plain"}),
    "judge": (LBL_JUDGE, {}), "held": (LBL_HELD, {}), "tf": (LBL_TF, {}), "cap": (LBL_CAP, {}),
    "stats": (LBL_STATS, {}),
    "key": (KEY, {"shape": "plain"}),
}
ROW1 = ["models", "opt", "sel", "exp"]
ROW2 = ["var", "gm", "s2", "cm"]          # gm, s2 feed the optimisation (arrows up); var, cm the evaluation (down)
ROW3 = ["judge", "held", "tf", "cap"]     # held-out set in the middle: responses -> judges, original's answers -> TF
GAP_H = 0.18        # in, min gap between boxes joined by an arrow
GAP_V = 0.22        # in, vertical gap between rows joined by arrows
CL_PAD = 0.08       # in, padding inside the evaluation box
EPS = 0.06          # in, min clearance between unrelated boxes
ORIG_X = 0.11       # in, x of the 'originals' line from stage 1 straight down into stage 5


def attrs(d) -> str:
    return ", ".join(f"{k}={v}" if str(v).startswith("<") else f'{k}="{v}"' for k, v in d.items())


def node_stmt(name, **extra) -> str:
    label, kw = NODES[name]
    lab = label if label else '""'
    d = {**NODE, **kw, **extra}
    return f"  {name} [{attrs(d)}, label={lab}];"


def run(cmd, **kw):
    p = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if p.returncode:
        sys.exit(f"{' '.join(cmd)} failed:\n{p.stderr}")
    if p.stderr.strip():
        print(f"  [fig1] {cmd[0]}: {p.stderr.strip()}")
    return p


def measure() -> dict:
    """Natural (width, height) in inches of every node label, as Graphviz sizes it."""
    src = ["digraph measure {", f'  node [fontname="{FONT}", fontsize={FS_BODY}, fontcolor="{INK}"];']
    src += [node_stmt(n) for n in NODES if n != "evbox"] + ["}"]
    objs = json.loads(run(["dot", "-Tjson0"], input="\n".join(src)).stdout)["objects"]
    return {o["name"]: (float(o["width"]), float(o["height"])) for o in objs if "width" in o}


def layout(sz):
    """Box geometry in inches, origin top-left, y downwards: name -> (x0, y0, w, h)."""
    W_avail = TARGET_W - 2 * PAD
    need = {
        "row 1": sum(sz[n][0] for n in ROW1) + 3 * GAP_H,
        "row 3": sum(sz[n][0] for n in ROW3) + 2 * GAP_H + 2 * EPS + 2 * CL_PAD,
        "eval label": sz["evlabel"][0] + 2 * CL_PAD,
        "statistics": sz["stats"][0],
        "key": sz["key"][0],
    }
    W = max(W_avail, *need.values())
    if W > W_avail + 0.005:
        print(f"  [fig1] WARNING: content needs {W:.2f} in > {W_avail:.2f} in "
              f"({max(need, key=need.get)}); the PDF will be scaled down")
    g = {}
    # row 1: core pipeline, justified
    h1 = max(sz[n][1] for n in ROW1)
    gap1 = (W - sum(sz[n][0] for n in ROW1)) / (len(ROW1) - 1)
    x = 0.0
    for n in ROW1:
        g[n] = (x, 0.0, sz[n][0], h1)
        x += sz[n][0] + gap1
    # 'originals' label in the gap under stage 1, right of the originals line
    ow_, oh_ = sz["origlbl"]
    g["origlbl"] = (ORIG_X + 0.04, h1 + (GAP_V - oh_) / 2, ow_, oh_)
    # row 2 (additional conditions), between the originals line (left) and the export -> evaluation arrow (right):
    # small-pair variants at the left (dashed arrow down into stage 5), gamma-MPOA and seed 2 under the optimisation
    # box (dashed arrows up into stage 2), community models after them (dashed arrow down)
    y2 = h1 + GAP_V
    h2 = max(sz[n][1] for n in ROW2)
    ox, _, ow, _ = g["opt"]
    ocx = ox + ow / 2
    x_left = ORIG_X + 0.10
    x_right = g["exp"][0] + g["exp"][2] / 2 - 0.08
    w = {n: sz[n][0] for n in ROW2}
    gap2 = 2 * EPS
    var_x = x_left
    gm_x = max(var_x + w["var"] + gap2, ocx - EPS - w["gm"])
    s2_x = gm_x + w["gm"] + gap2
    cm_x = max(s2_x + w["s2"] + gap2, min(g["sel"][0] + g["sel"][2] / 2 - w["cm"] / 2, x_right - w["cm"]))
    if cm_x + w["cm"] > x_right + 1e-6:
        print(f"  [fig1] WARNING: community box ends at {cm_x + w['cm']:.2f} in, past the export arrow "
              f"({x_right:.2f} in)")
    for n, x0 in zip(ROW2, (var_x, gm_x, s2_x, cm_x)):
        g[n] = (x0, y2, sz[n][0], h2)
    # row 3: evaluation box spanning the width; label top-left; four equal-height boxes, justified
    yc = y2 + h2 + GAP_V
    lw, lh = sz["evlabel"]
    g["evlabel"] = (CL_PAD, yc + 0.05, lw, lh)
    y3 = yc + 0.05 + lh + 0.07
    h3 = max(sz[n][1] for n in ROW3)
    inner = W - 2 * CL_PAD
    gap3 = (inner - sum(sz[n][0] for n in ROW3)) / (len(ROW3) - 1)
    x = CL_PAD
    for n in ROW3:
        g[n] = (x, y3, sz[n][0], h3)
        x += sz[n][0] + gap3
    hc = y3 + h3 + CL_PAD - yc
    g["evbox"] = (0.0, yc, W, hc)
    # row 4: statistics band, full width; key underneath
    y4 = yc + hc + GAP_V
    g["stats"] = (0.0, y4, W, sz["stats"][1])
    yk = y4 + sz["stats"][1] + 0.05
    g["key"] = (0.0, yk, sz["key"][0], sz["key"][1])
    H = yk + sz["key"][1]
    return g, W, H


def build_dot(g, W, H):
    def pt(x, y):                       # inches, y down  ->  points, y up (neato -n2)
        return f"{x * 72:.2f},{(H - y) * 72:.2f}!"

    out = ["digraph fig1_workflow {",
           f'  graph [splines=line, outputorder=nodesfirst, overlap=true, pad="{PAD}", margin=0, '
           f'bgcolor="white", size="{TARGET_W:.3f},{2 * H:.3f}!", fontname="{FONT}"];',
           f'  node [fontname="{FONT}", fontsize={FS_BODY}, fontcolor="{INK}"];',
           f'  edge [color="{EDGE}", penwidth=0.9, arrowsize=0.55];']
    for n in NODES:
        x0, y0, w, h = g[n]
        extra = {"pos": pt(x0 + w / 2, y0 + h / 2)}
        if NODES[n][1].get("shape") != "plain":
            extra.update(width=f"{w:.4f}", height=f"{h:.4f}")
        out.append(node_stmt(n, **extra))

    anchors = []

    def anchor(name, x, y):
        anchors.append(f'  {name} [shape=point, width=0.01, height=0.01, style=invis, label="", pos="{pt(x, y)}"];')
        return name

    def cx(n):
        return g[n][0] + g[n][2] / 2

    def top(n):
        return g[n][1]

    def bot(n):
        return g[n][1] + g[n][3]

    def vlink(lower, upper, style=""):
        """Vertical arrow from the top of `lower` to the bottom of `upper`, at the middle of their x overlap."""
        lo_ = max(g[lower][0], g[upper][0]) + 0.10
        hi_ = min(g[lower][0] + g[lower][2], g[upper][0] + g[upper][2]) - 0.10
        if hi_ < lo_:
            print(f"  [fig1] WARNING: {lower} and {upper} do not overlap horizontally; slanted arrow")
            xa, xb = cx(lower), min(max(cx(lower), g[upper][0] + 0.1), g[upper][0] + g[upper][2] - 0.1)
        else:
            xa = xb = (lo_ + hi_) / 2
        a, b = anchor(f"t_{lower}", xa, top(lower)), anchor(f"h_{lower}", xb, bot(upper))
        return f"  {a} -> {b}{style};"

    dash = f" [{attrs(DASH)}]"
    edges = []
    for a, b in zip(ROW1, ROW1[1:]):                                   # 1 -> 2 -> 3 -> 4
        edges.append(f"  {a} -> {b};")
    for n in ("gm", "s2"):                                             # additional runs -> optimisation
        edges.append(vlink(n, "opt", dash))
    yc = g["evbox"][1]
    edges.append(f"  exp -> {anchor('a_exp', cx('exp'), yc)};")       # 4 -> evaluation (selected models)
    for n in ("var", "cm"):                                            # extra models -> evaluation
        edges.append(f"  {anchor(f't_{n}', cx(n), bot(n))} -> {anchor(f'a_{n}', cx(n), yc)}{dash};")
    # 1 -> evaluation: the unedited originals are evaluated too (and give the teacher-forcing reference)
    edges.append(f"  {anchor('t_orig', ORIG_X, bot('models'))} -> {anchor('a_orig', ORIG_X, yc)};")
    edges.append("  held -> judge;")                                  # every model's responses -> judges
    edges.append("  held -> tf;")                                     # original's benign answers -> teacher forcing
    y4 = g["stats"][1]
    for n in ("judge", "tf", "cap"):                                   # outcomes -> statistics
        edges.append(f"  {n} -> {anchor(f'a_{n}', cx(n), y4)};")
    return "\n".join(out + anchors + edges + ["}"]) + "\n"


def main():
    for tool in ("dot", "neato", "pdftoppm", "pdfinfo", "pdffonts"):
        if not shutil.which(tool):
            sys.exit(f"missing tool: {tool}")
    S.PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

    sz = measure()
    g, W, H = layout(sz)
    DOT_PATH.write_text(build_dot(g, W, H))
    run(["neato", "-n2", "-Tpdf:cairo", "-o", str(PDF_PATH), str(DOT_PATH)])
    run(["pdftoppm", "-r", str(PREVIEW_DPI), "-png", "-singlefile", str(PDF_PATH), str(PNG_PATH.with_suffix(""))])

    m = re.search(r"Page size:\s+([\d.]+) x ([\d.]+) pts", run(["pdfinfo", str(PDF_PATH)]).stdout)
    pw, ph = float(m.group(1)) / 72, float(m.group(2)) / 72
    scale = pw / (W + 2 * PAD)
    eff = {"title": FS_HEAD * scale, "body": FS_BODY * scale, "note": FS_NOTE * scale}
    fonts = run(["pdffonts", str(PDF_PATH)]).stdout.splitlines()[2:]
    print(f"  [fig1] layout {W + 2 * PAD:.2f} x {H + 2 * PAD:.2f} in -> PDF {pw:.3f} x {ph:.3f} in "
          f"(scale {scale:.3f}); effective pt: " + ", ".join(f"{k} {v:.2f}" for k, v in eff.items()))
    print("  [fig1] fonts: " + "; ".join(" ".join(f.split()[:3]) for f in fonts))
    if min(eff.values()) < S.FONT_SMALL - 0.01:
        print(f"  [fig1] WARNING: smallest text {min(eff.values()):.2f} pt < {S.FONT_SMALL} pt at final size")
    if abs(pw - TARGET_W) > 0.02:
        print(f"  [fig1] WARNING: page width {pw:.3f} in != {TARGET_W} in")
    if any("Type 3" in f for f in fonts):
        print("  [fig1] WARNING: Type 3 fonts in the PDF")
    print(f"  wrote {PDF_PATH.relative_to(S.ROOT)}, {PNG_PATH.relative_to(S.ROOT)} ({PREVIEW_DPI} dpi), "
          f"{DOT_PATH.relative_to(S.ROOT)}")
    write_caption()


def write_caption():
    small_only = ", ".join(S.METHOD_LABELS[m] for m in methods_small_only)
    snap = S.snapshot_utc()
    caption = (
        "**Study workflow.** (1) Five earlier-to-later pairs of instruction-tuned models. (2) Each model is edited "
        f"with each refusal-removal method in Heretic 2.0, one {n_tr}-trial TPE study per model and method (seed 1; "
        f"{n_studies} studies) that minimises dev-set keyword refusals and first-token KL divergence. (3) One rule "
        f"selects a trial per study: the fewest dev keyword refusals among trials with KL $\\le$ {KL_MAX:g}. (4) The "
        "selected trial is exported as a standalone checkpoint. (5) Originals and selected models are evaluated on "
        "held-out sets that were not used for search or selection: judged refusal on "
        f"{HELD['total']:,} prompts, five capability benchmarks and the teacher-forced divergence from the original "
        f"model. {HELD['overlap']} harmful held-out items match mlabonne/harmful\\_behaviors prompts; they are "
        "flagged and results are reported with and without them. (6) Statistics test H1--H3 and compare dev-set "
        "with held-out refusal. Dashed boxes are additional conditions; grey strips give the status of unfinished "
        f"steps at the data snapshot ({snap})."
    )
    s2_small_txt = f"small pair {len(s2_small)} runs: {s2_small_state}"
    s2_main_txt = f"Qwen + Gemma DiM/MPOA/ARA {s2_main_n} runs: {s2_main_state}"
    detail = (
        "*For the text (moved out of the caption):* "
        f"methods {', '.join(S.METHOD_LABELS[m] for m in methods['main'])} on all models"
        + (f", {small_only} on the small pair only" if small_only else "")
        + f"; directions and SOM/ARA fits use train[:400] of mlabonne/harmful\\_behaviors and harmless\\_alpaca; "
        "the dev set is test[:100] of each. Optimisation ran on an H200 (main study) and an RTX 4090 (small pair); "
        "evaluation on an RTX PRO 6000 (Qwen, Gemma), an H200 NVL (OLMo, Mistral) and an RTX 4090 (small pair). "
        "Heretic's reproduce manifest is published for DiM, MPOA and ARA. Held-out refusal set: "
        f"{HELD['harmful']} harmful and {HELD['benign']} benign prompts (JailbreakBench, XSTest, StrongREJECT, "
        "HarmBench, code-as-weapon), greedy responses of at most "
        f"{HELD['max_tokens']} new tokens, labelled by WildGuard (primary), the HarmBench classifier, "
        "Qwen3Guard-Gen-8B, two encoder classifiers and 33 keyword markers. Teacher-forced divergence: token-averaged "
        "$\\log p_{\\mathrm{orig}}(y)-\\log p_{\\mathrm{mod}}(y)$ over the original model's greedy answers $y$ to "
        f"the {n_tf_items} benign prompts. Additional conditions: $\\gamma$-corrected MPOA on the four post-norm "
        f"Gemma and OLMo models; seed 2 ({s2_small_txt}; {s2_main_txt} -- the paper must state which seed-2 design "
        "it reports); small-pair reference variants (OBLITERATUS v0.1.4 presets basic, advanced, "
        "spectral\\_cascade and som, a paper-faithful SOM configuration, J-space as a forward hook with "
        f"$\\alpha = 1$); {len(cm)} public community releases ({cm_new} of later and {cm_old} of earlier models) "
        "under the same evaluation protocol."
    )
    gm_txt, cm_txt = gm_state or "complete", cm_state or "complete"
    source = (
        "Workflow diagram; no outcome values are plotted. The dev set (stages 2-3: mlabonne/harmful_behaviors "
        "test[:100] keyword refusals, mlabonne/harmless_alpaca test[:100] first-token KL) is used for search "
        "and selection; every stage-5 outcome comes from held-out sets (overlapping items flagged). Counts and status "
        f"from paper/figures/data (runs.csv, results.csv, trials.csv: {n_studies} seed-1 studies, "
        f"{n_trials_complete:,} completed trials); held-out set composition and overlap flags from "
        f"scripts/bench/refusal_sets.json. Snapshot status ({snap}): capability + teacher forcing finished for "
        f"{n_cells_done}/{n_cells} originals + selected models; judges {'run' if judges_done else 'not run yet'}; "
        f"gamma-MPOA {gm_txt}; seed 2 {s2_small_txt}, {s2_main_txt}; small-pair variants {var_state}; "
        f"community models {cm_txt}."
    )
    S.write_caption(NAME, caption + "\n\n" + detail, source=source)
    print(f"  [fig1] caption -> {S.CAPTIONS.relative_to(S.ROOT)} ({len(caption.split())} words)")


if __name__ == "__main__":
    main()
