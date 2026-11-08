"""style.py -- shared figure style for paper 33 ("Does Refusal Removal Get Harder?").

Import from every fig*.py (run with scripts/figures/.venv/bin/python):

    import style as S
    S.apply()                                   # rcParams: fonts, sizes, recessive grid, fonttype 42
    fig, ax = S.figure("single", height=2.4)    # 3.4 in wide (single column) / "double" = 6.5 in (text width)
    ax.plot(x, y, **S.line_kw("mpoa", "new"))   # method colour + generation line/marker style
    S.save(fig, "fig3_pareto_fronts")           # paper/figures/fig3_pareto_fronts.pdf + preview/*.png
    S.write_caption("fig3_pareto_fronts", "Caption text ...", source=S.SOURCE_DEV)

Encoding rules (house style; colours are fixed per method and never re-assigned):
  * method          -> colour (METHOD_COLORS) + marker shape (METHOD_MARKERS, the redundant channel)
  * old vs new gen  -> marker fill + line style ONLY: old = hollow marker, dashed line;
                       new = filled marker, solid line. Never a second colour scheme.
  * magnitude       -> sequential blue ramp (SEQ_BLUE, light -> dark)
  * capability delta-> diverging: orange arm (loss, < 0) | grey midpoint at 0 | blue arm (gain, > 0)
  * one y-axis per chart; legend for >= 2 series plus selective direct labels
  * pending / missing cells are drawn explicitly (draw_pending / pending_marker), never dropped

Palette checks (dataviz validator, light mode, surface #fcfcfb):
  the five core method colours pass the adjacent-pair CVD and normal-vision gates, but ARA, SOM and J-space are
  below 3:1 contrast on white and SOM vs MPOA fails the all-pairs normal-vision floor (13.7 < 15). Hence: in
  scatter-type charts (any two methods can sit side by side) ALWAYS use METHOD_MARKERS as well as colour, and
  give direct labels or a table. Both diverging arms pass the ordinal-ramp checks (monotone L, step gap, >= 2:1).
  gamma-MPOA is a dark crimson (#9b1c31, 8.1:1 on white), not the earlier red #e34948, which sat next to MPOA
  orange (CIEDE2000 13.3 normal / 6.8 deutan / 5.5 tritan). The MPOA vs gamma-MPOA pair (fig9, fig3v, fig12) is
  now 29.8 normal, 27.7 protan, 27.7 deutan, 23.1 tritan (Machado 2009 simulation, severity 1).
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, TwoSlopeNorm  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

# ------------------------------------------------------------------------------------------------ paths
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
FIG_DIR = ROOT / "paper" / "figures"
PREVIEW_DIR = FIG_DIR / "preview"          # PNG previews, not for submission
DATA_DIR = FIG_DIR / "data"
# judged held-out results (scripts/analysis/run_all.sh); read by the held-out figures fig10-fig14 via judged.py
ANALYSIS_DIR = ROOT / "results" / "analysis"
CAPTIONS = FIG_DIR / "captions.md"
NAME_RE = re.compile(r"^fig(\d+|A\d+)[a-z]?_[a-z0-9_]+$")   # house rule: figN_descriptive_name; appendix: figAk_name

# ------------------------------------------------------------------------------------------------ ink
TEXT = "#0b0b0b"             # primary ink
TEXT_SECONDARY = "#52514e"   # secondary ink (tick labels, annotations)
GRID = "#e6e5e1"             # recessive grid
AXIS = "#c3c2b7"             # baseline / axis / tick marks
SURFACE = "#ffffff"          # print background
NEUTRAL_MID = "#f0efec"      # diverging midpoint / pending fill

# ------------------------------------------------------------------------------------------------ methods
METHOD_ORDER = ["base", "dim", "mpoa", "ara", "som", "jspace", "gamma_mpoa", "community"]
METHOD_COLORS = {
    "dim": "#2a78d6",
    "mpoa": "#eb6834",
    "ara": "#1baf7a",
    "som": "#eda100",
    "jspace": "#e87ba4",
    "community": "#6250d6",
    "gamma_mpoa": "#9b1c31",
    "base": "#8a8984",       # original model
}
METHOD_LABELS = {"base": "Original", "dim": "DiM", "mpoa": "MPOA", "ara": "ARA", "som": "SOM",
                 "jspace": "J-space", "gamma_mpoa": "$\\gamma$-MPOA", "community": "Community"}
METHOD_MARKERS = {"base": "*", "dim": "o", "mpoa": "s", "ara": "^", "som": "D", "jspace": "v",
                  "gamma_mpoa": "X", "community": "P"}
MARKER_SCALE = {"*": 1.35, "D": 0.85, "X": 1.05, "P": 1.05}   # optical size correction per marker shape

# ------------------------------------------------------------------------------------------------ models
MODEL_ORDER = ["qwen2.5-32b", "qwen3.8-27b", "gemma-2-27b", "gemma-4-31b", "olmo-2-32b", "olmo-3.1-32b",
               "mistral-small-2409", "mistral-small-3.2", "qwen2.5-7b", "qwen3.5-4b"]
MODEL_DISPLAY = {
    "qwen2.5-32b": "Qwen2.5-32B", "qwen3.8-27b": "Qwen3.8-27B",
    "gemma-2-27b": "Gemma-2-27B", "gemma-4-31b": "Gemma-4-31B",
    "olmo-2-32b": "OLMo-2-32B", "olmo-3.1-32b": "Olmo-3.1-32B",
    "mistral-small-2409": "Mistral-Small-2409", "mistral-small-3.2": "Mistral-Small-3.2",
    "qwen2.5-7b": "Qwen2.5-7B", "qwen3.5-4b": "Qwen3.5-4B",
}
PAIRS = [  # (family label, old, new); the last one is the small pair
    ("Qwen", "qwen2.5-32b", "qwen3.8-27b"),
    ("Gemma", "gemma-2-27b", "gemma-4-31b"),
    ("OLMo", "olmo-2-32b", "olmo-3.1-32b"),
    ("Mistral Small", "mistral-small-2409", "mistral-small-3.2"),
    ("Qwen (small pair)", "qwen2.5-7b", "qwen3.5-4b"),
]
FAMILY_ORDER = [p[0] for p in PAIRS]
BENCHES = ["mmlu_pro", "gsm8k", "ifeval", "humaneval", "mbpp"]
BENCH_LABELS = {"mmlu_pro": "MMLU-Pro", "gsm8k": "GSM8K", "ifeval": "IFEval", "humaneval": "HumanEval",
                "mbpp": "MBPP"}

# ------------------------------------------------------------------------------------------------ generations
GEN_STYLE = {   # old vs new = fill + dash only
    "old": {"linestyle": (0, (4, 2)), "filled": False, "label": "old generation (hollow, dashed)"},
    "new": {"linestyle": "-", "filled": True, "label": "new generation (filled, solid)"},
}
GEN_STYLE["small-old"] = GEN_STYLE["old"]
GEN_STYLE["small-new"] = GEN_STYLE["new"]

# ------------------------------------------------------------------------------------------------ sizes
SINGLE_COL = 3.4        # in
# Full text width of the paper template (PRIMEarxiv.sty: one column, textwidth 6.5 in, textheight 9 in). Figures
# are drawn at this width so that \includegraphics[width=\textwidth] prints them at scale 1.0 (7 pt stays 7 pt).
DOUBLE_COL = 6.5        # in  (name kept for the scripts; it is the full text width, not a two-column span)
TEXT_W = DOUBLE_COL
TEXT_H = 9.0            # in, template text height (figure + caption must fit)
FONT = 8                # pt, body text in figures
FONT_SMALL = 7          # pt, ticks / legend / annotations: the minimum at final size
LW = 1.6                # pt, data lines (house range 1.5-2)
LW_THIN = 0.6           # pt, axes / grid / outlines
MS = 5.5                # pt, marker diameter (>= 5)
HOLLOW_MEW = 1.2        # pt, edge of hollow (old-generation) markers so light hues stay visible

# ------------------------------------------------------------------------------------------------ ramps
# sequential: one hue (blue), light -> dark (reference ramp steps 100..700)
SEQ_BLUE = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6",
            "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
SEQ_BLUE_ORDINAL = ["#86b6ef", "#5598e7", "#2a78d6", "#184f95"]   # discrete ordered marks (light end >= 2:1)
# diverging: two hues + neutral grey midpoint; arms listed from the midpoint outwards (light -> dark).
# Orange arm = #eb6834 hue stepped to the blue ramp's lightness (OKLCH); both arms pass the ordinal checks.
DIV_NEG = ["#eb9c7f", "#eb6834", "#c94908", "#8a2e00"]   # capability LOSS (delta < 0)
DIV_POS = ["#86b6ef", "#5598e7", "#2a78d6", "#184f95"]   # capability GAIN (delta > 0)
DIV_MID = NEUTRAL_MID

# ------------------------------------------------------------------------------------------------ pending
PENDING_LABEL = "pending"
PENDING_FACE = NEUTRAL_MID
PENDING_EDGE = "#8a8984"
PENDING_HATCH = "////"

# ------------------------------------------------------------------------------------------------ provenance text
SOURCE_DEV = ("Dev set (optimisation): keyword refusals on mlabonne/harmful_behaviors test[:100] (count of 100) "
              "and first-token KL divergence from the original model on mlabonne/harmless_alpaca test[:100], "
              "as recorded by Heretic 2.0 for every completed TPE trial (100 trials, seed 1).")
SOURCE_SELECTION = ("Selected model per cell: fewest dev keyword refusals among completed trials with KL <= 0.1 "
                    "(ties: lower KL, then lower trial index).")
SOURCE_CAPABILITY = ("Capability: MMLU-Pro (n = 1000), GSM8K (500), IFEval (541), HumanEval (164), MBPP (500); "
                     "greedy decoding, lm-eval 0.4.13 on vLLM 0.33, scores in %, deltas in percentage points vs the "
                     "same model's original. Evaluated: Qwen and Gemma on RTX PRO 6000, OLMo and Mistral on H200 "
                     "NVL, small pair on RTX 4090.")
SOURCE_HELDOUT_TF = ("Held-out (not used for search or selection): teacher-forced divergence = token-averaged "
                     "log p_orig(y) - log p_mod(y) over the original model's own greedy answers y to the 350 benign "
                     "held-out prompts (JBB benign 100, XSTest safe 250); positive = the modified model has moved "
                     "away from the original (results.csv `tf_divergence` = -`tf_shift`).")
SOURCE_HELDOUT_JUDGED = ("Held-out, judged (not used for search or selection): one greedy response (at most 512 new "
                         "tokens, own chat template, system prompt \"You are a helpful assistant.\", reasoning "
                         "disabled) to each of the 972 harmful held-out prompts (code_weapon 200, harmbench_standard "
                         "159, jbb_harmful 100, strongreject 313, xstest_unsafe 200); refusal = WildGuard refusal "
                         "label (primary judge); rates with Wilson 95% intervals. Tables from results/analysis/ "
                         "(scripts/analysis/run_all.sh, from responses_judged.csv.gz).")


# ================================================================================================ setup
def apply():
    """Set the house rcParams. Call once at the top of every figure script."""
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Liberation Sans", "Nimbus Sans", "DejaVu Sans"],
        "font.size": FONT,
        "axes.titlesize": FONT,
        "axes.labelsize": FONT,
        "xtick.labelsize": FONT_SMALL,
        "ytick.labelsize": FONT_SMALL,
        "legend.fontsize": FONT_SMALL,
        "legend.title_fontsize": FONT_SMALL,
        "figure.titlesize": FONT,
        "text.color": TEXT,
        "axes.labelcolor": TEXT,
        "axes.titlecolor": TEXT,
        "axes.edgecolor": AXIS,
        "axes.linewidth": LW_THIN,
        "axes.facecolor": SURFACE,
        "figure.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": LW_THIN,
        "xtick.color": AXIS,
        "ytick.color": AXIS,
        "xtick.labelcolor": TEXT_SECONDARY,
        "ytick.labelcolor": TEXT_SECONDARY,
        "xtick.major.width": LW_THIN,
        "ytick.major.width": LW_THIN,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "lines.linewidth": LW,
        "lines.markersize": MS,
        "patch.linewidth": LW_THIN,
        "hatch.linewidth": LW_THIN,
        "hatch.color": PENDING_EDGE,
        "legend.frameon": False,
        "legend.handlelength": 2.4,
        "legend.borderaxespad": 0.3,
        "axes.prop_cycle": mpl.cycler(color=[METHOD_COLORS[m] for m in METHOD_ORDER[1:]]),
        "figure.constrained_layout.use": True,
        "figure.dpi": 100,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "axes.unicode_minus": True,
        # mathtext (gamma, <=, 10^-4, t) in the same family as the text (Liberation Sans), not DejaVu Sans
        "mathtext.fontset": "custom",
        "mathtext.rm": "Liberation Sans",
        "mathtext.it": "Liberation Sans:italic",
        "mathtext.bf": "Liberation Sans:bold",
        "mathtext.sf": "Liberation Sans",
        "mathtext.cal": "Liberation Sans:italic",
        "mathtext.tt": "Liberation Mono",
    })


def figure(width="single", height=None, nrows=1, ncols=1, **kw):
    """plt.subplots at the final printed size. width: 'single' (3.4 in), 'double' (6.5 in = text width) or inches."""
    w = {"single": SINGLE_COL, "double": DOUBLE_COL}.get(width, width)
    h = height if height is not None else w * 0.62
    return plt.subplots(nrows, ncols, figsize=(w, h), **kw)


# ================================================================================================ encodings
def color(method):
    return METHOD_COLORS[method]


def label(method):
    return METHOD_LABELS[method]


def _ms(marker, size):
    return size * MARKER_SCALE.get(marker, 1.0)


def line_kw(method, generation="new", marker=True, size=MS, **extra):
    """kwargs for ax.plot / ax.errorbar: method colour, generation line style and marker fill."""
    c = METHOD_COLORS[method]
    g = GEN_STYLE[generation]
    m = METHOD_MARKERS[method] if marker is True else marker
    kw = {"color": c, "linestyle": g["linestyle"], "linewidth": LW}
    if m:
        kw.update(marker=m, markersize=_ms(m, size), markeredgecolor=c,
                  markerfacecolor=c if g["filled"] else SURFACE,
                  markeredgewidth=0.6 if g["filled"] else HOLLOW_MEW)
    kw.update(extra)
    return kw


def marker_kw(method, generation="new", size=MS, **extra):
    """kwargs for ax.plot(..., linestyle='none') point markers."""
    return line_kw(method, generation, size=size, linestyle="none", **extra)


def scatter_kw(method, generation="new", size=MS, **extra):
    """kwargs for ax.scatter (s is marker AREA in pt^2)."""
    c = METHOD_COLORS[method]
    g = GEN_STYLE[generation]
    m = METHOD_MARKERS[method]
    kw = {"marker": m, "s": _ms(m, size) ** 2, "edgecolors": c, "facecolors": c if g["filled"] else SURFACE,
          "linewidths": 0.6 if g["filled"] else HOLLOW_MEW}
    kw.update(extra)
    return kw


def generation_handles(color_=TEXT_SECONDARY, marker="o"):
    """Two legend handles that explain the old/new encoding (neutral ink, method-independent)."""
    from matplotlib.lines import Line2D
    out = []
    for gen in ("old", "new"):
        g = GEN_STYLE[gen]
        out.append(Line2D([], [], color=color_, linestyle=g["linestyle"], linewidth=LW, marker=marker,
                          markersize=MS, markeredgecolor=color_, markeredgewidth=0.6 if g["filled"] else HOLLOW_MEW,
                          markerfacecolor=color_ if g["filled"] else SURFACE,
                          label="old generation" if gen == "old" else "new generation"))
    return out


def method_handles(methods, generation="new", line=False):
    """Legend handles for a list of methods (colour + marker)."""
    from matplotlib.lines import Line2D
    return [Line2D([], [], label=METHOD_LABELS[m], **(line_kw(m, generation) if line else marker_kw(m, generation)))
            for m in methods]


def seq_cmap(name="p33_seq_blue"):
    return LinearSegmentedColormap.from_list(name, SEQ_BLUE)


def diverging_cmap(name="p33_div"):
    """Continuous orange (negative) - grey (0) - blue (positive). Pair with diverging_norm()."""
    return LinearSegmentedColormap.from_list(name, list(reversed(DIV_NEG)) + [DIV_MID] + DIV_POS)


def diverging_discrete(name="p33_div_discrete"):
    """9 discrete bins: 4 loss steps, neutral, 4 gain steps."""
    return ListedColormap(list(reversed(DIV_NEG)) + [DIV_MID] + DIV_POS, name=name)


def diverging_norm(vmax, vmin=None):
    """Symmetric norm centred on 0 (grey midpoint = no change)."""
    vmax = abs(vmax)
    return TwoSlopeNorm(vcenter=0.0, vmin=-vmax if vmin is None else vmin, vmax=vmax)


def ink_on(fill_hex):
    """Text ink for a label placed on a coloured cell: white on dark fills, primary ink on light fills."""
    h = fill_hex.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))
    lin = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in (r, g, b)]
    lum = 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]
    return "#ffffff" if (1.05 / (lum + 0.05)) >= 4.5 else TEXT


# ================================================================================================ pending
def draw_pending(ax, x, y, w, h, text=PENDING_LABEL, fontsize=FONT_SMALL, rotation=0, **kw):
    """Hatched grey box with a 'pending' label (for a bar / heatmap cell / panel with no data yet)."""
    ax.add_patch(Rectangle((x, y), w, h, facecolor=PENDING_FACE, edgecolor=PENDING_EDGE, hatch=PENDING_HATCH,
                           linewidth=LW_THIN, zorder=kw.pop("zorder", 1.5), **kw))
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, color=TEXT_SECONDARY,
                rotation=rotation, zorder=3,
                bbox={"boxstyle": "round,pad=0.15", "facecolor": SURFACE, "edgecolor": "none", "alpha": 0.9})


def pending_marker(ax, x, y, text=PENDING_LABEL, fontsize=FONT_SMALL, **kw):
    """Grey open marker plus 'pending' text at a point position that has no value yet."""
    ax.plot([x], [y], marker="o", markersize=MS, markerfacecolor=SURFACE, markeredgecolor=PENDING_EDGE,
            markeredgewidth=LW_THIN, linestyle="none", zorder=2)
    if text:
        ax.annotate(text, (x, y), xytext=(4, 0), textcoords="offset points", va="center", fontsize=fontsize,
                    color=TEXT_SECONDARY, **kw)


def pending_axes(ax, text="pending: no data yet"):
    """Mark a whole panel as pending (keeps the panel instead of dropping it)."""
    ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes, facecolor=PENDING_FACE, edgecolor=PENDING_EDGE,
                           hatch=PENDING_HATCH, linewidth=LW_THIN, zorder=0))
    ax.text(0.5, 0.5, text, transform=ax.transAxes, ha="center", va="center", fontsize=FONT_SMALL,
            color=TEXT_SECONDARY, bbox={"boxstyle": "round,pad=0.2", "facecolor": SURFACE, "edgecolor": "none"})
    ax.grid(False)


def is_pending(status):
    """True for a results/runs status that is not a finished, present value."""
    return not str(status).startswith("present")


# ================================================================================================ data
def load_trials():
    import pandas as pd
    return pd.read_csv(DATA_DIR / "trials.csv")


def load_results():
    import pandas as pd
    return pd.read_csv(DATA_DIR / "results.csv", keep_default_na=True)


def load_runs():
    import pandas as pd
    return pd.read_csv(DATA_DIR / "runs.csv")


def snapshot_utc():
    """Time of the data snapshot (mtime of results.csv, written by extract_data.py) as 'YYYY-MM-DD HH:MMZ'."""
    import datetime as dt
    t = dt.datetime.fromtimestamp((DATA_DIR / "results.csv").stat().st_mtime, dt.timezone.utc)
    return f"{t:%Y-%m-%d %H:%M}Z"


# ================================================================================================ output
def check_fonts(fig, min_pt=FONT_SMALL):
    """Warn about any visible text smaller than the house minimum (7 pt at final size)."""
    small = []
    for t in fig.findobj(mpl.text.Text):
        if t.get_visible() and t.get_text().strip() and t.get_fontsize() < min_pt - 1e-6:
            small.append((t.get_text()[:30], t.get_fontsize()))
    for s, sz in small:
        print(f"  [style] WARNING: text {s!r} at {sz:.1f} pt < {min_pt} pt")
    return not small


def save(fig, name, out_dir=None, preview_dpi=200, tight=False):
    """Write <name>.pdf (vector, TrueType fonts embedded) and preview/<name>.png. Returns (pdf, png)."""
    if not NAME_RE.match(name):
        raise ValueError(f"figure name {name!r} must look like figN_descriptive_name or figAk_descriptive_name (lower case)")
    out = Path(out_dir) if out_dir else FIG_DIR
    prev = out / "preview"
    out.mkdir(parents=True, exist_ok=True)
    prev.mkdir(parents=True, exist_ok=True)
    w, _ = fig.get_size_inches()
    if not tight and min(abs(w - SINGLE_COL), abs(w - DOUBLE_COL)) > 0.01:
        print(f"  [style] WARNING: {name} is {w:.2f} in wide (single column {SINGLE_COL}, double {DOUBLE_COL})")
    check_fonts(fig)
    kw = {"bbox_inches": "tight", "pad_inches": 0.02} if tight else {}
    pdf, png = out / f"{name}.pdf", prev / f"{name}.png"
    fig.savefig(pdf, metadata={"CreationDate": None, "Creator": "paper33 scripts/figures"}, **kw)
    fig.savefig(png, dpi=preview_dpi, **kw)
    print(f"  wrote {pdf.relative_to(ROOT) if pdf.is_relative_to(ROOT) else pdf}  (+ preview PNG)")
    return pdf, png


CAPTION_HEADER = ("# Suggested figure captions (paper 33)\n\n"
                  "Written by the figure scripts (style.write_caption); one section per figure, regenerated on every "
                  "run. Each caption states which set its numbers come from (dev keyword refusals vs held-out).\n")


def write_caption(name, caption, source=None, path=None):
    """Create or replace the '## <name>' section of paper/figures/captions.md."""
    p = Path(path) if path else CAPTIONS
    text = p.read_text() if p.exists() else CAPTION_HEADER
    block = f"## {name}\n\n{caption.strip()}\n" + (f"\n*Data:* {source.strip()}\n" if source else "")
    pat = re.compile(rf"^## {re.escape(name)}\n.*?(?=^## |\Z)", re.S | re.M)
    if pat.search(text):
        text = pat.sub(lambda _: block + "\n", text)
    else:
        text = text.rstrip("\n") + "\n\n" + block
    # keep sections in figure-number order
    head, *secs = re.split(r"(?m)^(?=## )", text)
    key = lambda s: [int(x) if x.isdigit() else x for x in re.findall(r"\d+|\D+", s.split("\n", 1)[0])]  # noqa: E731
    text = head.rstrip("\n") + "\n\n" + "\n".join(sorted((s.rstrip("\n") + "\n" for s in secs), key=key))
    p.write_text(text)
