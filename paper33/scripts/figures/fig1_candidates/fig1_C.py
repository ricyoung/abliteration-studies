#!/usr/bin/env python3
"""fig1_C.py -- Figure 1, candidate C: the study workflow as a refined Graphviz `dot` layout.

Run (any cwd):
    scripts/figures/.venv/bin/python scripts/figures/fig1_candidates/fig1_C.py

Writes
    scripts/figures/fig1_candidates/fig1_C.dot                 generated Graphviz source (final pass)
    scripts/figures/fig1_candidates/fig1_C.pdf                 vector PDF, exactly 7.0 in wide, TrueType fonts
    paper/figures/preview/fig1_candidates/fig1_C.png           200 dpi preview (pdftoppm)

Design
  * Three tinted bands read top to bottom -- SEARCH on dev prompts (stages 1-4) -> EVALUATION on held-out
    prompts (stage 5) -> STATISTICS (stage 6) -- on one 4-column grid (column 1 is wider: it carries the
    model table and the refusal-set composition). Rows are `rank=same`; invisible heavy edges pin every row
    to the grid, so columns line up across all bands.
  * Every card = bold title row (numbered badge for the stages) + light detail rows. Cards are borderless
    white on the tint and share width per column and height per band: structure comes from the grid,
    weight and whitespace, not from outlines.
  * Colour is identity only: method chips use the fixed method colours used in every other figure.
  * Additional conditions sit in one dashed cluster in the gap between the bands, at the point where they
    enter the pipeline: gamma-MPOA and the seed-2 replicate feed stage 2, community models feed stage 5.
  * splines=ortho; arrows run in gutters, inter-band arrows are vertical and clip at the band edge.
  * Every text run is measured with the embedded font (PIL) and checked against its cell, so nested HTML
    tables get exact widths and Graphviz never stretches them; the natural drawing is <= 7.0 in wide and is
    only ever scaled up (fonts never drop below their nominal size).
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

from PIL import ImageFont

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]                              # .../33_2611_does_refusal_removal_get_harder
DOT_PATH = HERE / "fig1_C.dot"
PDF_PATH = HERE / "fig1_C.pdf"
PNG_PATH = ROOT / "paper" / "figures" / "preview" / "fig1_candidates" / "fig1_C.png"

TARGET_W_IN = 7.0
PREVIEW_DPI = 200

# ---------------------------------------------------------------- style tokens
FONT = "Liberation Sans"
FONT_FILES = {False: "/usr/share/fonts/liberation/LiberationSans-Regular.ttf",
              True: "/usr/share/fonts/liberation/LiberationSans-Bold.ttf"}
INK, INK2 = "#0b0b0b", "#52514e"
GREY, GREY_LT = "#8a8984", "#e6e5e1"
TINT = "#efeeeb"                    # band fill: one light step of GREY_LT
METHOD = {"DiM": "#2a78d6", "MPOA": "#eb6834", "ARA": "#1baf7a", "SOM": "#eda100", "J-space": "#e87ba4"}
COMMUNITY, GAMMA = "#6250d6", "#e34948"

FS_TITLE, FS_BODY, FS_NOTE = 8.0, 7.2, 7.0  # nominal pt; drawing is scaled by >= 1, so these are minima

COL_W = [1.76, 1.46, 1.46, 1.46]    # grid columns (in)
GAP = 0.20                          # gutter between cards (in); horizontal arrows live here
RANKSEP = 0.36                      # in; sets the gutters between bands (arrows need length to read)
PAD_X, PAD_Y = 0.075, 0.065         # inner card padding (in)
BAND_MARGIN = 8                     # pt
CHIP = 7                            # pt
X_TOP = 6                           # pt, air above the additional-condition items (arrow tails stop here)
BADGE = 11                          # pt


# ---------------------------------------------------------------- text runs and measurement
_fonts: dict[bool, ImageFont.FreeTypeFont] = {}


def tw(text: str, size: float, bold: bool = False) -> float:
    """Advance width in pt (Liberation Sans, the font cairo embeds)."""
    if bold not in _fonts:
        _fonts[bold] = ImageFont.truetype(FONT_FILES[bold], 1000)
    return _fonts[bold].getlength(text) / 1000 * size


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


class Line:
    """A line of text made of styled runs; knows its own width."""

    def __init__(self, *runs: tuple[str, float, str, bool]):
        self.runs = runs

    @property
    def width(self) -> float:
        return sum(tw(t, s, b) for t, s, _, b in self.runs)

    def html(self) -> str:
        out = []
        for t, s, c, b in self.runs:
            x = esc(t)
            out.append(f'<FONT POINT-SIZE="{s}" COLOR="{c}">{"<B>" + x + "</B>" if b else x}</FONT>')
        return "".join(out)

    def __add__(self, other: "Line") -> "Line":
        return Line(*self.runs, *other.runs)


def body(t: str) -> Line:
    return Line((t, FS_BODY, INK2, False))


def ink(t: str, bold: bool = False) -> Line:
    return Line((t, FS_BODY, INK, bold))


def note(t: str) -> Line:
    return Line((t, FS_NOTE, GREY, False))


def title(t: str) -> Line:
    return Line((t, FS_TITLE, INK, True))


def td(content: str = "", width: float | None = None, align: str = "LEFT", extra: str = "") -> str:
    w = f' WIDTH="{int(round(width))}"' if width is not None else ""
    return f'<TD ALIGN="{align}" BALIGN="{align}"{w}{extra}>{content}</TD>'


def table(rows: list[str], width: float | None = None) -> str:
    w = f' WIDTH="{int(round(width))}"' if width is not None else ""
    return (f'<TABLE BORDER="0" CELLSPACING="0" CELLPADDING="0"{w}>'
            + "".join(f"<TR>{r}</TR>" for r in rows) + "</TABLE>")


def badge(n: str) -> str:
    return (f'<TD BGCOLOR="{INK}" STYLE="rounded" WIDTH="{BADGE}" HEIGHT="{BADGE}" FIXEDSIZE="TRUE">'
            f'<FONT POINT-SIZE="7" COLOR="white"><B>{n}</B></FONT></TD>')


def chip(color: str) -> str:
    return f'<TD BGCOLOR="{color}" STYLE="rounded" WIDTH="{CHIP}" HEIGHT="{CHIP}" FIXEDSIZE="TRUE"></TD>'


def check(line: Line, avail: float, where: str) -> None:
    if line.width > avail + 0.5:
        print(f"  [fig1_C] overflow {line.width - avail:.1f} pt in {where}: "
              f"{''.join(r[0] for r in line.runs)!r}", file=sys.stderr)


# ---------------------------------------------------------------- composite blocks (all exact width)
def full_w(width_in: float) -> int:
    """Text width for transparent (box-less) nodes: the whole column."""
    return int(round(width_in * 72))


def inner_w(width_in: float) -> int:
    return int(round((width_in - 2 * PAD_X) * 72))


def head_row(num: str | None, head: Line, right: Line | None, inner: int) -> str:
    """[badge] title ........ right-note, exactly `inner` pt wide."""
    lead = BADGE + 5 if num else 0
    rw = int(round(right.width)) + 1 if right else 0
    avail = inner - lead - rw
    check(head, avail, "title row")
    cells = [badge(num), td("", 5)] if num else []
    cells.append(td(head.html(), avail))
    if right:
        cells.append(td(right.html(), rw, "RIGHT"))
    return table(["".join(cells)])


def chips(items: list[tuple[Line, str]], label_w: list[float], inner: int) -> str:
    """One row of [chip] label pairs; label columns at fixed widths, the last one takes the rest."""
    cells, used = [], 0
    for k, ((lab, col), w) in enumerate(zip(items, list(label_w) + [0] * len(items))):
        last = k == len(items) - 1
        cells += [chip(col), td("", 3)]
        used += CHIP + 3
        w_eff = inner - used if last else int(round(w))
        check(lab, w_eff, "chip label")
        cells.append(td(lab.html(), w_eff))
        used += w_eff
    return table(["".join(cells)])


Row = Line | str | None             # Line, raw HTML block, or None for a small vertical gap


def card(num: str | None, head: str, rows: list[Row], width_in: float, right: str = "") -> str:
    inner = inner_w(width_in)
    out = [td(head_row(num, title(head), note(right) if right else None, inner)), td("", extra=' HEIGHT="5"')]
    for r in rows:
        if r is None:
            out.append(td("", extra=' HEIGHT="5"'))
        elif isinstance(r, Line):
            check(r, inner, f"card {head!r}")
            out.append(td(r.html()))
        else:
            out.append(td(r))
    return table(out, inner)


# ---------------------------------------------------------------- content
def model_table(inner: int) -> str:
    pairs = [("Qwen", "2.5-32B", "3.8-27B"), ("Gemma", "2-27B", "4-31B"), ("OLMo", "2-32B", "3.1-32B"),
             ("Mistral Small", "2409", "3.2 (24B)"), ("Qwen", "2.5-7B", "3.5-4B")]
    w_fam = int(round(max(tw(p[0], FS_BODY) for p in pairs))) + 5
    w_a = int(round(max(tw(p[1], FS_BODY) for p in pairs))) + 1
    w_arr = int(round(tw("→", FS_BODY))) + 6
    w_b = inner - w_fam - w_a - w_arr
    rows = []
    for k, (fam, a, b) in enumerate(pairs):
        small = k == len(pairs) - 1
        if small:
            rows.append(td("", extra=' COLSPAN="4" HEIGHT="4"'))
        fam_l = ink(fam) + note("  small") if small else ink(fam)
        check(fam_l, w_fam, "model table")
        check(body(b), w_b, "model table")
        rows.append(td(fam_l.html(), w_fam) + td(body(a).html(), w_a, "RIGHT")
                    + td(note("→").html(), w_arr, "CENTER") + td(body(b).html(), w_b))
    return table(rows)


def build_nodes() -> dict[str, tuple[str, float, str]]:
    """name -> (HTML label, width in, kind) with kind in {card, plain}."""
    W1, W = COL_W[0], COL_W[1]
    i1, i = inner_w(W1), inner_w(W)
    n: dict[str, tuple[str, float, str]] = {}

    # ---- band 1: search on dev prompts (stages 1-4)
    n["s1"] = (card("1", "Model pairs", [model_table(i1), None, note("instruction-tuned · bf16")],
                    W1, right="earlier → later"), W1, "card")
    lw = [tw("MPOA", FS_BODY) + 9]
    n["s2"] = (card("2", "Optimisation", [
        chips([(ink("DiM"), METHOD["DiM"]), (ink("ARA"), METHOD["ARA"])], lw, i),
        chips([(ink("MPOA"), METHOD["MPOA"]), (ink("SOM"), METHOD["SOM"])], lw, i),
        chips([(ink("J-space ") + note("small pair"), METHOD["J-space"])], [], i),
        None,
        body("TPE · 100 trials · seed 1"),
        body("↓ keyword refusals"),
        body("↓ first-token KL"),
    ], W), W, "card")
    n["s3"] = (card("3", "Trial selection", [
        body("fewest refusals among"),
        body("trials with KL ≤ 0.1,"),
        body("same rule, every study"),
        None,
        note("sensitivity: 0.05 · 0.2 · 0.5"),
    ], W), W, "card")
    lw4 = [tw("DiM", FS_BODY) + 6, tw("MPOA", FS_BODY) + 6]
    n["s4"] = (card("4", "Export & release", [
        body("adapter merged into"),
        body("standalone weights,"),
        body("public on Hugging Face"),
        None,
        note("reproduce manifest"),
        chips([(ink("DiM"), METHOD["DiM"]), (ink("MPOA"), METHOD["MPOA"]), (ink("ARA"), METHOD["ARA"])], lw4, i),
    ], W), W, "card")

    # ---- additional conditions (dashed cluster in the gap; col 1 carries its title)
    def extra_item(col: str, head: str, subs: list[str], w_in: float) -> list[str]:
        inner = full_w(w_in)
        rows = [chip(col) + td("", 4) + td(ink(head, True).html(), inner - CHIP - 4)]
        for s in subs:
            check(note(s), inner - CHIP - 4, f"extra {head!r}")
            rows.append(td("", CHIP) + td("", 4) + td(note(s).html(), inner - CHIP - 4))
        return rows
    xa = (extra_item(GAMMA, "γ-corrected MPOA", ["post-norm Gemma, OLMo"], W)
          + [td("", extra=' COLSPAN="3" HEIGHT="5"')]
          + extra_item(GREY, "Seed-2 replicate", ["small pair, 5 methods"], W))
    n["xa"] = (table(xa, full_w(W)), W, "plain")
    n["xb"] = (table(extra_item(COMMUNITY, "8 community models", ["public abliterated releases"], W), full_w(W)),
               W, "plain")
    n["xh"] = (table([td(Line(("Additional conditions", FS_BODY, INK2, True)).html()),
                      td(note("same protocol").html())], full_w(W1)), W1, "plain")

    # ---- band 2: evaluation on held-out prompts (stage 5)
    n["e1"] = (card(None, "Refusal prompts", [
        body("972 harmful-type, 350 benign"),
        body("greedy, ≤ 512 tokens"),
    ], W1, right="1,322"), W1, "card")
    n["e2"] = (card(None, "Judges", [
        body("WildGuard (primary),"),
        body("HarmBench, Qwen3Guard,"),
        body("2 encoders, keywords"),
        None,
        note("refusal · compliance ·"),
        note("degenerate · incomplete"),
    ], W), W, "card")
    n["e3"] = (card(None, "Divergence", [
        body("log-prob shift on the"),
        body("original's answers to"),
        body("the 350 benign prompts"),
    ], W, right="teacher-forced"), W, "card")
    n["e4"] = (card(None, "Capability", [
        body("MMLU-Pro 1,000"),
        body("GSM8K 500 · IFEval 541"),
        body("HumanEval 164 · MBPP 500"),
        None,
        note("lm-eval · vLLM"),
    ], W), W, "card")

    # ---- band 3: statistics (stage 6) -- title in column 1, text facets aligned to the card columns
    stat_title = head_row("6", title("Statistics"), None, full_w(W1))
    facets: list[tuple[str | None, list[Line]]] = [
        (None, [body("Wilson intervals; McNemar vs"), body("original; Holm correction")]),
        ("H1, H2", [body("mixed-effects logistic,"), body("generation × method class;"),
                    note("random: prompt, family")]),
        ("H3", [body("keyword–judge"), body("agreement (κ)")]),
        ("Dev vs held-out", [body("Spearman correlation")]),
    ]
    for k, ((head, lines), w) in enumerate(zip(facets, COL_W)):
        inner = full_w(w)
        head_html = stat_title if k == 0 else ink(head, True).html()
        rows = [td(head_html, inner, extra=f' HEIGHT="{BADGE}"'), td("", extra=' HEIGHT="4"')]
        for ln in lines:
            check(ln, inner, f"stats facet {k}")
            rows.append(td(ln.html()))
        n[f"t{k + 1}"] = (table(rows, inner), w, "plain")
    return n


def band_label(num: str | None, head: str, qual: str) -> str:
    """Cluster label: [badge] bold title   grey qualifier (single text cell, never stretched)."""
    text = (title(head) + Line(("   " + qual, FS_NOTE, GREY, False))).html()
    cells = ([badge(num), td("", 5)] if num else []) + [td(text)]
    span = 3 if num else 1
    return "<" + table([td("", extra=f' COLSPAN="{span}" HEIGHT="3"'), "".join(cells)]) + ">"


def build_dot(heights: dict[str, float] | None = None) -> str:
    n = build_nodes()
    heights = heights or {}
    L: list[str] = []
    a = L.append
    a("digraph fig1 {")
    a(f'  graph [rankdir=TB, splines=ortho, compound=true, nodesep={GAP}, ranksep={RANKSEP}, pad=0.0, '
      f'margin=0, fontname="{FONT}", bgcolor="white", newrank=true, outputorder=edgesfirst];')
    a(f'  node [shape=box, style="rounded,filled", fillcolor="white", color="white", penwidth=0, '
      f'fontname="{FONT}", margin="{PAD_X},0", labelloc=t, fixedsize=false];')
    a(f'  edge [color="{INK2}", penwidth=0.8, arrowsize=0.5, arrowhead=normal];')

    def node(name: str) -> None:
        label, w, kind = n[name]
        h = heights.get(name, 0.2)
        if kind == "plain":   # text straight on the background: transparent box, text spans the column
            if name.startswith("x"):
                label = (f'<TABLE BORDER="0" CELLSPACING="0" CELLPADDING="0"><TR><TD HEIGHT="{X_TOP}"></TD></TR>'
                         f'<TR><TD>{label}</TD></TR></TABLE>')
            a(f'    {name} [label=<{label}>, width={w}, height={h:.3f}, margin="0,0", style="", fillcolor="none"];')
        else:                 # white card; vertical padding inside the label (labelloc=t ignores margin y)
            pad = f'<TR><TD HEIGHT="{PAD_Y * 72:.0f}"></TD></TR>'
            label = f'<TABLE BORDER="0" CELLSPACING="0" CELLPADDING="0">{pad}<TR><TD>{label}</TD></TR>{pad}</TABLE>'
            a(f"    {name} [label=<{label}>, width={w}, height={h:.3f}];")

    def band(cname: str, label: str | None) -> None:
        a(f"  subgraph {cname} {{")
        a(f'    style="rounded,filled"; fillcolor="{TINT}"; color="{TINT}"; penwidth=0; '
          f'margin={BAND_MARGIN}; labeljust=l; label={label or "\"\""};')

    band("cluster_search", band_label(None, "Search", "dev prompts: 100 harmful, 100 harmless · Heretic 2.0"))
    for k in ("s1", "s2", "s3", "s4"):
        node(k)
    a("    { rank=same; s1 -> s2 -> s3 -> s4; }")
    a("  }")

    a("  subgraph cluster_extra {")
    a(f'    style="rounded,dashed"; color="{GREY}"; penwidth=0.7; margin={BAND_MARGIN}; label="";')
    for k in ("xh", "xa", "xb"):
        node(k)
    a("    { rank=same; xh -> xa -> xb [style=invis]; }")
    a("  }")

    band("cluster_eval", band_label("5", "Evaluation of originals and selected models", "held-out prompts only"))
    for k in ("e1", "e2", "e3", "e4"):
        node(k)
    a("    { rank=same; e1 -> e2; e2 -> e3 -> e4 [style=invis]; }")
    a("  }")

    band("cluster_stats", None)
    for k in ("t1", "t2", "t3", "t4"):
        node(k)
    a("    { rank=same; t1 -> t2 -> t3 -> t4 [style=invis]; }")
    a("  }")

    # grid: pin each column vertically (invisible, heavy)
    for col in (("s1", "xh", "e1", "t1"), ("s2", "xa", "e2", "t2"), ("s3", "xb", "e3", "t3"), ("s4", "e4", "t4")):
        for u, v in zip(col, col[1:]):
            a(f"  {u} -> {v} [style=invis, weight=50];")
    # visible flow between bands
    a("  s4 -> e4 [lhead=cluster_eval, weight=0];")
    a(f'  s2 -> xa [dir=back, lhead=cluster_extra, style=dashed, color="{GREY}", weight=0];')
    a(f'  xb -> e3 [lhead=cluster_eval, style=dashed, color="{GREY}", weight=0];')
    a("  e2 -> t2 [ltail=cluster_eval, lhead=cluster_stats, weight=0];")
    a("}")
    return "\n".join(L) + "\n"


def run(cmd: list[str]) -> str:
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        sys.exit(f"command failed: {' '.join(cmd)}\n{p.stderr}")
    if p.stderr.strip():
        print(p.stderr.strip(), file=sys.stderr)
    return p.stdout


def plain_layout(src: str) -> tuple[float, float, dict[str, float]]:
    DOT_PATH.write_text(src)
    out = run(["dot", "-Tplain", str(DOT_PATH)]).splitlines()
    w, h = map(float, out[0].split()[2:4])
    hs = {p[1]: float(p[5]) for p in (ln.split() for ln in out[1:]) if p and p[0] == "node"}
    return w, h, hs


def main() -> None:
    for tool in ("dot", "pdftoppm", "pdfinfo", "pdftotext"):
        if not shutil.which(tool):
            sys.exit(f"missing tool: {tool}")
    # pass 1: natural heights -> equal heights per row
    _, _, hs = plain_layout(build_dot())
    heights: dict[str, float] = {}
    for group in (("s1", "s2", "s3", "s4"), ("xh", "xa", "xb"), ("e1", "e2", "e3", "e4"), ("t1", "t2", "t3", "t4")):
        hmax = max(hs[g] for g in group)
        heights.update({g: hmax for g in group})
    src = build_dot(heights)
    # pass 2: crop to the union of the band/cluster boxes (dot adds an 8 pt cluster offset around the
    # drawing) and zoom so that box is exactly TARGET_W_IN wide -- via `viewport`, so the PDF stays vector
    DOT_PATH.write_text(src)
    lay = json.loads(run(["dot", "-Tjson0", str(DOT_PATH)]))
    boxes = [list(map(float, o["bb"].split(","))) for o in lay["objects"] if o.get("name", "").startswith("cluster")]
    bleed = 1.0                     # pt, keeps the half of the outermost strokes that lies outside the boxes
    x0, y0 = min(b[0] for b in boxes) - bleed, min(b[1] for b in boxes) - bleed
    x1, y1 = max(b[2] for b in boxes) + bleed, max(b[3] for b in boxes) + bleed
    nat_w, nat_h = (x1 - x0) / 72, (y1 - y0) / 72
    scale = TARGET_W_IN / nat_w
    viewport = f"{(x1 - x0) * scale:.3f},{(y1 - y0) * scale:.3f},{scale:.5f},{(x0 + x1) / 2:.3f},{(y0 + y1) / 2:.3f}"
    src = src.replace("digraph fig1 {", f'digraph fig1 {{\n  graph [viewport="{viewport}"];', 1)
    DOT_PATH.write_text(src)
    run(["dot", "-Tpdf:cairo", "-o", str(PDF_PATH), str(DOT_PATH)])
    PNG_PATH.parent.mkdir(parents=True, exist_ok=True)
    run(["pdftoppm", "-r", str(PREVIEW_DPI), "-png", "-singlefile", str(PDF_PATH), str(PNG_PATH.with_suffix(""))])
    m = re.search(r"Page size:\s+([\d.]+) x ([\d.]+)", run(["pdfinfo", str(PDF_PATH)]))
    w_in, h_in = float(m.group(1)) / 72, float(m.group(2)) / 72
    print(f"content {nat_w:.3f} x {nat_h:.3f} in; scale {scale:.3f}; PDF {w_in:.3f} x {h_in:.3f} in")
    print(f"effective font sizes: title {FS_TITLE * scale:.2f}, body {FS_BODY * scale:.2f}, "
          f"note {FS_NOTE * scale:.2f} pt")
    if scale < 1 - 1e-3:
        print("WARNING: drawing was shrunk; text below nominal size")
    words = [w for w in run(["pdftotext", "-raw", str(PDF_PATH), "-"]).split() if re.search(r"[A-Za-z0-9]", w)]
    print(f"words in figure: {len(words)}")
    print(f"wrote {DOT_PATH}\n      {PDF_PATH}\n      {PNG_PATH}")


if __name__ == "__main__":
    main()
