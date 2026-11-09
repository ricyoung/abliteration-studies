"""judged.py -- shared loaders and formatting for the held-out (judged) figures fig10-fig14 (paper 33).

Not a figure script (make_figures.sh only runs fig*.py). Every number these figures draw or print is read from the
tables that scripts/analysis/run_all.sh writes to results/analysis/; nothing is re-estimated here except simple
counts and selections of rows. The functions fail loudly when a table or row they need is missing, so a figure is
never drawn from a partial table.

Conventions (as FINAL_NOTES.md): WildGuard (`wg`) is the primary judge; `kw` = Heretic's 33 refusal keywords on the
full (<= 512-token) response; `q3g` = Qwen3Guard-Gen-8B; harmful set = 972 items (`subset == "all"` in h1h2 tables,
`set_group == "harmful_all"` in cell tables); pp = percentage points.
"""
from __future__ import annotations

import datetime as dt
import math
from decimal import ROUND_HALF_UP, Decimal

import numpy as np
import pandas as pd
from matplotlib.colors import to_hex, to_rgb

import style as S

PAIR_KEY = {"Qwen": "qwen", "Gemma": "gemma", "OLMo": "olmo", "Mistral Small": "mistral",
            "Small pair": "qwen-small"}
# held-out figures order the methods by H2 class: direction-based first, then the optimisation-based ARA
DIRECTION = ["dim", "mpoa", "som", "jspace"]
METHODS_HELDOUT = ["dim", "mpoa", "som", "jspace", "ara"]
SEED2_STRENGTH = 0.45       # seed 2 = method colour mixed with white at this strength (as fig3/fig8)


# ================================================================================================ loading
def load(name, **kw):
    p = S.ANALYSIS_DIR / name
    if not p.exists():
        raise FileNotFoundError(f"{p} missing: run scripts/analysis/run_all.sh first")
    return pd.read_csv(p, **kw)


def one(df, what=""):
    """The single row of `df` (raises if there are 0 or several)."""
    if len(df) != 1:
        raise ValueError(f"expected exactly one row for {what}, found {len(df)}")
    return df.iloc[0]


def judged_cells():
    """dir_status.csv rows of the judged cells that the assembly used."""
    d = load("dir_status.csv")
    d = d[(d["used"].astype(str) == "True") & (d["judge_status"] == "judged")]
    return d


def snapshot():
    """Time the assembly ran (from assemble_log.txt), e.g. '2026-11-06 06:02Z'; else the mtime of cell_rates.csv."""
    log = S.ANALYSIS_DIR / "assemble_log.txt"
    if log.exists():
        for line in log.read_text().splitlines():
            if "assembl" in line.lower() and "20" in line:
                import re
                m = re.search(r"(20\d\d-\d\d-\d\d)[T ](\d\d:\d\d)", line)
                if m:
                    return f"{m.group(1)} {m.group(2)}Z"
    t = dt.datetime.fromtimestamp((S.ANALYSIS_DIR / "cell_rates.csv").stat().st_mtime, dt.timezone.utc)
    return f"{t:%Y-%m-%d %H:%M}Z"


def source_text(extra=""):
    n = len(judged_cells())
    return (S.SOURCE_HELDOUT_JUDGED + f" Judged cells at the assembly of {snapshot()}: {n}." +
            (" " + extra.strip() if extra else ""))


# ================================================================================================ formatting
def half_up(x, nd=1):
    """Round half away from zero (as a reader would round)."""
    q = Decimal(1).scaleb(-nd)
    return float(Decimal(repr(float(x))).quantize(q, rounding=ROUND_HALF_UP))


def num(x, nd=1):
    return f"{half_up(x, nd):.{nd}f}".replace("-", "\u2212")


def signed(x, nd=1):
    v = half_up(x, nd) + 0.0          # + 0.0 turns -0.0 into 0.0
    return f"{v:+.{nd}f}".replace("-", "\u2212")


def pp(x, nd=1):
    """A proportion difference as signed percentage points."""
    return signed(100 * x, nd)


def pct(x, nd=1):
    return num(100 * x, nd)


def tex(s):
    """Unicode minus -> LaTeX-safe minus for captions."""
    return s.replace("\u2212", "$-$")


def p_text(p):
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return "n/a"
    if p < 0.001:
        return "< 0.001"
    if p < 0.01:
        return f"{p:.3f}"
    return f"{p:.2f}"


def tint(hex_, strength=SEED2_STRENGTH):
    """Opaque equivalent of `hex_` at alpha=strength on the white surface."""
    c = np.array(to_rgb(hex_))
    return to_hex(strength * c + (1 - strength) * np.array(to_rgb(S.SURFACE)))


def uploader(repo, variant=""):
    """Hugging Face uploader of a community release (as fig5/fig6), else its variant token."""
    repo = str(repo or "")
    return repo.split("/")[0] if "/" in repo else str(variant or "community")
