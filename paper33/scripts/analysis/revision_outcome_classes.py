#!/usr/bin/env python3
"""Paper 33 -- revision 2026-11-08 (ARS review, REV-34): per-model distribution of the four outcome classes of
Methods (Refusal and outcome classes): degenerate, refusal (WildGuard), complete compliance and incomplete
compliance (generation stopped at the 512-token limit), on the 972 harmful held-out prompts.

Reads (read-only): results/analysis/cell_rates.csv (set_group harmful_all; out_* columns written by assemble.py,
  where out_compliance counts complete and incomplete compliance together).
Writes: results/analysis/revision_outcome_classes.csv (one row per original and primary model),
        results/analysis/revision_outcome_classes_summary.md

Run (from the paper root):
    scripts/figures/.venv/bin/python scripts/analysis/revision_outcome_classes.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "analysis"
ORDER = ["original", "dim", "mpoa", "som", "jspace", "ara"]


def main() -> None:
    c = pd.read_csv(OUT / "cell_rates.csv")
    c = c[(c.set_group == "harmful_all") & c.role.isin(["original", "primary"])].copy()
    c["refusal"] = c.out_refusal_k / c.out_refusal_n
    c["incomplete_compliance"] = c.out_compliance_incomplete_k / c.out_refusal_n
    c["complete_compliance"] = (c.out_compliance_k - c.out_compliance_incomplete_k) / c.out_refusal_n
    c["degenerate"] = c.out_degenerate_k / c.out_refusal_n
    c["unscored"] = c.out_unscored_k / c.out_refusal_n
    tot = c[["refusal", "incomplete_compliance", "complete_compliance", "degenerate", "unscored"]].sum(axis=1)
    assert np.allclose(tot, 1.0), tot
    c["method_order"] = c.method.map({m: i for i, m in enumerate(ORDER)})
    keep = ["label", "pair", "study", "generation", "method", "role", "out_refusal_n", "refusal", "complete_compliance",
            "incomplete_compliance", "degenerate", "unscored"]
    o = c.sort_values(["study", "pair", "method_order", "gen_new"])[keep]
    o.to_csv(OUT / "revision_outcome_classes.csv", index=False)
    prim = o[o.role == "primary"]
    L = ["# REV-34: outcome classes (972 harmful held-out prompts)\n",
         f"- primary models: degenerate {100*prim.degenerate.min():.1f}-{100*prim.degenerate.max():.1f}% "
         f"(median {100*prim.degenerate.median():.1f}%); incomplete compliance {100*prim.incomplete_compliance.min():.1f}-"
         f"{100*prim.incomplete_compliance.max():.1f}% (median {100*prim.incomplete_compliance.median():.1f}%); "
         f"complete compliance {100*prim.complete_compliance.min():.1f}-{100*prim.complete_compliance.max():.1f}% "
         f"(median {100*prim.complete_compliance.median():.1f}%)",
         f"- originals: incomplete compliance {100*o[o.role=='original'].incomplete_compliance.min():.1f}-"
         f"{100*o[o.role=='original'].incomplete_compliance.max():.1f}%",
         f"- share of the non-refused responses of primary models that were incomplete (pooled): "
         f"{100*(prim.incomplete_compliance*prim.out_refusal_n).sum()/((1-prim.refusal)*prim.out_refusal_n).sum():.1f}%",
         ]
    (OUT / "revision_outcome_classes_summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print(o.round(3).to_string())


if __name__ == "__main__":
    main()
