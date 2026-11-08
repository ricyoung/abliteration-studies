#!/usr/bin/env python3
"""Summarise smoke outputs: PROVENANCE.json + smoke_generate results -> smoke/SUMMARY.md and SUMMARY.json."""
import json
from pathlib import Path

S = Path("/data/heretic_2/tools_ext/smoke")
rows = []
for d in sorted((S / "models").iterdir()):
    pf = d / "PROVENANCE.json"
    if not pf.exists():
        continue
    p = json.loads(pf.read_text())
    if "tool" not in p:  # e.g. the rewrap test
        continue
    g = S / "generations" / f"{d.name}.json"
    gen = json.loads(g.read_text()) if g.exists() else None
    qm = p.get("tool_reported_metrics") or {}
    if p["tool"] == "OBLITERATUS":
        metrics = {k: qm.get(k) for k in ("refusal_rate", "perplexity", "baseline_perplexity", "coherence",
                                          "coherence_retention", "kl_divergence", "first_token_kl_divergence",
                                          "degenerate_fraction", "capability_score") if k in qm}
        if not metrics:
            metrics = {k: v for k, v in qm.items() if isinstance(v, (int, float))}
    else:
        metrics = {"n_harmful_used": p["prompt_sets"].get("n_harmful_used"),
                   "hybrid": (p.get("hybrid_detection") or {}).get("is_hybrid")}
    rows.append({
        "name": d.name, "tool": p["tool"], "preset": p["preset"], "base": f"{p['base_repo']}@{p['base_revision'][:7]}",
        "patches": [r["name"] for r in p.get("runtime_patches") or []],
        "tool_min": round(p["wall_seconds_tool"] / 60, 1), "n_tensors": p["verify"]["n_tensors"],
        "non_bf16": len(p["verify"]["non_bf16"]), "keys_missing": p["finalize_report"].get("keys_missing_vs_base"),
        "metrics": metrics,
        "gen": None if gen is None else {k: {"refusal": v["keyword_refusal"], "reply": v["reply"][:110].replace("\n", " ")}
                                         for k, v in gen["replies"].items()},
    })
for b in sorted((S / "generations").glob("*__BASE.json")):
    gen = json.loads(b.read_text())
    rows.append({"name": b.stem, "tool": "base", "preset": "-", "gen": {k: {"refusal": v["keyword_refusal"], "reply": v["reply"][:110].replace("\n", " ")} for k, v in gen["replies"].items()}})
json.dump(rows, open(S / "SUMMARY.json", "w"), indent=1)
lines = ["| output | patches | tool min (CPU) | tensors | tool metrics | harmless reply | harmful reply |", "|---|---|---|---|---|---|---|"]
for r in rows:
    g = r.get("gen") or {}
    h, x = g.get("harmless", {}), g.get("harmful", {})
    fmt = lambda v: "" if not v else (("REFUSAL: " if v["refusal"] else "") + v["reply"].replace("|", "/"))
    lines.append(f"| {r['name']} | {', '.join(r.get('patches') or []) or '-'} | {r.get('tool_min', '-')} | {r.get('n_tensors', '-')} | "
                 f"{json.dumps(r.get('metrics', ''))[:160].replace('|', '/')} | {fmt(h)} | {fmt(x)} |")
(S / "SUMMARY.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
