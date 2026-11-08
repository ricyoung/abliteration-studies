#!/usr/bin/env python3
"""Recompute MBPP pass@1 from logged lm-eval samples with the patched extractor (see LM_EVAL_PATCHES.md).
Usage: HF_ALLOW_CODE_EVAL=1 python rescore_mbpp.py RESULT_DIR [...]   (updates summary.json; keeps the old value)"""
import json, sys, math, glob, os
os.environ.setdefault("HF_ALLOW_CODE_EVAL", "1")
from lm_eval.tasks.mbpp import utils as U
assert "\"```python\\n\" + text" in open(U.__file__).read(), "lm-eval mbpp extractor not patched"
for d in sys.argv[1:]:
    f = glob.glob(f"{d}/lmeval/mbpp/samples_*.jsonl")
    if not f:
        print("no mbpp samples:", d); continue
    rows = [json.loads(l) for l in open(f[0])]
    vals = []
    for r in rows:
        code = U.extract_code_blocks(r["resps"][0][0])
        vals.append(float(U.pass_at_1(references=[r["target"]], predictions=[[code]])))
    n = len(vals); p = sum(vals) / n; se = math.sqrt(p * (1 - p) / (n - 1))
    sp = f"{d}/summary.json"; s = json.load(open(sp)); e = s["benchmarks"]["mbpp"]
    e.setdefault("score_unpatched_extractor", e.get("score")); e["score"] = p; e["stderr"] = se; e["n"] = n
    e["rescored"] = "rescore_mbpp.py (patched extract_code_blocks)"
    json.dump(s, open(sp, "w"), indent=1)
    print(d, f"mbpp pass@1 = {p:.3f} (n={n}; was {e['score_unpatched_extractor']})")
