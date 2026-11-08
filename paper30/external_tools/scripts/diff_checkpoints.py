#!/usr/bin/env python3
"""Which tensors differ between two safetensors checkpoints (CPU, streaming).
Usage: python -I diff_checkpoints.py DIR_A DIR_B [OUT.json]
Prints counts of identical/changed tensors grouped by module type, and relative Frobenius change."""
import json, re, sys
from collections import defaultdict
from pathlib import Path
from safetensors import safe_open


def keys(d):
    out = {}
    for f in sorted(Path(d).glob("*.safetensors")):
        with safe_open(str(f), "pt") as h:
            for k in h.keys():
                out[k] = f
    return out


a, b = sys.argv[1], sys.argv[2]
ka, kb = keys(a), keys(b)
common = sorted(set(ka) & set(kb))
groups = defaultdict(lambda: {"same": 0, "changed": 0, "max_rel": 0.0})
for k in common:
    with safe_open(str(ka[k]), "pt") as ha, safe_open(str(kb[k]), "pt") as hb:
        ta, tb = ha.get_tensor(k).float(), hb.get_tensor(k).float()
    g = re.sub(r"\.\d+\.", ".N.", k)
    if ta.shape == tb.shape and bool((ta == tb).all()):
        groups[g]["same"] += 1
    else:
        groups[g]["changed"] += 1
        rel = float((ta - tb).norm() / ta.norm().clamp_min(1e-12)) if ta.shape == tb.shape else float("nan")
        groups[g]["max_rel"] = max(groups[g]["max_rel"], rel)
res = {"a": a, "b": b, "only_in_a": sorted(set(ka) - set(kb))[:10], "only_in_b": sorted(set(kb) - set(ka))[:10],
       "groups": {g: v for g, v in sorted(groups.items())}}
for g, v in sorted(groups.items()):
    if v["changed"]:
        print(f"{g:70s} changed {v['changed']:3d} same {v['same']:3d} max_rel {v['max_rel']:.4f}")
print("unchanged groups:", sum(1 for v in groups.values() if not v["changed"]), "| only_in_a", res["only_in_a"][:3], "| only_in_b", res["only_in_b"][:3])
if len(sys.argv) > 3:
    json.dump(res, open(sys.argv[3], "w"), indent=1)
