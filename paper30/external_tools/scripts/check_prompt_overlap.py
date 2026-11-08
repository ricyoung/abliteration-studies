#!/usr/bin/env python3
"""Check the prompt sets an external tool trains on against paper 33's data.

Compares each tool prompt set with
  * the frozen held-out set (scripts/bench/refusal_sets.json, 972 harmful + 350 benign),
  * AdvBench (all 520 mlabonne/harmful_behaviors prompts, train+test),
  * paper 33's dev set (mlabonne/harmful_behaviors test[:100]),
  * harmless_alpaca (train+test), for the harmless side.
Overlap rule is the one in scripts/build_refusal_sets.py: exact after normalisation, or
fuzzy = token Jaccard >= 0.6 and difflib ratio >= 0.8.

Usage: python -I check_prompt_overlap.py OUT.json NAME=FILE.json [NAME=FILE.json ...]
       python -I check_prompt_overlap.py OUT.json --outputs MODEL_DIR [MODEL_DIR ...]
FILE.json is a JSON list of strings. With --outputs, the harmful/harmless lists are read from
MODEL_DIR/tool_artifacts/prompts_used.json (written by run_obliteratus.py / run_jwest33.py).
"""
import difflib, json, re, sys
from pathlib import Path
import pandas as pd

P33 = Path("/data/heretic_2/33_2611_does_refusal_removal_get_harder")
HB = Path("/data/huggingface/hub/datasets--mlabonne--harmful_behaviors/snapshots/01cead01398926d81f7c52bdb790ee8cf77ebba7/data")
HA = Path("/data/huggingface/hub/datasets--mlabonne--harmless_alpaca/snapshots/02c6a92cfcf11bb0c387334f8146d149d65b587f/data")

norm = lambda s: re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", s.lower())).strip()


def build(prompts):
    ns = [norm(p) for p in prompts]
    return dict(set=set(ns), list=[(n, set(n.split())) for n in ns])


def match(q, ref):
    qn = norm(q)
    if qn in ref["set"]:
        return "exact"
    toks = set(qn.split())
    for a, at in ref["list"]:
        if len(toks & at) / max(1, len(toks | at)) >= 0.6 and difflib.SequenceMatcher(None, qn, a).ratio() >= 0.8:
            return "fuzzy"
    return ""


held = json.load(open(P33 / "scripts/bench/refusal_sets.json"))["items"]
adv_train = pd.read_parquet(HB / "train-00000-of-00001.parquet")["text"].tolist()
adv_test = pd.read_parquet(HB / "test-00000-of-00001.parquet")["text"].tolist()
alp = pd.read_parquet(HA / "train-00000-of-00001.parquet")["text"].tolist() + pd.read_parquet(HA / "test-00000-of-00001.parquet")["text"].tolist()
refs = {
    "heldout_harmful_972": build([i["prompt"] for i in held if i["kind"] == "harmful"]),
    "heldout_benign_350": build([i["prompt"] for i in held if i["kind"] == "benign"]),
    "advbench_520": build(adv_train + adv_test),
    "p33_dev_test100": build(adv_test[:100]),
    "harmless_alpaca_all": build(alp),
}
held_by_set = {}
for i in held:
    held_by_set.setdefault(i["set"], []).append(i["prompt"])
held_refs = {k: build(v) for k, v in held_by_set.items()}

out = {"rule": "exact after normalisation, or token-Jaccard>=0.6 and SequenceMatcher>=0.8 (as build_refusal_sets.py)", "sets": {}}
jobs = []
if len(sys.argv) > 2 and sys.argv[2] == "--outputs":
    for d in sys.argv[3:]:
        f = Path(d) / "tool_artifacts" / "prompts_used.json"
        if not f.exists():
            continue
        pu = json.load(open(f))
        for kind in ("harmful", "harmless"):
            if isinstance(pu.get(kind), list):
                jobs.append((f"{Path(d).name}:{kind}", pu[kind]))
else:
    for arg in sys.argv[2:]:
        name, f = arg.split("=", 1)
        jobs.append((name, json.load(open(f))))
cache = {}
for name, prompts in jobs:
    key = json.dumps(prompts)
    if key in cache:
        out["sets"][name] = dict(cache[key], same_as=cache[key]["name"])
        print(name, "same list as", cache[key]["name"], flush=True)
        continue
    rep = {"n": len(prompts), "n_unique": len(set(prompts)), "overlap": {}, "heldout_by_subset": {}, "examples": {}}
    for rname, ref in refs.items():
        hits = [(p, match(p, ref)) for p in prompts]
        hits = [(p, m) for p, m in hits if m]
        rep["overlap"][rname] = {"exact": sum(m == "exact" for _, m in hits), "fuzzy": sum(m == "fuzzy" for _, m in hits)}
        rep["examples"][rname] = [p for p, _ in hits[:5]]
    for sname, ref in held_refs.items():
        n = sum(bool(match(p, ref)) for p in prompts)
        if n:
            rep["heldout_by_subset"][sname] = n
    out["sets"][name] = rep
    cache[key] = dict(rep, name=name)
    print(name, json.dumps(rep["overlap"]), rep["heldout_by_subset"], flush=True)
json.dump(out, open(sys.argv[1], "w"), indent=1)
