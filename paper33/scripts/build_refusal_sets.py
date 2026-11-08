#!/usr/bin/env python3
"""Build the frozen held-out refusal prompt file (paper 33, v2). Run once; never regenerate mid-study.
Usage: HF_TOKEN=... python -I build_refusal_sets.py OUT.json"""
import json, re, sys, hashlib, csv, io, random, urllib.request, difflib
from collections import Counter
from datasets import load_dataset
from huggingface_hub import HfApi

api = HfApi()
norm = lambda s: re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", s.lower())).strip()
mh = load_dataset("mlabonne/harmful_behaviors")
adv = [norm(x) for sp in mh for x in mh[sp]["text"]]
advset = set(adv)


def overlap(p):
    q = norm(p)
    if q in advset:
        return "exact"
    toks = set(q.split())
    for a in adv:
        at = set(a.split())
        if len(toks & at) / max(1, len(toks | at)) >= 0.6 and difflib.SequenceMatcher(None, q, a).ratio() >= 0.8:
            return "fuzzy"
    return ""


items = []


def add(set_, id_, kind, prompt, category, source):
    items.append(dict(set=set_, id=id_, kind=kind, prompt=prompt, category=category, source=source))


jbb = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors")
for sp in ("harmful", "benign"):
    for r in jbb[sp]:
        add(f"jbb_{sp}", f"jbb_{sp}_{r['Index']}", "harmful" if sp == "harmful" else "benign", r["Goal"], r["Category"], r["Source"])
for r in load_dataset("Paul/XSTest")["train"]:
    add(f"xstest_{r['label']}", f"xstest_{r['id']}", "harmful" if r["label"] == "unsafe" else "benign", r["prompt"], r["type"], "XSTest")
sr = urllib.request.urlopen("https://raw.githubusercontent.com/alexandrasouly/strongreject/main/strongreject_dataset/strongreject_dataset.csv").read().decode()
for i, r in enumerate(csv.DictReader(io.StringIO(sr))):
    add("strongreject", f"strongreject_{i}", "harmful", r["forbidden_prompt"], r["category"], r.get("source", "StrongREJECT"))
HB_URL = "https://raw.githubusercontent.com/centerforaisafety/HarmBench/main/data/behavior_datasets/harmbench_behaviors_text_test.csv"
hbtxt = urllib.request.urlopen(HB_URL).read().decode()
for r in csv.DictReader(io.StringIO(hbtxt)):
    if r["FunctionalCategory"] == "standard":
        add("harmbench_standard", f"harmbench_{r['BehaviorID']}", "harmful", r["Behavior"], r["SemanticCategory"], "HarmBench test/standard")
cb = load_dataset("richardyoung/code-as-weapon-bank", data_files="code_safety/train.parquet")["train"]
cand = [r for r in cb if r["source_dataset"] not in ("harmful_behaviors", "jailbreakbench") and r["agreement_tier"] in ("5/5", "4/4")]
rng = random.Random(20261105)
by = {}
for r in cand:
    by.setdefault(r["source_dataset"], []).append(r)
pick = []
for src, rows in sorted(by.items()):
    rng.shuffle(rows)
    pick += rows[: max(1, round(200 * len(rows) / len(cand)))]
for r in pick[:200]:
    add("code_weapon", f"cw_{r['uid']}", "harmful", r["prompt"], r["category"], r["source_dataset"])
for it in items:
    it["advbench_overlap"] = overlap(it["prompt"]) if it["kind"] == "harmful" else ""

meta = dict(
    created="2026-11-05", version=2, system_prompt="You are a helpful assistant.", max_tokens=512, greedy=True,
    datasets={d: api.dataset_info(d).sha for d in ["JailbreakBench/JBB-Behaviors", "Paul/XSTest", "mlabonne/harmful_behaviors", "richardyoung/code-as-weapon-bank"]},
    strongreject_csv_sha256=hashlib.sha256(sr.encode()).hexdigest(),
    harmbench_csv=HB_URL, harmbench_csv_sha256=hashlib.sha256(hbtxt.encode()).hexdigest(),
    code_weapon_sampling="code_safety split, agreement 5/5 or 4/4, excluding harmful_behaviors/jailbreakbench sources, "
                         "proportional by source, seed 20261105, n=200",
    note="advbench_overlap: exact | fuzzy (token Jaccard>=0.6 and SequenceMatcher>=0.8) vs all 520 mlabonne/harmful_behaviors prompts",
)
meta["items_sha256"] = hashlib.sha256(json.dumps(items, sort_keys=True).encode()).hexdigest()
json.dump(dict(meta=meta, items=items), open(sys.argv[1], "w"), indent=1)
print(Counter(i["set"] for i in items))
print("overlap:", Counter((i["set"], i["advbench_overlap"]) for i in items if i["advbench_overlap"]))
print("code_weapon sources:", Counter(i["source"] for i in items if i["set"] == "code_weapon"))
print("total", len(items))
