"""Generate the fixed evaluation subsets (run ONCE; output is committed as subsets.json).

Every model in the paper is evaluated on exactly these document indices.
Indices are positions in lm-eval's `task.eval_docs` (after the task's process_docs),
which is what lm-eval's `samples=` argument expects. For each picked doc we also store
a stable id (question_id / task_id / key / question hash) so bench.py can verify that
the dataset has not shifted underneath us.

Usage: HF_ALLOW_CODE_EVAL=1 .venv/bin/python make_subsets.py [--out subsets.json]
"""

import os
import argparse
import hashlib
import json
import random
from datetime import datetime, timezone

os.environ.setdefault("HF_ALLOW_CODE_EVAL", "1")  # only to load the humaneval/mbpp task objects; no code is run here
from lm_eval.tasks import TaskManager, get_task_dict

SEED = 20261104
MMLU_PRO_N = 1000
GSM8K_N = 500
SMOKE_N = 20

MMLU_PRO_SUBJECTS = [
    "biology", "business", "chemistry", "computer_science", "economics", "engineering",
    "health", "history", "law", "math", "other", "philosophy", "physics", "psychology",
]


def doc_id(task_name: str, doc: dict) -> str:
    if task_name.startswith("mmlu_pro"):
        return str(doc["question_id"])
    if task_name.startswith("humaneval"):
        return doc["task_id"]
    if task_name.startswith("mbpp"):
        return str(doc["task_id"])
    if task_name == "ifeval":
        return str(doc["key"])
    if task_name == "gsm8k":
        return hashlib.sha1(doc["question"].encode()).hexdigest()[:12]
    raise KeyError(task_name)


def load_docs(task_names):
    td = get_task_dict(task_names, TaskManager())
    return {name: list(task.eval_docs) for name, task in td.items()}


def largest_remainder(sizes: dict, total: int) -> dict:
    """Proportional allocation of `total` over strata, summing exactly to total."""
    n = sum(sizes.values())
    raw = {k: total * v / n for k, v in sizes.items()}
    alloc = {k: int(r) for k, r in raw.items()}
    rest = total - sum(alloc.values())
    for k in sorted(raw, key=lambda k: raw[k] - alloc[k], reverse=True)[:rest]:
        alloc[k] += 1
    return alloc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="subsets.json")
    args = ap.parse_args()
    rng = random.Random(SEED)
    out = {
        "seed": SEED,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "note": "indices = positions in lm-eval task.eval_docs; ids = stable doc ids for verification",
        "benchmarks": {},
    }

    # --- MMLU-Pro: stratified (proportional) by subject, 1000 of 12032 test questions
    mp_tasks = [f"mmlu_pro_{s}" for s in MMLU_PRO_SUBJECTS]
    docs = load_docs(mp_tasks)
    alloc = largest_remainder({t: len(docs[t]) for t in mp_tasks}, MMLU_PRO_N)
    paper, order = {}, []
    for t in mp_tasks:
        idx = sorted(rng.sample(range(len(docs[t])), alloc[t]))
        paper[t] = idx
        order += [(t, i) for i in idx]
    rng.shuffle(order)  # smoke = first 20 of a shuffled global order (spans subjects)
    smoke = {}
    for t, i in order[:SMOKE_N]:
        smoke.setdefault(t, []).append(i)
    out["benchmarks"]["mmlu_pro"] = {
        "population": {t: len(docs[t]) for t in mp_tasks},
        "paper": paper,
        "smoke": {t: sorted(v) for t, v in smoke.items()},
        "ids": {t: [doc_id(t, docs[t][i]) for i in paper[t]] for t in mp_tasks},
    }

    # --- GSM8K: simple random 500 of 1319 test
    docs = load_docs(["gsm8k"])["gsm8k"]
    idx = rng.sample(range(len(docs)), GSM8K_N)
    out["benchmarks"]["gsm8k"] = {
        "population": {"gsm8k": len(docs)},
        "paper": {"gsm8k": sorted(idx)},
        "smoke": {"gsm8k": sorted(idx[:SMOKE_N])},
        "ids": {"gsm8k": [doc_id("gsm8k", docs[i]) for i in sorted(idx)]},
    }

    # --- full sets: IFEval 541, HumanEval 164, MBPP 500 (test)
    for bench, task in [("ifeval", "ifeval"), ("humaneval", "humaneval_instruct"), ("mbpp", "mbpp_instruct")]:
        docs = load_docs([task])[task]
        full = list(range(len(docs)))
        sm = sorted(rng.sample(full, SMOKE_N))
        out["benchmarks"][bench] = {
            "population": {task: len(docs)},
            "paper": {task: full},
            "smoke": {task: sm},
            "ids": {task: [doc_id(task, d) for d in docs]},
        }

    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    for b, v in out["benchmarks"].items():
        print(b, "paper n =", sum(len(x) for x in v["paper"].values()),
              "smoke n =", sum(len(x) for x in v["smoke"].values()))


if __name__ == "__main__":
    main()
