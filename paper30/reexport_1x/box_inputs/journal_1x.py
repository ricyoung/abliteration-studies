"""Stdlib-only reader for the Optuna JournalFileBackend files written by Heretic 1.x.

Heretic 1.4.0 stores, per trial, the user attrs: index (1-based trial number shown in
the UI and in result.json), direction_index (None for the "per layer" scope),
parameters ({component: {max_weight, max_weight_position, min_weight, min_weight_distance}},
min_weight already multiplied by max_weight), kl_divergence, refusals, base_refusals,
n_bad_prompts. Study user attr "settings" holds Settings.model_dump_json().
"""

import glob
import json
import os

STATE_NAMES = {0: "RUNNING", 1: "COMPLETE", 2: "PRUNED", 3: "FAIL", 4: "WAITING"}


def find_journal(study_dir):
    """study_dir may be the result dir (containing study/) or the journal file itself."""
    if os.path.isfile(study_dir):
        return study_dir
    cands = glob.glob(os.path.join(study_dir, "study", "*.jsonl")) or glob.glob(os.path.join(study_dir, "*.jsonl"))
    if len(cands) != 1:
        raise FileNotFoundError(f"expected exactly one journal under {study_dir}, found {cands}")
    return cands[0]


def read_journal(path):
    """Returns (study_user_attrs, trials) where trials is a list ordered by trial_id."""
    study_attrs = {}
    trials = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            op = r["op_code"]
            if op == 2:
                study_attrs.update(r["user_attr"])
            elif op == 4:
                trials.append({"trial_id": len(trials), "state": 0, "values": None,
                               "params_internal": {}, "distributions": {}, "user_attrs": {},
                               "datetime_start": r.get("datetime_start"), "datetime_complete": None})
            elif op == 5:
                t = trials[r["trial_id"]]
                t["params_internal"][r["param_name"]] = r["param_value_internal"]
                t["distributions"][r["param_name"]] = json.loads(r["distribution"])
            elif op == 6:
                t = trials[r["trial_id"]]
                t["state"] = r["state"]
                t["values"] = r.get("values")
                t["datetime_complete"] = r.get("datetime_complete")
            elif op == 8:
                trials[r["trial_id"]]["user_attrs"].update(r["user_attr"])
    for t in trials:
        t["state_name"] = STATE_NAMES.get(t["state"], str(t["state"]))
        t["params"] = {}
        for k, v in t["params_internal"].items():
            d = t["distributions"][k]
            if d["name"] == "CategoricalDistribution":
                t["params"][k] = d["attributes"]["choices"][int(v)]
            else:
                t["params"][k] = v
    return study_attrs, trials


def settings_dict(study_attrs):
    return json.loads(study_attrs["settings"])


def complete_trials(trials):
    return [t for t in trials if t["state"] == 1 and "kl_divergence" in t["user_attrs"]]


def by_index(trials, index):
    """Look up a trial by Heretic's 1-based 'index' user attr (the number shown in the UI/result.json)."""
    hits = [t for t in trials if t["user_attrs"].get("index") == index]
    if len(hits) != 1:
        raise KeyError(f"trial index {index}: {len(hits)} matches")
    return hits[0]


def pareto_front(trials):
    """Heretic 1.4.0's own Pareto construction (main.py): sort by (refusals, KL), keep strictly decreasing KL."""
    ct = sorted(complete_trials(trials), key=lambda t: (t["user_attrs"]["refusals"], t["user_attrs"]["kl_divergence"]))
    front, best = [], float("inf")
    for t in ct:
        kl = t["user_attrs"]["kl_divergence"]
        if kl < best:
            best = kl
            front.append(t)
    return front


def select_min_refusals_under_kl(trials, kl_max):
    """Paper-33 rule: among COMPLETE trials with KL <= kl_max, fewest refusals; ties -> lower KL
    (remaining ties -> lower index, deterministic)."""
    under = [t for t in complete_trials(trials) if t["user_attrs"]["kl_divergence"] <= kl_max]
    if not under:
        return None, 0
    best = min(under, key=lambda t: (t["user_attrs"]["refusals"], t["user_attrs"]["kl_divergence"], t["user_attrs"]["index"]))
    return best, len(under)


def trial_summary(t):
    a = t["user_attrs"]
    return {
        "index": a["index"],
        "optuna_trial_id": t["trial_id"],
        "state": t["state_name"],
        "refusals": a["refusals"],
        "n_bad_prompts": a.get("n_bad_prompts"),
        "base_refusals": a.get("base_refusals"),
        "kl_divergence": a["kl_divergence"],
        "objective_values": t["values"],
        "direction_scope": t["params"].get("direction_scope"),
        "direction_index": a["direction_index"],
        "parameters": a["parameters"],
        "optuna_params_raw": t["params"],
        "datetime_complete": t["datetime_complete"],
    }
