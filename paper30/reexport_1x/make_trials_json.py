#!/usr/bin/env python3
"""Build trials.json: original (KL<=0.01) selection vs paper-33 rule (KL<=0.1) for the 1.x studies."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import journal_1x as J

ROOT = "/data/heretic_2/results/heretic"
STUDIES = sys.argv[1:] or ["Qwen__Qwen2.5-32B-Instruct__general", "Qwen__Qwen2.5-32B-Instruct__code"]
EXTRA = ["Qwen__Qwen2.5-7B-Instruct__general", "Qwen__Qwen2.5-7B-Instruct__code"]

out = {"rule": "paper-33: among COMPLETE trials with dev KL <= 0.1, fewest dev refusals (study objective, keyword markers on 100 dev prompts); ties -> lower KL; then lower index",
       "original_rule": "paper-32 harness heretic_batch.select_trial: Pareto front, fewest refusals with KL <= 0.01 (ties -> lower KL)",
       "studies": {}}
for name in STUDIES + EXTRA:
    d = os.path.join(ROOT, name)
    path = J.find_journal(d)
    attrs, trials = J.read_journal(path)
    res = json.load(open(os.path.join(d, "result.json")))
    ct = J.complete_trials(trials)
    # Re-derive the original selection the way the harness did (on Heretic's Pareto front).
    front = J.pareto_front(trials)
    front_m = [{"index": t["user_attrs"]["index"], "refusals": t["user_attrs"]["refusals"], "kl_divergence": t["user_attrs"]["kl_divergence"]} for t in front]
    under = [m for m in front_m if m["kl_divergence"] <= res["kl_max"]]
    orig_rederived = min(under, key=lambda m: (m["refusals"], m["kl_divergence"])) if under else None
    # Same rule applied over ALL complete trials (equivalent on the front)
    orig_all, n_001 = J.select_min_refusals_under_kl(trials, 0.01)
    sel = J.by_index(trials, res["selected"]["index"])
    rule, n_01 = J.select_min_refusals_under_kl(trials, 0.1)
    checks = {
        "result_selected_matches_journal": (sel["user_attrs"]["refusals"] == res["selected"]["refusals"] and sel["user_attrs"]["kl_divergence"] == res["selected"]["kl_divergence"]),
        "rederived_on_pareto_front_equals_result": (orig_rederived is not None and orig_rederived["index"] == res["selected"]["index"]),
        "rule_over_all_complete_trials_at_0.01_equals_result": orig_all["user_attrs"]["index"] == res["selected"]["index"],
        "pareto_front_equals_result_pareto": sorted(front_m, key=lambda m: m["index"]) == sorted(res["pareto"], key=lambda m: m["index"]),
        "rule_trial_on_pareto_front": rule["user_attrs"]["index"] in [m["index"] for m in front_m],
    }
    out["studies"][name] = {
        "journal": path,
        "result_json": os.path.join(d, "result.json"),
        "model_in_journal": J.settings_dict(attrs)["model"],
        "n_trials_total": len(trials),
        "n_complete": len(ct),
        "n_complete_kl_le_0.01": n_001,
        "n_complete_kl_le_0.1": n_01,
        "base_refusals": sel["user_attrs"].get("base_refusals"),
        "n_dev_bad_prompts": sel["user_attrs"].get("n_bad_prompts"),
        "original_selected": J.trial_summary(sel),
        "original_selected_result_json": res["selected"],
        "rule_kl_le_0.1": J.trial_summary(rule),
        "checks": checks,
        "role": "deliverable" if name in STUDIES else "validation/reference",
    }
    print(f"{name}: orig #{sel['user_attrs']['index']} ref={sel['user_attrs']['refusals']} kl={sel['user_attrs']['kl_divergence']:.5f} | "
          f"rule #{rule['user_attrs']['index']} ref={rule['user_attrs']['refusals']} kl={rule['user_attrs']['kl_divergence']:.5f} "
          f"scope={rule['params']['direction_scope']} | complete={len(ct)} kl<=0.1: {n_01} | checks={checks}")
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "trials.json"), "w"), indent=2)
