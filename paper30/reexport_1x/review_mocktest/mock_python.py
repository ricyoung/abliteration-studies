import sys, os, json, shutil
args = sys.argv[1:]
if args[:1] == ["-c"]:
    code = args[1]
    if "heretic-llm" in code and "assert" in code:
        sys.exit(0 if os.path.exists(os.path.join(os.path.dirname(os.path.dirname(sys.executable)), "x")) else 0)
    if "snapshot_download" in code:
        print(os.environ["MOCK_BASE"]); sys.exit(0)
    exec(code); sys.exit(0)
if args[:1] == ["-"]:
    src = sys.stdin.read()
    if "sm_120 kernel smoke test" in src:
        print("venv ok (mock)"); sys.exit(0)
    sys.argv = ["-"] + args[1:]
    exec(compile(src, "<stdin>", "exec")); sys.exit(0)
if args and args[0].endswith("export_1x_trial.py"):
    a = args[1:]; jobs = []; rec = None; gate = []; study = None; adapter_only = []
    for i, x in enumerate(a):
        if x == "--job": k, v = a[i+1].split("=", 1); jobs.append((int(k), v))
        if x == "--record-dir": rec = a[i+1]
        if x == "--gate": gate.append(int(a[i+1]))
        if x == "--study": study = a[i+1]
        if x == "--adapter-only": adapter_only.append(int(a[i+1]))
    name = "general" if study.endswith("general") else "code"
    print("MOCK EXPORT", name, jobs, "gate", gate, "adapter_only", adapter_only, "argv-has-tol", "--max-refusal-diff" in a, a[a.index("--max-refusal-diff")+1], a[a.index("--max-refusal-diff-journal")+1])
    os.makedirs(rec, exist_ok=True)
    for idx, d in jobs:
        os.makedirs(os.path.join(d, "adapter"), exist_ok=True)
        open(os.path.join(d, "adapter", "adapter_model.safetensors"), "w").write("a")
        verdict = os.environ.get(f"MOCK_VERDICT_{name}_{idx}", "PASS (mock)")
        prov = {"trial_index": idx, "study_path": study, "journal_sha256": "j", "trial_direction_index": 1.0,
                "trial_parameters": {}, "journal_values": {}, "reevaluated": {"refusals": 50, "kl_divergence": 0.06, "refusal_flags": [1]},
                "adapter_sha256": "s", "base_model_path": os.environ["MOCK_BASE"], "base_model_revision": "r",
                "refusal_directions_sha256": "d", "validation_verdict": verdict, "validation_reasons": [],
                "validation": {"eval": {"refusals_journal": 50, "refusals_reeval": 52, "kl_journal": 0.064, "kl_reeval": 0.063},
                               "adapter": {"n_edited_new": 3, "n_edited_ref": 3}},
                "datasets": {"good_prompts": {"journal": "x"}}}
        if idx not in adapter_only:
            open(os.path.join(d, "tensor_hashes.json"), "w").write("{}")
            open(os.path.join(d, "model-00001-of-00001.safetensors"), "w").write("m")
        json.dump(prov, open(os.path.join(d, "PROVENANCE.json"), "w"))
        rd = os.path.join(rec, f"trial{idx}__{os.path.basename(d.rstrip('/'))}"); os.makedirs(rd, exist_ok=True)
        shutil.copy(os.path.join(d, "PROVENANCE.json"), rd)
        if idx in gate:
            if os.path.exists(os.path.join(d, "model-00001-of-00001.safetensors")): os.remove(os.path.join(d, "model-00001-of-00001.safetensors"))
            if verdict == "FAIL": print("MOCK GATE FAIL"); sys.exit(3)
    sys.exit(0)
print("mock python: unhandled", args); sys.exit(1)
