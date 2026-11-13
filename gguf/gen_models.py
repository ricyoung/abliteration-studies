#!/usr/bin/env python3
"""Build models.json: the list of paper 33 / paper 30 models to convert to GGUF, with the numbers shown on the cards.

Sources (read-only):
  paper 33: results/bench*/<label>/hf_repo.txt (46 repos), results/analysis/h1h2_cells.csv (WildGuard, all 972 harmful),
            review_sel_kl_runs.csv (dev refusals and first-token KL of the selected trial), final_gmpoa_control.csv
  paper 30: hf_release/stage/<repo>/ (13 repos), analysis/cells.csv
Excluded: the five Gemma-2-27B variants (withheld; private repositories).
Ollama: OLMo-2-32B variants are gated on Hugging Face, and the Ollama registry cannot gate, so they are not pushed there.
"""
import csv, glob, json, os, re

P33 = "/data/heretic_2/33_2611_does_refusal_removal_get_harder"
P30 = "/data/heretic_2/paper30/v3_2611"
OUT = "/data/heretic_2/gguf/models.json"
T33 = ("Is Abliteration Getting Harder? Not Uniformly: Refusal Removal by Direction-Based Ablation and ARA in Heretic "
       "on the Qwen3.8, Gemma 4, Olmo 3.1 and Mistral Small 3.2 LLMs Versus Their Predecessors")
T30 = ("The Selection Rule Can Reverse LLM Abliteration Tool Comparisons: Heretic 1.x, Heretic 2.0, OBLITERATUS and "
       "Community Releases on Qwen2.5 and Qwen2.5-Coder")

FAM = {  # analysis model key -> base repo, display, params (B), licence family (Ollama library name), flags
    "qwen2.5-32b": ("Qwen/Qwen2.5-32B-Instruct", "Qwen2.5-32B-Instruct", 32.8, "qwen2.5", {}),
    "qwen3.8-27b": ("Qwen/Qwen3.8-27B", "Qwen3.8-27B", 27.8, "qwen3.8", {"image_text": True}),
    "gemma-4-31b": ("google/gemma-4-31B-it", "Gemma-4-31B-it", 31.3, "gemma4", {"image_text": True}),
    "olmo-2-32b": ("allenai/OLMo-2-0325-32B-Instruct", "OLMo-2-0325-32B-Instruct", 32.2, "olmo-3.1", {"gated": True}),
    "olmo-3.1-32b": ("allenai/Olmo-3.1-32B-Instruct", "Olmo-3.1-32B-Instruct", 32.2, "olmo-3.1", {}),
    "mistral-small-2409": ("mistralai/Mistral-Small-Instruct-2409", "Mistral-Small-Instruct-2409", 22.2, "mistral-small", {}),
    "mistral-small-3.2": ("mistralai/Mistral-Small-3.2-24B-Instruct-2506", "Mistral-Small-3.2-24B-Instruct-2506", 24.0,
                          "mistral-small3.2", {"image_text": True}),
    "qwen2.5-7b": ("Qwen/Qwen2.5-7B-Instruct", "Qwen2.5-7B-Instruct", 7.6, "qwen2.5", {}),
    "qwen3.5-4b": ("Qwen/Qwen3.5-4B", "Qwen3.5-4B", 4.7, "qwen3.5", {"image_text": True}),
    "qwen2.5-14b": ("Qwen/Qwen2.5-14B-Instruct", "Qwen2.5-14B-Instruct", 14.8, "qwen2.5", {}),
    "qwen2.5-coder-7b": ("Qwen/Qwen2.5-Coder-7B-Instruct", "Qwen2.5-Coder-7B-Instruct", 7.6, "qwen2.5-coder", {}),
    "qwen2.5-coder-14b": ("Qwen/Qwen2.5-Coder-14B-Instruct", "Qwen2.5-Coder-14B-Instruct", 14.8, "qwen2.5-coder", {}),
    "qwen2.5-coder-32b": ("Qwen/Qwen2.5-Coder-32B-Instruct", "Qwen2.5-Coder-32B-Instruct", 32.8, "qwen2.5-coder", {}),
}
METHOD = {
    "dim": "difference-in-means (DiM) refusal-direction ablation, tuned by Heretic's search",
    "mpoa": "projected, magnitude-preserving abliteration (MPOA, Heretic's default)",
    "som": "self-organising-map (SOM) multi-direction ablation (plug-in of the study's Heretic fork)",
    "ara": "Arbitrary-Rank Ablation (ARA, Heretic's optimisation-based method)",
    "jspace": "Jacobian-space (J-space) direction ablation (plug-in of the study's Heretic fork)",
    "gmpoa": "MPOA corrected for the post-norm output gains (gamma-MPOA control; plug-in of the study's Heretic fork)",
}

def pct(x):
    return None if x in (None, "") else round(100 * float(x), 1)

models = []
# ---------------- paper 33
cells = {r["label"]: r for r in csv.DictReader(open(f"{P33}/results/analysis/h1h2_cells.csv"))
         if r["judge"] == "wg" and r["subset"] == "all"}
sel = {r["label"]: r for r in csv.DictReader(open(f"{P33}/results/analysis/review_sel_kl_runs.csv")) if r["role"] == "primary"}
gm = {r["cell"]: r for r in csv.DictReader(open(f"{P33}/results/analysis/final_gmpoa_control.csv"))}
repos = {}
for f in glob.glob(f"{P33}/results/**/hf_repo.txt", recursive=True) + glob.glob(f"{P33}/runs/**/hf_repo.txt", recursive=True):
    label = os.path.basename(os.path.dirname(f)); repos[label] = open(f).read().strip()
assert len(repos) == 46, len(repos)
for label, repo in sorted(repos.items()):
    mk, meth = label.split("__")
    if mk == "gemma-2-27b":
        continue  # withheld
    base, disp, params, lic, flags = FAM[mk]
    name = repo.split("/", 1)[1]
    local = sorted(glob.glob(f"/data/heretic_2/models/paper33/ours/richardyoung__{name}@*"))
    if meth == "gmpoa":
        c = gm[label]; rate, k, n = pct(c["wg_rate"]), int(c["wg_k"]), int(c["wg_n"]); orig = pct(gm[f"{mk}__base"]["wg_rate"])
        s = None
    else:
        c = cells[label]; rate, k, n, orig = pct(c["rate"]), int(c["k"]), int(c["n"]), pct(c["orig_rate"])
        s = sel.get(label)
    models.append(dict(
        name=name, src_repo=repo, paper=33, paper_title=T33, label=label, base_repo=base, base_display=disp, params_b=params,
        license_family=lic, method=METHOD[meth], tool="Heretic 2.0", gated=bool(flags.get("gated")),
        image_text=bool(flags.get("image_text")), ollama=not flags.get("gated"), ollama_name="richardyoung/" + name.lower(),
        local_src=local[0] if local else None,
        heldout_refusal_pct=rate, heldout_k=k, heldout_n=n, original_refusal_pct=orig,
        dev_refusals=(int(float(s["sel_refusals"])) if s else None), dev_kl=(round(float(s["sel_kl"]), 3) if s else None),
        selection_rule="fewest dev keyword refusals at first-token KL <= 0.1 (100-trial search)"))
assert len([m for m in models if m["paper"] == 33]) == 41

# ---------------- paper 30
c30 = {r["label"]: r for r in csv.DictReader(open(f"{P30}/analysis/cells.csv"))}
for d in sorted(glob.glob(f"{P30}/hf_release/stage/*/")):
    name = os.path.basename(d.rstrip("/"))
    m = re.match(r"(Qwen2\.5(?:-Coder)?-(\d+)B)-Instruct-heretic1x-(general|code)(-kl0\.1)?$", name); assert m, name
    mk = m.group(1).lower(); line = m.group(3); kl = bool(m.group(4))
    label = f"{mk}__heretic1x-{line}" + ("-kl0.1" if kl else "")
    base, disp, params, lic, flags = FAM[mk]
    c = c30[label]; b = c30[f"{mk}__base"]
    local = f"/data/heretic_2/models/merged_models/{base.replace('/', '__')}__{line}" + ("__kl0.1" if kl else "")
    models.append(dict(
        name=name, src_repo="richardyoung/" + name, paper=30, paper_title=T30, label=label, base_repo=base, base_display=disp,
        params_b=params, license_family=lic,
        method=("rank-1 directional ablation with " + ("general" if line == "general" else "code-derived") + " refusal directions"
                + (", re-exported from the same search at a dev-KL cap of 0.1" if kl else "")),
        tool="Heretic 1.x (heretic-llm 1.4.0)", gated=False, image_text=False, ollama=True,
        ollama_name="richardyoung/" + name.lower(), local_src=local if os.path.isdir(local) else None,
        heldout_refusal_pct=pct(c["wg_rate"]), heldout_k=int(float(c["wg_k"])), heldout_n=int(float(c["wg_n"])),
        original_refusal_pct=pct(b["wg_rate"]),
        dev_refusals=(int(float(c["dev_refusals"])) if c.get("dev_refusals") not in (None, "") else None),
        dev_kl=(round(float(c["dev_kl"]), 4) if c.get("dev_kl") not in (None, "") else None),
        selection_rule=c["selection_rule"]))
assert len([m for m in models if m["paper"] == 30]) == 13
models.sort(key=lambda m: (m["params_b"], m["paper"], m["name"]))
json.dump(models, open(OUT, "w"), indent=1)
print(len(models), "models ->", OUT)
print("to Ollama:", sum(m["ollama"] for m in models), "| gated on HF:", sum(m["gated"] for m in models),
      "| no local copy:", [m["name"] for m in models if not m["local_src"]])
for m in models[:3] + models[-2:]:
    print({k: m[k] for k in ("name", "label", "params_b", "heldout_refusal_pct", "original_refusal_pct", "dev_refusals", "dev_kl", "local_src")})
