"""Paper 30: verify and upload the 13 Heretic 1.x models (12 archived-cap merges + the KL<=0.1 re-export) to new public
HF repos richardyoung/<base>-heretic1x-<variant>. Resumable: a repo is skipped when its DONE marker exists.
Per model: (1) per-tensor sha256 of the local merge vs the reference hash file, (2) stage dir of symlinks + card +
licence + hashes, (3) upload_large_folder, (4) remote file list check, (5) DONE marker."""
import hashlib, json, os, sys, time, glob, torch
import pandas as pd
from safetensors import safe_open
from huggingface_hub import HfApi
H = "/data/heretic_2/paper30/v3_2611/hf_release"; MM = "/data/heretic_2/models/merged_models"
P33 = "/data/heretic_2/33_2611_does_refusal_removal_get_harder"
cells = pd.read_csv("/data/heretic_2/paper30/v3_2611/analysis/cells.csv").set_index("label")
api = HfApi(); LIC = os.path.join(H, "LICENSE_OpenRAIL_models.md")
def log(*a): print(time.strftime("%F %T"), *a, flush=True)
BASES = {"qwen2.5-7b": "Qwen2.5-7B-Instruct", "qwen2.5-coder-7b": "Qwen2.5-Coder-7B-Instruct",
         "qwen2.5-14b": "Qwen2.5-14B-Instruct", "qwen2.5-coder-14b": "Qwen2.5-Coder-14B-Instruct",
         "qwen2.5-32b": "Qwen2.5-32B-Instruct", "qwen2.5-coder-32b": "Qwen2.5-Coder-32B-Instruct"}
def ref_for(m, v):
    lab = f"{m}__heretic1x-{v}"
    if v == "general-kl0.1":
        return [f"/data/heretic_2/paper30/v3_2611/reexport_1x/results_rtx/{lab}/tensor_hashes.json"]
    if m == "qwen2.5-32b" and v == "code":
        return [f"{P33}/runs/mc-rtx-1/results/bench/{lab}/tensor_hashes_rebuild2.json"]
    if m.endswith("7b"):
        return [f"{P33}/results/bench-4090/{lab}/tensor_hashes_original.json", f"{P33}/results/bench-4090/{lab}/tensor_hashes_rebuilt.json"]
    return [f"{P33}/runs/mc-rtx-1/results/bench/{lab}/tensor_hashes_rebuilt.json"]
def local_dir(m, v):
    b = BASES[m]
    if v == "general-kl0.1": return f"{MM}/Qwen__{b}__general__kl0.1"
    if m == "qwen2.5-32b" and v == "code": return f"{MM}/Qwen__{b}__code__rebuilt_20261106"
    return f"{MM}/Qwen__{b}__{v}"
def load_ref(p):
    d = json.load(open(p)); return d.get("tensors", d)
def tensor_hashes(d):
    out = {}
    for s in sorted(glob.glob(os.path.join(d, "*.safetensors"))):
        with safe_open(s, "pt") as f:
            for k in f.keys():
                t = f.get_tensor(k); out[k] = [str(t.dtype), list(t.shape), hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()]
    return out
def pct(x): return f"{100*x:.1f}%"
def card(m, v, repo):
    base = "Qwen/" + BASES[m]; lab = f"{m}__heretic1x-{v}"; c = cells.loc[lab]; o = cells.loc[f"{m}__base"]
    kl01 = v == "general-kl0.1"; vv = "general" if kl01 else v
    variant = ("**general** direction: Heretic 1.x defaults (harmful prompts `mlabonne/harmful_behaviors`, harmless prompts `mlabonne/harmless_alpaca`)"
               if vv == "general" else
               "**code** direction: Heretic 1.x run with the code-abliteration prompt configuration (malicious vs benign coding requests)")
    sel = ("Selected by Heretic 1.x's rule: the trial with the fewest dev refusals among trials with dev KL divergence <= 0.01 (the archived cap)."
           if not kl01 else
           "**Post hoc re-export.** Same Heretic 1.x search as `Qwen2.5-32B-Instruct-heretic1x-general`, but the released trial was re-selected "
           "under the looser Heretic 2.0 cap: the trial with the fewest dev refusals among trials with dev KL <= 0.1 (trial #96). Built from the "
           "re-exported LoRA adapter (sha256 `88347ef1...`) with heretic-llm 1.4.0 (git v1.4.0 `6ea3b8d`); every tensor matches the evaluated model.")
    caps = " | ".join(f"{c[f'{b}_delta_pp']:+.1f}" for b in ["mmlu_pro", "gsm8k", "ifeval", "humaneval", "mbpp"])
    return f"""---
license: apache-2.0
license_link: https://huggingface.co/{base}/blob/main/LICENSE
language:
- en
pipeline_tag: text-generation
base_model: {base}
base_model_relation: finetune
library_name: transformers
tags:
- heretic
- heretic-1.x
- abliterated
- decensored
- research
- openrail++
---
# {repo.split('/')[1]}

A refusal-ablated ("abliterated") version of [{base}](https://huggingface.co/{base}), made with **Heretic 1.x**
(heretic-llm 1.4.0, git tag v1.4.0, `6ea3b8d`) as a rank-1 LoRA edit of `o_proj` and `down_proj`, merged into bf16 weights.
It is one of the models evaluated in:

> Richard J. Young. *The Selection Rule Can Reverse LLM Abliteration Tool Comparisons: Heretic 1.x, Heretic 2.0,
> OBLITERATUS and Community Releases on Qwen2.5 and Qwen2.5-Coder* (paper 30; arXiv link to be added).

It is released for research on abliteration and its evaluation, not as a general-purpose assistant.

## What this model is

- Variant: {variant}.
- Search: {int(c.h1x_n_trials)} Optuna trials (Heretic 1.x defaults).
- {sel}
- Selected trial: #{int(c.dev_trial)}; dev refusals {int(c.dev_refusals)}/100; dev KL divergence {c.dev_kl:.4f}
  (dev set: `{c.dev_set}`).

## Held-out evaluation (paper 30)

972 held-out harmful prompts (none used during the search), scored by the WildGuard judge; capability on five
benchmarks (percentage points, pp, relative to the original model).

| Metric | This model | Original |
|:--|:--:|:--:|
| Refusal rate, WildGuard judge [95% CI] | {pct(c.wg_rate)} [{pct(c.wg_lo)}, {pct(c.wg_hi)}] | {pct(o.wg_rate)} |
| Refusal rate, keyword markers | {pct(c.kw_rate)} | {pct(o.kw_rate)} |
| Over-refusal on benign prompts (WildGuard) | {pct(c.overrefusal_wg_rate)} | {pct(o.overrefusal_wg_rate)} |
| Mean capability change | {c.capability_mean_delta_pp:+.2f} pp | 0 |

Capability change by benchmark (pp) - MMLU-Pro | GSM8K | IFEval | HumanEval | MBPP: {caps}

## Caveats

- Each number comes from one search on one base model; the paper finds that search-to-search variation is large
  (seed-noise SD about 9 pp), so differences between single models should not be over-read.
- The selection cap matters: on Qwen2.5-32B, re-selecting the same search at dev KL <= 0.1 instead of 0.01 lowered
  residual refusal by 61.7 pp. Compare abliterated models only at a stated selection rule and realised divergence.
- Refusal is reduced, not removed, and the model can still produce harmful content. It has no added safety tuning.

## Integrity

`tensor_hashes.json` lists the SHA-256 of every tensor (layout independent). The weights here are bit-identical, tensor
by tensor, to the model evaluated in the paper. Code, run records and labels: GitHub `ricyoung/abliteration-studies`
(released with the paper).

## Licence

This model is a derivative of an Apache-2.0 base model and is released under the **Apache-2.0** licence of its base
model. In addition, it carries the use-based restrictions of the **OpenRAIL++ Licence v1.0** (`LICENSE_OpenRAIL++.md` in
this repository). Where the two conflict, the base licence prevails. It must not be deployed for, or used to generate
content for, any of the prohibited uses listed in the OpenRAIL++ licence.
"""
JOBS = [(m, v) for m in BASES for v in ("general", "code")] + [("qwen2.5-32b", "general-kl0.1")]
JOBS.sort(key=lambda j: ("kl0.1" in j[1], "32b" in j[0], j))  # small models first, re-export last
only = sys.argv[1:]
for m, v in JOBS:
    repo = f"richardyoung/{BASES[m]}-heretic1x-{v}"; tag = repo.split("/")[1]
    if only and tag not in only: continue
    done = os.path.join(H, "done", tag)
    if os.path.exists(done): log("skip (done)", repo); continue
    src = local_dir(m, v); log("START", repo, "from", src)
    hf = os.path.join(H, "hashes", tag + ".json")
    if not os.path.exists(hf):
        got = tensor_hashes(src); ok = None
        for r in ref_for(m, v):
            ref = load_ref(r)
            if set(ref) == set(got) and all(ref[k][2] == got[k][2] for k in ref): ok = r; break
        if not ok: log("FAILED hash check", repo, [os.path.basename(r) for r in ref_for(m, v)]); continue
        os.makedirs(os.path.dirname(hf), exist_ok=True); json.dump(got, open(hf, "w"), indent=0)
        log("hash OK vs", ok, len(got), "tensors")
    st = os.path.join(H, "stage", tag); os.makedirs(st, exist_ok=True)
    cfg_src = src if not v.endswith("kl0.1") else local_dir(m, "general")
    for f in os.listdir(src):
        if f.endswith(".safetensors") or f == "model.safetensors.index.json":
            d = os.path.join(st, f); os.path.lexists(d) or os.symlink(os.path.join(src, f), d)
    for f in ["config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja"]:
        d = os.path.join(st, f); os.path.lexists(d) or os.symlink(os.path.join(cfg_src, f), d)
    open(os.path.join(st, "README.md"), "w").write(card(m, v, repo))
    d = os.path.join(st, "LICENSE_OpenRAIL++.md"); os.path.lexists(d) or os.symlink(LIC, d)
    d = os.path.join(st, "tensor_hashes.json"); os.path.lexists(d) or os.symlink(hf, d)
    if "--dry" in sys.argv: log("dry", repo); continue
    api.create_repo(repo, private=False, exist_ok=True)
    api.upload_large_folder(repo_id=repo, folder_path=st, repo_type="model", print_report=False)
    remote = set(api.list_repo_files(repo)); want = {f for f in os.listdir(st) if not f.startswith(".")}
    if not want <= remote: log("FAILED remote check", repo, sorted(want - remote)); continue
    os.makedirs(os.path.dirname(done), exist_ok=True); open(done, "w").write(api.repo_info(repo).sha)
    log("DONE", repo, f"https://huggingface.co/{repo}")
log("ALL FINISHED")
