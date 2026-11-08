#!/usr/bin/env python3
"""
Non-interactive re-export of a Heretic 1.x (heretic-llm 1.4.0 = git v1.4.0 = 6ea3b8d) study trial.

Rebuilds exactly what the 1.x study did, using the 1.x code itself (imported from the heretic-llm 1.4.0 package
installed from heretic-1x/, i.e. `git archive v1.4.0`):

  1. Settings  <- the study's own user_attr "settings" (Settings.model_dump_json() written by 1.4.0 main.py),
                  with only `model` replaced by the local base path given on the command line.
  2. Model     <- heretic.model.Model(settings): same dtype list ("auto" -> bf16), device_map, LoRA rank-1 adapters.
  3. Prompts   <- heretic.utils.load_prompts(): same datasets/splits/system prompt/markers as the journal.
                  Relative local dataset paths (the "code" studies use configs/code_abliteration_data/*.txt) are
                  resolved against --datasets-root.
  4. Evaluator <- heretic.evaluator.Evaluator (only with --eval): base first-token logprobs (+ base refusals).
  5. Directions: verbatim copy of 1.4.0 main.py lines 493-534. The paper-32 configs set print_residual_geometry
                  and plot_residuals = true, so 1.x took the `needs_full_residuals` branch (get_residuals_batched +
                  fp32 .mean(dim=0)); that is --residual-mode batched (default). The Analyzer itself (plots /
                  geometry printout) does not touch the means and is skipped.
  6. Per job: model.reset_model(); model.abliterate(directions, trial.direction_index, trial.parameters)
                  exactly as 1.4.0's reset_trial_model(); optional re-evaluation with evaluator.get_score();
                  adapter saved with model.model.save_pretrained(..., max_shard_size=settings.max_shard_size)
                  (= 1.x export_strategy "adapter", which is how the original adapters were produced).
  7. Merge      : identical to paper32/scripts/merge_adapter.py (the procedure that built the original merged
                  models and the bit-identical rebuilds): bf16 base on CPU, PeftModel.from_pretrained(...)
                  .merge_and_unload(), save_pretrained(safe_serialization=True, max_shard_size="5GB") + tokenizer.

Validation (optional, per trial): compare the saved adapter with an original adapter (lora_A / lora_B of the edited
modules; direction cosine), compare per-tensor sha256 of the merged model with a reference hash file, and for
differing tensors report max-abs / relative differences against a reference merged model. --gate makes the script
stop with exit code 3 before touching later jobs when a gated job fails validation.

Usage (one process per study; jobs run in the order given):
  python export_1x_trial.py --study /path/results/heretic/Qwen__Qwen2.5-32B-Instruct__general \
      --model /path/to/hf/snapshot --model-revision 5ede1c97... \
      --job 84=/out/validate84 --job 96=/out/qwen2.5-32b__heretic1x-general-kl0.1 \
      --ref-adapter 84=/path/adapter --ref-hashes 84=/path/tensor_hashes_original.json --gate 84 \
      --eval trial --record-dir /out/records

Exit codes: 0 ok; 2 usage/input error; 3 validation failed (gated job or --fail-on-validation).
"""

import argparse
import datetime as _dt
import gc
import glob
import hashlib
import json
import os
import platform
import socket
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import journal_1x as J  # noqa: E402

EXPECTED_HERETIC_VERSION = "1.4.0"
HERETIC_COMMIT = "6ea3b8d778d047b4b3b7c5b843e21c5bea98ee8d"  # tag v1.4.0 (PyPI heretic-llm 1.4.0 wheel is byte-identical)


def log(msg):
    print(f"[export_1x {_dt.datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def sha256_file(path, bufsize=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(bufsize)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_text_list(items):
    h = hashlib.sha256()
    for s in items:
        h.update(s.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def hf_dataset_revision(repo):
    """Revision of a Hub dataset as resolved in the local HF cache (refs/main), if any."""
    if os.path.exists(repo) or repo.count("/") != 1:
        return None
    hub = os.environ.get("HF_HUB_CACHE") or os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub")
    ref = os.path.join(hub, "datasets--" + repo.replace("/", "--"), "refs", "main")
    try:
        return open(ref).read().strip()
    except OSError:
        return None


def parse_kv(values, what):
    out = {}
    for v in values or []:
        if "=" not in v:
            raise SystemExit(f"bad {what} '{v}', expected TRIAL=VALUE")
        k, val = v.split("=", 1)
        out[int(k)] = val
    return out


def tensor_hashes(model_dir):
    """Same as 33_*/scripts/tensor_hash.py: per-tensor sha256 of the raw bytes, layout independent."""
    import torch
    from safetensors import safe_open

    out = {}
    for f in sorted(glob.glob(os.path.join(model_dir, "*.safetensors"))):
        with safe_open(f, "pt") as s:
            for k in s.keys():
                t = s.get_tensor(k).contiguous()
                b = t.view(torch.uint8) if t.dtype != torch.uint8 else t
                out[k] = [str(t.dtype), list(t.shape), hashlib.sha256(b.numpy().tobytes()).hexdigest()]
    return out


def tensor_index(model_dir):
    from safetensors import safe_open

    idx = {}
    for f in sorted(glob.glob(os.path.join(model_dir, "*.safetensors"))):
        with safe_open(f, "pt") as s:
            for k in s.keys():
                idx[k] = f
    return idx


def load_tensor(idx, name):
    from safetensors import safe_open

    with safe_open(idx[name], "pt") as s:
        return s.get_tensor(name)


def compare_hashes(new, ref):
    ref = ref.get("tensors", ref)
    new_keys, ref_keys = set(new), set(ref)
    common = sorted(new_keys & ref_keys)
    differing = [k for k in common if list(new[k]) != list(ref[k])]
    return {
        "n_tensors_new": len(new_keys),
        "n_tensors_ref": len(ref_keys),
        "only_in_new": sorted(new_keys - ref_keys)[:20],
        "only_in_ref": sorted(ref_keys - new_keys)[:20],
        "n_common": len(common),
        "n_identical": len(common) - len(differing),
        "n_differing": len(differing),
        "differing": differing,
        "bit_identical": not differing and new_keys == ref_keys,
    }


class AdapterMergedRef:
    """Reference merged tensors computed in memory from base + an original adapter, with the same arithmetic as
    PEFT 0.19 LoraLayer.merge on a bf16 base: W_bf16 += (B @ A) * scaling (fp32 delta, one rounding to bf16)."""

    def __init__(self, base_dir, adapter_dir):
        from safetensors.torch import load_file

        self.bi = tensor_index(base_dir)
        self.sd = load_file(os.path.join(adapter_dir, "adapter_model.safetensors"))
        cfg = json.load(open(os.path.join(adapter_dir, "adapter_config.json")))
        self.scaling = cfg["lora_alpha"] / cfg["r"]

    def get(self, name):
        base = load_tensor(self.bi, name)
        key = "base_model.model." + name[: -len(".weight")]
        A, B = self.sd.get(key + ".lora_A.weight"), self.sd.get(key + ".lora_B.weight")
        if A is None or B is None:
            return base
        w = base.clone()
        w += (B.float() @ A.float()) * self.scaling
        return w


def diff_stats(new_dir, names, ref_dir=None, ref_adapter=None, base_dir=None):
    """For each differing tensor: element-level differences new vs reference merged tensor, also relative to the
    size of the edit (ref - base). Reference = an original merged model dir, or base + original adapter in memory."""
    import torch

    ni = tensor_index(new_dir)
    ri = tensor_index(ref_dir) if ref_dir else None
    am = AdapterMergedRef(base_dir, ref_adapter) if (ri is None and ref_adapter) else None
    bi = tensor_index(base_dir) if base_dir else {}
    stats = {"_reference": ref_dir if ri else f"in-memory merge of {ref_adapter} into {base_dir}"}
    for n in names:
        a = load_tensor(ni, n).float()
        r = (load_tensor(ri, n) if ri else am.get(n)).float()
        d = a - r
        nz = int((d != 0).sum())
        rec = {
            "n_elements": a.numel(),
            "n_elements_differing": nz,
            "frac_elements_differing": nz / a.numel(),
            "max_abs_diff": float(d.abs().max()),
            "max_abs_ref": float(r.abs().max()),
            "rel_fro_diff_vs_ref": float(d.norm() / r.norm()),
        }
        if n in bi:
            b = load_tensor(bi, n).double()
            e_ref, e_new = r.double() - b, a.double() - b
            en = float(e_ref.norm())
            rec["rel_fro_diff_vs_edit"] = float(d.double().norm()) / en if en > 0 else None
            rec["edit_cosine_new_vs_ref"] = float((e_new.flatten() @ e_ref.flatten()) / (e_new.norm() * e_ref.norm() + 1e-300))
            del b, e_ref, e_new
        if nz:
            # size of each element difference in units of the bf16 spacing at the larger of the two values
            # (1 = a single rounding flip of the merged weight)
            m = torch.maximum(a.abs(), r.abs())[d != 0]
            ulp = torch.pow(2.0, torch.floor(torch.log2(m.clamp_min(1e-38))) - 7)
            k = d[d != 0].abs() / ulp
            rec["max_diff_bf16_ulps"] = float(k.max())
            rec["frac_differing_within_1ulp"] = float((k <= 1.0).double().mean())
            del m, ulp, k
        stats[n] = rec
        del a, r, d
        gc.collect()
    return stats


def selfcheck_adapter_merge(out_dir, adapter_dir, base_dir, names):
    """Check that the in-memory merge formula reproduces the saved merged tensors bit-exactly (validates AdapterMergedRef)."""
    import torch

    am = AdapterMergedRef(base_dir, adapter_dir)
    ni = tensor_index(out_dir)
    bad = [n for n in names if not torch.equal(load_tensor(ni, n), am.get(n))]
    return {"n_checked": len(names), "n_mismatch": len(bad), "mismatch": bad[:10]}


def compare_adapters(new_dir, ref_dir):
    """Compare the lora_A/lora_B of edited modules (lora_B != 0). Unedited modules carry stale lora_A from
    earlier trials in 1.4.0 (lora_B == 0 there, so they do not change the merged weights) and are ignored."""
    import torch
    from safetensors.torch import load_file

    a = load_file(os.path.join(new_dir, "adapter_model.safetensors"))
    b = load_file(os.path.join(ref_dir, "adapter_model.safetensors"))

    def edited(sd):
        return sorted(k[: -len(".lora_B.weight")] for k, v in sd.items() if k.endswith(".lora_B.weight") and bool((v != 0).any()))

    ea, eb = edited(a), edited(b)
    res = {
        "n_edited_new": len(ea),
        "n_edited_ref": len(eb),
        "edited_sets_equal": ea == eb,
        "only_new": [m for m in ea if m not in eb][:10],
        "only_ref": [m for m in eb if m not in ea][:10],
    }
    rows = []
    for m in sorted(set(ea) & set(eb)):
        A1, A2 = a[m + ".lora_A.weight"].double(), b[m + ".lora_A.weight"].double()
        B1, B2 = a[m + ".lora_B.weight"].double(), b[m + ".lora_B.weight"].double()
        rows.append({
            "module": m,
            "A_bit_identical": bool(torch.equal(a[m + ".lora_A.weight"], b[m + ".lora_A.weight"])),
            "B_bit_identical": bool(torch.equal(a[m + ".lora_B.weight"], b[m + ".lora_B.weight"])),
            "A_rel_diff": float((A1 - A2).norm() / A2.norm()),
            "B_rel_diff": float((B1 - B2).norm() / B2.norm()),
            "A_max_abs_diff": float((A1 - A2).abs().max()),
            # lora_B = -weight * v, so cos(B_new, B_ref) is the cosine between the two refusal directions used.
            "direction_cosine": float((B1.flatten() @ B2.flatten()) / (B1.norm() * B2.norm())),
            "weight_ratio": float(B1.norm() / B2.norm()),
        })
    res["n_compared"] = len(rows)
    if rows:
        res["all_bit_identical"] = all(r["A_bit_identical"] and r["B_bit_identical"] for r in rows) and res["edited_sets_equal"]
        res["n_A_bit_identical"] = sum(r["A_bit_identical"] for r in rows)
        res["n_B_bit_identical"] = sum(r["B_bit_identical"] for r in rows)
        res["min_direction_cosine"] = min(r["direction_cosine"] for r in rows)
        res["max_A_rel_diff"] = max(r["A_rel_diff"] for r in rows)
        res["max_B_rel_diff"] = max(r["B_rel_diff"] for r in rows)
        res["max_abs_weight_ratio_dev"] = max(abs(r["weight_ratio"] - 1) for r in rows)
    else:
        res["all_bit_identical"] = False
    res["modules"] = rows
    return res


def env_info():
    import importlib.metadata as md

    import torch

    pk = {}
    for p in ["heretic-llm", "torch", "transformers", "peft", "accelerate", "optuna", "datasets", "safetensors",
              "huggingface-hub", "tokenizers", "numpy", "bitsandbytes", "kernels"]:
        try:
            pk[p] = md.version(p)
        except Exception:
            pk[p] = None
    info = {"python": sys.version.split()[0], "packages": pk, "host": socket.gethostname(), "platform": platform.platform(),
            "cuda_available": torch.cuda.is_available(), "torch_cuda": torch.version.cuda,
            "torch_num_threads": torch.get_num_threads()}
    if torch.cuda.is_available():
        info["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        info["gpu_capability"] = list(torch.cuda.get_device_capability(0))
        info["tf32_matmul"] = torch.backends.cuda.matmul.allow_tf32
    return info


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--study", required=True, help="Heretic 1.x result dir (with study/*.jsonl) or the journal file")
    ap.add_argument("--model", required=True, help="local path of the base model (the journal stores a relative path)")
    ap.add_argument("--model-revision", default=None, help="HF revision of --model (recorded; inferred from .../snapshots/<sha>)")
    ap.add_argument("--datasets-root", default=None,
                    help="directory against which relative local dataset paths in the journal settings are resolved "
                         "(code studies: configs/code_abliteration_data/*.txt)")
    ap.add_argument("--job", action="append", required=True, help="TRIAL=OUTDIR (TRIAL = Heretic's 1-based trial index)")
    ap.add_argument("--adapter-only", action="append", type=int, default=[], help="TRIAL: save/compare adapter, do not merge")
    ap.add_argument("--eval", choices=["none", "trial", "full"], default="none",
                    help="re-evaluate dev refusals + KL with the 1.x evaluator; 'full' also re-counts the base refusals "
                         "(otherwise the journal's base_refusals is used for the score only)")
    ap.add_argument("--residual-mode", choices=["batched", "mean"], default="batched",
                    help="batched = 1.x needs_full_residuals branch (paper-32 configs: plot_residuals/print_residual_geometry true)")
    ap.add_argument("--batch-size", type=int, default=None, help="override the journal batch size (NOT for faithful runs)")
    ap.add_argument("--ref-adapter", action="append", help="TRIAL=DIR original adapter to compare against")
    ap.add_argument("--ref-hashes", action="append", help="TRIAL=JSON per-tensor sha256 of the original merged model")
    ap.add_argument("--ref-model", action="append", help="TRIAL=DIR original merged model (diff stats for differing tensors)")
    ap.add_argument("--gate", action="append", type=int, default=[], help="TRIAL: abort (exit 3) before later jobs if it fails validation")
    ap.add_argument("--fail-on-validation", action="store_true", help="exit 3 at the end if any validation failed")
    ap.add_argument("--accept", choices=["bit-identical", "numerical"], default="numerical",
                    help="validation criterion: bit-identical adapters/merged tensors, or numerical equivalence (thresholds below)")
    ap.add_argument("--min-direction-cosine", type=float, default=0.9995)
    ap.add_argument("--max-adapter-rel-diff", type=float, default=0.02)
    ap.add_argument("--max-refusal-diff", type=int, default=3, help="ours vs ORIGINAL adapter evaluated on this machine")
    ap.add_argument("--max-kl-abs-diff", type=float, default=0.002)
    ap.add_argument("--max-kl-rel-diff", type=float, default=0.15)
    ap.add_argument("--max-refusal-diff-journal", type=int, default=5,
                    help="ours vs the journal values (June run on another machine/stack)")
    ap.add_argument("--max-kl-abs-diff-journal", type=float, default=0.002)
    ap.add_argument("--max-kl-rel-diff-journal", type=float, default=0.30)
    ap.add_argument("--eval-ref-adapter", action="store_true",
                    help="also evaluate the ORIGINAL adapter (--ref-adapter) on this machine: separates hardware/stack "
                         "noise (original adapter here vs journal) from reconstruction error (ours vs original adapter here)")
    ap.add_argument("--load-directions", default=None, help="diagnostics only: load refusal directions from a .pt instead of computing them")
    ap.add_argument("--defer-merge", action="store_true", help="free the heretic model before merging (low-RAM hosts)")
    ap.add_argument("--keep-merged", action="append", type=int, default=None,
                    help="TRIAL: keep merged model dir (default: keep all except gated validation jobs, whose merged copy is deleted after hashing)")
    ap.add_argument("--directions-file", default=None, help="save the refusal directions here (.pt)")
    ap.add_argument("--record-dir", default=None, help="also copy PROVENANCE/hashes/validation JSON here (survives model deletion)")
    ap.add_argument("--cpu", action="store_true", help="force CPU (sets CUDA_VISIBLE_DEVICES='')")
    ap.add_argument("--threads", type=int, default=None)
    args = ap.parse_args()

    if args.cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    t_start = time.time()
    jobs = []
    for j in args.job:
        k, v = j.split("=", 1)
        jobs.append((int(k), os.path.abspath(v)))
    ref_adapter = parse_kv(args.ref_adapter, "--ref-adapter")
    ref_hashes = parse_kv(args.ref_hashes, "--ref-hashes")
    ref_model = parse_kv(args.ref_model, "--ref-model")

    # ---- study / journal
    journal = J.find_journal(args.study)
    study_attrs, trials = J.read_journal(journal)
    settings_json = study_attrs["settings"]
    settings_raw = json.loads(settings_json)
    job_trials = {}
    for idx, _ in jobs:
        t = J.by_index(trials, idx)
        if t["state"] != 1:
            raise SystemExit(f"trial {idx} is not COMPLETE ({t['state_name']})")
        job_trials[idx] = t
    log(f"journal {journal}: {len(trials)} trials; jobs {[(i, job_trials[i]['user_attrs']['refusals'], round(job_trials[i]['user_attrs']['kl_divergence'], 5)) for i, _ in jobs]}")

    model_path = os.path.abspath(args.model)
    if not os.path.isfile(os.path.join(model_path, "config.json")):
        raise SystemExit(f"--model {model_path} has no config.json")
    revision = args.model_revision
    if revision is None and "/snapshots/" in model_path:
        revision = model_path.rstrip("/").split("/snapshots/")[-1].split("/")[0]

    # ---- heretic 1.x imports (after env setup)
    import importlib.metadata as md

    import torch
    import torch.nn.functional as F
    import transformers

    if args.threads:
        torch.set_num_threads(args.threads)
    hv = md.version("heretic-llm")
    if hv != EXPECTED_HERETIC_VERSION:
        raise SystemExit(f"heretic-llm {hv} installed, expected {EXPECTED_HERETIC_VERSION} (install ./heretic-1x)")
    from heretic.config import Settings
    from heretic.evaluator import Evaluator
    from heretic.model import AbliterationParameters, Model
    from heretic.system import empty_cache
    from heretic.utils import load_prompts, set_seed
    import heretic.model as heretic_model_mod

    # Same global setup as 1.4.0 main.run()
    torch.set_grad_enabled(False)
    torch._dynamo.config.cache_size_limit = 64
    transformers.logging.set_verbosity_error()

    # pydantic-settings 2.13 runs the Settings sources (CLI args, HERETIC_* env, ./config.toml) even inside
    # model_validate_json; the journal JSON is the init source and has top priority (as on 1.4.0 "continue"), but the
    # excluded fields would otherwise be read from our argv / cwd. Validate in a neutral context.
    import tempfile

    saved_argv, saved_cwd = sys.argv, os.getcwd()
    saved_env = {k: os.environ.pop(k) for k in list(os.environ) if k.startswith("HERETIC_")}
    with tempfile.TemporaryDirectory() as neutral:
        try:
            sys.argv = [sys.argv[0]]
            os.chdir(neutral)
            settings = Settings.model_validate_json(settings_json)
        finally:
            sys.argv = saved_argv
            os.chdir(saved_cwd)
            os.environ.update(saved_env)
    # Round-trip check: the reconstructed Settings dump must equal the journal's (proves the field set matches 1.4.0).
    roundtrip_ok = json.loads(settings.model_dump_json()) == settings_raw
    if not roundtrip_ok:
        raise SystemExit("Settings round-trip mismatch: installed heretic does not match the study's Settings schema")
    settings.model = model_path
    settings.model_commit = None
    if args.batch_size:
        log(f"WARNING: overriding batch_size {settings.batch_size} -> {args.batch_size} (not faithful)")
        settings.batch_size = args.batch_size

    dataset_resolution = {}
    for key in ["good_prompts", "bad_prompts", "good_evaluation_prompts", "bad_evaluation_prompts"]:
        spec = getattr(settings, key)
        p = spec.dataset
        if not os.path.isabs(p) and not os.path.exists(p) and args.datasets_root:
            cand = os.path.join(args.datasets_root, p)
            if os.path.exists(cand):
                spec.dataset = os.path.abspath(cand)
        dataset_resolution[key] = {"journal": p, "used": spec.dataset, "split": spec.split, "column": spec.column,
                                   "sha256_file": sha256_file(spec.dataset) if os.path.isfile(spec.dataset) else None,
                                   "hf_cached_revision": hf_dataset_revision(spec.dataset)}
        if not os.path.exists(spec.dataset) and (p.startswith("configs/") or p.endswith(".txt")):
            raise SystemExit(f"{key}: local dataset '{p}' not found (pass --datasets-root)")

    set_seed(settings.seed)
    log(f"settings: model={model_path} rev={revision} batch_size={settings.batch_size} dtypes={settings.dtypes} "
        f"device_map={settings.device_map} residual_mode={args.residual_mode}")

    # ---- model + prompts (1.4.0 main.run order)
    t0 = time.time()
    model = Model(settings)
    log(f"model loaded in {time.time() - t0:.0f}s; dtype={model.dtype}; device={model.model.device}; "
        f"layers={len(model.get_layers())}; components={model.get_abliterable_components()}")
    good_prompts = load_prompts(settings, settings.good_prompts)
    bad_prompts = load_prompts(settings, settings.bad_prompts)
    for key, pr in [("good_prompts", good_prompts), ("bad_prompts", bad_prompts)]:
        dataset_resolution[key]["n"] = len(pr)
        dataset_resolution[key]["sha256_prompts"] = sha256_text_list([x.system + "\x01" + x.user for x in pr])
    log(f"prompts: good={len(good_prompts)} bad={len(bad_prompts)}")

    first = job_trials[jobs[0][0]]["user_attrs"]
    journal_base_refusals = first.get("base_refusals")
    evaluator = None
    base_refusals_recounted = None
    if args.eval != "none":
        t0 = time.time()
        if args.eval == "full":
            evaluator = Evaluator(settings, model)
            base_refusals_recounted = evaluator.base_refusals
        else:
            orig = Evaluator.count_refusals
            Evaluator.count_refusals = lambda self: journal_base_refusals
            try:
                evaluator = Evaluator(settings, model)
            finally:
                Evaluator.count_refusals = orig
        for key, pr in [("good_evaluation_prompts", evaluator.good_prompts), ("bad_evaluation_prompts", evaluator.bad_prompts)]:
            dataset_resolution[key]["n"] = len(pr)
            dataset_resolution[key]["sha256_prompts"] = sha256_text_list([x.system + "\x01" + x.user for x in pr])
        log(f"evaluator ready in {time.time() - t0:.0f}s; base_refusals journal={journal_base_refusals} recounted={base_refusals_recounted}")
        n_bad_journal = first.get("n_bad_prompts")
        if n_bad_journal is not None and len(evaluator.bad_prompts) != n_bad_journal:
            raise SystemExit(f"bad evaluation prompts: {len(evaluator.bad_prompts)} loaded, journal says {n_bad_journal} -> wrong dataset files")

    # ---- refusal directions: verbatim logic of heretic 1.4.0 main.py (lines 493-534)
    t0 = time.time()
    if args.load_directions:
        refusal_directions = torch.load(args.load_directions)
        log(f"LOADED refusal directions from {args.load_directions} (diagnostic mode)")
    elif args.residual_mode == "batched":
        good_residuals = model.get_residuals_batched(good_prompts)
        bad_residuals = model.get_residuals_batched(bad_prompts)
        good_means = good_residuals.mean(dim=0)
        bad_means = bad_residuals.mean(dim=0)
        del good_residuals, bad_residuals
    else:
        good_means = model.get_residuals_mean(good_prompts)
        bad_means = model.get_residuals_mean(bad_prompts)
    if not args.load_directions:
        refusal_directions = F.normalize(bad_means - good_means, p=2, dim=1)
    if settings.orthogonalize_direction and not args.load_directions:
        good_directions = F.normalize(good_means, p=2, dim=1)
        projection_vector = torch.sum(refusal_directions * good_directions, dim=1)
        refusal_directions = refusal_directions - projection_vector.unsqueeze(1) * good_directions
        refusal_directions = F.normalize(refusal_directions, p=2, dim=1)
        del good_directions, projection_vector
    if not args.load_directions:
        del good_means, bad_means
    empty_cache()
    dir_sha = hashlib.sha256(refusal_directions.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
    log(f"refusal directions {tuple(refusal_directions.shape)} on {refusal_directions.device} in {time.time() - t0:.0f}s sha256={dir_sha[:16]}")
    if args.directions_file:
        os.makedirs(os.path.dirname(os.path.abspath(args.directions_file)), exist_ok=True)
        torch.save(refusal_directions.detach().cpu(), args.directions_file)

    env = env_info()
    common = {
        "tool": "heretic-llm 1.4.0 (git v1.4.0 6ea3b8d), non-interactive re-export",
        "heretic_commit": HERETIC_COMMIT,
        "heretic_version_installed": hv,
        "heretic_package_file": heretic_model_mod.__file__,
        "script": os.path.abspath(__file__),
        "script_sha256": sha256_file(os.path.abspath(__file__)),
        "study_path": os.path.abspath(args.study),
        "journal": os.path.abspath(journal),
        "journal_sha256": sha256_file(journal),
        "journal_settings": settings_raw,
        "settings_roundtrip_identical": roundtrip_ok,
        "settings_overrides": {"model": model_path, "batch_size": args.batch_size},
        "base_model_path": model_path,
        "base_model_revision": revision,
        "base_config_sha256": sha256_file(os.path.join(model_path, "config.json")),
        "datasets": dataset_resolution,
        "residual_mode": args.residual_mode,
        "directions_loaded_from": args.load_directions,
        "refusal_directions_sha256": dir_sha,
        "refusal_directions_shape": list(refusal_directions.shape),
        "eval_mode": args.eval,
        "base_refusals_journal": journal_base_refusals,
        "base_refusals_recounted": base_refusals_recounted,
        "environment": env,
    }

    keep = set(args.keep_merged) if args.keep_merged is not None else None
    any_fail = False
    pending_merges = []
    results = []

    def merge_and_check(idx, out_dir, adapter_dir, prov):
        """merge_adapter.py procedure + hashing + reference checks; returns validation dict entries."""
        nonlocal any_fail
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        t0 = time.time()
        tok = AutoTokenizer.from_pretrained(model_path)
        base = AutoModelForCausalLM.from_pretrained(model_path, dtype="bfloat16", device_map="cpu", low_cpu_mem_usage=True)
        merged = PeftModel.from_pretrained(base, adapter_dir).merge_and_unload()
        merged.save_pretrained(out_dir, safe_serialization=True, max_shard_size="5GB")
        tok.save_pretrained(out_dir)
        del merged, base
        gc.collect()
        prov["merge_seconds"] = round(time.time() - t0, 1)
        t0 = time.time()
        th = tensor_hashes(out_dir)
        prov["hash_seconds"] = round(time.time() - t0, 1)
        with open(os.path.join(out_dir, "tensor_hashes.json"), "w") as f:
            json.dump(th, f, indent=0, sort_keys=True)
        prov["merged_model_dir"] = out_dir
        prov["merged_n_tensors"] = len(th)
        prov["merged_shards_sha256"] = {os.path.basename(p): sha256_file(p) for p in sorted(glob.glob(os.path.join(out_dir, "*.safetensors")))}
        edited_w = [m + ".weight" for m in prov.get("edited_modules", [])]
        prov["merge_formula_selfcheck"] = selfcheck_adapter_merge(out_dir, adapter_dir, model_path, edited_w[:8])
        if idx in ref_hashes:
            ref = json.load(open(ref_hashes[idx]))
            cmp = compare_hashes(th, ref)
            cmp["ref_file"] = ref_hashes[idx]
            if cmp["n_differing"] and (idx in ref_model or idx in ref_adapter):
                log(f"trial {idx}: {cmp['n_differing']} tensors differ; computing diff stats")
                cmp["diff_stats"] = diff_stats(out_dir, cmp["differing"], ref_dir=ref_model.get(idx),
                                               ref_adapter=ref_adapter.get(idx), base_dir=model_path)
                st = [v for k, v in cmp["diff_stats"].items() if not k.startswith("_")]
                cmp["diff_summary"] = {
                    "max_abs_diff": max(x["max_abs_diff"] for x in st),
                    "max_rel_fro_diff_vs_ref": max(x["rel_fro_diff_vs_ref"] for x in st),
                    "max_rel_fro_diff_vs_edit": max((x.get("rel_fro_diff_vs_edit") or 0) for x in st),
                    "min_edit_cosine": min((x.get("edit_cosine_new_vs_ref") or 1) for x in st),
                    "max_frac_elements_differing": max(x["frac_elements_differing"] for x in st),
                    "max_diff_bf16_ulps": max((x.get("max_diff_bf16_ulps") or 0) for x in st),
                    "min_frac_differing_within_1ulp": min((x.get("frac_differing_within_1ulp") or 1) for x in st),
                }
                log(f"trial {idx}: diff summary {cmp['diff_summary']}")
            prov["validation"]["merged_hashes"] = cmp
            log(f"trial {idx}: merged tensors {cmp['n_identical']}/{cmp['n_common']} identical "
                f"(new {cmp['n_tensors_new']}, ref {cmp['n_tensors_ref']}); bit_identical={cmp['bit_identical']}")

    def finalize_validation(idx, prov):
        v = prov["validation"]
        verdict, reasons = "not-validated", []
        if v:
            ok = True
            ad = v.get("adapter")
            if ad is not None:
                if args.accept == "bit-identical":
                    if not ad["all_bit_identical"]:
                        ok = False
                        reasons.append("adapter not bit-identical")
                else:
                    if not ad["edited_sets_equal"]:
                        ok = False
                        reasons.append("edited module sets differ")
                    if ad.get("min_direction_cosine", 0) < args.min_direction_cosine:
                        ok = False
                        reasons.append(f"direction cosine {ad.get('min_direction_cosine')} < {args.min_direction_cosine}")
                    if ad.get("max_A_rel_diff", 1) > args.max_adapter_rel_diff or ad.get("max_B_rel_diff", 1) > args.max_adapter_rel_diff:
                        ok = False
                        reasons.append(f"adapter rel diff A={ad.get('max_A_rel_diff')} B={ad.get('max_B_rel_diff')} > {args.max_adapter_rel_diff}")
            mh = v.get("merged_hashes")
            if mh is not None:
                if mh["n_tensors_new"] != mh["n_tensors_ref"] or mh["only_in_new"] or mh["only_in_ref"]:
                    ok = False
                    reasons.append("merged tensor name sets differ")
                if args.accept == "bit-identical" and mh["n_differing"]:
                    ok = False
                    reasons.append(f"{mh['n_differing']} merged tensors differ")
                if args.accept == "numerical":
                    edited = set(prov.get("edited_modules", []))
                    stray = [n for n in mh["differing"] if n.rsplit(".weight", 1)[0] not in edited]
                    if stray:
                        ok = False
                        reasons.append(f"{len(stray)} differing tensors are not edited modules, e.g. {stray[:3]}")
            ev = v.get("eval")
            evo = v.get("eval_vs_original_adapter_here")
            if evo is not None:
                # same machine/stack: ours vs the original adapter -> isolates reconstruction error
                if abs(evo["refusals_diff"]) > args.max_refusal_diff:
                    ok = False
                    reasons.append(f"refusals ours {evo['refusals_ours']} vs original adapter here {evo['refusals_original_adapter']}")
                if abs(evo["kl_diff"]) > max(args.max_kl_abs_diff, args.max_kl_rel_diff * evo["kl_original_adapter"]):
                    ok = False
                    reasons.append(f"KL ours {evo['kl_ours']:.5f} vs original adapter here {evo['kl_original_adapter']:.5f}")
            # the journal values come from another machine/stack (June H200 run): always the looser tolerance
            tol_r, tol_ka, tol_kr = args.max_refusal_diff_journal, args.max_kl_abs_diff_journal, args.max_kl_rel_diff_journal
            if ev is not None:
                if abs(ev["refusals_diff"]) > tol_r:
                    ok = False
                    reasons.append(f"refusals {ev['refusals_reeval']} vs journal {ev['refusals_journal']} (tol {tol_r})")
                if abs(ev["kl_diff"]) > max(tol_ka, tol_kr * ev["kl_journal"]):
                    ok = False
                    reasons.append(f"KL {ev['kl_reeval']:.5f} vs journal {ev['kl_journal']:.5f}")
            has_cmp = ad is not None or mh is not None
            bit = has_cmp and (ad is None or ad["all_bit_identical"]) and (mh is None or mh["bit_identical"])
            if not ok:
                verdict = "FAIL"
            elif bit:
                verdict = "PASS (bit-identical)"
            elif has_cmp:
                verdict = "PASS (numerically equivalent, not bit-identical)"
            else:
                verdict = "PASS (dev re-eval within tolerance of the journal; no reference weights)"
        prov["validation_verdict"] = verdict
        prov["validation_reasons"] = reasons
        prov["validation_criteria"] = {k: getattr(args, k) for k in [
            "accept", "min_direction_cosine", "max_adapter_rel_diff", "max_refusal_diff", "max_kl_abs_diff", "max_kl_rel_diff",
            "max_refusal_diff_journal", "max_kl_abs_diff_journal", "max_kl_rel_diff_journal", "eval_ref_adapter"]}
        return verdict

    def write_records(idx, out_dir, prov):
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "PROVENANCE.json"), "w") as f:
            json.dump(prov, f, indent=1, default=str)
        if args.record_dir:
            rd = os.path.join(args.record_dir, f"trial{idx}__{os.path.basename(out_dir.rstrip('/'))}")
            os.makedirs(rd, exist_ok=True)
            for fn in ["PROVENANCE.json", "tensor_hashes.json"]:
                p = os.path.join(out_dir, fn)
                if os.path.exists(p):
                    with open(p) as src, open(os.path.join(rd, fn), "w") as dst:
                        dst.write(src.read())

    def run_eval():
        """1.4.0 Evaluator.get_score() on the current model state; also returns per-prompt refusal flags."""
        t0 = time.time()
        captured = {}
        orig_grb = model.get_responses_batched

        def grb(prompts, skip_special_tokens=False, _o=orig_grb):
            r = _o(prompts, skip_special_tokens=skip_special_tokens)
            captured["responses"] = r
            return r

        model.get_responses_batched = grb
        try:
            score, kl, refusals = evaluator.get_score()
        finally:
            model.get_responses_batched = orig_grb
        flags = [bool(evaluator.is_refusal(r)) for r in captured.get("responses", [])]
        return {"refusals": refusals, "kl_divergence": kl, "score": list(score), "refusal_flags": flags,
                "seconds": round(time.time() - t0, 1), "device": str(model.model.device)}

    def load_adapter_into_model(adapter_dir_ref):
        """Overwrite every LoRA A/B of the live 1.x model with an adapter file (same keys as save_pretrained)."""
        from safetensors.torch import load_file

        sd = load_file(os.path.join(adapter_dir_ref, "adapter_model.safetensors"))
        n = 0
        for name, module in model.model.named_modules():
            if hasattr(module, "lora_A") and "default" in getattr(module, "lora_A", {}):
                for ab in ("lora_A", "lora_B"):
                    w = getattr(module, ab)["default"].weight
                    w.data = sd[f"{name}.{ab}.weight"].to(device=w.device, dtype=w.dtype)
                n += 1
        return n

    for idx, out_dir in jobs:
        t = job_trials[idx]
        ua = t["user_attrs"]
        tj = time.time()
        os.makedirs(out_dir, exist_ok=True)
        adapter_dir = os.path.join(out_dir, "adapter")
        log(f"=== trial {idx}: scope={t['params'].get('direction_scope')} direction_index={ua['direction_index']} "
            f"journal refusals={ua['refusals']} KL={ua['kl_divergence']:.6f}")
        # 1.4.0 reset_trial_model()
        model.reset_model()
        model.abliterate(
            refusal_directions,
            ua["direction_index"],
            {k: AbliterationParameters(**v) for k, v in ua["parameters"].items()},
        )
        edited = []
        for name, module in model.model.named_modules():
            if hasattr(module, "lora_B") and "default" in getattr(module, "lora_B", {}):
                if bool((module.lora_B["default"].weight != 0).any()):
                    edited.append(name.replace("base_model.model.", "", 1))
        prov = dict(common)
        prov.update({
            "trial_index": idx,
            "optuna_trial_id": t["trial_id"],
            "trial_direction_scope": t["params"].get("direction_scope"),
            "trial_direction_index": ua["direction_index"],
            "trial_parameters": ua["parameters"],
            "trial_optuna_params_raw": t["params"],
            "journal_values": {"refusals": ua["refusals"], "kl_divergence": ua["kl_divergence"],
                               "base_refusals": ua.get("base_refusals"), "n_bad_prompts": ua.get("n_bad_prompts"),
                               "objective_values": t["values"], "datetime_complete": t["datetime_complete"]},
            "edited_modules": edited,
            "n_edited_modules": len(edited),
            "validation": {},
        })
        if evaluator is not None:
            r = run_eval()
            prov["reevaluated"] = r
            ev = {"refusals_journal": ua["refusals"], "refusals_reeval": r["refusals"], "refusals_diff": r["refusals"] - ua["refusals"],
                  "kl_journal": ua["kl_divergence"], "kl_reeval": r["kl_divergence"], "kl_diff": r["kl_divergence"] - ua["kl_divergence"],
                  "kl_rel_diff": (r["kl_divergence"] - ua["kl_divergence"]) / ua["kl_divergence"] if ua["kl_divergence"] else None}
            prov["validation"]["eval"] = ev
            log(f"trial {idx}: re-eval refusals {r['refusals']} (journal {ua['refusals']}), KL {r['kl_divergence']:.6f} "
                f"(journal {ua['kl_divergence']:.6f}) [{r['seconds']:.0f}s]")
        # 1.x export_strategy "adapter"
        model.model.save_pretrained(adapter_dir, max_shard_size=settings.max_shard_size)
        prov["adapter_dir"] = adapter_dir
        prov["adapter_sha256"] = sha256_file(os.path.join(adapter_dir, "adapter_model.safetensors"))
        if idx in ref_adapter:
            ad = compare_adapters(adapter_dir, ref_adapter[idx])
            ad["ref_dir"] = ref_adapter[idx]
            ad["ref_adapter_sha256"] = sha256_file(os.path.join(ref_adapter[idx], "adapter_model.safetensors"))
            ad["adapter_file_bit_identical"] = ad["ref_adapter_sha256"] == prov["adapter_sha256"]
            prov["validation"]["adapter"] = ad
            log(f"trial {idx}: adapter vs original: edited {ad['n_edited_new']}/{ad['n_edited_ref']} sets_equal={ad['edited_sets_equal']} "
                f"bit_identical={ad['all_bit_identical']} (A {ad.get('n_A_bit_identical')}/{ad['n_compared']}, B {ad.get('n_B_bit_identical')}/{ad['n_compared']}) "
                f"min_cos={ad.get('min_direction_cosine')} max_relA={ad.get('max_A_rel_diff')} max_relB={ad.get('max_B_rel_diff')}")
        if args.eval_ref_adapter and evaluator is not None and idx in ref_adapter:
            nmod = load_adapter_into_model(ref_adapter[idx])
            rr = run_eval()
            rr["n_modules_loaded"] = nmod
            prov["reevaluated_original_adapter_here"] = rr
            ours = prov["reevaluated"]
            fo = ours["refusal_flags"]
            prov["validation"]["eval_vs_original_adapter_here"] = {
                "refusals_ours": ours["refusals"], "refusals_original_adapter": rr["refusals"],
                "refusals_diff": ours["refusals"] - rr["refusals"],
                "n_prompts_flag_disagree": sum(a != b for a, b in zip(fo, rr["refusal_flags"])),
                "kl_ours": ours["kl_divergence"], "kl_original_adapter": rr["kl_divergence"],
                "kl_diff": ours["kl_divergence"] - rr["kl_divergence"],
                "stack_noise_refusals_vs_journal": rr["refusals"] - ua["refusals"],
                "stack_noise_kl_vs_journal": rr["kl_divergence"] - ua["kl_divergence"],
            }
            log(f"trial {idx}: ORIGINAL adapter evaluated here: refusals {rr['refusals']} KL {rr['kl_divergence']:.6f} "
                f"(journal {ua['refusals']} / {ua['kl_divergence']:.6f}; ours {ours['refusals']} / {ours['kl_divergence']:.6f}; "
                f"flag disagreements ours vs original {prov['validation']['eval_vs_original_adapter_here']['n_prompts_flag_disagree']})")
        prov["job_seconds_before_merge"] = round(time.time() - tj, 1)
        if idx in args.adapter_only:
            prov["merged"] = False
            verdict = finalize_validation(idx, prov)
            write_records(idx, out_dir, prov)
            log(f"trial {idx}: validation {verdict} {prov['validation_reasons']}")
            results.append((idx, out_dir, verdict))
            if verdict == "FAIL":
                any_fail = True
                if idx in args.gate:
                    log(f"GATE: trial {idx} failed validation -> abort before later jobs")
                    sys.exit(3)
            continue
        if args.defer_merge:
            pending_merges.append((idx, out_dir, adapter_dir, prov))
            write_records(idx, out_dir, prov)
            continue
        # gate on adapter/eval checks before spending time on the merge
        if idx in args.gate:
            pre = finalize_validation(idx, prov)
            if pre == "FAIL":
                write_records(idx, out_dir, prov)
                log(f"GATE: trial {idx} failed pre-merge validation {prov['validation_reasons']} -> abort")
                sys.exit(3)
        merge_and_check(idx, out_dir, adapter_dir, prov)
        verdict = finalize_validation(idx, prov)
        prov["merged"] = True
        write_records(idx, out_dir, prov)
        log(f"trial {idx}: validation {verdict} {prov['validation_reasons']}")
        results.append((idx, out_dir, verdict))
        if verdict == "FAIL":
            any_fail = True
        delete = (idx in args.gate) if keep is None else (idx not in keep)
        if delete:
            for p in glob.glob(os.path.join(out_dir, "*.safetensors")):
                os.remove(p)
            log(f"trial {idx}: merged weights deleted after hashing (records kept in {out_dir})")
        if verdict == "FAIL" and idx in args.gate:
            log(f"GATE: trial {idx} failed validation -> abort before later jobs")
            sys.exit(3)

    if pending_merges:
        log("freeing heretic model before deferred merges")
        del model, evaluator
        gc.collect()
        empty_cache()
        for idx, out_dir, adapter_dir, prov in pending_merges:
            merge_and_check(idx, out_dir, adapter_dir, prov)
            verdict = finalize_validation(idx, prov)
            prov["merged"] = True
            write_records(idx, out_dir, prov)
            log(f"trial {idx}: validation {verdict} {prov['validation_reasons']}")
            results.append((idx, out_dir, verdict))
            if verdict == "FAIL":
                any_fail = True

    log(f"done in {time.time() - t_start:.0f}s: " + "; ".join(f"trial {i} -> {d} [{v}]" for i, d, v in results))
    if any_fail and (args.fail_on_validation or any(i in args.gate for i, _, v in results if v == "FAIL")):
        sys.exit(3)


if __name__ == "__main__":
    main()
