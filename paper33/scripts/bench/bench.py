"""Capability benchmark runner for paper 33 (lm-eval-harness, vLLM backend, HF fallback).

Called by run.sh; see README.md. Loads the model once, then runs each benchmark with
its own fixed generation settings on the fixed subsets in subsets.json.

Exit codes: 0 ok, 1 error, 3 = model failed to load with this backend (run.sh retries with hf).
"""

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Fixed protocol. Change only with a new protocol version (recorded in summary.json).
PROTOCOL_VERSION = "1"
SEEDS = dict(random_seed=0, numpy_random_seed=1234, torch_random_seed=1234,
             fewshot_random_seed=1234, vllm_seed=1234)
GREEDY = dict(do_sample=False, temperature=0.0)
MMLU_PRO_SUBJECTS = [
    "biology", "business", "chemistry", "computer_science", "economics", "engineering",
    "health", "history", "law", "math", "other", "philosophy", "physics", "psychology",
]
BENCHMARKS = {
    # name: lm-eval tasks, primary (metric, filter), fixed max_gen_toks, few-shot (task default)
    "mmlu_pro": dict(tasks=[f"mmlu_pro_{s}" for s in MMLU_PRO_SUBJECTS],
                     metric="exact_match", filter="custom-extract", max_gen_toks=2048,
                     fewshot="5-shot CoT (task default)"),
    "gsm8k": dict(tasks=["gsm8k"], metric="exact_match", filter="flexible-extract",
                  secondary=[("exact_match", "strict-match")], max_gen_toks=512,
                  fewshot="5-shot (task default)"),
    "ifeval": dict(tasks=["ifeval"], metric="prompt_level_strict_acc", filter="none",
                   secondary=[("inst_level_strict_acc", "none"), ("prompt_level_loose_acc", "none")],
                   max_gen_toks=1280, fewshot="0-shot"),
    "humaneval": dict(tasks=["humaneval_instruct"], metric="pass@1", filter="create_test",
                      max_gen_toks=1024, fewshot="0-shot"),
    "mbpp": dict(tasks=["mbpp_instruct"], metric="pass_at_1", filter="extract_code",
                 max_gen_toks=1024, fewshot="3-shot (task default)"),
}
DOC_ID_KEY = {"mmlu_pro": "question_id", "humaneval": "task_id", "mbpp": "task_id", "ifeval": "key"}


def log(msg):
    print(f"[bench {datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def sha256_file(p: Path, buf=1 << 24) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(buf):
            h.update(chunk)
    return h.hexdigest()


def resolve_model(model: str) -> dict:
    """Local dir (or HF id resolved from cache) -> path, HF commit, tree hash later."""
    info = {"model": model}
    p = Path(model)
    if p.is_dir():
        info["source"] = "local"
        info["path"] = str(p.resolve())
        try:
            r = subprocess.run(["git", "-C", str(p), "rev-parse", "HEAD"], capture_output=True, text=True)
            if r.returncode == 0:
                info["git_commit"] = r.stdout.strip()
        except Exception:
            pass
        for manifest in ("reproduce/manifest.json", "reproduce.json", "heretic.json"):
            if (p / manifest).exists():
                info["manifest"] = manifest
    else:
        info["source"] = "hf"
        from huggingface_hub import snapshot_download
        path = snapshot_download(model, allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "*.jinja", "*.py", "*.tiktoken"])
        info["path"] = path
        info["hf_commit"] = Path(path).name
    return info


def tree_hash(path: str) -> dict:
    """sha256 over (relative name, sha256) of weight + config + tokenizer files."""
    root = Path(path)
    files = sorted(f for f in root.rglob("*") if f.is_file() and (
        f.suffix in (".safetensors", ".bin", ".pt", ".json", ".model", ".jinja", ".txt", ".tiktoken")))
    h = hashlib.sha256()
    per_file, total = {}, 0
    for f in files:
        rel = str(f.relative_to(root))
        if rel.startswith((".cache", "reproduce/")):
            continue
        d = sha256_file(f.resolve())
        per_file[rel] = d
        total += f.resolve().stat().st_size
        h.update(f"{rel} {d}\n".encode())
    return {"tree_sha256": h.hexdigest(), "files": per_file, "bytes": total}


def versions() -> dict:
    from importlib.metadata import version, PackageNotFoundError
    v = {"python": sys.version.split()[0], "platform": platform.platform(), "host": platform.node()}
    for pkg in ("lm_eval", "vllm", "torch", "transformers", "datasets", "accelerate", "flashinfer-python"):
        try:
            v[pkg] = version(pkg)
        except PackageNotFoundError:
            v[pkg] = None
    try:
        import torch
        v["torch_cuda"] = torch.version.cuda
        v["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:
        pass
    try:
        v["driver"] = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                                     capture_output=True, text=True).stdout.strip()
    except Exception:
        pass
    v["env"] = {k: os.environ.get(k) for k in ("VLLM_USE_FLASHINFER_SAMPLER", "VLLM_ATTENTION_BACKEND",
                                                "BENCH_VLLM_EXTRA", "CUDA_VISIBLE_DEVICES")}
    for name in ("uv.lock", "bench.py", "subsets.json", "run.sh"):
        v[f"sha256:{name}"] = sha256_file(HERE / name) if (HERE / name).exists() else None
    return v


def select(subsets: dict, bench: str, mode: str):
    """samples dict for lm-eval, or None for 'all docs'."""
    sel = subsets["benchmarks"][bench][mode]
    pop = subsets["benchmarks"][bench]["population"]
    if mode == "paper" and all(len(sel[t]) == pop[t] for t in sel):
        return None  # full set
    return sel


def verify_ids(subsets: dict, bench: str, tasks: list[str]):
    """Abort if the dataset changed: stored doc ids must match current eval_docs."""
    from lm_eval.tasks import TaskManager, get_task_dict
    stored = subsets["benchmarks"][bench]
    td = get_task_dict(tasks, TaskManager())
    for t, task in td.items():
        docs = list(task.eval_docs)
        if len(docs) != stored["population"][t]:
            raise RuntimeError(f"{t}: dataset size {len(docs)} != recorded {stored['population'][t]}")
        idx = stored["paper"][t]
        if bench == "gsm8k":
            cur = [hashlib.sha1(docs[i]["question"].encode()).hexdigest()[:12] for i in idx]
        else:
            cur = [str(docs[i][DOC_ID_KEY[bench]]) for i in idx]
        if cur != stored["ids"][t]:
            raise RuntimeError(f"{t}: doc ids differ from subsets.json - dataset revision changed")


def build_lm(args):
    common = dict(enable_thinking=False)  # same for every model; see README "Chat template"
    if args.backend == "vllm":
        from lm_eval.models.vllm_causallms import VLLM
        extra = json.loads(os.environ.get("BENCH_VLLM_EXTRA", "{}"))
        kw = dict(pretrained=args.model_path, dtype="auto", max_model_len=args.max_model_len,
                  gpu_memory_utilization=args.gpu_util, seed=SEEDS["vllm_seed"], batch_size="auto",
                  trust_remote_code=args.trust_remote_code, **common, **extra)
        log(f"vLLM args: {kw}")
        return VLLM(**kw), kw
    from lm_eval.models.huggingface import HFLM
    kw = dict(pretrained=args.model_path, dtype="bfloat16", max_length=args.max_model_len,
              batch_size=args.hf_batch_size, trust_remote_code=args.trust_remote_code, device="cuda", **common)
    log(f"HF args: {kw}")
    return HFLM(**kw), kw


def render_prompt(tok, system: str, user: str) -> str:
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except Exception:  # templates without a system role (e.g. Gemma 2): fold it into the user turn
        msgs = [{"role": "user", "content": system + "\n\n" + user}]
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)


def teacher_forced_logprobs(lm, args, out: Path) -> dict:
    """Score the ORIGINAL model's greedy answers to benign held-out prompts under this model.
    Per item: sum/mean log p(reference answer tokens | prompt). Divergence (computed in analysis) =
    mean_tokens[log p_orig(y) - log p_this(y)], y = original model's greedy answer (teacher forcing)."""
    from vllm import SamplingParams
    spec = json.loads(Path(args.refusal_sets).read_text())
    meta = spec["meta"]
    tok = lm.model.get_tokenizer()
    refs = [json.loads(l) for l in open(args.reference_responses)]
    refs = [r for r in refs if r["kind"] == "benign" and r["response"].strip()]
    texts, starts = [], []
    for r in refs:
        p = render_prompt(tok, meta["system_prompt"], r["prompt"])
        texts.append(p + r["response"])
        starts.append(len(tok(p, add_special_tokens=False)["input_ids"]))
    sp = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=0)
    t0 = time.time()
    outs = lm.model.generate(texts, sp, use_tqdm=True)
    rdir = out / "refusal"
    rdir.mkdir(parents=True, exist_ok=True)
    tot_lp = tot_n = 0
    with open(rdir / "teacher_forced.jsonl", "w") as f:
        for r, text, st, o in zip(refs, texts, starts, outs):
            ids, plp = o.prompt_token_ids, o.prompt_logprobs
            # vLLM may prepend special tokens (e.g. BOS) that the plain encoding lacks: shift by the difference.
            n_special = len(ids) - len(tok(text, add_special_tokens=False)["input_ids"])
            first = max(1, st + n_special)  # first answer token
            lp = [plp[i][ids[i]].logprob for i in range(first, len(ids))
                  if plp[i] is not None and ids[i] in plp[i]]
            tot_lp += sum(lp); tot_n += len(lp)
            f.write(json.dumps({"id": r["id"], "set": r["set"], "n_tokens": len(lp), "sum_logprob": sum(lp)}) + "\n")
    info = {"file": str(rdir / "teacher_forced.jsonl"), "n_items": len(refs), "n_tokens": tot_n,
            "mean_logprob": tot_lp / max(1, tot_n), "reference": str(args.reference_responses),
            "wall_seconds": round(time.time() - t0, 1)}
    log(f"=== teacher-forced: {info}")
    return info


def generate_refusal_set(lm, args, out: Path) -> dict:
    """Generate one greedy response per prompt of the frozen refusal-set file (vLLM backend only)."""
    from vllm import SamplingParams
    spec = json.loads(Path(args.refusal_sets).read_text())
    meta, items = spec["meta"], spec["items"]
    tok = lm.model.get_tokenizer()
    texts = [render_prompt(tok, meta["system_prompt"], it["prompt"]) for it in items]
    sp = SamplingParams(temperature=0.0, max_tokens=meta["max_tokens"], seed=SEEDS["vllm_seed"])
    t0 = time.time()
    outs = lm.model.generate(texts, sp, use_tqdm=True)
    rdir = out / "refusal"
    rdir.mkdir(parents=True, exist_ok=True)
    with open(rdir / "responses.jsonl", "w") as f:
        for it, text, o in zip(items, texts, outs):
            c = o.outputs[0]
            f.write(json.dumps({**it, "response": c.text, "n_tokens": len(c.token_ids),
                                "response_256": tok.decode(list(c.token_ids)[:256], skip_special_tokens=True),
                                "finish_reason": c.finish_reason,
                                "rendered_prompt_sha256": hashlib.sha256(text.encode()).hexdigest()}) + "\n")
    info = {"file": str(rdir / "responses.jsonl"), "n": len(items), "items_sha256": meta["items_sha256"],
            "max_tokens": meta["max_tokens"], "wall_seconds": round(time.time() - t0, 1)}
    log(f"=== refusal set: {info}")
    return info


def jdump(obj, path: Path):
    from lm_eval.utils import handle_non_serializable
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=handle_non_serializable)


def primary_from_samples(samples: dict, metric: str, flt: str):
    vals = []
    for task_samples in samples.values():
        for s in task_samples:
            if s.get("filter") == flt and metric in s:
                v = s[metric]
                vals.append(float(v[0] if isinstance(v, list) else v))
    n = len(vals)
    if n == 0:
        return None, None, 0
    p = sum(vals) / n
    se = math.sqrt(p * (1 - p) / (n - 1)) if n > 1 else None
    return p, se, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("out_dir")
    ap.add_argument("--limit-mode", choices=["paper", "smoke"], default="paper")
    ap.add_argument("--backend", choices=["vllm", "hf"], default="vllm")
    ap.add_argument("--benchmarks", default=",".join(BENCHMARKS))
    ap.add_argument("--method", default=None, help="condition label, e.g. base, M1, M2")
    ap.add_argument("--base-model", default=None, help="base model id/path this model was derived from")
    ap.add_argument("--gpu-util", type=float, default=float(os.environ.get("BENCH_GPU_UTIL", 0.85)))
    ap.add_argument("--max-model-len", type=int, default=int(os.environ.get("BENCH_MAX_MODEL_LEN", 8192)))
    ap.add_argument("--hf-batch-size", default=os.environ.get("BENCH_HF_BATCH", "16"))
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--no-hash", action="store_true", help="skip sha256 of weight files")
    ap.add_argument("--refusal-sets", default=None,
                    help="frozen prompt file (refusal_sets.json); responses -> <out>/refusal/responses.jsonl")
    ap.add_argument("--reference-responses", default=None,
                    help="ORIGINAL model's refusal/responses.jsonl; scores its benign answers under this model (teacher forcing). "
                         "Use 'self' for the original model itself (after generation).")
    args = ap.parse_args()
    if args.benchmarks.strip() in ("", "none"):
        args.benchmarks = ""

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    benches = [b.strip() for b in args.benchmarks.split(",") if b.strip()]
    for b in benches:
        if b not in BENCHMARKS:
            sys.exit(f"unknown benchmark {b}; choose from {list(BENCHMARKS)}")
    if any(b in ("humaneval", "mbpp") for b in benches) and os.environ.get("HF_ALLOW_CODE_EVAL") != "1":
        sys.exit("HF_ALLOW_CODE_EVAL=1 is required for humaneval/mbpp (run.sh sets it)")

    subsets = json.loads((HERE / "subsets.json").read_text())
    t_start = time.time()
    summary_path = out / "summary.json"
    summary = {
        "protocol_version": PROTOCOL_VERSION,
        "started": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "limit_mode": args.limit_mode,
        "method": args.method,
        "base_model": args.base_model,
        "backend": args.backend,
        "seeds": SEEDS,
        "subset_seed": subsets["seed"],
        "decoding": {"greedy": True, **GREEDY, "apply_chat_template": True,
                     "fewshot_as_multiturn": True, "enable_thinking": False,
                     "max_model_len": args.max_model_len},
        "versions": versions(),
        "benchmarks": {},
    }

    log(f"resolving model {args.model}")
    minfo = resolve_model(args.model)
    args.model_path = minfo["path"]
    summary["model"] = args.model
    summary["model_info"] = minfo
    jdump(summary, summary_path)

    hash_result = {}
    if not args.no_hash:
        th = threading.Thread(target=lambda: hash_result.update(tree_hash(args.model_path)), daemon=True)
        th.start()

    # verify subsets before spending GPU time
    for b in benches:
        verify_ids(subsets, b, BENCHMARKS[b]["tasks"])
    log("subset ids verified against current datasets")

    t0 = time.time()
    try:
        lm, lm_kwargs = build_lm(args)
    except Exception:
        traceback.print_exc()
        log(f"MODEL LOAD FAILED with backend={args.backend}")
        summary["load_error"] = traceback.format_exc()[-4000:]
        jdump(summary, summary_path)
        sys.exit(3)
    if args.backend == "vllm":
        # vLLM 0.33's incremental detokenizer leaves SentencePiece meta-space pieces (U+2581) in the text for some
        # tokenizers (Gemma-2: runs of spaces -> "▁▁▁▁"), which breaks code indentation and word counts. HF decode is
        # correct. U+2581 never occurs in normal output, so map it back to a space for every model.
        _orig_generate = lm.model.generate

        def _generate_fixed(*a, **k):
            outs = _orig_generate(*a, **k)
            for o in outs:
                for c in o.outputs:
                    if "\u2581" in c.text:
                        c.text = c.text.replace("\u2581", " ")
            return outs
        lm.model.generate = _generate_fixed
        summary["postprocess"] = "vLLM output text: U+2581 -> space (detokenizer workaround)"
    summary["load_seconds"] = round(time.time() - t0, 1)
    summary["lm_kwargs"] = lm_kwargs
    log(f"model loaded in {summary['load_seconds']} s")

    import lm_eval
    from lm_eval.tasks import TaskManager
    tm = TaskManager()
    failed = []
    for b in benches:
        cfg = BENCHMARKS[b]
        samples = select(subsets, b, args.limit_mode)
        # lm-eval runs ALL docs of a task missing from `samples`, so only pass sampled tasks
        tasks = cfg["tasks"] if samples is None else [t for t in cfg["tasks"] if samples.get(t)]
        gen = {**GREEDY, "max_gen_toks": cfg["max_gen_toks"]}
        log(f"=== {b}: tasks={tasks} n={'all' if samples is None else sum(map(len, samples.values()))} gen={gen}")
        bdir = out / "lmeval" / b
        bdir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        try:
            res = lm_eval.simple_evaluate(
                model=lm, tasks=tasks, samples=samples, task_manager=tm,
                apply_chat_template=True, fewshot_as_multiturn=True, gen_kwargs=gen,
                log_samples=True, confirm_run_unsafe_code=True, bootstrap_iters=1000,
                random_seed=SEEDS["random_seed"], numpy_random_seed=SEEDS["numpy_random_seed"],
                torch_random_seed=SEEDS["torch_random_seed"], fewshot_random_seed=SEEDS["fewshot_random_seed"],
            )
        except Exception:
            traceback.print_exc()
            failed.append(b)
            summary["benchmarks"][b] = {"error": traceback.format_exc()[-4000:]}
            jdump(summary, summary_path)
            continue
        wall = time.time() - t0
        smp = res.pop("samples", {})
        for task, rows in smp.items():
            with open(bdir / f"samples_{task}.jsonl", "w") as f:
                from lm_eval.utils import handle_non_serializable
                for r in rows:
                    f.write(json.dumps(r, default=handle_non_serializable) + "\n")
        jdump(res, bdir / "results.json")

        score, se, n = primary_from_samples(smp, cfg["metric"], cfg["filter"])
        key = f"{cfg['metric']},{cfg['filter']}"
        if len(tasks) == 1:  # use lm-eval's own aggregate where it exists
            r = res["results"][cfg["tasks"][0]]
            score = r.get(key, score)
            se = r.get(f"{cfg['metric']}_stderr,{cfg['filter']}", se)
        entry = {"metric": key, "score": score, "stderr": se, "n": n,
                 "wall_seconds": round(wall, 1), "max_gen_toks": cfg["max_gen_toks"], "fewshot": cfg["fewshot"],
                 "lm_eval_tasks": tasks}
        for m, fl in cfg.get("secondary", []):
            r = res["results"][cfg["tasks"][0]]
            entry.setdefault("secondary", {})[f"{m},{fl}"] = r.get(f"{m},{fl}")
        if b == "mmlu_pro":
            entry["per_subject"] = {t: res["results"][t].get(key) for t in tasks if t in res["results"]}
        summary["benchmarks"][b] = entry
        log(f"=== {b}: {key} = {score} (n={n}) in {wall:.0f} s")
        jdump(summary, summary_path)

    # Held-out refusal sets (paper 33): plain responses to a frozen prompt file, scored afterwards
    # by a separate script. Same system prompt as heretic; greedy; thinking disabled.
    if args.refusal_sets:
        try:
            summary["refusal_generation"] = generate_refusal_set(lm, args, out)
        except Exception:
            traceback.print_exc()
            summary["refusal_generation"] = {"error": traceback.format_exc()[-4000:]}
            failed.append("refusal_generation")
        jdump(summary, summary_path)
    if args.refusal_sets and args.reference_responses:
        try:
            if args.reference_responses == "self":
                args.reference_responses = str(out / "refusal" / "responses.jsonl")
            summary["teacher_forced"] = teacher_forced_logprobs(lm, args, out)
        except Exception:
            traceback.print_exc()
            summary["teacher_forced"] = {"error": traceback.format_exc()[-4000:]}
            failed.append("teacher_forced")
        jdump(summary, summary_path)

    if not args.no_hash:
        log("waiting for weight hash")
        th.join()
        summary["model_info"]["tree_sha256"] = hash_result.get("tree_sha256")
        summary["model_info"]["weights_bytes"] = hash_result.get("bytes")
        jdump(hash_result, out / "model_hash.json")
    summary["finished"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    summary["total_seconds"] = round(time.time() - t_start, 1)
    summary["failed"] = failed
    jdump(summary, summary_path)
    log(f"done in {summary['total_seconds']} s; failed={failed}")
    # Results are on disk; skip interpreter teardown (vLLM engine shutdown can SIGABRT after a clean run).
    sys.stdout.flush(); sys.stderr.flush()
    try:  # os._exit skips vLLM shutdown: kill its engine-core/helper children so they don't hold the GPU or our stdout
        import psutil
        for c in psutil.Process().children(recursive=True):
            c.kill()
    except Exception:
        pass
    os._exit(1 if failed else 0)


if __name__ == "__main__":
    main()
