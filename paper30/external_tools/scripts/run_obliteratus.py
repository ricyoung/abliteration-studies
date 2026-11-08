#!/usr/bin/env python3
"""Run one OBLITERATUS preset through its Python API and write an HF bf16 checkpoint + PROVENANCE.json.

The Python API (AbliterationPipeline) is used rather than `obliteratus obliterate` because the CLI's
--method choices omit `som` (and rdo/failspy/...); the CLI handler is a thin wrapper that passes the
same arguments, so default parameters are identical. Only these arguments are set explicitly:
model_name (local snapshot of the pinned base revision), output_dir, method (= preset), dtype
(bfloat16; the CLI default is float16), seed, device. Everything else is the preset's default.

Usage (inside venv-obliteratus):
  python -I run_obliteratus.py --repo Qwen/Qwen2.5-7B-Instruct --revision SHA --preset basic \
      --out /data/heretic_2/models/external/qwen2.5-7b__obliteratus-basic --seed 1 [--device auto]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("OBLITERATUS_TELEMETRY", "0")  # opt-in only, but make it explicit
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

TOOL_DIR = Path("/data/heretic_2/tools_ext/src/obliteratus")


def jsonable(v):
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if isinstance(v, (list, tuple)):
        return [jsonable(x) for x in v][:64]
    if isinstance(v, dict):
        return {str(k): jsonable(x) for k, x in list(v.items())[:256]}
    if isinstance(v, Path):
        return str(v)
    return repr(v)[:200]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--revision", required=True)
    ap.add_argument("--preset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--workdir", default="/data/heretic_2/tools_ext/work")
    ap.add_argument("--gpu-memory-utilization", type=float, default=None,
                    help="resource knob only (OBLITERATUS default reserves max(15%%, 2 GiB))")
    ap.add_argument("--qwen35-generic-contract", action="store_true",
                    help="OPT-IN: accept dense Qwen3.5 sizes other than Qwen3.8-27B (patches/obliteratus_qwen35_generic_contract.py)")
    a = ap.parse_args()

    final = Path(a.out)
    if (final / "PROVENANCE.json").exists():
        print(f"[skip] {final} already has PROVENANCE.json")
        return 0
    if final.exists():
        raise SystemExit(f"{final} exists without PROVENANCE.json; refusing to overwrite (inspect/remove it manually)")
    base = common.base_snapshot(a.repo, a.revision)
    work = Path(a.workdir) / (final.name + f".tmp{os.getpid()}")
    work.mkdir(parents=True)
    tool_out = work / "out"
    t0 = time.time()

    import torch
    import obliteratus
    from obliteratus.abliterate import METHODS, AbliterationPipeline

    runtime_patches = []
    if a.qwen35_generic_contract:
        sys.path.insert(0, "/data/heretic_2/tools_ext/patches")
        import obliteratus_qwen35_generic_contract as qpatch
        runtime_patches.append({"name": "qwen35_generic_contract", "what": qpatch.install(),
                                "file": "/data/heretic_2/tools_ext/patches/obliteratus_qwen35_generic_contract.py",
                                "sha256": common.sha256_file(Path(qpatch.__file__)),
                                "note": "outside the tool's validated scope (upstream supports only Qwen3.8-27B)"})
    if a.preset not in METHODS:
        raise SystemExit(f"unknown preset {a.preset}; OBLITERATUS has {sorted(METHODS)}")
    stages = []

    def on_log(m):
        print(f"[obliteratus] {m}", flush=True)

    def on_stage(r):
        d = {k: jsonable(getattr(r, k, None)) for k in ("stage", "status", "message", "duration")}
        stages.append(d)
        print(f"[stage] {d}", flush=True)

    kwargs = dict(model_name=str(base), output_dir=str(tool_out), device=a.device, dtype=a.dtype,
                  method=a.preset, seed=a.seed, on_log=on_log, on_stage=on_stage)
    if a.gpu_memory_utilization is not None:
        kwargs["gpu_memory_utilization"] = a.gpu_memory_utilization
    err = None
    try:
        # constructor inside try: v0.1.4 rejects som here ("som_directions is not shipped")
        pipe = AbliterationPipeline(**kwargs)
        resolved = {k: jsonable(v) for k, v in vars(pipe).items()
                    if not k.startswith("_") and k not in ("harmful_prompts", "harmless_prompts", "handle",
                                                            "evaluation_harmful_prompts", "evaluation_harmless_prompts",
                                                            "jailbreak_prompts", "hub_token")}
        prompts = {"harmful": list(pipe.harmful_prompts), "harmless": list(pipe.harmless_prompts),
                   "evaluation_harmful": list(pipe.evaluation_harmful_prompts),
                   "evaluation_harmless": list(pipe.evaluation_harmless_prompts),
                   "jailbreak_explicit": pipe.jailbreak_prompts}
        out_path = Path(pipe.run())
    except BaseException as e:  # record the failure, then re-raise
        err = f"{type(e).__name__}: {e}"
        traceback.print_exc()
        fail = {"tool": "OBLITERATUS", "preset": a.preset, "runtime_patches": runtime_patches, "base_repo": a.repo, "base_revision": a.revision,
                "error": err, "wall_seconds": round(time.time() - t0, 1), "stages": stages,
                "command": [sys.executable, *sys.argv]}
        (work / "FAILED.json").write_text(json.dumps(fail, indent=2, default=str))
        print(f"[failed] details in {work / 'FAILED.json'}", flush=True)
        raise
    t_tool = time.time() - t0
    if getattr(pipe, "use_jailbreak_contrast", False) and not pipe.jailbreak_prompts:
        try:
            prompts["jailbreak_generated_from_harmful"] = pipe._generate_jailbreak_prompts()
        except Exception as e:  # informational only
            prompts["jailbreak_generated_from_harmful"] = f"could not regenerate: {e}"
    meta_file = out_path / "abliteration_metadata.json"
    meta = json.loads(meta_file.read_text()) if meta_file.exists() else {}
    (out_path / "prompts_used.json").write_text(json.dumps(prompts, indent=1))

    rep = common.finalize(out_path, final, base)
    staging = Path(rep["staging_dir"])
    ver = common.verify_loadable(staging)
    gi = common.git_info(TOOL_DIR)
    prov = {
        "tool": "OBLITERATUS",
        "tool_version": getattr(obliteratus, "__version__", "0.1.4"),
        "repo_url": "https://github.com/elder-plinius/OBLITERATUS",
        "commit": gi["commit"], "tag": "v0.1.4", "local_modifications": gi,
        "runtime_patches": runtime_patches,
        "preset": a.preset,
        "preset_definition": METHODS[a.preset],
        "resolved_config": resolved,
        "tool_metadata": meta,
        "command": [sys.executable, *sys.argv],
        "env": {k: os.environ.get(k) for k in ("HF_HOME", "HF_HUB_OFFLINE", "OBLITERATUS_TELEMETRY",
                                                "CUDA_VISIBLE_DEVICES", "PYTORCH_CUDA_ALLOC_CONF")},
        "seed": a.seed,
        "base_repo": a.repo, "base_revision": a.revision, "base_local_path": str(base),
        "dtype": a.dtype, "device": a.device,
        "prompt_sets": {
            "source": "OBLITERATUS built-in pairs (obliteratus/prompts.py BUILTIN_HARMFUL / BUILTIN_HARMLESS)",
            "n_harmful": len(prompts["harmful"]), "n_harmless": len(prompts["harmless"]),
            "n_evaluation_harmful": len(prompts["evaluation_harmful"]),
            "full_lists": "tool_artifacts/prompts_used.json",
        },
        "wall_seconds_tool": round(t_tool, 1),
        "wall_seconds_total": round(time.time() - t0, 1),
        "stages": stages,
        "tool_reported_metrics": meta.get("quality_metrics"),
        "finalize_report": rep, "verify": ver,
        "torch": torch.__version__, "transformers": __import__("transformers").__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    common.write_provenance(staging, prov)
    common.promote(staging, final)
    shutil.rmtree(work, ignore_errors=True)  # own temp dir only
    print(f"[done] {final}  tool {t_tool/60:.1f} min, total {(time.time()-t0)/60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
