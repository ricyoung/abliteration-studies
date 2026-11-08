#!/usr/bin/env python3
"""Run jwest33/abliterator through its own non-interactive CLI (`abliterate --batch`) and write an HF
bf16 checkpoint + PROVENANCE.json.

What this wrapper adds around the unmodified CLI (src.cli:cli):
  * HOME is pointed at a sandbox (tools_ext/home/jwest33) because the tool creates ~/abliterate/ and
    reads its default prompt files from ~/abliterate/prompts/ first. The pinned commit (6ca3356) no
    longer ships prompts/ (deleted in deb3ac4, "added jspace abliteration wip"), so the sandbox gets
    the tool's own default files from the last commit that shipped them (a2a7940):
    tools_ext/prompts/jwest33_a2a7940/{harmful,harmless,preservation}.txt.
  * Python `random`, numpy and torch are seeded: the tool samples prompts with an unseeded
    random.sample/random.shuffle.
  * compute_refusal_directions is wrapped (observation only) to record the prompts actually used.
  * a fresh, non-existent output path is passed, so the model is not silently written to a
    versioned "<out>_v2" path (the March failure; see README).

Usage (inside venv-jwest33):
  python -I run_jwest33.py --repo Qwen/Qwen2.5-7B-Instruct --revision SHA \
      --out /data/heretic_2/models/external/qwen2.5-7b__jwest33-recommended --seed 1 -- \
      --dtype bfloat16 --device cuda --winsorize --null-space
Everything after `--` is passed verbatim to `abliterate --batch -m <base> -o <tmp>`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import re
import shutil
import sys
import time
import traceback
from pathlib import Path

ROOT = Path("/data/heretic_2/tools_ext")
TOOL_DIR = ROOT / "src/jwest33-abliterator"
PROMPTS = ROOT / "prompts/jwest33_a2a7940"
SANDBOX_HOME = ROOT / "home/jwest33"
os.environ["HOME"] = str(SANDBOX_HOME)  # before importing the tool (Path.home() is read lazily, but be safe)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402


def main():
    argv = sys.argv[1:]
    tool_args = argv[argv.index("--") + 1:] if "--" in argv else []
    own = argv[: argv.index("--")] if "--" in argv else argv
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--revision", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--workdir", default=str(ROOT / "work"))
    ap.add_argument("--label", default="recommended")
    ap.add_argument("--no-hybrid-fix", action="store_true",
                    help="do NOT set config._hybrid_info in the CLI path (tool as published: --hybrid-strategy auto is then a no-op)")
    ap.add_argument("--fix-square-writers", action="store_true",
                    help="OPT-IN runtime fix for the square o_proj bug (patches/jwest33_square_writer_fix.py)")
    a = ap.parse_args(own)

    final = Path(a.out)
    if (final / "PROVENANCE.json").exists():
        print(f"[skip] {final} already has PROVENANCE.json")
        return 0
    if final.exists():
        raise SystemExit(f"{final} exists without PROVENANCE.json; refusing to overwrite (inspect/remove it manually)")
    base = common.base_snapshot(a.repo, a.revision)
    work = Path(a.workdir) / (final.name + f".tmp{os.getpid()}")
    work.mkdir(parents=True)
    tool_out = work / "out"  # must not exist: the tool versions existing paths to <out>_v2

    # sandbox HOME with the tool's default prompt files
    upd = SANDBOX_HOME / "abliterate" / "prompts"
    upd.mkdir(parents=True, exist_ok=True)
    prompt_sha = {}
    for fn in ("harmful.txt", "harmless.txt", "preservation.txt"):
        shutil.copy2(PROMPTS / fn, upd / fn)
        prompt_sha[fn] = hashlib.sha256((upd / fn).read_bytes()).hexdigest()

    import numpy as np
    import torch
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)

    logging.basicConfig(level=logging.INFO, stream=sys.stdout, force=True,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    import src.abliterate as abl
    from src import cli as tool_cli

    runtime_patches = []
    sqfix = None
    if a.fix_square_writers:
        sys.path.insert(0, str(ROOT / "patches"))
        import jwest33_square_writer_fix as sqfix
        runtime_patches.append({"name": "square_writer_fix", "what": sqfix.install(),
                                "file": str(ROOT / "patches/jwest33_square_writer_fix.py"),
                                "sha256": common.sha256_file(Path(sqfix.__file__))})

    used = {}
    orig = abl.compute_refusal_directions

    hybrid = {}

    def recording(model, tokenizer, config, *x, **k):
        used["harmful"] = list(config.harmful_prompts)
        used["harmless"] = list(config.harmless_prompts)
        # Hybrid fix (default on): the CLI's run_abliteration never sets config._hybrid_info, so the
        # documented default --hybrid-strategy auto is silently ignored for Qwen3.5. The library entry
        # point src.abliterate.run_abliteration does exactly this before extraction (abliterate.py:3410-3418).
        if not a.no_hybrid_fix and getattr(config, "_hybrid_info", None) is None:
            config._hybrid_info = abl.detect_hybrid_architecture(config.model_path)
            hybrid["is_hybrid"] = bool(config._hybrid_info.is_hybrid)
            hybrid["full_attention_indices"] = list(config._hybrid_info.full_attention_indices)
        return orig(model, tokenizer, config, *x, **k)

    abl.compute_refusal_directions = recording  # cli.run_abliteration imports it at call time
    if not a.no_hybrid_fix:
        runtime_patches.append({"name": "cli_hybrid_detection",
                                "what": "set config._hybrid_info = detect_hybrid_architecture(model_path) before "
                                        "compute_refusal_directions, as src.abliterate.run_abliteration does; the CLI "
                                        "path omits it, so --hybrid-strategy auto (default) was a no-op. No effect on "
                                        "non-hybrid models.",
                                "file": __file__})

    cli_args = ["--batch", "-m", str(base), "-o", str(tool_out), *tool_args]
    print("[jwest33] abliterate " + " ".join(cli_args), flush=True)
    t0 = time.time()
    rc = 0
    try:
        tool_cli.cli.main(args=cli_args, prog_name="abliterate", standalone_mode=False)
    except SystemExit as e:
        rc = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
    except BaseException as e:
        rc = 1
        traceback.print_exc()
        used["exception"] = f"{type(e).__name__}: {e}"
    t_tool = time.time() - t0
    if rc != 0 or not (tool_out / "config.json").exists():
        fail = {"tool": "jwest33/abliterator", "rc": rc, "runtime_patches": runtime_patches, "base_repo": a.repo, "error": used.get("exception"),
                "tool_out_exists": tool_out.exists(), "wall_seconds": round(t_tool, 1),
                "command": [sys.executable, *sys.argv]}
        (work / "FAILED.json").write_text(json.dumps(fail, indent=2))
        print(f"[failed] rc={rc}; details in {work / 'FAILED.json'}", flush=True)
        return 1
    (tool_out / "prompts_used.json").write_text(json.dumps(used, indent=1))
    cfg_file = tool_out / "abliteration_config.json"
    tool_cfg = json.loads(cfg_file.read_text()) if cfg_file.exists() else {}

    rep = common.finalize(tool_out, final, base)
    staging = Path(rep["staging_dir"])
    ver = common.verify_loadable(staging)
    gi = common.git_info(TOOL_DIR)
    prov = {
        "tool": "jwest33/abliterator",
        "repo_url": "https://github.com/jwest33/abliterator",
        "commit": gi["commit"], "local_modifications": gi,
        "runtime_patches": runtime_patches,
        "square_writer_fix_report": sqfix.report() if sqfix else None,
        "hybrid_detection": hybrid or None,
        "preset": a.label,
        "preset_definition": "interactive CLI preset 'Recommended standard (winsorize + null-space)' / "
                             "'Recommended for hybrid model (hybrid-aware + null-space + winsorization)'; "
                             "batch flags: " + " ".join(tool_args),
        "resolved_config": tool_cfg,
        "command": [sys.executable, *sys.argv],
        "tool_cli_args": cli_args,
        "env": {k: os.environ.get(k) for k in ("HOME", "HF_HOME", "HF_HUB_OFFLINE", "CUDA_VISIBLE_DEVICES")},
        "seed": a.seed,
        "base_repo": a.repo, "base_revision": a.revision, "base_local_path": str(base),
        "prompt_sets": {
            "source": "jwest33 default prompt files from commit a2a7940 (prompts/ removed from the repo in deb3ac4)",
            "files_sha256": prompt_sha,
            "n_harmful_used": len(used.get("harmful", [])), "n_harmless_used": len(used.get("harmless", [])),
            "full_lists": "tool_artifacts/prompts_used.json",
        },
        "wall_seconds_tool": round(t_tool, 1),
        "wall_seconds_total": round(time.time() - t0, 1),
        "tool_reported_metrics": {k: tool_cfg.get(k) for k in tool_cfg if re.search(r"kl|refus|num_harm|quality", k)},
        "finalize_report": rep, "verify": ver,
        "torch": torch.__version__, "transformers": __import__("transformers").__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    common.write_provenance(staging, prov)
    common.promote(staging, final)
    shutil.rmtree(work, ignore_errors=True)
    print(f"[done] {final}  tool {t_tool/60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
