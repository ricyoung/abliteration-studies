"""Shared helpers for the external-tool wrappers (OBLITERATUS, jwest33/abliterator).

finalize() turns whatever a tool wrote into the layout paper 33 evaluates:
  * HF safetensors, bf16, shards <= 5 GB, model.safetensors.index.json;
  * tokenizer files, chat template, generation_config copied from the pinned base snapshot
    (abliteration does not touch the tokenizer; the tool's re-saved copies are kept in tool_artifacts/);
  * for Qwen3.5 (Qwen3_5ForConditionalGeneration): if the tool exported a text-only
    derivative (OBLITERATUS does), the language-model tensors are renamed into the
    model.language_model.* namespace and the base's (unmodified) vision tower and config are
    re-attached, so the checkpoint has the same layout as the Heretic exports. MTP tensors are
    absent, as in the Heretic exports (transformers does not load them).
  * every other file the tool wrote (directions, projectors, its own metadata) goes to tool_artifacts/.
Nothing here changes a language-model weight value except a dtype cast, which is recorded.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

HF_HOME = os.environ.get("HF_HOME", "/data/huggingface")
TOKENIZER_FILES = [
    "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "special_tokens_map.json",
    "added_tokens.json", "chat_template.jinja", "chat_template.json", "generation_config.json",
    "preprocessor_config.json", "video_preprocessor_config.json", "processor_config.json",
]
MAX_SHARD = 5 * 1024**3


def base_snapshot(repo: str, revision: str) -> Path:
    p = Path(HF_HOME) / "hub" / ("models--" + repo.replace("/", "--")) / "snapshots" / revision
    if not (p / "config.json").exists():
        raise FileNotFoundError(f"{repo}@{revision} not in HF cache at {p}; download it first")
    return p


def git_info(repo_dir: str | Path) -> dict:
    def g(*a):
        return subprocess.run(["git", "-C", str(repo_dir), *a], capture_output=True, text=True).stdout.strip()
    diff = g("diff", "HEAD")
    return {
        "commit": g("rev-parse", "HEAD"),
        "remote": g("remote", "get-url", "origin"),
        "dirty": bool(g("status", "--porcelain", "--untracked-files=no")),
        "local_diff_sha256": hashlib.sha256(diff.encode()).hexdigest() if diff else None,
        "local_diff_files": g("diff", "--name-only", "HEAD").split() if diff else [],
    }


def _shards(d: Path) -> list[Path]:
    return sorted(d.glob("*.safetensors"))


def _read_all_keys(d: Path) -> dict[str, Path]:
    out = {}
    for f in _shards(d):
        with safe_open(str(f), "pt") as h:
            for k in h.keys():
                out[k] = f
    return out


def _qwen35_text_to_vlm_key(k: str) -> str | None:
    """Map a text-only Qwen3_5ForCausalLM key to the Qwen3_5ForConditionalGeneration namespace."""
    if k.startswith("model.language_model.") or k.startswith("model.visual.") or k == "lm_head.weight":
        return k
    if k.startswith("model."):
        return "model.language_model." + k[len("model."):]
    return None


def finalize(tool_out: Path, final_dir: Path, base_dir: Path, log=print) -> dict:
    """Write the normalised checkpoint to <final_dir>.staging (final_dir must not exist).
    Returns a report dict; call write_provenance(staging) and promote(staging, final_dir) afterwards."""
    tool_out, final_dir, base_dir = Path(tool_out), Path(final_dir), Path(base_dir)
    assert not final_dir.exists(), final_dir
    staging = final_dir.parent / (final_dir.name + ".staging")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    rep = {"tool_output_files": sorted(p.name for p in tool_out.iterdir())}
    base_cfg = json.load(open(base_dir / "config.json"))
    tool_cfg = json.load(open(tool_out / "config.json"))
    is_qwen35 = base_cfg.get("model_type") == "qwen3_5"
    tool_keys = _read_all_keys(tool_out)
    if not tool_keys:
        raise RuntimeError(f"no safetensors weights in {tool_out}")
    # text-only export = no vision tower in the tool output (key names may be either namespace,
    # depending on whether transformers reverse-maps them on save)
    text_only = is_qwen35 and not any(k.startswith("model.visual.") for k in tool_keys)
    rep["tool_architectures"] = tool_cfg.get("architectures")
    rep["qwen35_rewrapped_from_text_only"] = text_only

    # 1. collect (name -> source) for the final checkpoint
    plan: list[tuple[str, Path, str]] = []  # (final_name, file, source_name)
    if text_only:
        base_keys = _read_all_keys(base_dir)
        for k, f in tool_keys.items():
            nk = _qwen35_text_to_vlm_key(k)
            if nk is None:
                raise RuntimeError(f"unexpected tensor {k} in text-only Qwen3.5 output")
            plan.append((nk, f, k))
        lang_base = {k for k in base_keys if k.startswith("model.language_model.") or k == "lm_head.weight"}
        got = {p[0] for p in plan}
        missing = sorted(lang_base - got)
        extra = sorted(got - lang_base)
        # tied embeddings: base has no lm_head.weight; a text-only export may or may not write it
        if base_cfg.get("tie_word_embeddings", base_cfg.get("text_config", {}).get("tie_word_embeddings")):
            extra = [e for e in extra if e != "lm_head.weight"]
            plan = [p for p in plan if p[0] != "lm_head.weight"]
        if missing or extra:
            raise RuntimeError(f"Qwen3.5 rewrap key mismatch: missing={missing[:5]} extra={extra[:5]}")
        vis = sorted(k for k in base_keys if k.startswith("model.visual."))
        plan += [(k, base_keys[k], k) for k in vis]
        rep["visual_tensors_from_base"] = len(vis)
        rep["mtp_tensors_dropped"] = sum(k.startswith("mtp.") for k in base_keys)
    else:
        plan = [(k, f, k) for k, f in tool_keys.items()]

    # 2. write shards (bf16), streaming tensor by tensor
    dtypes_seen: dict[str, int] = {}
    shard, size, n, weight_map, shard_files = {}, 0, 0, {}, []

    def flush():
        nonlocal shard, size, n
        if not shard:
            return
        n += 1
        name = f"model-{n:05d}.safetensors"
        save_file(shard, str(staging / name), metadata={"format": "pt"})
        shard_files.append(name)
        for k in shard:
            weight_map[k] = name
        shard, size = {}, 0

    by_file: dict[Path, list[tuple[str, str]]] = {}
    for nk, f, k in plan:
        by_file.setdefault(f, []).append((nk, k))
    for f, items in by_file.items():
        with safe_open(str(f), "pt") as h:
            for nk, k in items:
                t = h.get_tensor(k)
                dtypes_seen[str(t.dtype)] = dtypes_seen.get(str(t.dtype), 0) + 1
                if t.is_floating_point() and t.dtype != torch.bfloat16:
                    t = t.to(torch.bfloat16)
                b = t.numel() * t.element_size()
                if size and size + b > MAX_SHARD:
                    flush()
                shard[nk] = t.contiguous()
                size += b
    flush()
    total = len(shard_files)
    final_map = {}
    for i, name in enumerate(shard_files, 1):
        new = f"model-{i:05d}-of-{total:05d}.safetensors"
        (staging / name).rename(staging / new)
        for k, v in weight_map.items():
            if v == name:
                final_map[k] = new
    total_size = sum((staging / v).stat().st_size for v in set(final_map.values()))
    json.dump({"metadata": {"total_size": total_size}, "weight_map": dict(sorted(final_map.items()))},
              open(staging / "model.safetensors.index.json", "w"), indent=2)
    rep["tool_saved_dtypes"] = dtypes_seen
    rep["n_tensors"] = len(final_map)

    # 3. config: always the base config (architecture must match the base for vLLM/transformers),
    #    dtype set to bf16; differences between the tool's config and the base are recorded.
    cfg = dict(base_cfg)
    ign = {"torch_dtype", "dtype", "transformers_version", "_name_or_path", "architectures", "use_cache"}
    def _diff(t, b, prefix=""):
        out = {}
        for k in sorted(set(t) | set(b)):
            if k in ign:
                continue
            tv, bv = t.get(k), b.get(k)
            if isinstance(tv, dict) and isinstance(bv, dict):
                out.update(_diff(tv, bv, prefix + k + "."))
            elif tv != bv:
                out[prefix + k] = {"tool": tv, "base": bv}
        return out
    rep["tool_config_diff_vs_base"] = _diff(tool_cfg, base_cfg)
    cfg.pop("torch_dtype", None)
    cfg["dtype"] = "bfloat16"
    if "text_config" in cfg and isinstance(cfg["text_config"], dict):
        cfg["text_config"] = dict(cfg["text_config"], dtype="bfloat16")
    cfg.pop("quantization_config", None)
    json.dump(cfg, open(staging / "config.json", "w"), indent=2)
    rep["config_source"] = "base"
    # key-set comparison with the base checkpoint (MTP tensors are never loaded by transformers)
    base_keys_all = set(_read_all_keys(base_dir))
    base_cmp = {k for k in base_keys_all if not k.startswith("mtp.")}
    fin = set(final_map)
    rep["keys_missing_vs_base"] = sorted(base_cmp - fin)[:20]
    rep["keys_extra_vs_base"] = sorted(fin - base_cmp)[:20]

    # 4. tokenizer / chat template / processor files from the base snapshot
    copied = []
    for fn in TOKENIZER_FILES:
        if (base_dir / fn).exists():
            shutil.copyfile(base_dir / fn, staging / fn)
            copied.append(fn)
    rep["copied_from_base"] = copied
    if not (staging / "generation_config.json").exists() and (tool_out / "generation_config.json").exists():
        shutil.copyfile(tool_out / "generation_config.json", staging / "generation_config.json")
        rep["generation_config"] = "base has none; transformers-generated copy from the tool output (as in the Heretic exports)"
    for p in staging.iterdir():
        if p.is_file():
            p.chmod(0o644)

    # 5. everything else the tool wrote -> tool_artifacts/
    art = staging / "tool_artifacts"
    art.mkdir()
    for p in tool_out.iterdir():
        if p.suffix == ".safetensors" or p.name in ("model.safetensors.index.json",):
            continue
        shutil.copytree(p, art / p.name) if p.is_dir() else shutil.copy2(p, art / p.name)
    rep["staging_dir"] = str(staging)
    return rep


def promote(staging: Path, final_dir: Path) -> None:
    """Atomically publish: final_dir appears only once weights, metadata and PROVENANCE.json are complete."""
    assert (staging / "PROVENANCE.json").exists()
    staging.rename(final_dir)


def verify_loadable(final_dir: Path) -> dict:
    """Cheap structural check: config parses, every indexed tensor exists, all float tensors bf16."""
    from transformers import AutoConfig
    AutoConfig.from_pretrained(str(final_dir))
    idx = json.load(open(final_dir / "model.safetensors.index.json"))
    bad = []
    for shard in set(idx["weight_map"].values()):
        with safe_open(str(final_dir / shard), "pt") as h:
            for k in h.keys():
                sl = h.get_slice(k)
                if sl.get_dtype() not in ("BF16", "I64", "I32", "BOOL", "U8"):
                    bad.append((k, sl.get_dtype()))
    return {"n_tensors": len(idx["weight_map"]), "non_bf16": bad[:10]}


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def write_provenance(final_dir: Path, prov: dict) -> None:
    prov = dict(prov)
    prov["written_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    prov["weights_sha256"] = {p.name: sha256_file(p) for p in sorted(final_dir.glob("*.safetensors"))}
    json.dump(prov, open(final_dir / "PROVENANCE.json", "w"), indent=2, default=str)
