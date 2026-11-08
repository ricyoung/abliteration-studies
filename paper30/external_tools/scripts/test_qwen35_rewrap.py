#!/usr/bin/env python3
"""CPU test of common.finalize's Qwen3.5 text-only -> ConditionalGeneration rewrap (the path an OBLITERATUS
Qwen3.5 export would take). Loads Qwen3.5-4B the way OBLITERATUS does (AutoModelForCausalLM -> text-only
Qwen3_5ForCausalLM), saves it unmodified, finalises it, and checks that every language tensor equals the
tool output and every vision tensor equals the base.
Usage: python -I test_qwen35_rewrap.py WORKDIR OUT_DIR"""
import json, shutil, sys
from pathlib import Path
import torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common

work, out = Path(sys.argv[1]), Path(sys.argv[2])
base = common.base_snapshot("Qwen/Qwen3.5-4B", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a")
tool_out = work / "out"
if tool_out.exists():
    shutil.rmtree(tool_out)
m = AutoModelForCausalLM.from_pretrained(str(base), dtype=torch.bfloat16, device_map="cpu")
print("loaded as", type(m).__name__)
m.save_pretrained(str(tool_out), max_shard_size="5GB")
AutoTokenizer.from_pretrained(str(base)).save_pretrained(str(tool_out))
del m
keys = sorted(common._read_all_keys(tool_out))
print("text-only export:", len(keys), "tensors, e.g.", keys[:2], "visual:", sum(k.startswith("model.visual.") for k in keys))
rep = common.finalize(tool_out, out, base)
staging = Path(rep["staging_dir"])
ver = common.verify_loadable(staging)
fin, tk, bk = common._read_all_keys(staging), common._read_all_keys(tool_out), common._read_all_keys(base)
bad = 0
for k, f in fin.items():
    src = (tk, common._qwen35_text_to_vlm_key) if not k.startswith("model.visual.") else (bk, None)
    if k.startswith("model.visual."):
        with safe_open(str(bk[k]), "pt") as a, safe_open(str(f), "pt") as b:
            same = torch.equal(a.get_tensor(k).to(torch.bfloat16), b.get_tensor(k))
    else:
        tkey = next((t for t in tk if common._qwen35_text_to_vlm_key(t) == k), None)
        with safe_open(str(tk[tkey]), "pt") as a, safe_open(str(f), "pt") as b:
            same = torch.equal(a.get_tensor(tkey), b.get_tensor(k))
    bad += not same
res = {"rewrapped": rep["qwen35_rewrapped_from_text_only"], "n_final": len(fin), "visual_from_base": rep.get("visual_tensors_from_base"),
       "keys_missing_vs_base": rep["keys_missing_vs_base"], "keys_extra_vs_base": rep["keys_extra_vs_base"],
       "tensor_mismatches": bad, "verify": ver}
common.write_provenance(staging, {"test": "qwen35 rewrap", **res})
common.promote(staging, out)
print(json.dumps(res, indent=1))
