#!/usr/bin/env python3
"""CPU-only check of OBLITERATUS's Qwen3.5 surgery contract on a model skeleton (meta device, no weights).
Prints the exact error v0.1.4 raises for a given Qwen3.5 checkpoint, i.e. the error the pipeline turns into
'Qwen3.5/Qwen3.8 hybrid surgery remains blocked: ...; no weights were modified' at the baseline stage.
Usage: python -I check_obliteratus_qwen35_contract.py MODEL_DIR_OR_ID [MODEL_NAME_AS_PASSED]
"""
import json, sys, types
import torch
from transformers import AutoConfig, AutoModelForCausalLM
sys.path.insert(0, "/data/heretic_2/tools_ext/src/obliteratus")
from obliteratus.models.qwen35_contracts import Qwen35ContractError, validate_qwen38_27b_projection_contract

src = sys.argv[1]
name = sys.argv[2] if len(sys.argv) > 2 else src
cfg = AutoConfig.from_pretrained(src)
with torch.device("meta"):
    model = AutoModelForCausalLM.from_config(cfg)   # OBLITERATUS loads Qwen3.5 through AutoModelForCausalLM
layers = model.model.layers
handle = types.SimpleNamespace(model=model, config=cfg, model_name=name, architecture=cfg.model_type,
                               num_layers=len(layers))
res = {"model": src, "model_name_passed": name, "model_class": type(model).__name__,
       "text_dims": {k: getattr(cfg.text_config, k) for k in ("hidden_size", "intermediate_size", "num_hidden_layers")}}
try:
    m = validate_qwen38_27b_projection_contract(handle)
    res["result"] = f"accepted ({len(m)} layer targets)"
except Qwen35ContractError as e:
    res["result"] = f"Qwen35ContractError: {e}"
print(json.dumps(res, indent=1))
