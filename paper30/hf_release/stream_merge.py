"""Streaming per-shard LoRA merge (rank-1 Heretic 1.x adapters), replicating PEFT 0.19.1 CPU merge_and_unload
on a bf16 base: delta = (B.float() @ A.float()) * scaling, then W += delta. Verifies every tensor against a
per-tensor sha256 reference (tensor_hashes.json from the original merge). Mode chosen by --mode."""
import argparse, hashlib, json, os, shutil, sys, torch
from safetensors import safe_open
from safetensors.torch import save_file
ap = argparse.ArgumentParser()
ap.add_argument("--base"); ap.add_argument("--adapter"); ap.add_argument("--out"); ap.add_argument("--ref")
ap.add_argument("--mode", default="bf16delta", choices=["bf16delta", "fp32add"])
ap.add_argument("--test", action="store_true", help="only check the first edited tensor")
a = ap.parse_args()
cfg = json.load(open(os.path.join(a.adapter, "adapter_config.json"))); scaling = cfg["lora_alpha"] / cfg["r"]
ref = json.load(open(a.ref))
lora = {}
with safe_open(os.path.join(a.adapter, "adapter_model.safetensors"), "pt") as f:
    for k in f.keys():
        mod, ab = k.removeprefix("base_model.model.").rsplit(".lora_", 1)
        lora.setdefault(mod + ".weight", {})[ab.split(".")[0]] = f.get_tensor(k)
def h(t): return hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
def merge(w, ab):
    d = (ab["B"].float() @ ab["A"].float()) * scaling
    if a.mode == "bf16delta":
        return w + d.to(w.dtype)
    return (w.float() + d).to(w.dtype)
idx = json.load(open(os.path.join(a.base, "model.safetensors.index.json")))
shards = sorted(set(idx["weight_map"].values()))
os.makedirs(a.out, exist_ok=True)
bad = 0; n = 0; edited = 0
for s in shards:
    out = {}
    with safe_open(os.path.join(a.base, s), "pt") as f:
        for k in f.keys():
            t = f.get_tensor(k)
            if k in lora:
                t = merge(t, lora[k]); edited += 1
            ok = h(t) == ref[k][2]; n += 1; bad += (not ok)
            if not ok: print("MISMATCH", k, flush=True)
            if a.test and k in lora:
                print("test", k, "match" if ok else "MISMATCH", a.mode); sys.exit(0 if ok else 1)
            out[k] = t
    if not a.test:
        save_file(out, os.path.join(a.out, s), metadata={"format": "pt"}); print("wrote", s, flush=True)
    del out
if a.test: print("no edited tensor found"); sys.exit(2)
assert n == len(ref), (n, len(ref))
shutil.copy(os.path.join(a.base, "model.safetensors.index.json"), a.out)
print(f"tensors {n} edited {edited} mismatches {bad}")
sys.exit(1 if bad else 0)
