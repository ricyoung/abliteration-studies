"""Per-tensor sha256 of a safetensors model directory (layout-independent: shard boundaries/metadata do not matter).
Usage: tensor_hash.py <model_dir>  -> JSON {name: [dtype, shape, sha256]} on stdout."""
import hashlib, json, sys, glob, torch
from safetensors import safe_open
out = {}
for f in sorted(glob.glob(sys.argv[1].rstrip("/") + "/*.safetensors")):
    with safe_open(f, "pt") as s:
        for k in s.keys():
            t = s.get_tensor(k).contiguous()
            b = t.view(torch.uint8) if t.dtype != torch.uint8 else t
            out[k] = [str(t.dtype), list(t.shape), hashlib.sha256(b.numpy().tobytes()).hexdigest()]
json.dump(out, sys.stdout, indent=0, sort_keys=True)
