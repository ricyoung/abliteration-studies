"""Check every model in the HF cache shard-by-shard against its index. Run on the box."""
import json, os, glob
hub = os.environ.get("HF_HOME", "/workspace/paper33/hf") + "/hub"
for d in sorted(glob.glob(hub + "/models--*")):
    name = d.split("models--")[1].replace("--", "/")
    snaps = glob.glob(d + "/snapshots/*")
    if not snaps:
        print(f"{name:48s} NO SNAPSHOT"); continue
    s = snaps[0]
    have = {os.path.basename(p) for p in glob.glob(s + "/*.safetensors") if os.path.exists(os.path.realpath(p))}
    size = sum(os.path.getsize(os.path.realpath(os.path.join(s, f))) for f in have) / 1e9
    idx = os.path.join(s, "model.safetensors.index.json")
    if os.path.exists(idx):
        need = set(json.load(open(idx))["weight_map"].values())
        miss = need - have
        status = "OK" if not miss else f"MISSING {len(miss)}"
        print(f"{name:48s} {len(need) - len(miss):3d}/{len(need):3d} shards {size:6.1f} GB  {status}")
    else:
        print(f"{name:48s} single file      {size:6.1f} GB  {'OK' if have else 'MISSING'}")
