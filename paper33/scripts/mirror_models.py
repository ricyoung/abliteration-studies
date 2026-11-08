"""Mirror every model paper 33 depends on to /data at the EXACT revision used, and verify every LFS file's sha256 against
the Hugging Face record for that revision. Order: third-party first (most at risk of being changed/deleted):
base models (pinned hf_commit from the originals' evaluation summaries), community models (community_repo.json),
judge models (judge_revisions), then our own published repos (current commit, recorded in the manifest).
Output: /data/heretic_2/models/paper33/<category>/<org>__<name>@<sha12>/ and manifest.json (+ manifest.log).
Re-runnable: verified repos are skipped."""
import glob, hashlib, json, os, sys, time
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download

P = "/data/heretic_2/33_2611_does_refusal_removal_get_harder"
OUT = Path("/data/heretic_2/models/paper33"); OUT.mkdir(parents=True, exist_ok=True)
MAN = OUT / "manifest.json"; LOG = OUT / "manifest.log"
api = HfApi()
def log(m):
    line = f"{time.strftime('%F %T')} {m}"; print(line, flush=True); open(LOG, "a").write(line + "\n")

def collect():
    items = {}
    for f in glob.glob(P + "/runs/mc-*-1/results/bench/*__base/summary.json") + glob.glob(P + "/results/bench-4090/*__base/summary.json"):
        mi = json.load(open(f)).get("model_info", {})
        if mi.get("hf_commit") and "/" in mi.get("model", ""):
            items[mi["model"]] = ("base", mi["hf_commit"])
    for f in glob.glob(P + "/runs/mc-*-1/results/bench/*community*/community_repo.json"):
        d = json.load(open(f)); items[d["repo"]] = ("community", d["sha"])
    for f in glob.glob(P + "/runs/mc-rtx-1/results/*/*/refusal/summary.json"):
        for repo, sha in (json.load(open(f)).get("judge_revisions") or {}).items():
            if sha: items[repo] = ("judge", sha)
    keys = ("Qwen2.5-32B", "Qwen3.8-27B", "gemma-2-27b", "gemma-4-31B", "OLMo-2-0325", "Olmo-3.1", "Mistral-Small",
            "Qwen3.5-4B", "Qwen2.5-7B-Instruct")
    for m in api.list_models(author="richardyoung", limit=2000):
        if any(k in m.id for k in keys) and ("-heretic" in m.id) and m.id != "richardyoung/Qwen2.5-7B-Instruct-heretic":
            items[m.id] = ("ours", api.model_info(m.id).sha)
    order = {"base": 0, "community": 1, "judge": 2, "ours": 3}
    return sorted(items.items(), key=lambda kv: (order[kv[1][0]], kv[0]))

def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""): h.update(b)
    return h.hexdigest()

def main():
    man = json.load(open(MAN)) if MAN.exists() else {}
    todo = collect()
    log(f"{len(todo)} repos: " + ", ".join(f"{c}={sum(1 for _, (cc, _) in todo if cc == c)}" for c in ("base", "community", "judge", "ours")))
    for repo, (cat, rev) in todo:
        key = f"{repo}@{rev}"
        if man.get(key, {}).get("verified"):
            continue
        d = OUT / cat / f"{repo.replace('/', '__')}@{rev[:12]}"
        try:
            info = api.model_info(repo, revision=rev, files_metadata=True)
            t = time.time()
            snapshot_download(repo, revision=rev, local_dir=str(d), max_workers=8)
            lfs = {s.rfilename: s.lfs["sha256"] for s in info.siblings if s.lfs and s.lfs.get("sha256")}
            bad = [fn for fn, h in lfs.items() if not (d / fn).exists() or sha256(d / fn) != h]
            size = sum(f.stat().st_size for f in d.rglob("*") if f.is_file() and ".cache" not in f.parts)
            man[key] = {"repo": repo, "category": cat, "revision": rev, "path": str(d), "files": len(info.siblings),
                        "lfs_files": len(lfs), "bytes": size, "verified": not bad, "mismatched": bad,
                        "seconds": round(time.time() - t), "mirrored": time.strftime("%FT%T")}
            log(f"{'OK ' if not bad else 'BAD'} {cat:9s} {repo}@{rev[:12]} {size/1e9:.1f} GB, {len(lfs)} LFS files verified" + (f", MISMATCH {bad}" if bad else ""))
        except Exception as e:
            man[key] = {"repo": repo, "category": cat, "revision": rev, "verified": False, "error": f"{type(e).__name__}: {e}"[:300]}
            log(f"ERR {cat:9s} {repo}@{rev[:12]}: {type(e).__name__}: {str(e)[:200]}")
        json.dump(man, open(MAN, "w"), indent=1)
    ok = sum(1 for v in man.values() if v.get("verified"))
    log(f"DONE: {ok}/{len(man)} verified, {sum(v.get('bytes', 0) for v in man.values())/1e12:.2f} TB")

if __name__ == "__main__":
    main()
