"""Re-score the RELEASED merged exports on the dev set (paper 33 check, 2026-11-06).

For each selected run: `heretic --model <pinned base snapshot> --evaluate-model <merged export>` in a fresh workdir with
the run's own config.toml, code version (PYTHONPATH) and batch size. This is Heretic's built-in evaluation path: it
computes the base model's dev refusals / first-token logprobs exactly as the search did, then scores the merged export
with the same KeywordRate and KLDivergence scorers. Compares with the selected trial's refusal count and KL.

Usage: python rescore_dev.py JOBS.tsv OUT_ROOT [--heretic BIN] [--download] [--cleanup]
  JOBS.tsv columns: run model method base_repo base_rev merged merged_rev pythonpath config batch_size sel_trial sel_refusals sel_kl
  merged = local dir, or an HF repo id (downloaded at merged_rev into OUT_ROOT/models/<run> when --download).
  --cleanup: delete each downloaded merged export after scoring, and a base snapshot once no later job needs it
             (only bases listed in the TSV; never the 1.x Qwen2.5-32B base while the 1.x rebuild is still running).
Results: OUT_ROOT/rescore_dev.tsv (appended per job; finished runs are skipped on restart), OUT_ROOT/<run>/output.log.
"""
import csv, os, re, shutil, subprocess, sys, threading, time
from pathlib import Path

args = [a for a in sys.argv[1:] if not a.startswith("--")]
flags = set(a for a in sys.argv[1:] if a.startswith("--") and "=" not in a)
opts = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
JOBS, OUT = args[0], Path(args[1])
HERETIC = opts.get("heretic", "heretic")
OUT.mkdir(parents=True, exist_ok=True)
RES = OUT / "rescore_dev.tsv"
COLS = ["run", "model", "method", "sel_trial", "sel_refusals", "sel_kl", "rescore_refusals", "rescore_kl",
        "baseline_refusals", "d_refusals", "d_kl", "base_repo", "base_rev", "merged", "merged_rev", "batch_size",
        "pythonpath", "prefix", "rc", "seconds", "host", "finished"]
jobs = list(csv.DictReader(open(JOBS), delimiter="\t"))
done = {r["run"] for r in csv.DictReader(open(RES), delimiter="\t") if r["rc"] == "0"} if RES.exists() else set()
if not RES.exists():
    RES.write_text("\t".join(COLS) + "\n")


def strip(s):
    return re.sub(r"\x1b\[[0-9;]*m", "", s)


def base_path(j):
    from huggingface_hub import snapshot_download
    return snapshot_download(j["base_repo"], revision=j["base_rev"])


def merged_path(j):
    m = j["merged"]
    if os.path.isdir(m):
        return m
    from huggingface_hub import snapshot_download
    d = OUT / "models" / j["run"]
    return snapshot_download(m, revision=j["merged_rev"], local_dir=str(d))


prefetch = {}


def start_prefetch(j):
    if j and j["run"] not in done and not os.path.isdir(j["merged"]) and j["run"] not in prefetch:
        t = threading.Thread(target=lambda: prefetch.__setitem__(j["run"] + ":path", merged_path(j)), daemon=True)
        prefetch[j["run"]] = t; t.start()


def delete_base(repo):
    hub = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub"
    d = hub / ("models--" + repo.replace("/", "--"))
    if repo == "Qwen/Qwen2.5-32B-Instruct":
        log = Path("/workspace/paper33/logs/rebuild_1x_upload.log")
        if log.exists() and "END rebuild+upload" not in log.read_text() and "FAILED" not in log.read_text():
            print("keep Qwen2.5-32B base: 1.x rebuild still running", flush=True); return
    if d.is_dir():
        shutil.rmtree(d)
        prune = Path("/workspace/paper33/scripts/prune_hf_blobs.py")
        if prune.exists():
            subprocess.run([sys.executable, str(prune), "--delete"])
        print("deleted base cache", repo, flush=True)


for k, j in enumerate(jobs):
    nxt = jobs[k + 1] if k + 1 < len(jobs) else None
    if j["run"] in done:
        print("skip (done)", j["run"], flush=True); continue
    t0 = time.time(); wd = OUT / j["run"]; wd.mkdir(parents=True, exist_ok=True)
    shutil.copy(j["config"], wd / "config.toml")
    bp = base_path(j)
    if j["run"] in prefetch:
        prefetch[j["run"]].join(); mp = prefetch.get(j["run"] + ":path") or merged_path(j)
    else:
        mp = merged_path(j)
    start_prefetch(nxt)  # download the next export while this one is scored
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    if j["pythonpath"]:
        env["PYTHONPATH"] = j["pythonpath"]
    cmd = [HERETIC, "--model", bp, "--evaluate-model", mp, "--batch-size", j["batch_size"]]
    (wd / "cmd.txt").write_text(" ".join(cmd) + f"\nPYTHONPATH={j['pythonpath']}\nbase={j['base_repo']}@{j['base_rev']}\n"
                                f"merged={j['merged']}@{j['merged_rev']}\n")
    with open(wd / "output.log", "w") as f:
        rc = subprocess.run(cmd, cwd=wd, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
    log = strip((wd / "output.log").read_text(errors="replace"))
    ref = re.findall(r"^\s*\* Refusals: (\d+)/(\d+)", log, re.M)
    kl = re.findall(r"^\s*\* KL divergence: ([0-9.eE+-]+)", log, re.M)
    base = re.search(r"Baseline Refusals: (\d+)/(\d+)", log)
    pre = re.search(r"Extended prefix found: (.*)", log) or re.search(r"[Pp]refix found: (.*)", log)
    rr = int(ref[-1][0]) if ref else None; rk = float(kl[-1]) if kl else None
    sr = round(float(j["sel_refusals"])); sk = float(j["sel_kl"])
    row = {"run": j["run"], "model": j["model"], "method": j["method"], "sel_trial": j["sel_trial"], "sel_refusals": sr,
           "sel_kl": f"{sk:.4f}", "rescore_refusals": rr if rr is not None else "", "rescore_kl": rk if rk is not None else "",
           "baseline_refusals": base.group(1) if base else "", "d_refusals": (rr - sr) if rr is not None else "",
           "d_kl": f"{rk - sk:+.4f}" if rk is not None else "", "base_repo": j["base_repo"], "base_rev": j["base_rev"],
           "merged": j["merged"], "merged_rev": j["merged_rev"], "batch_size": j["batch_size"], "pythonpath": j["pythonpath"],
           "prefix": (pre.group(1).strip() if pre else "").replace("\t", " "), "rc": rc if (rr is not None and rk is not None) else (rc or 99),
           "seconds": int(time.time() - t0), "host": os.uname().nodename, "finished": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    with open(RES, "a") as f:
        f.write("\t".join(str(row[c]) for c in COLS) + "\n")
    print(f"{j['run']}: rescore {rr}/100 kl {rk} vs selected {sr}/100 kl {sk:.4f} rc={row['rc']} ({row['seconds']}s)", flush=True)
    if "--cleanup" in flags:
        if not os.path.isdir(j["merged"]) or str(OUT / "models") in mp:
            shutil.rmtree(OUT / "models" / j["run"], ignore_errors=True)
        if not any(x["base_repo"] == j["base_repo"] for x in jobs[k + 1:]):
            delete_base(j["base_repo"])
print("ALL DONE", flush=True)
