"""Shared helpers for the GGUF build and upload pipeline (papers 30 and 33). 2026-11-10."""
import datetime, fcntl, hashlib, json, os, re, subprocess, time

ROOT = "/data/heretic_2/gguf"
LC = "/data/heretic_2/llama.cpp"                      # llama.cpp source, commit 468fde9
BIN = LC + "/build-cpu/bin"                           # CPU build (no GPU is used anywhere in this pipeline)
PY = "/data/heretic_2/33_2611_does_refusal_removal_get_harder/scripts/bench/.venv/bin/python"   # read-only use
CALIB = ROOT + "/calib/calibration_datav3.txt"
QUANTS = ["Q4_K_M", "Q8_0", "Q5_K_M", "IQ4_XS", "Q6_K", "IQ3_M"]     # six sizes, as in the account's other GGUF repos
IMATRIX_QUANTS = {"Q4_K_M", "Q5_K_M", "IQ4_XS", "IQ3_M"}               # the rest (Q6_K, Q8_0) are static
UPLOAD_PRIORITY = ["Q4_K_M", "Q5_K_M", "IQ4_XS", "Q6_K", "IQ3_M", "Q8_0"]
LLAMA_COMMIT = "468fde9"

def now():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def log(name, msg):
    line = f"{now()} {msg}"
    with open(f"{ROOT}/logs/{name}.log", "a") as f:
        f.write(line + "\n")
    print(line, flush=True)

def models():
    return json.load(open(ROOT + "/models.json"))

def bdir(name):
    d = f"{ROOT}/build/{name}"; os.makedirs(d, exist_ok=True); return d

def read_json(path, default):
    try:
        return json.load(open(path))
    except FileNotFoundError:
        return default

def update_json(path, fn):
    """Read-modify-write under a lock (builder and uploader both touch the state files)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        d = read_json(path, {})
        fn(d)
        tmp = path + ".tmp"
        json.dump(d, open(tmp, "w"), indent=1); os.replace(tmp, path)
    return d

def sha256(path, buf=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(buf)
            if not b: break
            h.update(b)
    return h.hexdigest()

def run(cmd, logfile, timeout, env=None):
    """Run a command, appending output to logfile. Returns seconds; raises on a non-zero exit."""
    t0 = time.time()
    e = dict(os.environ); e.update(env or {})
    with open(logfile, "a") as lf:
        lf.write(f"\n=== {now()} $ {' '.join(map(str, cmd))}\n"); lf.flush()
        p = subprocess.run(list(map(str, cmd)), stdout=lf, stderr=subprocess.STDOUT, timeout=timeout, env=e)
    if p.returncode != 0:
        raise RuntimeError(f"exit {p.returncode}: {os.path.basename(str(cmd[0]))} (see {os.path.basename(logfile)})")
    return int(time.time() - t0)

def fail(name, model, step, err):
    """Record a failure. (The private notification step used during the build is not part of this copy.)"""
    msg = f"{model} | {step} | {type(err).__name__}: {str(err)[:300]}"
    with open(f"{ROOT}/logs/failures.log", "a") as f:
        f.write(f"{now()} [{name}] {msg}\n")
    log(name, "FAILED " + msg)

def gguf_path(name, quant):
    return f"{bdir(name)}/{name}-{quant}.gguf"
