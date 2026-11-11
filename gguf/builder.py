#!/usr/bin/env python3
"""GGUF builder for the paper 30 / paper 33 models. CPU only; resumable; no uploads (uploader.py does those).

Phase A (every model, smallest first): stage the published weights -> bf16 GGUF -> Q8_0 -> importance matrix -> Q4_K_M
          -> load test.
Phase B (every model): Q5_K_M, IQ4_XS, Q6_K, IQ3_M.
The weights are staged from the local mirror only when every weight file matches the SHA-256 that Hugging Face lists for
the published repository; anything else is downloaded. Stop cleanly by creating state/STOP.
"""
import json, os, shutil, subprocess, sys, time, urllib.request
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from huggingface_hub import HfApi, hf_hub_download

api = HfApi()
ENV = {"PYTHONPATH": LC + "/gguf-py", "LD_LIBRARY_PATH": BIN}
ME = "builder"

def stop_requested():
    return os.path.exists(ROOT + "/state/STOP")

def state_path(name):
    return f"{bdir(name)}/build_state.json"

def done(name, key):
    return bool(read_json(state_path(name), {}).get(key))

def mark(name, key, val=True):
    update_json(state_path(name), lambda d: d.__setitem__(key, val))

def stage(m):
    """Stage dir with the exact published files: weights verified against the Hub's LFS SHA-256."""
    name = m["name"]; sd = f"{ROOT}/stage/{name}"
    if done(name, "staged"):
        return sd
    os.makedirs(sd, exist_ok=True)
    info = api.model_info(m["src_repo"], files_metadata=True)
    linked = fetched = 0
    for s in info.siblings:
        f = s.rfilename
        if "/" in f or f in (".gitattributes",):
            continue                                   # adapter/, reproduce/ and other sub-folders are not needed
        dst = os.path.join(sd, f)
        if f.endswith(".safetensors"):
            want = (s.lfs or {}).get("sha256") if isinstance(s.lfs, dict) else getattr(s.lfs, "sha256", None)
            assert want, f"no LFS hash for {f}"
            src = os.path.join(m["local_src"], f) if m.get("local_src") else None
            if os.path.lexists(dst):
                os.remove(dst)
            if src and os.path.isfile(src) and os.path.getsize(src) == s.size and sha256(src) == want:
                os.symlink(os.path.realpath(src), dst); linked += 1
            else:
                hf_hub_download(m["src_repo"], f, revision=info.sha, local_dir=sd)
                assert sha256(dst) == want, f"downloaded {f} does not match the Hub hash"
                fetched += 1
        else:
            hf_hub_download(m["src_repo"], f, revision=info.sha, local_dir=sd)
    shutil.copy2(os.path.join(sd, "README.md"), f"{bdir(name)}/source_README.md")
    mark(name, "source_revision", info.sha)
    mark(name, "staged", {"linked_local": linked, "downloaded": fetched})
    log(ME, f"{name}: staged at {info.sha[:8]} ({linked} weight files from the local mirror, {fetched} downloaded)")
    return sd

def load_test(name, gguf):
    """Start llama-server on the CPU, ask one question, require a non-empty answer (or reasoning for thinking models)."""
    logf = f"{bdir(name)}/loadtest.log"; port = 18099
    with open(logf, "a") as lf:
        p = subprocess.Popen([BIN + "/llama-server", "-m", gguf, "--port", str(port), "--host", "127.0.0.1", "-ngl", "0",
                              "-t", "16", "-c", "2048"], stdout=lf, stderr=subprocess.STDOUT, env={**os.environ, **ENV})
    try:
        for _ in range(450):
            if p.poll() is not None:
                raise RuntimeError("llama-server exited during load")
            try:
                if b'"ok"' in urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2).read(): break
            except Exception:
                time.sleep(2)
        else:
            raise RuntimeError("llama-server did not become ready")
        body = json.dumps({"messages": [{"role": "user", "content": "In one sentence, what is the capital of France?"}],
                           "max_tokens": 400, "temperature": 0}).encode()
        req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=body,
                                     headers={"Content-Type": "application/json"})
        msg = json.loads(urllib.request.urlopen(req, timeout=1800).read())["choices"][0]["message"]
        text = (msg.get("content") or "") + " " + (msg.get("reasoning_content") or "")
        if "Paris" not in text:
            raise RuntimeError(f"load test answer did not mention Paris: {text[:120]!r}")
        return text.strip()[:120]
    finally:
        p.terminate()
        try: p.wait(timeout=60)
        except Exception: p.kill()

def record(name, quant, path, secs):
    a = {"file": os.path.basename(path), "size": os.path.getsize(path), "sha256": sha256(path), "built": now(), "seconds": secs}
    update_json(f"{bdir(name)}/artifacts.json", lambda d: d.__setitem__(quant, a))

def quantize(m, quant):
    name = m["name"]; out = gguf_path(name, quant)
    if quant in read_json(f"{bdir(name)}/artifacts.json", {}):
        return
    cmd = [BIN + "/llama-quantize"]
    if quant in IMATRIX_QUANTS:
        cmd += ["--imatrix", gguf_path(name, "imatrix")]
    cmd += [gguf_path(name, "bf16"), out, quant, "32"]
    secs = run(cmd, f"{bdir(name)}/quantize_{quant}.log", 6 * 3600, ENV)
    record(name, quant, out, secs)
    log(ME, f"{name}: {quant} built in {secs}s ({os.path.getsize(out)/1e9:.1f} GB)")

def phase_a(m):
    name = m["name"]; b = bdir(name)
    sd = stage(m)
    if not done(name, "bf16"):
        secs = run([PY, LC + "/convert_hf_to_gguf.py", sd, "--outtype", "bf16", "--outfile", gguf_path(name, "bf16")],
                   f"{b}/convert.log", 4 * 3600, ENV)
        mark(name, "bf16", {"seconds": secs, "size": os.path.getsize(gguf_path(name, "bf16"))})
        log(ME, f"{name}: bf16 GGUF in {secs}s")
    quantize(m, "Q8_0")
    if not done(name, "imatrix"):
        out = gguf_path(name, "imatrix")
        secs = run([BIN + "/llama-imatrix", "-m", gguf_path(name, "Q8_0"), "-f", CALIB, "-o", out, "-ngl", "0", "-t", "16",
                    "-c", "512"], f"{b}/imatrix.log", 10 * 3600, ENV)
        mark(name, "imatrix", {"seconds": secs, "size": os.path.getsize(out), "sha256": sha256(out)})
        log(ME, f"{name}: importance matrix in {secs}s")
    quantize(m, "Q4_K_M")
    if not done(name, "loadtest"):
        ans = load_test(name, gguf_path(name, "Q4_K_M"))
        mark(name, "loadtest", ans); log(ME, f"{name}: load test ok: {ans[:70]!r}")
    mark(name, "phase_a", now())

def phase_b(m):
    for q in ("Q5_K_M", "IQ4_XS", "Q6_K", "IQ3_M"):
        if stop_requested(): return
        quantize(m, q)
    mark(m["name"], "phase_b", now())

def main():
    ms = models()
    only = sys.argv[1:]                                 # optional: model names to restrict to
    if only: ms = [m for m in ms if m["name"] in only]
    log(ME, f"start: {len(ms)} models")
    for phase, fn in (("phase_a", phase_a), ("phase_b", phase_b)):
        for m in ms:
            if stop_requested():
                log(ME, "STOP requested; exiting"); return
            if done(m["name"], phase): continue
            if phase == "phase_b" and not done(m["name"], "phase_a"): continue
            if done(m["name"], "failed_" + phase) and not only: continue
            try:
                fn(m)
            except Exception as e:
                mark(m["name"], "failed_" + phase, f"{type(e).__name__}: {str(e)[:200]}")
                fail(ME, m["name"], phase, e)
    if not only:
        open(ROOT + "/state/builder_done", "w").write(now())
    log(ME, "finished")

if __name__ == "__main__":
    main()
