#!/usr/bin/env python3
"""Uploader for the GGUF pipeline: sends each finished file to Hugging Face (<name>-GGUF) and to the Ollama account.

Runs beside builder.py. Picks finished, unpublished files (Q4_K_M first), uploads to Hugging Face with a card and
SHA256SUMS, then creates and pushes the Ollama tag (Q4_K_M also as `latest`). OLMo-2-32B models are gated on Hugging
Face and are not pushed to Ollama. Exits when the builder is done and nothing is left. Stop with state/STOP.
"""
import json, os, subprocess, sys, time, urllib.request
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from card import make_card, sums
from huggingface_hub import HfApi

api = HfApi(); ME = "uploader"
BLOCK = ROOT + "/state/ollama_blocked"

def pub_path(name): return f"{bdir(name)}/published.json"

def pending():
    out = []
    for m in models():
        arts = read_json(f"{bdir(m['name'])}/artifacts.json", {}); pub = read_json(pub_path(m["name"]), {})
        bs = read_json(f"{bdir(m['name'])}/build_state.json", {})
        if not bs.get("loadtest"):
            continue                                    # nothing leaves this machine before the load test passed
        for q, a in arts.items():
            p = pub.get(q, {})
            need_hf = not p.get("hf")
            need_ol = (m["ollama"] and not p.get("ollama") and not os.path.exists(BLOCK)
                       and (q == "Q4_K_M" or pub.get("Q4_K_M", {}).get("ollama")))   # other sizes only after Q4_K_M is up
            if (need_hf or need_ol) and not p.get("failed"):
                out.append((UPLOAD_PRIORITY.index(q), m["params_b"], m["name"], q, m, a, need_hf, need_ol))
    return sorted(out, key=lambda x: x[:4])

def retry(fn, tries=3, wait=120):
    for i in range(tries):
        try:
            return fn()
        except Exception as e:
            if i == tries - 1: raise
            log(ME, f"retry {i+1} after {type(e).__name__}: {str(e)[:150]}"); time.sleep(wait * (i + 1))

def refresh_card(m):
    name = m["name"]; repo = f"richardyoung/{name}-GGUF"
    arts = read_json(f"{bdir(name)}/artifacts.json", {}); pub = read_json(pub_path(name), {})
    bs = read_json(f"{bdir(name)}/build_state.json", {})
    live = {q: a for q, a in arts.items() if pub.get(q, {}).get("hf")}
    im = bs.get("imatrix") if pub.get("imatrix", {}).get("hf") else None
    on_ollama = bool(m["ollama"] and pub.get("Q4_K_M", {}).get("ollama"))
    card = make_card(m, live, im, bs.get("source_revision", ""), on_ollama)
    retry(lambda: api.upload_file(path_or_fileobj=card.encode(), path_in_repo="README.md", repo_id=repo,
                                  commit_message="Update model card"))
    retry(lambda: api.upload_file(path_or_fileobj=sums(m, live, im).encode(), path_in_repo="SHA256SUMS", repo_id=repo,
                                  commit_message="Update SHA256SUMS"))

def hf_publish(m, q, a):
    name = m["name"]; repo = f"richardyoung/{name}-GGUF"; t0 = time.time()
    retry(lambda: api.create_repo(repo, repo_type="model", private=False, exist_ok=True))
    if m["gated"]:
        retry(lambda: api.update_repo_settings(repo, gated="auto"))
    pub = read_json(pub_path(name), {})
    if not pub.get("imatrix", {}).get("hf"):
        retry(lambda: api.upload_file(path_or_fileobj=gguf_path(name, "imatrix"), path_in_repo=f"{name}-imatrix.gguf",
                                      repo_id=repo, commit_message="Add importance matrix"))
        update_json(pub_path(name), lambda d: d.setdefault("imatrix", {}).__setitem__("hf", now()))
    retry(lambda: api.upload_file(path_or_fileobj=gguf_path(name, q), path_in_repo=a["file"], repo_id=repo,
                                  commit_message=f"Add {q}"))
    update_json(pub_path(name), lambda d: d.setdefault(q, {}).__setitem__("hf", now()))
    refresh_card(m)
    secs = time.time() - t0
    log(ME, f"{name}: {q} on Hugging Face in {secs:.0f}s ({a['size']/1e6/max(secs,1):.1f} MB/s)")

def ollama(args, logf, timeout):
    return run(["ollama"] + args, logf, timeout)

def ollama_chat_ok(tag):
    body = json.dumps({"model": tag, "stream": False, "keep_alive": 0, "options": {"temperature": 0, "num_predict": 300, "num_gpu": 0},
                       "messages": [{"role": "user", "content": "In one sentence, what is the capital of France?"}]}).encode()
    req = urllib.request.Request("http://127.0.0.1:11434/api/chat", data=body, headers={"Content-Type": "application/json"})
    msg = json.loads(urllib.request.urlopen(req, timeout=3600).read()).get("message", {})
    text = (msg.get("content") or "") + " " + str(msg.get("thinking") or "")
    if "Paris" not in text:
        raise RuntimeError(f"Ollama test answer did not mention Paris: {text[:120]!r}")

def ollama_publish(m, q, a):
    name = m["name"]; tag = f"{m['ollama_name']}:{q}"; logf = f"{bdir(name)}/ollama.log"; t0 = time.time()
    mf = f"{bdir(name)}/Modelfile.{q}"
    lic = open(f"{ROOT}/licenses/{m['license_family']}.LICENSE.txt").read().replace('"""', "'''")
    open(mf, "w").write(f'FROM {gguf_path(name, q)}\nLICENSE """{lic}"""\n')
    ollama(["create", tag, "-f", mf], logf, 3 * 3600)
    try:
        if q == "Q4_K_M":
            ollama_chat_ok(tag)
        retry(lambda: ollama(["push", tag], logf, 12 * 3600), tries=3, wait=180)
        if q == "Q4_K_M":
            latest = f"{m['ollama_name']}:latest"
            ollama(["cp", tag, latest], logf, 600)
            retry(lambda: ollama(["push", latest], logf, 3600), tries=3, wait=60)
            ollama(["rm", latest], logf, 600)
    finally:
        try: ollama(["rm", tag], logf, 600)
        except Exception: pass
    update_json(pub_path(name), lambda d: d.setdefault(q, {}).__setitem__("ollama", now()))
    secs = time.time() - t0
    log(ME, f"{name}: {q} on Ollama as {tag} in {secs:.0f}s")

def main():
    log(ME, "start")
    while True:
        if os.path.exists(ROOT + "/state/STOP"):
            log(ME, "STOP requested; exiting"); return
        todo = pending()
        if not todo:
            if os.path.exists(ROOT + "/state/builder_done"):
                log(ME, "finished: nothing left to publish"); return
            time.sleep(60); continue
        _, _, name, q, m, a, need_hf, need_ol = todo[0]
        try:
            if need_hf:
                hf_publish(m, q, a)
            if need_ol:
                try:
                    ollama_publish(m, q, a)
                    if q == "Q4_K_M": refresh_card(m)          # the card now shows the Ollama command
                except Exception as e:
                    tail = ""
                    try: tail = open(f"{bdir(name)}/ollama.log").read()[-1500:].lower()
                    except Exception: pass
                    if any(s in tail for s in ("unauthorized", "not authorized", "sign in", "signin", "public key", "permission", "401", "403")):
                        open(BLOCK, "w").write(now() + " " + str(e)[:200])
                        fail(ME, name, f"ollama push {q} (account not authorised; Ollama uploads paused)", e)
                    else:
                        raise
        except Exception as e:
            update_json(pub_path(name), lambda d: d.setdefault(q, {}).__setitem__("failed", f"{type(e).__name__}: {str(e)[:200]}"))
            fail(ME, name, f"publish {q}", e)

if __name__ == "__main__":
    main()
