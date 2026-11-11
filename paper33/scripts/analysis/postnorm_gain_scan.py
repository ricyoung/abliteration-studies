"""Post-sublayer RMSNorm gain scan and Gemma-4-31B DiM trial diagnosis (2026-11-10, ARS re-review NEW-6).

Part 1 (network, run once): fetches only the post_attention_layernorm / post_feedforward_layernorm weight tensors
of the four post-norm base models from Hugging Face via HTTP range reads of the safetensors shards
(no full download) -> results/analysis/postnorm_gains_raw.json.  Skipped if that file exists.
Effective gain: Gemma-2 RMSNorm multiplies by (1 + w); Gemma-4, OLMo-2 and Olmo-3 multiply by w
(transformers models/{gemma2,gemma4,olmo2,olmo3}/modeling_*.py).
Part 2: per model and norm, coefficient of variation of the effective gain, max/median gain, and the channels
whose gain exceeds 20x the median -> results/analysis/postnorm_gain_scan.csv.
Part 3: refusals (dev, keyword, 100 prompts) by first-token-KL band for every trial of the Gemma-4-31B DiM study
-> results/analysis/gemma4_dim_kl_bands.csv.
Read-only on all inputs.
"""
import json, struct, collections, pathlib
import numpy as np
P = pathlib.Path(__file__).resolve().parents[2]; A = P / "results/analysis"
RAW = A / "postnorm_gains_raw.json"
MODELS = {"gemma-2-27b-it": "google/gemma-2-27b-it", "gemma-4-31B-it": "google/gemma-4-31B-it",
          "OLMo-2-32B": "allenai/OLMo-2-0325-32B-Instruct", "Olmo-3.1-32B": "allenai/Olmo-3.1-32B-Instruct"}
OFFSET = {"gemma-2-27b-it": 1.0, "gemma-4-31B-it": 0.0, "OLMo-2-32B": 0.0, "Olmo-3.1-32B": 0.0}

def fetch():
    import requests
    from huggingface_hub import hf_hub_url, get_token
    tok = get_token(); H = {"Authorization": f"Bearer {tok}"} if tok else {}
    def get(url, rng=None):
        h = dict(H)
        if rng: h["Range"] = f"bytes={rng[0]}-{rng[1]}"
        r = requests.get(url, headers=h, allow_redirects=True, timeout=120); r.raise_for_status(); return r
    out = {}
    for name, repo in MODELS.items():
        idx = get(hf_hub_url(repo, "model.safetensors.index.json")).json()["weight_map"]
        keys = [k for k in idx if ("post_attention_layernorm" in k or "post_feedforward_layernorm" in k)
                and "vision" not in k and "audio" not in k]
        byfile = collections.defaultdict(list)
        for k in keys: byfile[idx[k]].append(k)
        res = {}
        for f, ks in byfile.items():
            url = hf_hub_url(repo, f)
            n = struct.unpack("<Q", get(url, (0, 7)).content)[0]
            hdr = json.loads(get(url, (8, 8 + n - 1)).content)
            for k in ks:
                m = hdr[k]; a, b = m["data_offsets"]
                raw = get(url, (8 + n + a, 8 + n + b - 1)).content
                if m["dtype"] == "BF16":
                    v = (np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)
                else:
                    v = np.frombuffer(raw, dtype={"F32": np.float32, "F16": np.float16}[m["dtype"]]).astype(np.float32)
                res[k] = v.tolist()
        out[name] = res
    RAW.write_text(json.dumps(out))

if not RAW.exists(): fetch()
d = json.loads(RAW.read_text())
rows = ["model,norm,n_vectors,cv_median,cv_p10,cv_p90,maxmed_median,maxmed_max,n_vectors_channel_gt20x,top_outlier_channels"]
for m, t in d.items():
    for kind in ["post_attention_layernorm", "post_feedforward_layernorm"]:
        cv, mx, top = [], [], collections.Counter()
        for k, v in t.items():
            if not k.endswith(kind + ".weight"): continue
            g = np.array(v) + OFFSET[m]; ga = np.abs(g)
            cv.append(np.std(g) / abs(np.mean(g))); r = ga.max() / np.median(ga); mx.append(r)
            if r > 20: top[int(ga.argmax())] += 1
        rows.append(f"{m},{kind},{len(cv)},{np.median(cv):.3f},{np.percentile(cv,10):.3f},{np.percentile(cv,90):.3f},"
                    f"{np.median(mx):.2f},{np.max(mx):.1f},{sum(top.values())},"
                    + ";".join(f"{c}x{n}" for c, n in top.most_common(3)))
(A / "postnorm_gain_scan.csv").write_text("\n".join(rows) + "\n")

J = P / "runs/mc-h200-1/runs/gemma-4-31b__dim__core-s1/checkpoints/google--gemma-4-31B-it.jsonl"
if not J.exists():  # layout of the released repository
    J = P / "runs/mc-h200-1/gemma-4-31b__dim__core-s1/optuna_journal/google--gemma-4-31B-it.jsonl"
vals = {}
for l in J.read_text().splitlines():
    x = json.loads(l)
    if x.get("op_code") == 6 and x.get("values"): vals[x["trial_id"]] = x["values"]
bands = [(0, 0.1), (0.1, 0.5), (0.5, 1), (1, 3), (3, 8), (8, 1e9)]
out = ["kl_lo,kl_hi,n_trials,min_refusal_rate,max_refusal_rate"]
for lo, hi in bands:
    s = [v[0] for v in vals.values() if lo <= v[1] < hi]
    out.append(f"{lo},{hi if hi < 1e9 else 'inf'},{len(s)},{min(s) if s else ''},{max(s) if s else ''}")
(A / "gemma4_dim_kl_bands.csv").write_text("\n".join(out) + "\n")
print(open(A / "postnorm_gain_scan.csv").read()); print("\n".join(out))
