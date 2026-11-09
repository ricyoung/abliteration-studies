#!/usr/bin/env python3
"""Paper 30 revision analyses (2026-11-08, response to the ARS review). Existing data only; no GPU; every input is
read-only and no file under analysis/ (outside this folder) is modified.

Run:
    /data/heretic_2/33_2611_does_refusal_removal_get_harder/scripts/figures/.venv/bin/python \
        /data/heretic_2/paper30/v3_2611/analysis/revision_20261108/revision_analyses.py

Outputs (this folder):
  seednoise_derivation.csv   R1: the seed-noise threshold recomputed from the companion study's seed pairs, with the
                             chi-square interval, a cell bootstrap and the batch-size sensitivity (7 pairs).
  seednoise_pairs.csv        R1: the ten seed pairs with the batch size of each run.
  h1_vs_threshold.csv        R8: every H1 / selection-cap contrast against the threshold, its interval and the
                             batch-size sensitivity, with the search-level verdict defined in Methods.
  dev_matched_divergence.csv R9: fewest dev refusals of the 1.x general and 2.0 DiM searches at matched dev-KL caps
                             (each arm's realised dev KL and a grid), from dev_points.csv (1.x Pareto fronts) and the
                             companion study's trials.csv (2.0, all trials).
  search_space.csv           R12: search distributions recorded in the Optuna journals of the 1.x general and 2.0 DiM
                             searches on Qwen2.5-7B and Qwen2.5-32B, and the parameter values of the selected trials.
  search_cost.csv            R15: trials, wall time (journal timestamps) and GPU of each search behind a Heretic cell.
  direction_cosine.csv       R13: per-layer cosine between the 1.x general and code refusal directions, read from the
                             archived rank-1 adapters (lora_B = -weight * v in heretic-llm 1.4.0, model.py l.565-567).
"""
import json, math, struct, glob, os
from datetime import datetime
import numpy as np
import pandas as pd
from scipy import stats

OUT = os.path.dirname(os.path.abspath(__file__))
P30 = "/data/heretic_2/paper30/v3_2611"
P33 = "/data/heretic_2/33_2611_does_refusal_removal_get_harder"
H1X = "/data/heretic_2/results/heretic"
rng = np.random.default_rng(20261108)


# ------------------------------------------------------------------ R1 seed-noise threshold
def sigma_from_d(d):
    """Per-cell run-to-run SD from paired seed differences: sigma^2 = mean(d^2)/2 (mean difference assumed 0)."""
    d = np.asarray(d, float)
    n = len(d)
    s = math.sqrt(np.mean(d ** 2) / 2)
    # sum(d^2)/(2 sigma^2) ~ chi2(n)  ->  95% interval for sigma
    ss = np.sum(d ** 2) / 2
    lo = math.sqrt(ss / stats.chi2.ppf(0.975, n))
    hi = math.sqrt(ss / stats.chi2.ppf(0.025, n))
    boot = [math.sqrt(np.mean(rng.choice(d, n, replace=True) ** 2) / 2) for _ in range(20000)]
    blo, bhi = np.percentile(boot, [2.5, 97.5])
    return n, s, lo, hi, blo, bhi


pairs = pd.read_csv(f"{P33}/results/analysis/seednoise_seed_pairs.csv")
runs = pd.read_csv(f"{P33}/results/analysis/review_sel_seed_runs.csv")
bs = {(r.model, r.method, int(r.seed)): (r.chosen_batch_size, r.oom_fallback) for r in runs.itertuples()}
pairs["batch_s1"] = [bs[(r.model, r.method, 1)][0] for r in pairs.itertuples()]
pairs["batch_s2"] = [bs[(r.model, r.method, 2)][0] for r in pairs.itertuples()]
pairs["same_batch"] = pairs.batch_s1 == pairs.batch_s2
pairs.to_csv(f"{OUT}/seednoise_pairs.csv", index=False)

rows = []
for scope, sub in [("all 10 seed pairs", pairs), ("7 pairs with equal batch size (seed-2 batch-64 runs removed)",
                                                  pairs[pairs.same_batch])]:
    n, s, lo, hi, blo, bhi = sigma_from_d(sub.d_pp)
    rows.append(dict(scope=scope, n_pairs=n, mean_d_pp=sub.d_pp.mean(), median_abs_d_pp=sub.d_pp.abs().median(),
                     max_abs_d_pp=sub.d_pp.abs().max(), sigma_pp=s, sigma_lo_chi2=lo, sigma_hi_chi2=hi,
                     sigma_lo_boot=blo, sigma_hi_boot=bhi, sd_diff_pp=s * math.sqrt(2),
                     sd_diff_lo=lo * math.sqrt(2), sd_diff_hi=hi * math.sqrt(2),
                     threshold_pp=2 * s * math.sqrt(2), threshold_lo=2 * lo * math.sqrt(2),
                     threshold_hi=2 * hi * math.sqrt(2),
                     estimator="sigma=sqrt(mean(d^2)/2); SD_diff=sigma*sqrt(2); threshold=2*SD_diff; "
                               "interval: chi-square(df=n) for sigma, propagated; boot: 20000 cell resamples"))
sd = pd.DataFrame(rows)
sd.to_csv(f"{OUT}/seednoise_derivation.csv", index=False)
T, TLO, THI = sd.loc[0, ["threshold_pp", "threshold_lo", "threshold_hi"]]
TB = sd.loc[1, "threshold_pp"]

# ------------------------------------------------------------------ R8 contrasts vs threshold
tr = pd.read_csv(f"{P30}/analysis/tests_refusal.csv")
want = [("h1_1x_vs_2.0", "qwen2.5-32b__heretic1x-general", "qwen2.5-32b__dim", "H1 primary, Qwen2.5-32B"),
        ("h1_1x_vs_2.0", "qwen2.5-7b__heretic1x-general", "qwen2.5-7b__dim", "H1 primary, Qwen2.5-7B"),
        ("kl_cap_1x", "qwen2.5-32b__heretic1x-general-kl0.1", "qwen2.5-32b__heretic1x-general",
         "same search, KL<=0.1 vs KL<=0.01 export"),
        ("h1_kl_matched", "qwen2.5-32b__heretic1x-general-kl0.1", "qwen2.5-32b__dim", "rule-matched vs 2.0 DiM"),
        ("h1_kl_matched", "qwen2.5-32b__heretic1x-general-kl0.1", "qwen2.5-32b__som", "rule-matched vs SOM"),
        ("h1_kl_matched", "qwen2.5-32b__heretic1x-general-kl0.1", "qwen2.5-32b__mpoa", "rule-matched vs MPOA"),
        ("h1_kl_matched", "qwen2.5-32b__heretic1x-general-kl0.1", "qwen2.5-32b__ara", "rule-matched vs ARA")]
rows = []
for fam, a, b, lab in want:
    r = tr[(tr.family == fam) & (tr.a == a) & (tr.b == b) & (tr.judge == "wg") & (tr.group == "harmful_all")]
    assert len(r) == 1, (fam, a, b)
    r = r.iloc[0]
    e = abs(r["diff"]) * 100
    if fam == "kl_cap_1x":
        verdict = "within one search: no search-to-search variance enters (prompt-level test decides)"
    elif e > THI:
        verdict = "exceeds the threshold and its upper limit (search-level support)"
    elif e > T:
        verdict = "exceeds the point threshold but lies inside its interval (inconclusive at search level)"
    else:
        verdict = "does not exceed the point threshold (inconclusive at search level)"
    rows.append(dict(contrast=lab, family=fam, a=a, b=b, diff_pp=r["diff"] * 100, lo_pp=r.lo * 100, hi_pp=r.hi * 100,
                     a_only=r.a_only, b_only=r.b_only, p_holm=r.p_holm, sig_holm=r.sig_holm, abs_diff_pp=e,
                     threshold_pp=T, threshold_lo=TLO, threshold_hi=THI, threshold_equal_batch_pp=TB,
                     exceeds_point=e > T, exceeds_upper=e > THI, exceeds_lower=e > TLO,
                     exceeds_equal_batch=e > TB, search_level_verdict=verdict))
pd.DataFrame(rows).to_csv(f"{OUT}/h1_vs_threshold.csv", index=False)

# ------------------------------------------------------------------ R9 dev comparison at matched divergence
dp = pd.read_csv(f"{P30}/analysis/dev_points.csv")
t2 = pd.read_csv(f"{P33}/paper/figures/data/trials.csv")
t2 = t2[(t2.study == "main") | (t2.model.isin(["qwen2.5-7b", "qwen2.5-32b"]))]
rows = []
for model in ["qwen2.5-32b", "qwen2.5-7b"]:
    x = dp[(dp.model == model) & (dp.method == "1x_general")]
    y = t2[(t2.model == model) & (t2.method == "dim") & (t2.seed == 1)]
    sel_x = x[x.is_selected].iloc[0]
    sel_y = y[y["is_selected_kl0.1"]].iloc[0]
    trial96 = x[(x.trial == 96)] if model == "qwen2.5-32b" else None
    caps = {"1.x selected dev KL": sel_x.kl, "2.0 DiM selected dev KL": sel_y.kl, "0.01": 0.01, "0.02": 0.02,
            "0.05": 0.05, "0.1": 0.1}
    if trial96 is not None and len(trial96):
        caps["1.x trial #96 dev KL"] = float(trial96.kl.iloc[0])
    for name, c in caps.items():
        xx, yy = x[x.kl <= c + 1e-12], y[y.kl <= c + 1e-12]
        bx = xx.sort_values(["refusals", "kl"]).iloc[0] if len(xx) else None
        by = yy.sort_values(["refusals", "kl"]).iloc[0] if len(yy) else None
        rows.append(dict(model=model, cap_name=name, cap=c,
                         x1_min_refusals=None if bx is None else bx.refusals, x1_trial=None if bx is None else bx.trial,
                         x1_kl=None if bx is None else bx.kl,
                         dim_min_refusals=None if by is None else by.refusals,
                         dim_trial=None if by is None else by.trial, dim_kl=None if by is None else by.kl,
                         note="dev counts scored under each search's own system prompt and marker list; "
                              "not on a common scale"))
    # 1.x Pareto-front trial nearest the DiM selected dev KL
    near = x.iloc[(x.kl - sel_y.kl).abs().argsort()[:1]].iloc[0]
    rows.append(dict(model=model, cap_name="1.x front trial nearest the 2.0 DiM selected dev KL", cap=sel_y.kl,
                     x1_min_refusals=near.refusals, x1_trial=near.trial, x1_kl=near.kl,
                     dim_min_refusals=sel_y.refusals, dim_trial=sel_y.trial, dim_kl=sel_y.kl,
                     note="held-out evaluation of this 1.x trial would need a new re-export (not done)"))
pd.DataFrame(rows).to_csv(f"{OUT}/dev_matched_divergence.csv", index=False)


# ------------------------------------------------------------------ journals: R12 search space, R15 cost
def read_journal(path):
    trials, dists, start, comp, attrs = {}, {}, {}, {}, {}
    tid = -1
    settings = None
    for line in open(path):
        o = json.loads(line)
        oc = o["op_code"]
        if oc == 2 and "settings" in o.get("user_attr", {}):
            settings = json.loads(o["user_attr"]["settings"])
        elif oc == 4:
            tid += 1
            start[tid] = o["datetime_start"]
            trials[tid] = {}
        elif oc == 5:
            trials[o["trial_id"]][o["param_name"]] = o["param_value_internal"]
            d = json.loads(o["distribution"])
            dists[o["param_name"]] = d
        elif oc == 6:
            comp[o["trial_id"]] = o.get("datetime_complete")
        elif oc == 8:
            attrs.setdefault(o["trial_id"], {}).update(o.get("user_attr", {}))
    return settings, trials, dists, start, comp, attrs


def ts(s):
    return datetime.fromisoformat(s)


arms = [("qwen2.5-32b", "1.x general", glob.glob(f"{H1X}/Qwen__Qwen2.5-32B-Instruct__general/study/*.jsonl")[0], 84),
        ("qwen2.5-32b", "1.x general (trial #96)", glob.glob(f"{H1X}/Qwen__Qwen2.5-32B-Instruct__general/study/*.jsonl")[0], 96),
        ("qwen2.5-7b", "1.x general", glob.glob(f"{H1X}/Qwen__Qwen2.5-7B-Instruct__general/study/*.jsonl")[0], 193),
        ("qwen2.5-32b", "2.0 DiM", f"{P33}/runs/mc-h200-1/runs/qwen2.5-32b__dim__core-s1/checkpoints/Qwen--Qwen2--5-32B-Instruct.jsonl", None),
        ("qwen2.5-7b", "2.0 DiM", f"{P33}/runs/local-4090/qwen2.5-7b__dim__core-s1/checkpoints/Qwen--Qwen2--5-7B-Instruct.jsonl", None)]
cells = pd.read_csv(f"{P30}/analysis/cells.csv").set_index("label")
rows = []
for model, arm, path, sel in arms:
    settings, trials, dists, start, comp, attrs = read_journal(path)
    if sel is None:  # 2.0: the selected trial number is in cells.csv (dev_trial, 1-based index as in the journal)
        sel = int(cells.loc[f"{model}__dim", "dev_trial"])
    # trial "index" user attr is 1-based; journal trial ids are 0-based
    tid = next(k for k, v in attrs.items() if int(v.get("index", -1)) == sel)
    for p, d in dists.items():
        a = d["attributes"]
        lo, hi = (a.get("low"), a.get("high")) if d["name"] != "CategoricalDistribution" else (None, None)
        val = trials[tid].get(p)
        if d["name"] == "CategoricalDistribution":
            val = a["choices"][int(val)]
        rows.append(dict(model=model, arm=arm, journal=path, param=p, dist=d["name"], low=lo, high=hi,
                         choices=";".join(a.get("choices", [])) if d["name"] == "CategoricalDistribution" else "",
                         selected_trial_index=sel, selected_value=val,
                         selected_value_effective=(max(0.0, val) if p.endswith("max_weight") and isinstance(val, float)
                                                   else val)))
ss = pd.DataFrame(rows)
ss.to_csv(f"{OUT}/search_space.csv", index=False)

rows = []
for j in sorted(glob.glob(f"{H1X}/Qwen__Qwen2.5-*__*/study/*.jsonl")):
    settings, trials, dists, start, comp, attrs = read_journal(j)
    t0 = min(ts(s) for s in start.values()); t1 = max(ts(s) for s in comp.values() if s)
    rows.append(dict(search=j.split("/")[-3], version="1.x (heretic-llm 1.4.0)", n_trials=len(trials),
                     wall_minutes=(t1 - t0).total_seconds() / 60, gpu="not recorded",
                     source="journal timestamps (first trial start to last trial completion)"))
for meta in sorted(glob.glob(f"{P33}/runs/mc-h200-1/runs/qwen2.5-32b__*__core-s1/meta.txt") +
                   glob.glob(f"{P33}/runs/local-4090/qwen2.5-7b__*__core-s1/meta.txt")):
    run = meta.split("/")[-2]
    j = glob.glob(os.path.dirname(meta) + "/checkpoints/*.jsonl")[0]
    settings, trials, dists, start, comp, attrs = read_journal(j)
    t0 = min(ts(s) for s in start.values()); t1 = max(ts(s) for s in comp.values() if s)
    m = open(meta).read()
    gpu = "NVIDIA H200 NVL" if "H200" in m else ("NVIDIA GeForce RTX 4090" if "4090" in m else "unknown")
    complete = len(trials) >= 100
    rows.append(dict(search=run, version="2.0 / fork (companion study)", n_trials=len(trials),
                     wall_minutes=(t1 - t0).total_seconds() / 60 if complete else None, gpu=gpu,
                     source="journal timestamps; GPU from meta.txt" if complete else
                     "journal in this copy holds fewer than 100 trials (restarted run); wall time not reported"))
pd.DataFrame(rows).to_csv(f"{OUT}/search_cost.csv", index=False)


# ------------------------------------------------------------------ R13 direction cosine (general vs code)
def load_safetensors(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
        base = 8 + n
        out = {}
        for k, v in hdr.items():
            if k == "__metadata__":
                continue
            s, e = v["data_offsets"]
            f.seek(base + s)
            buf = f.read(e - s)
            if v["dtype"] == "BF16":
                a = (np.frombuffer(buf, np.uint16).astype(np.uint32) << 16).view(np.float32)
            elif v["dtype"] == "F16":
                a = np.frombuffer(buf, np.float16).astype(np.float32)
            elif v["dtype"] == "F32":
                a = np.frombuffer(buf, np.float32)
            else:
                raise ValueError(v["dtype"])
            out[k] = a.reshape(v["shape"])
    return out


rows = []
for name in ["Qwen2.5-7B", "Qwen2.5-Coder-7B", "Qwen2.5-14B", "Qwen2.5-Coder-14B", "Qwen2.5-32B", "Qwen2.5-Coder-32B"]:
    g = load_safetensors(f"{H1X}/Qwen__{name}-Instruct__general/adapter/adapter_model.safetensors")
    c = load_safetensors(f"{H1X}/Qwen__{name}-Instruct__code/adapter/adapter_model.safetensors")
    for k in sorted(g):
        if "lora_B" not in k or k not in c:
            continue
        bg, bc = g[k].ravel(), c[k].ravel()
        ng, nc = np.linalg.norm(bg), np.linalg.norm(bc)
        if ng == 0 or nc == 0:
            continue
        layer = int(k.split(".layers.")[1].split(".")[0])
        comp = "attn.o_proj" if "o_proj" in k else ("mlp.down_proj" if "down_proj" in k else k)
        # lora_B = -weight * v, weight >= 0 in 1.4.0, so v = -B/|B| and cos(v_g, v_c) = cos(B_g, B_c)
        rows.append(dict(model=name, layer=layer, component=comp, cosine=float(bg @ bc / (ng * nc)),
                         weight_general=float(ng), weight_code=float(nc)))
dc = pd.DataFrame(rows)
dc.to_csv(f"{OUT}/direction_cosine.csv", index=False)

print(sd.to_string())
print(pd.read_csv(f"{OUT}/h1_vs_threshold.csv")[["contrast", "diff_pp", "exceeds_point", "exceeds_upper",
                                                 "exceeds_equal_batch", "search_level_verdict"]].to_string())
print(pd.read_csv(f"{OUT}/dev_matched_divergence.csv").to_string())
print(ss[ss.param.str.contains("max_weight$|direction")].to_string())
print(pd.read_csv(f"{OUT}/search_cost.csv").to_string())
if len(dc):
    print(dc.groupby(["model", "component"]).cosine.describe().to_string())


# ------------------------------------------------------------------ S7 clustered correlations; S8 equivalence (TOST)
# (appended 2026-11-08) Main-design edited cells as in assemble_p30.correlations(): Qwen2.5 bases, not original, not
# added (KL<=0.1 re-export, external tools); 23 cells.
from scipy.stats import spearmanr
c = pd.read_csv(f"{P30}/analysis/cells.csv")
added = c.label.str.contains("kl0.1|obliteratus|jwest33") | c.model.str.startswith("qwen3.5")
md = c[(c.cond != "base") & ~added & (c.status == "judged")].copy()
assert len(md) == 23, len(md)
md["dev_kl_general"] = np.where(md.method == "1x_code", np.nan, md.dev_kl)
pairs_c = [("wg_rate", "tf_divergence"), ("wg_rate", "dev_kl"), ("wg_rate", "dev_kl_general"),
           ("wg_rate", "capability_mean_delta_pp"), ("kw_minus_wg_pp", "wg_reduction_pp")]
models = sorted(md.model.unique())
rows = []
for a, b in pairs_c:
    x = md[["model", a, b]].dropna()
    rho = spearmanr(x[a], x[b])[0]
    # cluster bootstrap: resample base models with replacement, keep all their cells
    bs = []
    for _ in range(4000):
        pick = rng.choice(models, len(models), replace=True)
        xi = pd.concat([x[x.model == m] for m in pick])
        if xi[a].nunique() > 1 and xi[b].nunique() > 1 and len(xi) >= 4:
            bs.append(spearmanr(xi[a], xi[b])[0])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    # base-model fixed effects: Spearman of within-model centred values (models with >= 2 cells)
    xc = x.groupby("model").filter(lambda g: len(g) >= 2).copy()
    xc[a] = xc[a] - xc.groupby("model")[a].transform("mean")
    xc[b] = xc[b] - xc.groupby("model")[b].transform("mean")
    rfe, pfe = spearmanr(xc[a], xc[b])
    rows.append(dict(x=a, y=b, n_cells=len(x), n_models=x.model.nunique(), spearman_rho=rho,
                     cluster_boot_lo=lo, cluster_boot_hi=hi, n_boot_valid=len(bs),
                     within_model_rho=rfe, within_model_p=pfe, n_cells_within=len(xc)))
pd.DataFrame(rows).to_csv(f"{OUT}/correlations_clustered.csv", index=False)

MARGIN = 2.0  # pp on the mean of the five benchmarks (chosen 2026-11-08, after the results were known)
rows = []
for r in c[(c.cond != "base")].itertuples():
    m, lo95, hi95 = r.capability_mean_delta_pp, r.capability_mean_delta_lo, r.capability_mean_delta_hi
    if pd.isna(lo95):
        continue
    se = (hi95 - lo95) / (2 * 1.959964)
    lo90, hi90 = m - 1.644854 * se, m + 1.644854 * se
    rows.append(dict(label=r.label, added=bool(added[r.Index]), mean_delta_pp=m, lo95=lo95, hi95=hi95, lo90=lo90,
                     hi90=hi90, margin_pp=MARGIN, equivalent_tost=(lo90 > -MARGIN) and (hi90 < MARGIN),
                     mde80_pp=2.8016 * se))
eq = pd.DataFrame(rows)
eq.to_csv(f"{OUT}/capability_equivalence.csv", index=False)
print(pd.read_csv(f"{OUT}/correlations_clustered.csv").to_string())
print(eq.to_string())
