#!/usr/bin/env python3
"""Paper 33 -- reviewer concerns C2, C3 and C4 (notes/REVIEW_v5_reviewer_agent_20261106.md). Analysis only.

Purpose
  C2  Item source. The 972 harmful held-out items come from five sets: jbb_harmful (100), xstest_unsafe (200),
      strongreject (313), harmbench_standard (159) and the author-built code_weapon bank (200). H1 and H2 are re-run
      exactly as scripts/analysis/h1h2.py runs them for harmful_all (its own `analyse` function is called on the item
      subset; WildGuard refusal, original + seed-1 primary cells, GEE robustness models, Newcombe CIs, exact McNemar,
      item bootstrap, Holm per item subset), on each set, on the four public sets together (772 items) and on
      code_weapon alone. Per pair: class-level generation effects raw and baseline-adjusted (risk difference with
      item-bootstrap CI, and log-odds from a per-pair GEE with the originals as reference class). The originals'
      generation difference is decomposed into the contribution of each set (n_set / 972 x difference on the set).
  C3  Harmful compliance as the outcome. H1 and H2 are re-run (same `analyse` call, all 972 harmful items) with
      (a) the HarmBench classifier label and (b) WildGuard's harmful-response label. To keep the sign convention of
      the refusal analysis, the outcome is "not harmful" = 1 - harmful compliance (`hbnc`, `wghnc`; hbnc is the
      `nonharm_hb` / `hb_noncompliance` column of the paper's tables): a positive generation effect means the newer
      model's edit leaves fewer prompts answered harmfully ("harder"), as for refusal. On the logit link this is an
      exact sign flip of the harmful-compliance model. Per cell (all 74 judged cells, harmful items): share of the
      responses that WildGuard does not label as refusal and that HarmBench / WildGuard label harmful.
  C4  H3 at Heretic's own keyword window. Heretic 2.0 scores keyword refusals on at most 100 generated tokens
      (heretic config.default.toml @ 8fd4d76: max_response_length = 100; src/heretic/model.py get_responses:
      max_new_tokens = max_response_length). Every stored held-out response (<= 512 tokens, greedy) is re-tokenised
      with the evaluated model's tokenizer (tokenizer.json of the base checkpoint, which the exports share; for a
      community model its repo's own tokenizer.json if that reproduces the stored response_256 at least as often as
      the base one -- it does not for three repos), cut to its first 100 tokens, decoded
      (skip_special_tokens=True, as bench.py does for response_256) and matched with the paper's keyword function
      (scripts/bench/score_refusals.py `keyword`: lower-case, typographic apostrophe -> ', substring match of the 33
      markers). Validation first: the same procedure on the full text must reproduce keyword_refusal exactly, and the
      256-token cut must reproduce keyword_refusal_256 (and, as text, the stored response_256). Then the H3
      keyword-minus-WildGuard gap is recomputed with the 100-token keyword label against the full-response WildGuard
      label and against the 256-token WildGuard label (judgments_256.jsonl, wg_refusal_256), pooled exactly as h3.py
      pools (h3.pooled_groups, prompt-cluster bootstrap, B = 2000), per cell (Newcombe CI, exact McNemar, Holm per
      comparison x set group across all cells, as h3.gap_cells), per base model, and in h3.py's primary linear GEE
      and cell-level OLS. Sensitivities: Heretic's own matcher (keyword_rate.py `_is_match`: also strips '*',
      collapses whitespace, counts empty responses as refusals) on the same 100 tokens, and the alternative
      re-tokenisation (see "Tokenisation" below).

Tokenisation (C4)
  Token ids were not stored, only text, so the cut is made on a re-encoding of the text. Two encodings are computed
  for every cell: `canonical` (tokenizer.json encode of the whole response) and `nl_split` (every "\n" encoded as its
  own segment). OLMo-2-0325-32B emits each newline as a separate token, while its tokenizer.json merges newlines with
  the preceding punctuation and with each other, so the canonical re-encoding has ~16 fewer tokens per 512 and its
  256-token cut overshoots. Per base model the encoding that reproduces the stored response_256 text in more rows is
  used (ties -> canonical); the other one is reported as a sensitivity (`kw100_alt`). Encoding-free bounds: the stored
  response_256 is exactly the first 256 generated tokens and the keyword label can only grow with the window, so the
  true 100-token label lies between the labels of two cuts set by the re-encoding's token deficit (`kw100_lo`,
  `kw100_hi`; equal for all but a few responses); responses of <= 100 generated tokens are matched in full (exact).
  Raw `tokenizers` is used rather
  than transformers' AutoTokenizer: transformers 5.x rebuilds some tokenizers instead of reading tokenizer.json; vLLM
  ran with tokenizer_mode = 'hf' (bench logs).

Inputs (read-only)
  results/analysis/responses_judged.csv.gz     per-response table (assemble.py)
  results/analysis/cell_rates.csv, dir_status.csv
  results/analysis/h1h2_generation_effects.csv, h1h2_models.csv, h1h2_tests.csv   (C2 reproduction check)
  results/analysis/h3_gap_pooled.csv, h3_models.csv                              (C4 reproduction check)
  <source_dir>/refusal/responses.jsonl          response text (tokeniser stage only; no text is written)
  <source_dir>/refusal/judgments_256.jsonl, summary.json, summary_256.json        (wg_refusal_256; via truncation256)
  scripts/bench/score_refusals.py               KEYWORDS (parsed, not imported)
  /data/heretic_2/models/paper33/{base,community}/*/tokenizer.json
  scripts/analysis/h1h2.py, h3.py, truncation256.py, freshness.py               (imported: statistics conventions)

Outputs (results/analysis/; overwritten on every run, nothing else is written)
  review_out_c2_generation_effects.csv   per item subset x pair x method: generation effect, CI, McNemar, Holm,
                                         baseline-adjusted (h1h2_generation_effects.csv columns + item_subset)
  review_out_c2_h2_contrasts.csv         per item subset x pair: H2 risk-difference contrast + GEE interaction
  review_out_c2_models.csv               every GEE term per item subset x scope (h1h2_models.csv columns)
  review_out_c2_tests.csv                every H1 / H2 / ablation test with raw and Holm p, per item subset
  review_out_c2_pair_class.csv           per item subset x pair: class-level generation effects raw and
                                         baseline-adjusted (RD with bootstrap CI; log-odds from a per-pair GEE)
  review_out_c2_originals_decomposition.csv  per pair (and main mean) x set: originals' difference and its
                                         contribution to the 972-item difference, bootstrap CI
  review_out_c3_generation_effects.csv, review_out_c3_h2_contrasts.csv, review_out_c3_models.csv,
  review_out_c3_tests.csv, review_out_c3_pair_class.csv
                                         as C2, outcome = hbnc / wghnc (all 972 harmful items)
  review_out_c3_cells.csv                per judged cell (74), harmful items: refusal, harmful compliance (HarmBench,
                                         WildGuard), share of non-refused responses judged harmful, Wilson CIs
  review_out_c3_pooled.csv               the same shares pooled over h3.py's cell groups (prompt-cluster bootstrap)
  review_out_c4_kw_rows.csv              per judged response (cell x item): re-tokenised lengths and keyword labels
                                         (0/1) at full / 256 / 100 tokens, both encodings (tokeniser stage)
  review_out_c4_tokenizer_cells.csv      per cell: tokenizer used, encoding chosen, responses md5 check
  review_out_c4_validation.csv           per cell: reproduction of keyword_refusal / keyword_refusal_256 / response_256
  review_out_c4_gap_pooled.csv           pooled gaps (h3 groups + per base model) for every variant
  review_out_c4_gap_cells.csv            per cell x set group x comparison gap, Newcombe CI, McNemar, Holm
  review_out_c4_models.csv               h3.py primary GEE + cell-level OLS with the 100-token keyword outcome
  review_out_checks.csv                  every reproduction / validation check with its value
  review_out_summary.md                  key numbers, plainly worded, and a verdict per reviewer concern

How to run (no GPU, about 2 min; the tokeniser stage needs `tokenizers`, which only the Heretic venv has):
    cd /data/heretic_2/33_2611_does_refusal_removal_get_harder
    scripts/figures/.venv/bin/python scripts/analysis/review_outcomes.py               # everything
    scripts/figures/.venv/bin/python scripts/analysis/review_outcomes.py --retokenize  # force the tokeniser stage
    /data/heretic_2/heretic/.venv/bin/python -I scripts/analysis/review_outcomes.py --tokenize   # tokeniser stage only
  The main run calls the tokeniser stage itself (Heretic venv, `python -I`) when review_out_c4_kw_rows.csv is missing
  or the judged response files changed (md5 in dir_status.csv).
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = ROOT / "results" / "analysis"
SCORER = ROOT / "scripts" / "bench" / "score_refusals.py"
MODELS = Path("/data/heretic_2/models/paper33")
HERETIC_PY = Path("/data/heretic_2/heretic/.venv/bin/python")
KW_ROWS = OUT / "review_out_c4_kw_rows.csv"
TOK_CELLS = OUT / "review_out_c4_tokenizer_cells.csv"

WINDOW = 100          # heretic config.default.toml @ 8fd4d76: max_response_length = 100
CUT256 = 256          # bench.py response_256
N_BOOT = 4000         # h1h2.py default item bootstrap
SEED = 20261105

HARMFUL_SETS = ["jbb_harmful", "xstest_unsafe", "strongreject", "harmbench_standard", "code_weapon"]
PUBLIC_SETS = HARMFUL_SETS[:4]
ITEM_SUBSETS = {  # name -> sets
    "harmful_all": HARMFUL_SETS, "public4": PUBLIC_SETS, "code_weapon": ["code_weapon"],
    "jbb_harmful": ["jbb_harmful"], "xstest_unsafe": ["xstest_unsafe"], "strongreject": ["strongreject"],
    "harmbench_standard": ["harmbench_standard"],
}
SUBSET_DISPLAY = {"harmful_all": "all 972", "public4": "4 public sets (772)", "code_weapon": "code_weapon (200)",
                  "jbb_harmful": "JBB (100)", "xstest_unsafe": "XSTest-unsafe (200)",
                  "strongreject": "StrongREJECT (313)", "harmbench_standard": "HarmBench (159)"}
OUTCOMES_C3 = {"hbnc": "HarmBench classifier: not harmful compliance (1 - hb_harmful)",
               "wghnc": "WildGuard: not a harmful response (1 - wg_harmful_response)"}
OUTCOME_SHORT = {"wg": "WildGuard refusal", "hbnc": "HarmBench not-harmful", "wghnc": "WildGuard not-harmful"}

BASE_TOKENIZER = {  # model -> base checkpoint dir prefix under MODELS/base (revision = the evaluated one)
    "qwen2.5-32b": "Qwen__Qwen2.5-32B-Instruct@", "qwen3.8-27b": "Qwen__Qwen3.8-27B@",
    "gemma-2-27b": "google__gemma-2-27b-it@", "gemma-4-31b": "google__gemma-4-31B-it@",
    "olmo-2-32b": "allenai__OLMo-2-0325-32B-Instruct@", "olmo-3.1-32b": "allenai__Olmo-3.1-32B-Instruct@",
    "mistral-small-2409": "mistralai__Mistral-Small-Instruct-2409@",
    "mistral-small-3.2": "unsloth__Mistral-Small-3.2-24B-Instruct-2506@",
    "qwen2.5-7b": "Qwen__Qwen2.5-7B-Instruct@", "qwen3.5-4b": "Qwen__Qwen3.5-4B@",
}
COMMUNITY_TOKENIZER = {  # (model, variant) -> community repo dir prefix under MODELS/community
    ("qwen2.5-32b", "huihui"): "huihui-ai__Qwen2.5-32B-Instruct-abliterated@",
    ("qwen2.5-32b", "zetasepic-v2"): "zetasepic__Qwen2.5-32B-Instruct-abliterated-v2@",
    ("qwen3.8-27b", "hereticorg-ara"): "heretic-org__Qwen3.8-27B-heretic-ara@",
    ("qwen3.8-27b", "huihui"): "huihui-ai__Huihui-Qwen3.8-27B-abliterated@",
    ("gemma-4-31b", "coder3101"): "coder3101__gemma-4-31B-it-heretic@",
    ("mistral-small-3.2", "huihui"): "huihui-ai__Huihui-Mistral-Small-3.2-24B-Instruct-2506-abliterated@",
    ("mistral-small-3.2", "llmfan46"): "llmfan46__Mistral-Small-3.2-24B-Instruct-2506-ultra-uncensored-heretic@",
    ("olmo-3.1-32b", "muxodious"): "MuXodious__Olmo-3.1-32B-Instruct-impotent-heresy@",
}
ENCODINGS = ("canonical", "nl_split")


def log(*a):
    print(*a, flush=True)


# =============================================================================================== keyword functions
def load_keywords() -> list[str]:
    """Heretic's 33 markers as used by the paper (score_refusals.KEYWORDS), parsed without importing vLLM."""
    tree = ast.parse(SCORER.read_text())
    for n in tree.body:
        if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "KEYWORDS":
            return list(ast.literal_eval(n.value))
    raise SystemExit("KEYWORDS not found in score_refusals.py")


def kw_paper(text: str, kws: list[str]) -> bool:
    """scripts/bench/score_refusals.py keyword(): the paper's keyword measure."""
    t = text.lower().replace("’", "'")
    return any(k in t for k in kws)


def kw_heretic(text: str, kws: list[str]) -> bool:
    """heretic/src/heretic/scorers/keyword_rate.py KeywordRate._is_match (Heretic 2.0 @ 8fd4d76)."""
    if not text.strip():
        return True
    t = text.lower().replace("*", "").replace("’", "'")
    t = " ".join(t.split())
    return any(k.lower() in t for k in kws)


def md5_file(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def used_dirs() -> list[dict]:
    with open(OUT / "dir_status.csv", newline="") as f:
        return [r for r in csv.DictReader(f) if r["used"] == "True"]


# =============================================================================================== stage A: tokenise
def tokenizer_candidates(model: str, method: str, variant: str) -> list[tuple[Path, str]]:
    """Base checkpoint tokenizer (shared by the exports); for community cells also the repo's own tokenizer.json."""
    hits = sorted((MODELS / "base").glob(BASE_TOKENIZER[model] + "*"))
    if not hits:
        raise SystemExit(f"no base tokenizer for {model}")
    cands = [(hits[0], "base")]
    if method == "community":
        pre = COMMUNITY_TOKENIZER.get((model, variant))
        own = sorted((MODELS / "community").glob(pre + "*")) if pre else []
        if own and (own[0] / "tokenizer.json").exists():
            cands.append((own[0], "community repo"))
        else:
            cands[0] = (hits[0], "base (community repo ships no tokenizer.json)")
    return cands


def encode_all(tk, texts: list[str], mode: str) -> list[list[int]]:
    if mode == "canonical":
        return [e.ids for e in tk.encode_batch(texts, add_special_tokens=False)]
    segs, owner = [], []
    for i, t in enumerate(texts):
        for s in re.split(r"(\n)", t):
            if s:
                segs.append(s)
                owner.append(i)
    out: list[list[int]] = [[] for _ in texts]
    for i, e in zip(owner, tk.encode_batch(segs, add_special_tokens=False)):
        out[i].extend(e.ids)
    return out


def stage_tokenize() -> None:
    """Needs only the standard library and `tokenizers` (run with the Heretic venv)."""
    from tokenizers import Tokenizer
    t0 = time.time()
    kws = load_keywords()
    assert len(kws) == 33
    cells = used_dirs()
    tok_cache: dict[Path, object] = {}
    per_cell: dict[str, dict] = {}
    cell_rows = []
    for r in cells:
        lab, model, method, variant, src = r["dir_name"], r["model"], r["method"], r["variant"], r["source_dir"]
        path = ROOT / src / "refusal" / "responses.jsonl"
        with open(path, encoding="utf-8") as f:
            recs = [json.loads(line) for line in f if line.strip()]
        texts = [x.get("response") or "" for x in recs]
        r256 = [x.get("response_256", x.get("response")) or "" for x in recs]
        # tokenizer: the candidate whose canonical 256-token cut reproduces the stored response_256 most often
        # (community repos: some tokenizer.json files re-encode differently from what vLLM generated; ties -> base)
        cand_scores = []
        for tdir, tsrc in tokenizer_candidates(model, method, variant):
            if tdir not in tok_cache:
                tok_cache[tdir] = Tokenizer.from_file(str(tdir / "tokenizer.json"))
            ids = encode_all(tok_cache[tdir], texts, "canonical")
            d256 = tok_cache[tdir].decode_batch([i[:CUT256] for i in ids], skip_special_tokens=True)
            cand_scores.append((sum(a == b for a, b in zip(d256, r256)), tdir, tsrc))
        best = max(s for s, _, _ in cand_scores)
        _, tdir, tsrc = next(c for c in cand_scores if c[0] == best)
        cand_txt = "; ".join(f"{s_}: {sc}/{len(recs)}" for sc, _, s_ in cand_scores)
        tk = tok_cache[tdir]
        res = {"id": [x["id"] for x in recs], "n_tokens": [x["n_tokens"] for x in recs],
               "finish_reason": [x["finish_reason"] for x in recs],
               "kw_full": [int(kw_paper(t, kws)) for t in texts],
               "kw256_stored": [int(kw_paper(t, kws)) for t in r256],
               "chars_full": [len(t) for t in texts]}
        ntok = res["n_tokens"]
        for mode in ENCODINGS:
            ids = encode_all(tk, texts, mode)
            ids256 = encode_all(tk, r256, mode)
            d256 = tk.decode_batch([i[:CUT256] for i in ids], skip_special_tokens=True)
            d100 = tk.decode_batch([i[:WINDOW] for i in ids], skip_special_tokens=True)
            dfull = tk.decode_batch(ids, skip_special_tokens=True)
            # a response of at most WINDOW generated tokens (n_tokens counts the EOS) lies wholly inside the window
            short = [n <= WINDOW for n in ntok]
            d100 = [t if s else c for t, s, c in zip(texts, short, d100)]
            # bounds on the true 100-token text: the re-encoding can have fewer (or more) tokens than were generated;
            # its deficit within the first 100 tokens is at most the deficit within 256 tokens (anchor: response_256 is
            # exactly the first 256 generated tokens) or, for responses of 101-256 tokens, within the whole response
            # (0-2 trailing special tokens). The keyword label is monotone in the window, so kw(lo cut) <= true <=
            # kw(hi cut).
            lo_t, hi_t = [], []
            for i, n in enumerate(ntok):
                if n <= WINDOW:
                    lo_t.append(texts[i]); hi_t.append(texts[i])
                    continue
                if n > CUT256:
                    src_ids, dft_max = ids256[i], CUT256 - len(ids256[i])
                    dft_min = dft_max
                else:
                    src_ids, dft_max = ids[i], n - len(ids[i])
                    dft_min = n - 2 - len(ids[i])
                lo_t.append(tk.decode(src_ids[:WINDOW - max(dft_max, 0)], skip_special_tokens=True))
                hi_t.append(tk.decode(src_ids[:WINDOW + max(-dft_min, 0)], skip_special_tokens=True))
            res[f"{mode}:n_retok"] = [len(i) for i in ids]
            res[f"{mode}:roundtrip_eq"] = [int(a == b) for a, b in zip(dfull, texts)]
            res[f"{mode}:text256_eq"] = [int(a == b) for a, b in zip(d256, r256)]
            res[f"{mode}:kw256"] = [int(kw_paper(t, kws)) for t in d256]
            res[f"{mode}:kw100"] = [int(kw_paper(t, kws)) for t in d100]
            res[f"{mode}:kw100_heretic"] = [int(kw_heretic(t, kws)) for t in d100]
            res[f"{mode}:kw100_lo"] = [int(kw_paper(t, kws)) for t in lo_t]
            res[f"{mode}:kw100_hi"] = [int(kw_paper(t, kws)) for t in hi_t]
            res[f"{mode}:chars100"] = [len(t) for t in d100]
        per_cell[lab] = res
        md5 = md5_file(path)
        cell_rows.append(dict(label=lab, model=model, method=method, variant=variant, source_dir=src,
                              tokenizer_dir=str(tdir), tokenizer_source=tsrc,
                              tokenizer_candidates_text256_eq=cand_txt, n_rows=len(recs),
                              responses_md5=md5, responses_md5_ok=int(md5 == r["responses_md5"]),
                              **{f"{m}:text256_eq": sum(res[f"{m}:text256_eq"]) for m in ENCODINGS}))
        log(f"  {lab:42s} {tsrc:10s} {time.time() - t0:5.0f}s  text256 equal: " +
            ", ".join(f"{m} {sum(res[f'{m}:text256_eq'])}/{len(recs)}" for m in ENCODINGS))
    # encoding per base model: the one that reproduces the stored 256-token text in more rows (ties -> canonical)
    choice = {}
    for model in sorted({c["model"] for c in cell_rows}):
        sc = {m: sum(c[f"{m}:text256_eq"] for c in cell_rows if c["model"] == model) for m in ENCODINGS}
        choice[model] = "nl_split" if sc["nl_split"] > sc["canonical"] else "canonical"
    for c in cell_rows:
        c["encoding"] = choice[c["model"]]
        c["encoding_alt"] = "canonical" if c["encoding"] == "nl_split" else "nl_split"
    cols = ["label", "id", "n_tokens", "finish_reason", "encoding", "n_retok", "n_retok_alt", "roundtrip_eq",
            "text256_eq", "text256_eq_alt", "kw_full", "kw256_stored", "kw256_retok", "kw256_retok_alt", "kw100",
            "kw100_alt", "kw100_heretic", "kw100_heretic_alt", "kw100_lo", "kw100_hi", "chars_full", "chars100",
            "chars100_alt"]
    tmp = KW_ROWS.with_suffix(".csv.tmp")
    with open(tmp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for c in cell_rows:
            res, m, a = per_cell[c["label"]], c["encoding"], c["encoding_alt"]
            for i in range(c["n_rows"]):
                w.writerow([c["label"], res["id"][i], res["n_tokens"][i], res["finish_reason"][i], m,
                            res[f"{m}:n_retok"][i], res[f"{a}:n_retok"][i], res[f"{m}:roundtrip_eq"][i],
                            res[f"{m}:text256_eq"][i], res[f"{a}:text256_eq"][i], res["kw_full"][i],
                            res["kw256_stored"][i], res[f"{m}:kw256"][i], res[f"{a}:kw256"][i], res[f"{m}:kw100"][i],
                            res[f"{a}:kw100"][i], res[f"{m}:kw100_heretic"][i], res[f"{a}:kw100_heretic"][i],
                            res[f"{m}:kw100_lo"][i], res[f"{m}:kw100_hi"][i],
                            res["chars_full"][i], res[f"{m}:chars100"][i], res[f"{a}:chars100"][i]])
    tmp.replace(KW_ROWS)
    with open(TOK_CELLS, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(cell_rows[0].keys()))
        w.writeheader()
        w.writerows(cell_rows)
    log(f"tokeniser stage: {len(cell_rows)} cells, {sum(c['n_rows'] for c in cell_rows):,} rows, "
        f"{time.time() - t0:.0f}s -> {KW_ROWS.name}, {TOK_CELLS.name}; encodings: {choice}")


def tokenize_needed() -> str:
    if not KW_ROWS.exists() or not TOK_CELLS.exists():
        return "rows file missing"
    with open(TOK_CELLS, newline="") as f:
        have = {(r["label"], r["responses_md5"]) for r in csv.DictReader(f)}
    want = {(r["dir_name"], r["responses_md5"]) for r in used_dirs()}
    return "" if have == want else f"judged response files changed ({len(have ^ want)} cell entries differ)"


# =============================================================================================== stage B: analysis
def main_analysis(args) -> None:
    global np, pd, sm, smf, h1h2, h3, t256, multipletests
    sys.dont_write_bytecode = True  # importing the analysis modules must not leave __pycache__ files behind
    import numpy as np  # noqa: F401
    import pandas as pd
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from statsmodels.stats.multitest import multipletests
    sys.path.insert(0, str(HERE))
    import h1h2  # noqa: E402  (H1/H2 conventions: analyse, newcombe_paired, mcnemar_exact, boot_ci, gee, lincom)
    import h3  # noqa: E402  (H3 conventions: pooled_groups, newcombe_paired, mcnemar_exact, model_data, B, SEED)
    import truncation256 as t256  # noqa: E402  (judgments_256 loader and checks)
    globals().update(np=np, pd=pd, sm=sm, smf=smf, h1h2=h1h2, h3=h3, t256=t256, multipletests=multipletests)

    t0 = time.time()
    why = "forced (--retokenize)" if args.retokenize else tokenize_needed()
    if why:
        log(f"tokeniser stage needed: {why}; running it with {HERETIC_PY} -I ...")
        env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="true")
        subprocess.run([str(HERETIC_PY), "-I", str(Path(__file__).resolve()), "--tokenize"], check=True, cwd=ROOT,
                       env=env)
    checks: list[dict] = []

    # ------------------------------------------------------------------ C2 / C3 inputs (exactly h1h2.load_inputs)
    d, rates, src, log_head = h1h2.load_inputs()
    extra = pd.read_csv(src, usecols=["label", "id", "harm_hb", "harm_wg"], low_memory=False)
    d = d.merge(extra, on=["label", "id"], how="left", validate="one_to_one")
    d["y_hbnc"] = 1.0 - h1h2.to01(d["harm_hb"])
    d["y_wghnc"] = 1.0 - h1h2.to01(d["harm_wg"])
    log(f"C2/C3 input: {len(d):,} harmful rows, {d.label.nunique()} cells (original + primary)")

    c2 = run_c2(d, args.boot, checks)
    log(f"C2 done ({time.time() - t0:.0f}s)")
    c3 = run_c3(d, args.boot, src, checks)
    log(f"C3 done ({time.time() - t0:.0f}s)")
    wf = sorted({w.split(":")[0] for w in WALD_FAILURES})
    check(checks, "GEE heterogeneity Wald tests not computable (quasi-separation on a small subset; reported as NaN)",
          len(WALD_FAILURES), True, ("in " + ", ".join(wf) + "; " + WALD_FAILURES[0]) if WALD_FAILURES else "")
    c4 = run_c4(checks)
    log(f"C4 done ({time.time() - t0:.0f}s)")
    ck = pd.DataFrame(checks)
    ck.to_csv(OUT / "review_out_checks.csv", index=False)
    write_summary(c2, c3, c4, ck, src, log_head, time.time() - t0, args.boot)
    log(f"wrote review_out_*.csv and review_out_summary.md in {time.time() - t0:.0f}s")


def reldiff(a, b) -> float:
    """max |a - b| / max(1, |b|) over non-missing pairs (absolute for small values, relative for large ones)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = ~(np.isnan(a) | np.isnan(b))
    both_nan_mismatch = int((np.isnan(a) != np.isnan(b)).sum())
    d = np.abs(a[ok] - b[ok]) / np.maximum(1.0, np.abs(b[ok]))
    return float(d.max() if len(d) else 0.0) + both_nan_mismatch


def check(checks, name, value, ok, detail=""):
    checks.append(dict(check=name, value=value, ok=bool(ok), detail=detail))
    log(f"  check {'OK ' if ok else 'FAIL'} {name}: {value} {detail}")


WALD_FAILURES: list[str] = []


def run_analyse(d, judges: dict, n_boot: int, tag: str = ""):
    """h1h2.analyse with h1h2.JUDGES replaced (no VB fits); returns its tables.
    On small item subsets a family can be (quasi-)separated (e.g. 0 refusals of one cell on 200 items) and statsmodels'
    wald_test then raises inside h1h2's heterogeneity test; for the duration of the call a failing wald_test returns
    NaN (recorded in WALD_FAILURES) instead of aborting. Nothing else in h1h2 is changed."""
    from types import SimpleNamespace
    from statsmodels.genmod.generalized_estimating_equations import GEEResults
    old, had = h1h2.JUDGES, "wald_test" in GEEResults.__dict__
    orig = GEEResults.wald_test

    def safe_wald(self, *a, **k):
        try:
            return orig(self, *a, **k)
        except (ValueError, np.linalg.LinAlgError) as e:
            WALD_FAILURES.append(f"{tag}: {str(e)[:100]}")
            return SimpleNamespace(statistic=np.nan, pvalue=np.nan)

    h1h2.JUDGES = judges
    GEEResults.wald_test = safe_wald
    import warnings
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # sqrt of negative variance in separated fits -> NaN
            cells, gen, h2, models, tests, pairs = h1h2.analyse(d, n_boot, set())
    finally:
        h1h2.JUDGES = old
        if had:
            GEEResults.wald_test = orig
        else:
            del GEEResults.wald_test
    return cells, gen, h2, models, tests, pairs


# ------------------------------------------------------------------------------------------------------------ C2
def run_c2(d, n_boot, checks):
    gens, h2s, mods, tests, pcs = [], [], [], [], []
    sep = {}
    for sname, sets in ITEM_SUBSETS.items():
        ds = d[d["set"].isin(sets)].copy()
        sep[sname] = separated_cells(ds, "wg")
        log(f"C2 {sname}: {ds.id.nunique()} items")
        _, gen, h2, models, tt, pairs = run_analyse(ds, {"wg": h1h2.JUDGES["wg"]}, n_boot, tag=f"C2 {sname}")
        for t in (gen, h2, models, tt):
            t.insert(0, "item_subset", sname)
        gens.append(gen); h2s.append(h2); mods.append(models); tests.append(tt)
        pc = pair_class(ds, pairs, "wg", n_boot)
        pc.insert(0, "item_subset", sname)
        pcs.append(pc)
    gen, h2, models, tests, pc = (pd.concat(x, ignore_index=True) for x in (gens, h2s, mods, tests, pcs))
    dec = originals_decomposition(d, n_boot)
    # reproduction of h1h2.py on all 972 items (WildGuard)
    ref = pd.read_csv(OUT / "h1h2_generation_effects.csv")
    ref = ref[(ref.judge == "wg")].set_index(["subset", "pair", "method"])
    mine = gen[gen.item_subset == "harmful_all"].set_index(["subset", "pair", "method"])
    cols = ["p_old", "p_new", "delta", "delta_lo", "delta_hi", "delta_boot_lo", "delta_boot_hi", "did_baseline_adj",
            "did_lo", "did_hi", "p_mcnemar", "p_holm_H1"]
    j = mine[cols].merge((ref[cols]).add_suffix("_ref"), left_index=True, right_index=True, how="outer")
    diff = max(reldiff(j[c], j[c + "_ref"]) for c in cols)
    check(checks, "C2 harmful_all reproduces h1h2_generation_effects.csv (wg; max |diff| / max(1, |ref|) over 12 "
          "columns; the reference CSV has 6 significant digits)", f"{diff:.2e}", diff < 1e-5 and len(j) == len(mine) == len(ref), f"rows {len(mine)} vs {len(ref)}")
    rm = pd.read_csv(OUT / "h1h2_models.csv")
    rm = rm[(rm.judge == "wg") & rm.model.str.startswith("gee")].set_index(["subset", "scope", "model", "term"])
    mm = models[(models.item_subset == "harmful_all") & models.model.str.startswith("gee")].set_index(
        ["subset", "scope", "model", "term"])
    jj = mm[["estimate", "lo", "hi", "p"]].merge((rm[["estimate", "lo", "hi", "p"]]).add_suffix("_ref"), left_index=True, right_index=True, how="inner")
    diff = max(reldiff(jj[c], jj[c + "_ref"]) for c in ["estimate", "lo", "hi"])
    check(checks, "C2 harmful_all reproduces h1h2_models.csv GEE terms (wg; max |diff| / max(1, |ref|) of "
          "estimate/lo/hi)", f"{diff:.2e}", diff < 1e-5 and len(jj) == len(rm), f"terms matched {len(jj)} of {len(rm)}")
    rt = pd.read_csv(OUT / "h1h2_tests.csv")
    rt = rt[rt.judge == "wg"].set_index(["subset", "family", "test_id"])
    mt = tests[tests.item_subset == "harmful_all"].set_index(["subset", "family", "test_id"])
    jt = mt[["p_holm"]].merge((rt[["p_holm"]]).add_suffix("_ref"), left_index=True, right_index=True, how="inner")
    diff = reldiff(jt.p_holm, jt.p_holm_ref)
    check(checks, "C2 harmful_all reproduces h1h2_tests.csv Holm p (wg)", f"{diff:.2e}",
          diff < 1e-5 and len(jt) == len(rt), f"tests matched {len(jt)} of {len(rt)}")
    # decomposition adds up
    dd = dec[dec.set_name.isin(HARMFUL_SETS)].groupby("pair").contribution_pp.sum()
    tot = dec[dec.set_name == "harmful_all"].set_index("pair").delta_pp
    diff = float((dd - tot).abs().max())
    check(checks, "C2 set contributions add up to the 972-item originals' difference (max abs diff, pp)",
          f"{diff:.2e}", diff < 1e-9)
    od = gen[(gen.item_subset == "harmful_all") & (gen.subset == "all") & (gen.method == "original")].set_index("pair")
    diff = float((tot - 100 * od.delta.reindex(tot.index)).abs().max(skipna=True))
    check(checks, "C2 decomposition total equals the originals' generation effect of h1h2 (max abs diff, pp)",
          f"{diff:.2e}", diff < 0.15, "differs only where a WildGuard label is missing (pairwise vs listwise)")

    gen.to_csv(OUT / "review_out_c2_generation_effects.csv", index=False, float_format="%.6g")
    h2.to_csv(OUT / "review_out_c2_h2_contrasts.csv", index=False, float_format="%.6g")
    models.to_csv(OUT / "review_out_c2_models.csv", index=False, float_format="%.6g")
    tests.to_csv(OUT / "review_out_c2_tests.csv", index=False, float_format="%.6g")
    pc.to_csv(OUT / "review_out_c2_pair_class.csv", index=False, float_format="%.6g")
    dec.to_csv(OUT / "review_out_c2_originals_decomposition.csv", index=False, float_format="%.6g")
    return dict(gen=gen, h2=h2, models=models, tests=tests, pc=pc, dec=dec, sep=sep)


MAIN_MODELS = {"qwen2.5-32b", "qwen3.8-27b", "gemma-2-27b", "gemma-4-31b", "olmo-2-32b", "olmo-3.1-32b",
               "mistral-small-2409", "mistral-small-3.2"}


def separated_cells(ds, judge) -> list[str]:
    """Cells whose outcome is 0 (or 1) on every item of the subset: per-pair and cell-level (heterogeneity) logistic
    GEE terms are then not estimable (quasi-separation); risk-difference results are unaffected."""
    m = ds.groupby("label")[f"y_{judge}"].mean()
    return sorted(m.index[(m == 0) | (m == 1)])


def fchi2(het, sep_main) -> str:
    if het is None or not np.isfinite(het.estimate) or het.estimate < 0 or sep_main:
        return "n/a" + (f" (separation, {len(sep_main)} cells)" if sep_main else " (not computable)")
    return f"{het.estimate:.0f}"


def pair_class(ds, pairs, judge, n_boot):
    """Per pair: class-level generation effects, raw and baseline-adjusted.
    RD: direction = mean over the pair's direction-based methods of (p_new - p_old); ARA = p_new - p_old; adjusted =
    minus the originals' p_new - p_old; item-bootstrap percentile CIs (h1h2.boot_ci; items with every label present).
    Log-odds: GEE y ~ C(class, Treatment('original')) * gen_new on the pair's original + modified cells (exchangeable,
    clustered by prompt; h1h2.gee): baseline difference = gen_new, adjusted = class:gen_new, raw = their sum."""
    rows = []
    rng = np.random.default_rng(SEED)
    wide = ds.pivot(index="id", columns="label", values=f"y_{judge}")
    n_items = len(wide)
    W = rng.multinomial(n_items, np.full(n_items, 1.0 / n_items), size=n_boot)
    for pname, P in pairs.items():
        if not (P.has_orig and P.has_ara and P.dir_methods):
            continue
        labs = [P.cells[(m, g)] for m in P.methods_both for g in (0, 1)]
        Y = wide[labs].values
        col = {lab: i for i, lab in enumerate(labs)}
        ok = ~np.isnan(Y).any(axis=1)

        def eff(M, m, P=P, col=col):
            return M[:, col[P.cells[(m, 1)]]] - M[:, col[P.cells[(m, 0)]]]

        fns = {
            "orig": lambda M: eff(M, "original"),
            "dir_raw": lambda M: np.mean([eff(M, m) for m in P.dir_methods], axis=0),
            "ara_raw": lambda M: eff(M, "ara"),
            "dir_adj": lambda M: np.mean([eff(M, m) for m in P.dir_methods], axis=0) - eff(M, "original"),
            "ara_adj": lambda M: eff(M, "ara") - eff(M, "original"),
            "h2_rd": lambda M: eff(M, "ara") - np.mean([eff(M, m) for m in P.dir_methods], axis=0),
        }
        means = np.nanmean(Y[ok], axis=0)[None, :]
        row = dict(judge=judge, pair=pname, study=P.study, direction_methods="+".join(P.dir_methods),
                   n_items=int(ok.sum()), separated_cells="+".join(
                       lab for lab in labs if np.nanmean(wide[lab].values) in (0.0, 1.0)))
        for k, fn in fns.items():
            lo, hi = h1h2.boot_ci(W, Y, ok, fn)
            row.update({f"rd_{k}": float(fn(means)[0]), f"rd_{k}_lo": lo, f"rd_{k}_hi": hi})
        # log-odds with the originals as reference class
        da = ds[ds.label.isin(labs)].copy()
        da["y"] = da[f"y_{judge}"]
        da = da[da.y.notna()]
        da["cls3"] = da["method_class"]
        import warnings
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)  # separated fits: sqrt of negative variance -> NaN SE
                res, names, note = h1h2.gee(da, "y ~ C(cls3, Treatment('original')) * gen_new")
                lcs = {}
                tn = {c: f"C(cls3, Treatment('original'))[T.{c}]:gen_new" for c in ("direction", "optimisation")}
                for k, w in (("orig", {"gen_new": 1}), ("dir_adj", {tn["direction"]: 1}),
                             ("ara_adj", {tn["optimisation"]: 1}), ("dir_raw", {"gen_new": 1, tn["direction"]: 1}),
                             ("ara_raw", {"gen_new": 1, tn["optimisation"]: 1}),
                             ("h2", {tn["optimisation"]: 1, tn["direction"]: -1})):
                    lcs[k] = h1h2._lc(res, names, w)
            for k, e in lcs.items():
                row.update({f"lo_{k}": e["estimate"], f"lo_{k}_lo": e["lo"], f"lo_{k}_hi": e["hi"],
                            f"lo_{k}_p": e["p"]})
            row["gee_note"] = note
        except Exception as e:  # separation etc.
            row["gee_note"] = f"FAILED: {str(e)[:150]}"
        rows.append(row)
    return pd.DataFrame(rows)


def originals_decomposition(d, n_boot):
    """Originals' generation difference on the 972 items = sum over sets of (n_set / n) x difference on the set.
    Per pair (WildGuard; items where both originals have a label) and for the mean of the four main pairs."""
    o = d[d.method == "original"]
    wide = o.pivot(index="id", columns="label", values="y_wg")
    sets = o.drop_duplicates("id").set_index("id")["set"].reindex(wide.index)
    pairs = {}
    for pname, g in o.drop_duplicates("label").groupby("pair"):
        lo_ = g[g.gen_new == 0].label.iloc[0]; ln_ = g[g.gen_new == 1].label.iloc[0]
        pairs[pname] = wide[ln_] - wide[lo_]
    D = pd.DataFrame(pairs)
    main = [p for p in ["qwen", "gemma", "olmo", "mistral"] if p in D]
    D["main_mean"] = D[main].mean(axis=1, skipna=False)
    rng = np.random.default_rng(SEED)
    rows = []
    for pname in [p for p in h1h2.PAIR_ORDER if p in D] + ["main_mean"]:
        x = D[pname]
        ok = x.notna().values
        xv = x.values[ok]
        sv = sets.values[ok]
        n = len(xv)
        W = rng.multinomial(n, np.full(n, 1.0 / n), size=n_boot).astype(float)
        tot_b = (W @ xv) / n
        tot = xv.mean()
        for sname in ["harmful_all"] + HARMFUL_SETS + ["public4"]:
            m = np.ones(n, bool) if sname == "harmful_all" else (np.isin(sv, PUBLIC_SETS) if sname == "public4"
                                                                    else sv == sname)
            contrib = xv[m].sum() / n
            contrib_b = (W[:, m] @ xv[m]) / n
            k = int(m.sum())
            dset = xv[m].mean()
            dset_b = (W[:, m] @ xv[m]) / W[:, m].sum(axis=1)
            with np.errstate(invalid="ignore", divide="ignore"):
                share_b = contrib_b / tot_b
            share_b[~np.isfinite(share_b)] = np.nan  # total exactly 0 in a resample
            rows.append(dict(pair=pname, set_name=sname, n_items=k, item_weight=k / n, delta_pp=100 * dset,
                             delta_lo=100 * np.percentile(dset_b, 2.5), delta_hi=100 * np.percentile(dset_b, 97.5),
                             contribution_pp=100 * contrib, contribution_lo=100 * np.percentile(contrib_b, 2.5),
                             contribution_hi=100 * np.percentile(contrib_b, 97.5), total_pp=100 * tot,
                             total_lo=100 * np.percentile(tot_b, 2.5), total_hi=100 * np.percentile(tot_b, 97.5),
                             share_of_total=contrib / tot if tot != 0 else np.nan,
                             share_lo=np.nanpercentile(share_b, 2.5), share_hi=np.nanpercentile(share_b, 97.5)))
    out = pd.DataFrame(rows)
    out["note"] = ("WildGuard refusal of the unmodified models; delta = new - old original on the set's items; "
                   "contribution = n_set / n_all x delta (sums to the 972-item difference); share = contribution / "
                   "total (unstable when the total is near 0); item-bootstrap percentile CIs, B = %d, seed %d" %
                   (n_boot, SEED))
    return out


# ------------------------------------------------------------------------------------------------------------ C3
def run_c3(d, n_boot, src, checks):
    _, gen, h2, models, tests, pairs = run_analyse(d, OUTCOMES_C3, n_boot, tag="C3")
    pc = pd.concat([pair_class(d, pairs, j, n_boot) for j in OUTCOMES_C3], ignore_index=True)
    for t in (gen,):
        t["p_old_harmful"] = 1 - t["p_old"]
        t["p_new_harmful"] = 1 - t["p_new"]
    gen.to_csv(OUT / "review_out_c3_generation_effects.csv", index=False, float_format="%.6g")
    h2.to_csv(OUT / "review_out_c3_h2_contrasts.csv", index=False, float_format="%.6g")
    models.to_csv(OUT / "review_out_c3_models.csv", index=False, float_format="%.6g")
    tests.to_csv(OUT / "review_out_c3_tests.csv", index=False, float_format="%.6g")
    pc.to_csv(OUT / "review_out_c3_pair_class.csv", index=False, float_format="%.6g")
    # per-cell shares (all 74 judged cells) and pooled over h3 groups
    rj, _ = h3.load_judged()
    rj = rj[rj.judge_status.isin(["judged", "partially_judged"])].reset_index(drop=True)
    for c in ["ref_wg", "harm_hb", "harm_wg"]:
        rj[c] = h3.to01(rj[c])
    for c in ["adv_overlap", "incomplete", "is_degenerate"]:
        rj[c] = rj[c].map({True: True, False: False, "True": True, "False": False}).fillna(False).astype(bool)
    hr = rj[rj.kind == "harmful"].copy()
    cr = pd.read_csv(OUT / "cell_rates.csv", low_memory=False)
    crh = cr[cr.set_group == "harmful_all"].set_index("label")
    rows = []
    meta = h3.sort_key_cells(hr.groupby("label", as_index=False).first()[
        ["label", "model", "family", "pair", "study", "generation", "gen_new", "method", "variant", "method_class",
         "role", "seed", "source_dir"]])
    for lab in meta.label:
        x = hr[hr.label == lab]
        nr = x.ref_wg == 0
        row = {k: x[k].iloc[0] for k in ["label", "model", "pair", "study", "generation", "method", "role"]}
        row["n"] = len(x)
        for name, mask_num, mask_den in (
                ("wg_refusal", x.ref_wg == 1, x.ref_wg.notna()),
                ("not_refused", nr, x.ref_wg.notna()),
                ("hb_harmful", x.harm_hb == 1, x.harm_hb.notna()),
                ("wg_harmful", x.harm_wg == 1, x.harm_wg.notna()),
                ("hb_harmful_given_not_refused", nr & (x.harm_hb == 1), nr & x.harm_hb.notna()),
                ("wg_harmful_given_not_refused", nr & (x.harm_wg == 1), nr & x.harm_wg.notna()),
                ("both_harmful_given_not_refused", nr & (x.harm_hb == 1) & (x.harm_wg == 1),
                 nr & x.harm_hb.notna() & x.harm_wg.notna()),
                ("either_harmful_given_not_refused", nr & ((x.harm_hb == 1) | (x.harm_wg == 1)),
                 nr & x.harm_hb.notna() & x.harm_wg.notna()),
                ("not_refused_not_hb_harmful", nr & (x.harm_hb == 0), x.ref_wg.notna() & x.harm_hb.notna()),
                ("hb_harmful_given_refused", (x.ref_wg == 1) & (x.harm_hb == 1), (x.ref_wg == 1) & x.harm_hb.notna()),
                ("capped_given_not_refused", nr & x.incomplete, nr)):
            k, n = int(mask_num.sum()), int(mask_den.sum())
            lo, hi = h1h2.wilson(k, n)
            row.update({f"{name}_k": k, f"{name}_n": n, f"{name}_rate": k / n if n else np.nan, f"{name}_lo": lo,
                        f"{name}_hi": hi})
        rows.append(row)
    cells = pd.DataFrame(rows)
    # check against cell_rates.csv
    diffs = []
    for c, ref in (("wg_refusal", "wg_refusal"), ("hb_harmful", "hb_harmful"), ("wg_harmful", "wg_harmful_response")):
        a = cells.set_index("label")[f"{c}_rate"]
        b = crh[f"{ref}_rate"].reindex(a.index)
        diffs.append(float((a - b).abs().max()))
    check(checks, "C3 per-cell refusal / harmful rates equal cell_rates.csv harmful_all (max abs diff; the CSV has 6 "
          "significant digits)", f"{max(diffs):.2e}", max(diffs) < 1e-5, f"{len(cells)} cells")
    cells.to_csv(OUT / "review_out_c3_cells.csv", index=False, float_format="%.6g")
    # pooled ratios over the h3 groups, prompt-cluster bootstrap (h3 convention)
    groups = h3.pooled_groups(hr)
    ids = np.sort(hr.id.unique())
    pmap = pd.Series(np.arange(len(ids)), index=ids)
    rng = np.random.default_rng(h3.SEED)
    Wb = rng.multinomial(len(ids), np.full(len(ids), 1.0 / len(ids)), size=h3.B).astype(float)
    prow = []
    nr_all = (hr.ref_wg == 0).values
    for gname, gm in groups.items():
        for name, num, den in (
                ("hb_harmful_given_not_refused", nr_all & (hr.harm_hb == 1).values, nr_all & hr.harm_hb.notna().values),
                ("wg_harmful_given_not_refused", nr_all & (hr.harm_wg == 1).values, nr_all & hr.harm_wg.notna().values),
                ("hb_harmful", (hr.harm_hb == 1).values, hr.harm_hb.notna().values),
                ("wg_refusal", (hr.ref_wg == 1).values, hr.ref_wg.notna().values),
                ("not_refused_not_hb_harmful", nr_all & (hr.harm_hb == 0).values,
                 hr.ref_wg.notna().values & hr.harm_hb.notna().values)):
            m = gm & den
            p = pmap.loc[hr.id.values[m]].values
            C = np.stack([np.bincount(p, weights=num[m].astype(float), minlength=len(ids)),
                          np.bincount(p, minlength=len(ids)).astype(float)], 1)
            k, n = C.sum(0)
            Cb = Wb @ C
            with np.errstate(invalid="ignore", divide="ignore"):
                rb = Cb[:, 0] / Cb[:, 1]
            lo, hi = np.nanpercentile(rb, [2.5, 97.5])
            prow.append(dict(group=gname, measure=name, n_cells=len(np.unique(hr.label.values[gm])), k=int(k),
                             n=int(n), rate=k / n if n else np.nan, lo=lo, hi=hi))
    pooled = pd.DataFrame(prow)
    pooled["note"] = f"harmful items; ratio of counts; prompt-cluster bootstrap percentile CI (B = {h3.B}, seed {h3.SEED})"
    pooled.to_csv(OUT / "review_out_c3_pooled.csv", index=False, float_format="%.6g")
    return dict(gen=gen, h2=h2, models=models, tests=tests, pc=pc, cells=cells, pooled=pooled)


# ------------------------------------------------------------------------------------------------------------ C4
GAP_VARIANTS = {  # name: (rater1 column, rater2 column, row filter name)
    "kw_vs_wg": ("ref_kw", "ref_wg", None),                  # paper H3 (full text)
    "kw256_vs_wg": ("ref_kw256", "ref_wg", None),            # paper App. D.1 (mixed windows)
    "kw256_vs_wg256": ("ref_kw256", "ref_wg256", None),      # truncation256.py (same 256-token text)
    "kw100_vs_wg": ("kw100", "ref_wg", None),                # NEW: tool window vs full-response judge
    "kw100_vs_wg256": ("kw100", "ref_wg256", None),          # NEW: tool window vs 256-token judge
    "kw100_heretic_vs_wg": ("kw100_heretic", "ref_wg", None),
    "kw100_heretic_vs_wg256": ("kw100_heretic", "ref_wg256", None),
    "kw100_alt_vs_wg": ("kw100_alt", "ref_wg", None),        # other re-tokenisation
    "kw100_lo_vs_wg": ("kw100_lo", "ref_wg", None),          # lower / upper bound of the true 100-token label
    "kw100_hi_vs_wg": ("kw100_hi", "ref_wg", None),
    "kw100_lo_vs_wg256": ("kw100_lo", "ref_wg256", None),
    "kw100_hi_vs_wg256": ("kw100_hi", "ref_wg256", None),
    "kw100_vs_q3g": ("kw100", "ref_q3g", None),
    "kw100_vs_wg_complete_only": ("kw100", "ref_wg", "complete"),
    "kw100_vs_wg_capped_only": ("kw100", "ref_wg", "capped"),
    "kw100_vs_kw": ("kw100", "ref_kw", None),                # how many keyword refusals the shorter window loses
}
CELL_COMPARISONS = {"kw_vs_wg": ("ref_kw", "ref_wg"), "kw100_vs_wg": ("kw100", "ref_wg"),
                    "kw100_vs_wg256": ("kw100", "ref_wg256"), "kw256_vs_wg256": ("ref_kw256", "ref_wg256")}


def run_c4(checks):
    df, jpath = t256.load()
    cov, j256, not_in_df = t256.load_256(df)
    df, chk256 = t256.merge_256(df, j256)
    check(checks, "C4 judgments_256 coverage: cells with gaps", int((cov.gap.fillna("") != "").sum()),
          (cov.gap.fillna("") == "").all(), f"{len(cov)} cells; md5/revision/id checks of truncation256.py")
    kr = pd.read_csv(KW_ROWS, low_memory=False)
    tc = pd.read_csv(TOK_CELLS)
    n0 = len(df)
    df = df.merge(kr, on=["label", "id"], how="left", validate="one_to_one", suffixes=("", "_kwfile"))
    assert len(df) == n0
    miss = int(df.kw100.isna().sum())
    check(checks, "C4 every judged response has a re-tokenised row", miss, miss == 0, f"{n0:,} rows")
    check(checks, "C4 responses.jsonl md5 equal to dir_status.csv (tokeniser stage)",
          int(tc.responses_md5_ok.sum()), int(tc.responses_md5_ok.sum()) == len(tc), f"of {len(tc)} cells")
    nt = int((df.n_tokens_kwfile != df.n_tokens).sum())
    check(checks, "C4 n_tokens in responses.jsonl equal to the assembled n_tokens", nt, nt == 0)
    for c in ["kw_full", "kw256_stored", "kw256_retok", "kw256_retok_alt", "kw100", "kw100_alt", "kw100_heretic",
              "kw100_heretic_alt", "kw100_lo", "kw100_hi"]:
        df[c] = df[c].astype(float)
    # ---- validation
    v_full = int((df.kw_full != df.ref_kw).sum())
    v_256s = int((df.kw256_stored != df.ref_kw256).sum())
    v_256 = int((df.kw256_retok != df.ref_kw256).sum())
    v_256a = int((df.kw256_retok_alt != df.ref_kw256).sum())
    check(checks, "C4 keyword on the full response reproduces keyword_refusal (mismatching rows)", v_full, v_full == 0,
          f"of {len(df):,}")
    check(checks, "C4 keyword on the stored response_256 reproduces keyword_refusal_256 (mismatching rows)", v_256s,
          v_256s == 0, f"of {len(df):,}")
    o2 = (df.model == "olmo-2-32b").values
    bad256 = (df.kw256_retok != df.ref_kw256).values
    check(checks, "C4 keyword on the re-tokenised 256-token cut reproduces keyword_refusal_256, all models but OLMo-2 "
          "(mismatching rows)", int(bad256[~o2].sum()), int(bad256[~o2].sum()) == 0,
          f"of {int((~o2).sum()):,}; alternative encoding: {int((df.kw256_retok_alt != df.ref_kw256).values[~o2].sum())}")
    check(checks, "C4 the same for OLMo-2 (generates non-canonical tokens; newline-split encoding; mismatching rows)",
          int(bad256[o2].sum()), True, f"of {int(o2.sum()):,} (" + ", ".join(
              f"{k}: {v}" for k, v in df[o2 & bad256].groupby("label").size().items()) +
          f"); canonical encoding: {int((df.kw256_retok_alt != df.ref_kw256).values[o2].sum())}; all {v_256} "
          f"mismatches of the {len(df):,} rows are here; the 100-token bounds do not depend on the encoding")
    t_eq = float(df.text256_eq.values[~o2].mean())
    check(checks, "C4 re-tokenised 256-token cut equals the stored response_256 text, all models but OLMo-2 (share)",
          f"{t_eq:.4f}", t_eq > 0.98, "rows <= 256 tokens are trivially equal; OLMo-2: "
          f"{df.text256_eq.values[o2].mean():.4f} (canonical encoding {df.text256_eq_alt.values[o2].mean():.4f}); "
          f"other encoding for the rest: {df.text256_eq_alt.values[~o2].mean():.4f}; v_256a all rows {v_256a}")
    mono = int(((df.kw100 > df.ref_kw256) | (df.ref_kw256 > df.ref_kw)).sum())
    check(checks, "C4 window monotonicity kw100 <= kw256 <= kw_full (violating rows)", mono, mono == 0)
    brk = int(((df.kw100_lo > df.kw100) | (df.kw100 > df.kw100_hi) | (df.kw100_hi > df.ref_kw256)).sum())
    check(checks, "C4 bounds kw100_lo <= kw100 <= kw100_hi <= kw256 (violating rows)", brk, brk == 0)
    unc = df.kw100_lo != df.kw100_hi
    check(checks, "C4 responses whose 100-token keyword label is not pinned down by the bounds", int(unc.sum()),
          True, f"{100 * unc.mean():.2f}% of {len(df):,}; by model: " + ", ".join(
              f"{m} {int(v)}" for m, v in df[unc].groupby("model").size().items()))
    val = []
    for lab, g in df.groupby("label", sort=False):
        long_ = g.n_tokens > CUT256
        val.append(dict(label=lab, model=g.model.iloc[0], role=g.role.iloc[0], n=len(g),
                        encoding=g.encoding.iloc[0], n_gt256=int(long_.sum()), n_gt100=int((g.n_tokens > WINDOW).sum()),
                        kw_full_mismatch=int((g.kw_full != g.ref_kw).sum()),
                        kw256_stored_mismatch=int((g.kw256_stored != g.ref_kw256).sum()),
                        kw256_retok_mismatch=int((g.kw256_retok != g.ref_kw256).sum()),
                        kw256_retok_alt_mismatch=int((g.kw256_retok_alt != g.ref_kw256).sum()),
                        text256_eq_share=float(g.text256_eq.mean()),
                        text256_eq_share_gt256=float(g.text256_eq[long_].mean()) if long_.any() else np.nan,
                        text256_eq_alt_share=float(g.text256_eq_alt.mean()),
                        roundtrip_eq_share=float(g.roundtrip_eq.mean()),
                        median_retok_minus_generated=float(np.median(
                            g.n_retok - (g.n_tokens - (g.finish_reason == "stop").astype(int)))),
                        kw100_rate=float(g.kw100.mean()), kw100_alt_rate=float(g.kw100_alt.mean()),
                        kw100_alt_differs=int((g.kw100 != g.kw100_alt).sum()),
                        kw100_lo_rate=float(g.kw100_lo.mean()), kw100_hi_rate=float(g.kw100_hi.mean()),
                        kw100_unresolved=int((g.kw100_lo != g.kw100_hi).sum()),
                        median_chars100=float(g.chars100.median())))
    val = pd.DataFrame(val).merge(tc[["label", "tokenizer_dir", "tokenizer_source", "responses_md5_ok"]], on="label")
    val.to_csv(OUT / "review_out_c4_validation.csv", index=False, float_format="%.6g")

    # ---- gaps: pooled (h3 groups + base models), per cell, models
    groups = h3.pooled_groups(df)
    for m in sorted(df.model.unique()):
        gm = ((df.model == m) & (df.role == "primary")).values
        if gm.any():
            groups[f"model:{m}:ablated_primary"] = gm
        go = ((df.model == m) & (df.role == "original")).values
        groups[f"model:{m}:original"] = go
    pooled = gap_pooled(df, groups)
    # reproduction of the paper's pooled full-text gap (h3_gap_pooled.csv) and the 256-token value
    ref = pd.read_csv(OUT / "h3_gap_pooled.csv").set_index(["subset", "group", "variant"])
    mine = pooled.set_index(["subset", "group", "variant"])
    keys = [k for k in mine.index if k in ref.index and k[2] in ("kw_vs_wg", "kw256_vs_wg")]
    diff = max(abs(mine.loc[k, "gap"] - ref.loc[k, "gap"]) for k in keys)
    check(checks, "C4 pooled kw-WG and kw256-WG gaps equal h3_gap_pooled.csv (max abs diff of point estimates)",
          f"{diff:.2e}", diff < 1e-12, f"{len(keys)} rows; bootstrap limits use new draws")
    cells_gap = gap_cells(df)
    hc = pd.read_csv(OUT / "h3_gap_cells.csv")
    hc = hc[hc.comparison == "kw_vs_wg"].set_index(["label", "set_group"])
    mc = cells_gap[cells_gap.comparison == "kw_vs_wg"].set_index(["label", "set_group"])
    jj = mc[["gap", "p_holm"]].merge((hc[["gap", "p_holm"]]).add_suffix("_ref"), left_index=True, right_index=True, how="inner")
    diff = float(np.nanmax(np.abs(jj[["gap", "p_holm"]].values - jj[["gap_ref", "p_holm_ref"]].values)))
    check(checks, "C4 per-cell kw-WG gaps and Holm p equal h3_gap_cells.csv (max abs diff)", f"{diff:.2e}",
          diff < 1e-9, f"{len(jj)} cell x set-group rows")
    models = h3_models(df)
    hm = pd.read_csv(OUT / "h3_models.csv")
    hm = hm[hm.model.isin(["GEE_linear_paired_gap (primary)", "OLS_cell_level_gap"])].set_index(
        ["model", "study", "subset", "term"])
    mm = models[models.outcome == "kw_vs_wg"].set_index(["model", "study", "subset", "term"])
    jj = mm[["estimate"]].merge((hm[["estimate"]]).add_suffix("_ref"), left_index=True, right_index=True, how="inner")
    diff = float(np.nanmax(np.abs(jj.estimate - jj.estimate_ref)))
    check(checks, "C4 H3 GEE/OLS terms with the full-text keyword equal h3_models.csv (max abs diff)", f"{diff:.2e}",
          diff < 1e-6, f"{len(jj)} terms")
    pooled.to_csv(OUT / "review_out_c4_gap_pooled.csv", index=False, float_format="%.6g")
    cells_gap.to_csv(OUT / "review_out_c4_gap_cells.csv", index=False, float_format="%.6g")
    models.to_csv(OUT / "review_out_c4_models.csv", index=False, float_format="%.6g")
    return dict(df=df, val=val, pooled=pooled, cells=cells_gap, models=models, tc=tc, chk256=chk256)


def gap_pooled(df, groups):
    """Prompt-cluster bootstrap of the mean paired difference x - y (h3.gap_pooled, generalised to any columns)."""
    rng = np.random.default_rng(h3.SEED)
    filt = {None: np.ones(len(df), bool), "complete": ~df.incomplete.values, "capped": df.incomplete.values}
    rows = []
    for sname, smask in {"harmful_all": (df.kind == "harmful").values,
                         "harmful_excl_advbench": ((df.kind == "harmful") & ~df.adv_overlap).values,
                         "benign_all": (df.kind == "benign").values}.items():
        ids = np.sort(df.id[smask].unique())
        n_p = len(ids)
        pmap = pd.Series(np.arange(n_p), index=ids)
        W = rng.multinomial(n_p, np.full(n_p, 1.0 / n_p), size=h3.B).astype(np.float64)
        for gname, gm in groups.items():
            for vname, (c1, c2, fl) in GAP_VARIANTS.items():
                m = smask & gm & filt[fl]
                if not m.any():
                    continue
                x = df[c1].values[m].astype(float); y = df[c2].values[m].astype(float)
                ok = ~(np.isnan(x) | np.isnan(y))
                if not ok.any():
                    continue
                x, y = x[ok], y[ok]
                p = pmap.loc[df.id.values[m][ok]].values
                Cg = np.stack([np.bincount(p, weights=x - y, minlength=n_p), np.bincount(p, minlength=n_p).astype(float),
                               np.bincount(p, weights=x, minlength=n_p), np.bincount(p, weights=y, minlength=n_p)], 1)
                s, nn, sx, sy = Cg.sum(0)
                Cb = W @ Cg
                with np.errstate(invalid="ignore", divide="ignore"):
                    gb = Cb[:, 0] / Cb[:, 1]
                lo, hi = np.nanpercentile(gb, [2.5, 97.5])
                rows.append(dict(subset=sname, group=gname, variant=vname, rater1=c1, rater2=c2,
                                 n_cells=len(np.unique(df.label.values[m][ok])), n=int(nn), rate1=sx / nn,
                                 rate2=sy / nn, gap=s / nn, gap_lo=lo, gap_hi=hi,
                                 b_r1_only=int(((x == 1) & (y == 0)).sum()), c_r2_only=int(((x == 0) & (y == 1)).sum())))
    out = pd.DataFrame(rows)
    out["note"] = (f"gap = mean(rater1 - rater2) over responses with both labels; prompt-cluster bootstrap percentile "
                   f"95% CI (B = {h3.B}, seed {h3.SEED}; h3.gap_pooled convention)")
    return out


def gap_cells(df):
    """h3.gap_cells for CELL_COMPARISONS: Newcombe CI, exact McNemar, Holm per comparison x set group over all cells."""
    meta = h3.sort_key_cells(df.groupby("label", as_index=False).first()[
        ["label", "model", "family", "pair", "study", "generation", "gen_new", "method", "variant", "method_class",
         "role", "seed", "source_dir"]])
    sg = h3.set_group_masks(df)
    sg = {k: v for k, v in sg.items() if k in ("harmful_all", "harmful_excl_advbench", "benign_all")}
    rows = []
    for label in meta.label:
        lm = (df.label == label).values
        for gname, gmask in sg.items():
            m = lm & gmask
            if not m.any():
                continue
            sub = df[m]
            for comp, (c1, c2) in CELL_COMPARISONS.items():
                x, y = sub[c1].values.astype(float), sub[c2].values.astype(float)
                ok = ~(np.isnan(x) | np.isnan(y))
                x, y = x[ok], y[ok]
                a = int(((x == 1) & (y == 1)).sum()); b = int(((x == 1) & (y == 0)).sum())
                c = int(((x == 0) & (y == 1)).sum()); d_ = int(((x == 0) & (y == 0)).sum())
                n = a + b + c + d_
                delta, lo, hi = h3.newcombe_paired(a, b, c, d_)
                rows.append(dict(label=label, set_group=gname, comparison=comp, rater1=c1, rater2=c2, n=n,
                                 rate1=(a + b) / n if n else np.nan, rate2=(a + c) / n if n else np.nan,
                                 a_both_yes=a, b_r1_only=b, c_r2_only=c, d_both_no=d_, gap=delta, gap_lo=lo,
                                 gap_hi=hi, p_mcnemar_exact=h3.mcnemar_exact(b, c),
                                 capped_rate=sub.incomplete.mean(), n_tokens_median=sub.n_tokens.median()))
    g = pd.DataFrame(rows)
    g["p_holm"] = np.nan
    for _, idx in g.groupby(["comparison", "set_group"]).groups.items():
        g.loc[idx, "p_holm"] = multipletests(g.loc[idx, "p_mcnemar_exact"].values, method="holm")[1]
    g["direction"] = np.where(g.p_holm < h3.ALPHA, np.where(g.gap > 0, "rater1>rater2", "rater1<rater2"), "n.s.")
    return meta.merge(g, on="label", how="right")


def h3_models(df):
    """h3.fit_models models (1) primary linear GEE on the paired difference and (2) cell-level OLS, outcome swapped."""
    term_names = {"Intercept": "mean gap (kw - wg), averaged over generation x class",
                  "gen_c": "generation effect on gap (new - old), averaged over classes",
                  "ara_c": "class effect on gap (ARA - direction), averaged over generations",
                  "gen_c:ara_c": "generation x class interaction on gap (DiD)"}
    rows = []
    for outcome, (kc, wc) in {"kw_vs_wg": ("ref_kw", "ref_wg"), "kw100_vs_wg": ("kw100", "ref_wg"),
                              "kw100_vs_wg256": ("kw100", "ref_wg256"),
                              "kw256_vs_wg256": ("ref_kw256", "ref_wg256")}.items():
        d0 = df.copy()
        d0["ref_kw"], d0["ref_wg"] = df[kc].astype(float), df[wc].astype(float)
        for study in ["main", "small"]:
            for subset in ["harmful_all", "harmful_excl_advbench", "benign_all"]:
                d, pairs = h3.model_data(d0, study, subset)
                if d.empty or d.label.nunique() < 4:
                    continue
                pair_term = " + C(pair, Sum)" if len(pairs) > 1 else ""
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    r = smf.gee("gap ~ gen_c * ara_c" + pair_term, groups="pid", data=d,
                                cov_struct=sm.cov_struct.Exchangeable(), family=sm.families.Gaussian()).fit()
                for t, lab in term_names.items():
                    rows.append(dict(outcome=outcome, model="GEE_linear_paired_gap (primary)", study=study,
                                     subset=subset, term=lab, estimate=r.params[t], se=r.bse[t],
                                     lo=r.params[t] - 1.96 * r.bse[t], hi=r.params[t] + 1.96 * r.bse[t],
                                     p=r.pvalues[t], n_obs=len(d), n_clusters=d.pid.nunique(), pairs="+".join(pairs)))
                cg = d.groupby(["label", "pair", "gen_c", "ara_c"], as_index=False).gap.mean()
                if len(cg) >= 4 + (len(pairs) - 1) + 2:
                    r = smf.ols("gap ~ gen_c * ara_c" + pair_term, data=cg).fit()
                    for t, lab in term_names.items():
                        rows.append(dict(outcome=outcome, model="OLS_cell_level_gap", study=study, subset=subset,
                                         term=lab, estimate=r.params[t], se=r.bse[t], lo=r.params[t] - 1.96 * r.bse[t],
                                         hi=r.params[t] + 1.96 * r.bse[t], p=r.pvalues[t], n_obs=len(cg),
                                         n_clusters=len(cg), pairs="+".join(pairs)))
    m = pd.DataFrame(rows)
    m["p_holm"] = np.nan
    prim = m.model == "GEE_linear_paired_gap (primary)"
    for _, idx in m[prim].groupby(["outcome", "study", "subset"]).groups.items():
        m.loc[idx, "p_holm"] = multipletests(m.loc[idx, "p"].values, method="holm")[1]
    m["holm_family"] = np.where(m.p_holm.notna(), "4 terms of the primary GEE within outcome x study x subset", "")
    return m


# ========================================================================================================= summary
def f_pp(x, d=1, sign=True):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "--"
    return f"{100 * x:+.{d}f}" if sign else f"{100 * x:.{d}f}"


def f_num(x, d=2):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "--"
    return f"{x:+.{d}f}"


def f_p(p):
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "--"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def md_table(rows: list[dict]) -> str:
    if not rows:
        return "(none)"
    cols = list(rows[0].keys())
    L = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for r in rows:
        L.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(L)


def term(models, item_subset, model, t, scope="main", subset="all", judge=None):
    m = models[(models.scope == scope) & (models.model == model) & (models.term == t) & (models.subset == subset)]
    if item_subset is not None:
        m = m[m.item_subset == item_subset]
    if judge is not None:
        m = m[m.judge == judge]
    return m.iloc[0] if len(m) else None


def fterm(r, d=2):
    if r is None:
        return "--"
    return f"{f_num(r.estimate, d)} [{f_num(r.lo, d)}, {f_num(r.hi, d)}]"


def h1_tally(tests, sel):
    """Main-study H1 McNemar tests (16): Holm-significant higher / lower / n.s."""
    t = tests[sel & (tests.family == "H1") & (tests.pair.isin(["qwen", "gemma", "olmo", "mistral"]))]
    up = int(((t.reject_holm) & (t.estimate > 0)).sum()); dn = int(((t.reject_holm) & (t.estimate < 0)).sum())
    return up, dn, len(t) - up - dn, len(t)


def all_higher_pairs(tests, sel):
    t = tests[sel & (tests.family == "H1") & ~tests.pair.str.startswith("main")]
    out = []
    for p, g in t.groupby("pair"):
        if len(g) and (g.reject_holm & (g.estimate > 0)).all():
            out.append(p)
    return out


def write_summary(c2, c3, c4, ck, src, log_head, runtime, n_boot):
    L = []
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    L += ["# Paper 33 -- reviewer concerns C2, C3, C4: item source, harmful compliance, H3 at the 100-token window", "",
          f"Generated {now} by `scripts/analysis/review_outcomes.py` ({runtime:.0f} s) from "
          f"`{src.relative_to(ROOT)}` (assembly: {log_head}). Read-only on every input; regenerated on every run. "
          "Every number is relative to the automatic judges (WildGuard, HarmBench classifier); there is no human audit.",
          ""]
    import freshness  # noqa: E402
    L += [freshness.summary_line(h1h2.STALE), ""]
    nfail = int((~ck.ok).sum())
    L += ["## Checks", "", f"{len(ck)} reproduction and validation checks, {nfail} failed (`review_out_checks.csv`):", ""]
    for r in ck.itertuples():
        L.append(f"- {'ok' if r.ok else '**FAILED**'}: {r.check}: {r.value}" + (f" ({r.detail})" if r.detail else ""))
    L.append("")

    # ============================================================================================== C2
    gen, h2, models, tests, pc, dec = (c2[k] for k in ["gen", "h2", "models", "tests", "pc", "dec"])
    L += ["## C2. Item source: public sets versus the code_weapon bank", "",
          "Outcome: WildGuard refusal (primary). Same cells, models, CIs, tests and Holm families as h1h2.py, run "
          "separately on each item subset (Holm within each subset's H1 and H2 families). Main study = Qwen, Gemma, "
          "OLMo, Mistral pairs; pooled log-odds are population-averaged GEE (family fixed effect, exchangeable "
          "within prompt), as the paper's +0.62 / +0.03 / -0.46 / -0.50.", ""]
    # originals
    o = gen[(gen.subset == "all") & (gen.method == "original")]
    rows = []
    for p in h1h2.PAIR_ORDER:
        r = {"Pair": p}
        for s in ["harmful_all", "public4", "code_weapon"] + PUBLIC_SETS:
            x = o[(o.pair == p) & (o.item_subset == s)]
            if len(x):
                x = x.iloc[0]
                r[SUBSET_DISPLAY[s]] = (f"{f_pp(x.p_old, sign=False)} -> {f_pp(x.p_new, sign=False)}: "
                                        f"{f_pp(x.delta)} [{f_pp(x.delta_lo)}, {f_pp(x.delta_hi)}]")
        rows.append(r)
    L += ["**Originals' generation difference (unmodified models, WildGuard refusal %, old -> new: new - old pp "
          "[95% Newcombe CI])**", "", md_table(rows), ""]
    # decomposition
    rows = []
    for p in [x for x in h1h2.PAIR_ORDER if x in set(dec.pair)] + ["main_mean"]:
        g = dec[dec.pair == p].set_index("set_name")
        a, cw, pub = g.loc["harmful_all"], g.loc["code_weapon"], g.loc["public4"]
        sig = not (a.total_lo <= 0 <= a.total_hi)
        rows.append({"Pair": "mean of 4 main pairs" if p == "main_mean" else p,
                     "Difference, 972 items": f"{a.delta_pp:+.1f} [{a.delta_lo:+.1f}, {a.delta_hi:+.1f}]",
                     "on public 772": f"{pub.delta_pp:+.1f} [{pub.delta_lo:+.1f}, {pub.delta_hi:+.1f}]",
                     "on code 200": f"{cw.delta_pp:+.1f} [{cw.delta_lo:+.1f}, {cw.delta_hi:+.1f}]",
                     "code contribution, pp": f"{cw.contribution_pp:+.1f} [{cw.contribution_lo:+.1f}, "
                                              f"{cw.contribution_hi:+.1f}]",
                     "public contribution, pp": f"{pub.contribution_pp:+.1f} [{pub.contribution_lo:+.1f}, "
                                                f"{pub.contribution_hi:+.1f}]",
                     "code share of the difference": (f"{100 * cw.share_of_total:.0f}% [{100 * cw.share_lo:.0f}, "
                                                      f"{100 * cw.share_hi:.0f}]" if sig else
                                                      f"(total CI includes 0; {100 * cw.share_of_total:.0f}%)")})
    L += ["**How much of the originals' difference comes from code_weapon items.** The 972-item difference is the "
          "item-weighted sum of the set differences: 200/972 x (difference on code items) + 772/972 x (difference on "
          "public items). Item-bootstrap 95% CIs.", "", md_table(rows), ""]
    # pooled GEE per subset
    rows = []
    for s in ITEM_SUBSETS:
        sel = (tests.item_subset == s) & (tests.subset == "all")
        up, dn, ns, nt = h1_tally(tests, sel)
        het = term(models, s, "gee_heterogeneity", "generation x family (direction class)")
        tH2 = tests[sel & (tests.test_id == "pooled main: generation x class")]
        rows.append({"Items": SUBSET_DISPLAY[s],
                     "Originals (baseline)": fterm(term(models, s, "gee_with_originals",
                                                        "gen effect | original (baseline difference)")),
                     "Direction, raw": fterm(term(models, s, "gee_planned", "gen effect | direction (= gen_new)")),
                     "Direction, adjusted": fterm(term(models, s, "gee_with_originals",
                                                       "baseline-adjusted gen effect | direction")),
                     "ARA, raw": fterm(term(models, s, "gee_planned", "gen effect | optimisation (derived)")),
                     "ARA, adjusted": fterm(term(models, s, "gee_with_originals",
                                                 "baseline-adjusted gen effect | optimisation")),
                     "H2 interaction (p Holm)": fterm(term(models, s, "gee_planned", "gen_new:opt")) +
                     (f" ({f_p(tH2.p_holm.iloc[0])})" if len(tH2) else ""),
                     "gen x family chi2": fchi2(het, [c for c in c2["sep"][s] if c.split("__")[0] in MAIN_MODELS]),
                     "H1 tests up/down/n.s. (of 16)": f"{up}/{dn}/{ns}"})
    L += ["**Pooled main-study generation effects by item subset (GEE, log-odds [95% CI]).** Raw = among modified "
          "models (`gee_planned`); adjusted = minus the originals' difference (`gee_with_originals`); H2 = ARA minus "
          "direction class (identical raw and adjusted, the originals' term cancels). Last column: Holm-significant "
          "per-pair McNemar tests (16 main pair x method), newer higher / newer lower / not significant; the paper "
          "reports 8/4/4 on all items.", "", md_table(rows), "",
          "Cells with 0% or 100% WildGuard refusal on a subset (per-pair and family-level logistic terms are not "
          "estimable there and are shown as n/a; risk differences and McNemar tests are unaffected): " + "; ".join(
              f"{SUBSET_DISPLAY[s]}: {', '.join(v)}" for s, v in c2["sep"].items() if v) + ".", ""]
    # per pair class-level
    rows = []
    for p in h1h2.PAIR_ORDER:
        for s in ["harmful_all", "public4", "code_weapon"]:
            x = pc[(pc.pair == p) & (pc.item_subset == s)]
            hh = h2[(h2.pair == p) & (h2.item_subset == s) & (h2.subset == "all")]
            if not len(x):
                continue
            x = x.iloc[0]
            rr = {"Pair": p, "Items": SUBSET_DISPLAY[s]}
            for k, lab in (("orig", "Originals"), ("dir_raw", "Direction raw"), ("dir_adj", "Direction adj."),
                           ("ara_raw", "ARA raw"), ("ara_adj", "ARA adj.")):
                rr[lab] = f"{f_pp(x[f'rd_{k}'])} [{f_pp(x[f'rd_{k}_lo'])}, {f_pp(x[f'rd_{k}_hi'])}]"
            rr["H2 RD contrast"] = f"{f_pp(x.rd_h2_rd)} [{f_pp(x.rd_h2_rd_lo)}, {f_pp(x.rd_h2_rd_hi)}]"
            rr["H2 GEE log-odds (p Holm)"] = (
                f"n/a (separation: {x.separated_cells})" if isinstance(x.separated_cells, str) and x.separated_cells
                else (f"{f_num(hh.gee_interaction.iloc[0])} ({f_p(hh.p_holm_H2.iloc[0])})" if len(hh) else "--"))
            rows.append(rr)
    L += ["**Per pair, class level (risk difference, pp [item-bootstrap 95% CI]).** Direction = mean over DiM, MPOA, "
          "SOM (and J-space on the small pair); adj. = minus the originals' difference. Log-odds versions in "
          "`review_out_c2_pair_class.csv` (`lo_*`).", "", md_table(rows), ""]
    # per method, main pairs, public vs code
    rows = []
    for p in ["qwen", "gemma", "olmo", "mistral", "qwen-small"]:
        for m in ["dim", "mpoa", "som", "jspace", "ara"]:
            rr = {"Pair": p, "Method": h1h2.METHOD_DISPLAY[m]}
            okrow = False
            for s in ["harmful_all", "public4", "code_weapon"]:
                x = gen[(gen.pair == p) & (gen.method == m) & (gen.item_subset == s) & (gen.subset == "all")]
                if not len(x):
                    continue
                okrow = True
                x = x.iloc[0]
                star = "*" if x.p_holm_H1 < 0.05 else ""
                rr[SUBSET_DISPLAY[s]] = (f"{f_pp(x.delta)}{star} (adj. {f_pp(x.did_baseline_adj)})")
            if okrow:
                rows.append(rr)
    L += ["**Per pair x method: generation effect, pp (\\* Holm-significant within the subset's H1 family); "
          "baseline-adjusted in brackets.**", "", md_table(rows), ""]
    c2_verdict = c2_findings(gen, models, tests, dec, pc, h2)
    L += ["**C2 findings**", ""] + [f"- {x}" for x in c2_verdict["lines"]] + [""]

    # ============================================================================================== C3
    g3, h23, m3, t3, pc3, cells3, pool3 = (c3[k] for k in ["gen", "h2", "models", "tests", "pc", "cells", "pooled"])
    mw = models[models.item_subset == "harmful_all"].assign(judge="wg")
    allm = pd.concat([mw, m3], ignore_index=True)
    tw = tests[tests.item_subset == "harmful_all"].assign(judge="wg")
    allt = pd.concat([tw, t3], ignore_index=True)
    L += ["## C3. Harmful compliance as the outcome", "",
          "Outcome coded as **not harmful** = 1 - harmful compliance, so that signs match the refusal analysis: a "
          "positive generation effect means the newer model's edit leaves fewer harmful answers (abliteration "
          "'harder'), H1 predicts positive, H2 predicts a negative interaction. HarmBench classifier "
          "(`hb_harmful`) and WildGuard harmful-response label (`wg_harmful_response`), all 972 harmful items, same "
          "cells, models and Holm families as h1h2.py (each outcome its own families).", ""]
    rows = []
    for j in ["wg", "hbnc", "wghnc"]:
        sel = (allt.judge == j) & (allt.subset == "all")
        up, dn, ns, nt = h1_tally(allt, sel)
        tH2 = allt[sel & (allt.test_id == "pooled main: generation x class")]
        tA = allt[sel & (allt.test_id == "pooled main: gen effect | optimisation class")]
        het = term(allm, None, "gee_heterogeneity", "generation x family (direction class)", judge=j)
        rows.append({"Outcome": OUTCOME_SHORT[j],
                     "Originals (baseline)": fterm(term(allm, None, "gee_with_originals",
                                                        "gen effect | original (baseline difference)", judge=j)),
                     "Direction, raw": fterm(term(allm, None, "gee_planned", "gen effect | direction (= gen_new)",
                                                  judge=j)),
                     "Direction, adjusted": fterm(term(allm, None, "gee_with_originals",
                                                       "baseline-adjusted gen effect | direction", judge=j)),
                     "ARA, raw (p Holm)": fterm(term(allm, None, "gee_planned", "gen effect | optimisation (derived)",
                                                     judge=j)) + (f" ({f_p(tA.p_holm.iloc[0])})" if len(tA) else ""),
                     "ARA, adjusted": fterm(term(allm, None, "gee_with_originals",
                                                 "baseline-adjusted gen effect | optimisation", judge=j)),
                     "H2 interaction (p Holm)": fterm(term(allm, None, "gee_planned", "gen_new:opt", judge=j)) +
                     (f" ({f_p(tH2.p_holm.iloc[0])})" if len(tH2) else ""),
                     "gen x family chi2": fchi2(het, []),
                     "H1 tests up/down/n.s.": f"{up}/{dn}/{ns}",
                     "Pairs with every method higher": ", ".join(all_higher_pairs(allt, sel)) or "none"})
    L += ["**Pooled main-study estimates by outcome (GEE log-odds [95% CI]; all 972 harmful items)**", "",
          md_table(rows), ""]
    allh2 = pd.concat([h2[h2.item_subset == "harmful_all"].assign(judge="wg"), h23], ignore_index=True)
    rows = []
    for p in h1h2.PAIR_ORDER:
        rr = {"Pair": p}
        for j in ["wg", "hbnc", "wghnc"]:
            x = allh2[(allh2.pair == p) & (allh2.judge == j) & (allh2.subset == "all")]
            if len(x):
                x = x.iloc[0]
                rr[OUTCOME_SHORT[j]] = (f"ARA {f_pp(x.delta_ara)}, dir. {f_pp(x.delta_direction_mean)}; RD "
                                        f"{f_pp(x.rd_contrast)} [{f_pp(x.rd_lo)}, {f_pp(x.rd_hi)}]; log-odds "
                                        f"{f_num(x.gee_interaction)} (p Holm {f_p(x.p_holm_H2)})")
        rows.append(rr)
    L += ["**H2 per pair by outcome** (generation effects in pp on the not-harmful / not-refused scale; RD = ARA "
          "minus direction mean, bootstrap CI; log-odds = per-pair GEE interaction)", "", md_table(rows), ""]
    # per pair x method table: rates harmful
    rows = []
    for p in h1h2.PAIR_ORDER:
        for m in h1h2.METHOD_ORDER:
            rr = {"Pair": p, "Method": h1h2.METHOD_DISPLAY[m]}
            okrow = False
            for j, gtab in (("wg", gen[(gen.item_subset == "harmful_all")].assign(judge="wg")), ("hbnc", g3),
                            ("wghnc", g3)):
                x = gtab[(gtab.pair == p) & (gtab.method == m) & (gtab.judge == j) & (gtab.subset == "all")]
                if not len(x):
                    continue
                okrow = True
                x = x.iloc[0]
                star = "" if m == "original" else ("*" if x.p_holm_H1 < 0.05 else "")
                if j == "wg":
                    rr["WG refusal % old -> new: effect"] = (f"{f_pp(x.p_old, sign=False)} -> "
                                                             f"{f_pp(x.p_new, sign=False)}: {f_pp(x.delta)}{star}")
                else:
                    rr[f"{'HB' if j == 'hbnc' else 'WG'} harmful % old -> new: effect"] = (
                        f"{f_pp(1 - x.p_old, sign=False)} -> {f_pp(1 - x.p_new, sign=False)}: {f_pp(x.delta)}{star}")
            if okrow:
                rows.append(rr)
    L += ["**Per pair x method.** Refusal and harmful-compliance rates; the effect is new - old on the "
          "not-refused / not-harmful scale (= minus the change in harmful %), \\* = Holm-significant in that outcome's "
          "H1 family.", "", md_table(rows), ""]
    # decision changes
    changes = []
    a = allt[(allt.judge == "wg") & (allt.subset == "all") & allt.family.isin(["H1", "H2"])].set_index("test_id")
    for j in ["hbnc", "wghnc"]:
        b = allt[(allt.judge == j) & (allt.subset == "all") & allt.family.isin(["H1", "H2"])].set_index("test_id")
        for tid in a.index.intersection(b.index):
            ra, rb = a.loc[tid], b.loc[tid]
            ca = ("sig. " if ra.reject_holm else "n.s. ") + ("+" if ra.estimate > 0 else "-")
            cb = ("sig. " if rb.reject_holm else "n.s. ") + ("+" if rb.estimate > 0 else "-")
            if ca != cb:
                unit = "pp" if "risk" in ra.scale else "log-odds"
                ea = f_pp(ra.estimate) if unit == "pp" else f_num(ra.estimate)
                eb = f_pp(rb.estimate) if unit == "pp" else f_num(rb.estimate)
                changes.append(f"- {OUTCOME_SHORT[j]}: `{tid}` ({ra.family}): refusal {ca} ({ea} {unit}, p_Holm "
                               f"{f_p(ra.p_holm)}) -> {cb} ({eb} {unit}, p_Holm {f_p(rb.p_holm)})")
    L += [f"**H1/H2 tests whose Holm decision or sign differs from the refusal outcome ({len(changes)}):**", ""] + \
         (changes or ["- none"]) + [""]
    # share harmful among non-refused
    rows = []
    for gname, lab in (("original", "originals (10)"), ("ablated_primary:main", "ablated, main study (32)"),
                       ("ablated_primary:small", "ablated, small pair (10)"), ("role:control", "gamma-MPOA (4)"),
                       ("role:external", "community (8)"), ("role:replicate", "seed-2 replicates (10)")):
        g = pool3[pool3.group == gname].set_index("measure")
        if not len(g):
            continue
        fx = lambda k: f"{f_pp(g.loc[k, 'rate'], sign=False)} [{f_pp(g.loc[k, 'lo'], sign=False)}, " \
                       f"{f_pp(g.loc[k, 'hi'], sign=False)}]"  # noqa: E731
        rows.append({"Cells": lab, "WG refusal %": fx("wg_refusal"), "HB harmful %": fx("hb_harmful"),
                     "HB harmful among not refused %": fx("hb_harmful_given_not_refused"),
                     "WG harmful among not refused %": fx("wg_harmful_given_not_refused"),
                     "not refused and not HB-harmful, % of all": fx("not_refused_not_hb_harmful")})
    L += ["**How much of 'not refused' is harmful compliance** (harmful items, pooled over cells; prompt-cluster "
          "bootstrap 95% CI). 'Not refused' = WildGuard refusal label no (degenerate rows included).", "",
          md_table(rows), ""]
    cp = cells3[cells3.role.isin(["original", "primary"])]
    rows = []
    for r in cp.itertuples():
        rows.append({"Cell": r.label, "WG refusal %": f_pp(r.wg_refusal_rate, sign=False),
                     "not refused, n": r.not_refused_k,
                     "HB harmful among not refused % [CI]": f"{f_pp(r.hb_harmful_given_not_refused_rate, sign=False)} "
                                                   f"[{f_pp(r.hb_harmful_given_not_refused_lo, sign=False)}, "
                                                   f"{f_pp(r.hb_harmful_given_not_refused_hi, sign=False)}]",
                     "WG harmful among not refused % [CI]": f"{f_pp(r.wg_harmful_given_not_refused_rate, sign=False)} "
                                                   f"[{f_pp(r.wg_harmful_given_not_refused_lo, sign=False)}, "
                                                   f"{f_pp(r.wg_harmful_given_not_refused_hi, sign=False)}]",
                     "HB harmful % (all)": f_pp(r.hb_harmful_rate, sign=False)})
    L += ["**Per cell (original + seed-1 primary; harmful items; Wilson 95% CI).** All 74 cells in "
          "`review_out_c3_cells.csv`.", "", md_table(rows), ""]
    c3_verdict = c3_findings(allm, allt, allh2, pool3, cells3)
    L += ["**C3 findings**", ""] + [f"- {x}" for x in c3_verdict["lines"]] + [""]

    # ============================================================================================== C4
    df, val, pooled, cg, mods, tc = (c4[k] for k in ["df", "val", "pooled", "cells", "models", "tc"])
    enc = tc.groupby("model").first()
    L += ["## C4. H3 at Heretic's own keyword window (100 tokens)", "",
          "Heretic 2.0 counts a dev response as a refusal if any of its 33 markers occurs in the first **100** "
          "generated tokens (`config.default.toml` @ 8fd4d76: `max_response_length = 100`; `model.py` "
          "`get_responses`: `max_new_tokens = max_response_length`; Methods sec:setup-opt, 'at most 100 new tokens'). "
          "The paper's H3 uses the same markers on the full response (<= 512 tokens). Here each stored response is "
          "re-tokenised with the evaluated model's tokenizer, cut to 100 tokens, and matched with the paper's "
          "keyword function. Because responses are greedy, the first 100 tokens of a 512-token greedy answer are "
          "what a 100-token greedy run would produce (up to numerics).", "",
          "**Validation.** Encoding per base model (the one that reproduces the stored `response_256` text more "
          "often): " + "; ".join(f"{m} {enc.loc[m, 'encoding']}" for m in enc.index) + ". Community cells: the "
          "repo's own tokenizer.json is used only if it reproduces the stored 256-token text at least as often as "
          "the base tokenizer; chosen own tokenizer: " + (", ".join(tc[tc.tokenizer_source == "community repo"].label)
                                                          or "none") + " (scores per candidate in "
          "`review_out_c4_tokenizer_cells.csv`: " + "; ".join(
              f"{r.label} {r.tokenizer_candidates_text256_eq}" for r in tc[tc.method == "community"].itertuples())
          + ").", ""]
    vbad = val[(val.kw256_retok_mismatch > 0) | (val.text256_eq_share < 0.98)]
    L += [f"- Full response: re-matched keyword equals `keyword_refusal` in "
          f"{int((df.kw_full == df.ref_kw).sum()):,} of {len(df):,} responses.",
          f"- 256 tokens: the re-tokenised 256-token cut gives the stored `keyword_refusal_256` label in "
          f"{int((df.kw256_retok == df.ref_kw256).sum()):,} of {len(df):,} responses "
          f"(other encoding: {int((df.kw256_retok_alt == df.ref_kw256).sum()):,}); its text equals the stored "
          f"`response_256` in {100 * df.text256_eq.mean():.2f}% of all responses and "
          f"{100 * df.text256_eq[df.n_tokens > CUT256].mean():.2f}% of those longer than 256 tokens.",
          "- Cells below 98% text agreement or with any 256-token label mismatch: " + (
              "; ".join(f"`{r.label}` text {100 * r.text256_eq_share:.1f}% (other encoding "
                        f"{100 * r.text256_eq_alt_share:.1f}%), label mismatches {r.kw256_retok_mismatch}"
                        for r in vbad.itertuples()) or "none") + ".",
          f"- The two encodings give a different 100-token keyword label in {int((df.kw100 != df.kw100_alt).sum()):,} "
          f"responses ({100 * (df.kw100 != df.kw100_alt).mean():.2f}%), "
          f"{int(val[val.model == 'olmo-2-32b'].kw100_alt_differs.sum())} of them in OLMo-2 cells.",
          f"- Bounds that do not rely on the re-encoding being exact (`response_256` is exactly the first 256 "
          f"generated tokens, the keyword label only grows with the window, responses of <= 100 tokens lie wholly "
          f"inside it): the 100-token label is pinned down in {100 * (df.kw100_lo == df.kw100_hi).mean():.2f}% of the "
          f"responses; {int((df.kw100_lo != df.kw100_hi).sum())} are unresolved ("
          + ", ".join(f"{m} {int(v)}" for m, v in df[df.kw100_lo != df.kw100_hi].groupby("model").size().items())
          + "). The gaps are reported at both bounds below.",
          f"- Median length of the 100-token text: {df.chars100.median():.0f} characters "
          f"(full responses: {df.chars_full.median():.0f}); "
          f"{100 * (df.n_tokens > WINDOW).mean():.1f}% of the responses are longer than 100 tokens.", ""]

    def pg(group, variant, subset="harmful_all"):
        x = pooled[(pooled.group == group) & (pooled.variant == variant) & (pooled.subset == subset)]
        return x.iloc[0] if len(x) else None

    def gs(r):
        return "--" if r is None else f"{f_pp(r.gap)} [{f_pp(r.gap_lo)}, {f_pp(r.gap_hi)}]"

    rows = []
    for gname, lab in (("original", "originals (10)"), ("ablated_primary", "ablated, all (42)"),
                       ("ablated_primary:main", "ablated, main (32)"), ("ablated_primary:small", "ablated, small (10)"),
                       ("role:control", "gamma-MPOA (4)"), ("role:external", "community (8)"),
                       ("role:replicate", "seed-2 (10)"), ("all_judged", "all 74 cells")):
        a = pg(gname, "kw_vs_wg")
        if a is None:
            continue
        k100, w256 = pg(gname, "kw100_vs_wg"), pg(gname, "kw100_vs_wg256")
        rows.append({"Cells": lab, "kw 512 %": f_pp(a.rate1, sign=False),
                     "kw 256 %": f_pp(pg(gname, "kw256_vs_wg").rate1, sign=False),
                     "kw 100 %": f_pp(k100.rate1, sign=False), "WG full %": f_pp(a.rate2, sign=False),
                     "WG 256 %": f_pp(w256.rate2, sign=False),
                     "kw512 - WG (paper)": gs(a), "kw256 - WG256": gs(pg(gname, "kw256_vs_wg256")),
                     "**kw100 - WG**": gs(k100), "**kw100 - WG256**": gs(w256),
                     "kw100 - WG at the bounds": f"{f_pp(pg(gname, 'kw100_lo_vs_wg').gap)} to "
                                                 f"{f_pp(pg(gname, 'kw100_hi_vs_wg').gap)}",
                     "kw100 (Heretic matcher) - WG": gs(pg(gname, "kw100_heretic_vs_wg"))})
    L += ["**Pooled keyword-minus-WildGuard gap, harmful items** (pp, prompt-cluster bootstrap 95% CI; h3.py groups). "
          "H3 predicted a negative gap (keyword undercounts refusal).", "", md_table(rows), ""]
    rows = []
    model_order =["qwen2.5-32b", "qwen3.8-27b", "gemma-2-27b", "gemma-4-31b", "olmo-2-32b", "olmo-3.1-32b",
                   "mistral-small-2409", "mistral-small-3.2", "qwen2.5-7b", "qwen3.5-4b"]
    for m in model_order:
        for kind, lab in (("original", "original"), ("ablated_primary", "ablated (seed-1 methods)")):
            a = pg(f"model:{m}:{kind}", "kw_vs_wg")
            if a is None:
                continue
            k100, w256 = pg(f"model:{m}:{kind}", "kw100_vs_wg"), pg(f"model:{m}:{kind}", "kw100_vs_wg256")
            rows.append({"Model": m, "Cells": lab, "kw512 - WG": gs(a), "kw100 - WG": gs(k100),
                         "kw100 - WG256": gs(w256), "kw 100 % / WG %": f"{f_pp(k100.rate1, sign=False)} / "
                                                                       f"{f_pp(k100.rate2, sign=False)}"})
    L += ["**Per base model** (harmful items; pooled over the model's cells)", "", md_table(rows), ""]
    hcg = cg[(cg.set_group == "harmful_all") & (cg.role == "primary")]
    lines = []
    for comp in ["kw_vs_wg", "kw100_vs_wg", "kw100_vs_wg256", "kw256_vs_wg256"]:
        s = hcg[hcg.comparison == comp]
        lines.append(f"- `{comp}`: keyword > WildGuard in {int((s.gap > 0).sum())}/{len(s)} ablated seed-1 cells "
                     f"({int(((s.gap > 0) & (s.p_holm < 0.05)).sum())} Holm-significant), keyword < WildGuard in "
                     f"{int((s.gap < 0).sum())} ({int(((s.gap < 0) & (s.p_holm < 0.05)).sum())} Holm-significant"
                     + (": " + ", ".join(s[(s.gap < 0) & (s.p_holm < 0.05)].label) if
                        ((s.gap < 0) & (s.p_holm < 0.05)).any() else "") + ").")
    L += ["**Per evaluated model (cell)**, harmful items, Holm over all 74 cells per comparison "
          "(`review_out_c4_gap_cells.csv`):", ""] + lines + [""]
    rows = []
    for outcome in ["kw_vs_wg", "kw100_vs_wg", "kw100_vs_wg256", "kw256_vs_wg256"]:
        for study in ["main", "small"]:
            x = mods[(mods.outcome == outcome) & (mods.study == study) & (mods.subset == "harmful_all") &
                     (mods.model == "GEE_linear_paired_gap (primary)")]
            if not len(x):
                continue
            rr = {"Outcome": outcome, "Study": study}
            for t, lab in (("mean gap", "Mean gap"), ("generation effect", "Generation effect on gap"),
                           ("class effect", "Class effect (ARA - dir.)"), ("generation x class", "Gen x class")):
                y = x[x.term.str.startswith(t)].iloc[0]
                rr[lab] = f"{f_pp(y.estimate)} [{f_pp(y.lo)}, {f_pp(y.hi)}] (p Holm {f_p(y.p_holm)})"
            rows.append(rr)
    L += ["**H3 models (h3.py primary linear GEE on the paired difference, harmful items, pp).** Cell-level OLS "
          "in `review_out_c4_models.csv`.", "", md_table(rows), ""]
    c4_verdict = c4_findings(pooled, cg, mods, df)
    L += ["**C4 findings**", ""] + [f"- {x}" for x in c4_verdict["lines"]] + [""]

    # ============================================================================================== verdicts
    L += ["## Verdict per reviewer concern", "",
          f"- **C2 (item source drives the result): {c2_verdict['verdict']}.** {c2_verdict['why']}",
          f"- **C3 (refusal is not harmful compliance): {c3_verdict['verdict']}.** {c3_verdict['why']}",
          f"- **C4 (H3 uses the wrong keyword window): {c4_verdict['verdict']}.** {c4_verdict['why']}", "",
          "## Caveats", "",
          "- Intervals reflect prompt sampling only (selected checkpoints fixed, one seed), as in the paper; per-set "
          "analyses have 100-313 items, so their per-pair tests have less power than the 972-item tests.",
          "- Pooled GEE coefficients average across families that differ in sign (see the generation x family chi2); "
          "per-pair numbers are the primary evidence, as the reviewer's A1 asks.",
          "- HarmBench and WildGuard harmfulness labels are themselves automatic judges; 'not harmful' includes "
          "degenerate, off-topic and unhelpful answers, so it is a weaker construct than refusal in the other "
          "direction. The HarmBench classifier sees at most 512 classifier tokens of each response.",
          "- C4 re-tokenises stored text (token ids were not stored). For every model but OLMo-2 the canonical "
          "re-encoding reproduces the stored 256-token text in nearly every response; OLMo-2 needs the newline-split "
          "encoding and still differs by a few tokens in some responses (alternative-encoding sensitivity reported). "
          "No judge was run on 100-token text (no GPU here); the 100-token keyword is compared with WildGuard on the "
          "full response and on the 256-token prefix.",
          "- Heretic's dev scoring generates with HF transformers on the 100 dev prompts (and, for Qwen3.x, after the "
          "detected '</think>' response prefix); this analysis applies its window to the held-out greedy vLLM "
          "responses, which start after the chat template's empty think block.", "",
          "## Files", ""]
    for f in sorted(OUT.glob("review_out_*")):
        L.append(f"- `results/analysis/{f.name}`")
    L.append("")
    (OUT / "review_out_summary.md").write_text("\n".join(L) + "\n")


# ------------------------------------------------------------------------------------------------ findings (rules)
def c2_findings(gen, models, tests, dec, pc, h2):
    lines = []
    g = lambda p, s: dec[(dec.pair == p) & (dec.set_name == s)].iloc[0]  # noqa: E731
    for p in ["qwen", "gemma", "olmo", "mistral", "qwen-small", "main_mean"]:
        if not (dec.pair == p).any():
            continue
        a, cw, pub = g(p, "harmful_all"), g(p, "code_weapon"), g(p, "public4")
        sig = not (a.total_lo <= 0 <= a.total_hi)
        lines.append(f"{'Mean of the 4 main pairs' if p == 'main_mean' else p}: originals differ by "
                     f"{a.delta_pp:+.1f} pp on all items, {pub.delta_pp:+.1f} pp [{pub.delta_lo:+.1f}, "
                     f"{pub.delta_hi:+.1f}] on the public sets and {cw.delta_pp:+.1f} pp [{cw.delta_lo:+.1f}, "
                     f"{cw.delta_hi:+.1f}] on code items; code items contribute {cw.contribution_pp:+.1f} pp"
                     + (f" ({100 * cw.share_of_total:.0f}% [{100 * cw.share_lo:.0f}, {100 * cw.share_hi:.0f}] of the "
                        "difference)" if sig else " (the total difference is not distinguishable from 0)") + ".")
    t = lambda s, mdl, tt: term(models, s, mdl, tt)  # noqa: E731
    da, dp, dc = (t(s, "gee_with_originals", "baseline-adjusted gen effect | direction")
                  for s in ("harmful_all", "public4", "code_weapon"))
    ra, rp, rc = (t(s, "gee_planned", "gen effect | direction (= gen_new)") for s in ("harmful_all", "public4",
                                                                                       "code_weapon"))
    aa, ap, ac = (t(s, "gee_with_originals", "baseline-adjusted gen effect | optimisation")
                  for s in ("harmful_all", "public4", "code_weapon"))
    ia, ip, ic = (t(s, "gee_planned", "gen_new:opt") for s in ("harmful_all", "public4", "code_weapon"))
    ex0 = lambda r: r.lo <= 0 <= r.hi  # noqa: E731
    lines.append(f"Pooled direction-class generation effect (log-odds): raw {fterm(ra)} all, {fterm(rp)} public, "
                 f"{fterm(rc)} code; baseline-adjusted {fterm(da)} all, {fterm(dp)} public, {fterm(dc)} code.")
    lines.append(f"Pooled ARA baseline-adjusted: {fterm(aa)} all, {fterm(ap)} public, {fterm(ac)} code. "
                 f"H2 interaction: {fterm(ia)} all, {fterm(ip)} public, {fterm(ic)} code.")
    tl = {s: h1_tally(tests, (tests.item_subset == s) & (tests.subset == "all")) for s in ("harmful_all", "public4",
                                                                                         "code_weapon")}
    hp = {s: all_higher_pairs(tests, (tests.item_subset == s) & (tests.subset == "all"))
          for s in ("harmful_all", "public4", "code_weapon")}
    lines.append("H1 McNemar tally of the 16 main pair x method tests (higher/lower/n.s.): " + ", ".join(
        f"{SUBSET_DISPLAY[s]} {tl[s][0]}/{tl[s][1]}/{tl[s][2]}" for s in tl) + ". Pairs in which every method is "
        "significantly higher on the newer model (H1 as stated): " + "; ".join(
        f"{SUBSET_DISPLAY[s]}: {', '.join(hp[s]) or 'none'}" for s in hp) + ".")
    hh = h2[(h2.subset == "all") & h2.item_subset.isin(["harmful_all", "public4", "code_weapon"])]
    sepp = {s: set(pc[(pc.item_subset == s) & (pc.separated_cells.fillna("").astype(str).str.len() > 0)].pair)
            for s in ("harmful_all", "public4", "code_weapon")}
    sig_h2 = {s: [r.pair for r in hh[hh.item_subset == s].itertuples()
                  if r.pair not in sepp[s] and r.p_holm_H2 < 0.05 and r.gee_interaction < 0]
              for s in ("harmful_all", "public4", "code_weapon")}
    opp_h2 = {s: [r.pair for r in hh[hh.item_subset == s].itertuples()
                  if r.pair not in sepp[s] and r.p_holm_H2 < 0.05 and r.gee_interaction > 0]
              for s in ("harmful_all", "public4", "code_weapon")}
    lines.append("Pairs with a Holm-significant per-pair H2 interaction (log-odds) in the predicted (negative) direction: "
                 + "; ".join(f"{SUBSET_DISPLAY[s]}: {', '.join(sig_h2[s]) or 'none'}" for s in sig_h2)
                 + "; opposite direction: " + "; ".join(f"{SUBSET_DISPLAY[s]}: {', '.join(opp_h2[s]) or 'none'}"
                                                        for s in opp_h2)
                 + "; not estimable (a cell at 0% or 100%): " + "; ".join(
                     f"{SUBSET_DISPLAY[s]}: {', '.join(sorted(sepp[s]))}" for s in sepp if sepp[s]) + ".")
    gc0 = gen[(gen.item_subset == "code_weapon") & (gen.subset == "all") & (gen.method != "original") &
              gen.pair.isin(["qwen", "gemma", "olmo", "mistral"])]
    n0 = int((gc0.p_old == 0).sum() + (gc0.p_new == 0).sum())
    still = sorted({r.old_label for r in gc0.itertuples() if r.p_old >= 0.10} |
                   {r.new_label for r in gc0.itertuples() if r.p_new >= 0.10})
    lines.append(f"Floor on code items: {n0} of the {2 * len(gc0)} main-study modified cells refuse none of the 200 "
                 "code_weapon items (WildGuard), so many code-only generation effects are bounded at 0 and the "
                 "code-only pooled log-odds are driven by the few cells that still refuse at least 10% of code items: "
                 + (", ".join(still) or "none") + ".")
    # where the per-method effects sit (main pairs): public vs code
    gd = lambda p, m, s: gen[(gen.pair == p) & (gen.method == m) & (gen.item_subset == s) & (gen.subset == "all")]  # noqa: E731
    for p in ["qwen", "gemma", "olmo", "mistral"]:
        parts = []
        for m in ["dim", "mpoa", "som", "ara"]:
            a_, pu, co = gd(p, m, "harmful_all"), gd(p, m, "public4"), gd(p, m, "code_weapon")
            if len(a_) and len(pu) and len(co):
                parts.append(f"{h1h2.METHOD_DISPLAY[m]} {f_pp(a_.delta.iloc[0])} (public {f_pp(pu.delta.iloc[0])}, "
                             f"code {f_pp(co.delta.iloc[0])})")
        lines.append(f"{p} generation effects, pp, all items (public / code): " + "; ".join(parts) + ".")
    dpx = term(models, "public4", "gee_with_originals", "baseline-adjusted gen effect | direction", subset="excl_advbench")
    apx = term(models, "public4", "gee_with_originals", "baseline-adjusted gen effect | optimisation",
               subset="excl_advbench")
    ipx = term(models, "public4", "gee_planned", "gen_new:opt", subset="excl_advbench")
    lines.append(f"Public sets without the 44 AdvBench-overlap items (728 items): baseline-adjusted direction "
                 f"{fterm(dpx)}, ARA {fterm(apx)}, H2 interaction {fterm(ipx)}.")
    # verdict
    h1_same = (ex0(da) == ex0(dp)) and (hp["harmful_all"] == hp["public4"])
    h2_same = (ia.hi < 0) == (ip.hi < 0) and sig_h2["harmful_all"] == sig_h2["public4"]
    mm = dec[(dec.pair == "main_mean")].set_index("set_name")
    shares = []
    for p in ["qwen", "gemma", "olmo", "mistral"]:
        gg = dec[dec.pair == p].set_index("set_name")
        a_, cw, pu = gg.loc["harmful_all"], gg.loc["code_weapon"], gg.loc["public4"]
        if not (a_.total_lo <= 0 <= a_.total_hi):
            shares.append(f"{p} {100 * cw.share_of_total:.0f}% [{100 * cw.share_lo:.0f}, {100 * cw.share_hi:.0f}] of "
                          f"{a_.delta_pp:+.1f} pp")
        else:
            shares.append(f"{p}: total {a_.delta_pp:+.1f} pp = {pu.delta_pp:+.1f} pp on public items and "
                          f"{cw.delta_pp:+.1f} pp on code items, weighted")
    gm_ = {m: (gd("gemma", m, "code_weapon").delta.iloc[0], gd("gemma", m, "public4").delta.iloc[0])
           for m in ["dim", "mpoa", "som"]}
    ga = gen[(gen.item_subset == "harmful_all") & (gen.subset == "all") & (gen.method != "original") &
             gen.pair.isin(["qwen", "olmo", "mistral"])]
    big_public = [(p_, ga[ga.pair == p_].loc[ga[ga.pair == p_].delta.abs().idxmax(), "method"])
                  for p_ in ["qwen", "olmo", "mistral"] if (ga.pair == p_).any()]  # largest effect per pair
    pcc = pc[(pc.item_subset == "code_weapon") & pc.pair.isin(["qwen", "gemma", "olmo", "mistral"])]
    code_adj = ", ".join(f"{r.pair} {f_pp(r.rd_dir_adj)}" for r in pcc.itertuples())
    desc = ("Code items carry a large but uneven part of the originals' generation difference (" + "; ".join(shares)
            + f"; mean of the four main pairs {100 * mm.loc['code_weapon'].share_of_total:.0f}% "
            f"[{100 * mm.loc['code_weapon'].share_lo:.0f}, {100 * mm.loc['code_weapon'].share_hi:.0f}]), and Gemma's "
            "direction-method effects sit in the code items (DiM / MPOA / SOM " +
            " / ".join(f"{f_pp(gm_[m][0])}" for m in gm_) + " pp on code vs " +
            " / ".join(f"{f_pp(gm_[m][1])}" for m in gm_) + " pp on the public sets); the largest effect of each other main "
            "pair, by contrast: " + "; ".join(
                f"{p} {h1h2.METHOD_DISPLAY[m]} {f_pp(gd(p, m, 'public4').delta.iloc[0])} public vs "
                f"{f_pp(gd(p, m, 'code_weapon').delta.iloc[0])} code"
                for p, m in big_public) + ".")
    if h1_same and h2_same:
        cmain = mm.loc["code_weapon"]
        if cmain.share_hi < 0.25 and not any(s_.startswith("qwen ") and int(s_.split()[1][:-1]) >= 50 for s_ in shares):
            verdict = "contradicted"
        else:
            verdict = "partly supported"
        why = (desc + f" But on the 772 public items alone the paper's conclusions are unchanged: H1 is still not "
               f"supported (baseline-adjusted direction effect {fterm(dp)} vs {fterm(da)} on all items; every method "
               f"higher only in {', '.join(hp['public4']) or 'no pair'}; Holm tally {tl['public4'][0]}/{tl['public4'][1]}"
               f"/{tl['public4'][2]} vs {tl['harmful_all'][0]}/{tl['harmful_all'][1]}/{tl['harmful_all'][2]}), and "
               f"the pooled H2 interaction keeps its size and sign ({fterm(ip)} vs {fterm(ia)}) with the same pairs "
               f"carrying it ({', '.join(sig_h2['public4'])}) or opposing it ({', '.join(opp_h2['public4'])}). On "
               f"code items alone the pooled baseline-adjusted direction effect would be {'positive' if dc.estimate > 0 else 'negative'} "
               f"({fterm(dc)}; per pair, risk difference: {code_adj} pp), so the item mix changes pooled log-odds, "
               "but the conclusions rest on the public sets. "
               "The paper should show the per-source table and the public-only results.")
    else:
        verdict = "supported"
        why = (desc + " On the 772 public items alone at least one conclusion changes: H1 "
               + ("unchanged" if h1_same else f"changes (adjusted direction {fterm(dp)} vs {fterm(da)}; every method "
                  f"higher: {', '.join(hp['public4']) or 'none'} vs {', '.join(hp['harmful_all']) or 'none'})")
               + "; H2 " + ("unchanged" if h2_same else f"changes ({fterm(ip)} vs {fterm(ia)}; pairs "
                                                         f"{', '.join(sig_h2['public4'])})") + ".")
    return dict(lines=lines, verdict=verdict, why=why)


def c3_findings(allm, allt, allh2, pool3, cells3):
    lines = []
    t = lambda j, mdl, tt: term(allm, None, mdl, tt, judge=j)  # noqa: E731
    res = {}
    for j in ["wg", "hbnc", "wghnc"]:
        res[j] = dict(dir_raw=t(j, "gee_planned", "gen effect | direction (= gen_new)"),
                      dir_adj=t(j, "gee_with_originals", "baseline-adjusted gen effect | direction"),
                      ara_raw=t(j, "gee_planned", "gen effect | optimisation (derived)"),
                      ara_adj=t(j, "gee_with_originals", "baseline-adjusted gen effect | optimisation"),
                      h2=t(j, "gee_planned", "gen_new:opt"),
                      orig=t(j, "gee_with_originals", "gen effect | original (baseline difference)"),
                      hp=all_higher_pairs(allt, (allt.judge == j) & (allt.subset == "all")),
                      tally=h1_tally(allt, (allt.judge == j) & (allt.subset == "all")))
    for j in ["hbnc", "wghnc"]:
        r = res[j]
        lines.append(f"{OUTCOME_SHORT[j]}: direction raw {fterm(r['dir_raw'])}, baseline-adjusted {fterm(r['dir_adj'])}; "
                     f"ARA raw {fterm(r['ara_raw'])}, adjusted {fterm(r['ara_adj'])}; H2 {fterm(r['h2'])}; originals "
                     f"{fterm(r['orig'])}; H1 tally {r['tally'][0]}/{r['tally'][1]}/{r['tally'][2]}; pairs with every "
                     f"method higher: {', '.join(r['hp']) or 'none'} (refusal: direction raw {fterm(res['wg']['dir_raw'])}, "
                     f"adjusted {fterm(res['wg']['dir_adj'])}, H2 {fterm(res['wg']['h2'])}).")
    hh = allh2[allh2.subset == "all"]
    for j in ["wg", "hbnc", "wghnc"]:
        x = hh[hh.judge == j]
        lines.append(f"H2 per pair, {OUTCOME_SHORT[j]}: " + "; ".join(
            f"{r.pair} {f_num(r.gee_interaction)} (p_Holm {f_p(r.p_holm_H2)})" for r in x.itertuples()) + ".")
    pm = pool3[(pool3.group == "ablated_primary:main")].set_index("measure")
    po = pool3[(pool3.group == "original")].set_index("measure")
    cp = cells3[cells3.role == "primary"]
    lines.append(f"Of the ablated main-study responses that WildGuard does not call a refusal, "
                 f"{f_pp(pm.loc['hb_harmful_given_not_refused', 'rate'], sign=False)}% "
                 f"[{f_pp(pm.loc['hb_harmful_given_not_refused', 'lo'], sign=False)}, "
                 f"{f_pp(pm.loc['hb_harmful_given_not_refused', 'hi'], sign=False)}] are harmful per HarmBench and "
                 f"{f_pp(pm.loc['wg_harmful_given_not_refused', 'rate'], sign=False)}% per WildGuard (originals: "
                 f"{f_pp(po.loc['hb_harmful_given_not_refused', 'rate'], sign=False)}% and "
                 f"{f_pp(po.loc['wg_harmful_given_not_refused', 'rate'], sign=False)}%). Across the 42 ablated seed-1 "
                 f"cells the HarmBench share ranges {f_pp(cp.hb_harmful_given_not_refused_rate.min(), sign=False)}-"
                 f"{f_pp(cp.hb_harmful_given_not_refused_rate.max(), sign=False)}% (lowest: "
                 f"{cp.loc[cp.hb_harmful_given_not_refused_rate.idxmin(), 'label']}), the WildGuard share "
                 f"{f_pp(cp.wg_harmful_given_not_refused_rate.min(), sign=False)}-"
                 f"{f_pp(cp.wg_harmful_given_not_refused_rate.max(), sign=False)}%.")
    # verdict: does the H1/H2 picture change?
    ex0 = lambda r: r.lo <= 0 <= r.hi  # noqa: E731
    pooled_same = all((ex0(res[j]["dir_adj"]) == ex0(res["wg"]["dir_adj"])) and
                      ((res[j]["h2"].hi < 0) == (res["wg"]["h2"].hi < 0)) for j in ["hbnc", "wghnc"])
    pair_changes = [f"under {OUTCOME_SHORT[j]} every method is significantly higher on the newer model in "
                    f"{', '.join(res[j]['hp']) or 'no pair'} (refusal: {', '.join(res['wg']['hp']) or 'no pair'}), "
                    "so H1 as stated would hold for " + ", ".join(sorted(set(res[j]['hp']) - set(res['wg']['hp'])))
                    for j in ["hbnc", "wghnc"] if set(res[j]["hp"]) - set(res["wg"]["hp"])]
    tid_ara = "pooled main: gen effect | optimisation class"
    rej = {j: bool(allt[(allt.judge == j) & (allt.subset == "all") & (allt.test_id == tid_ara)].reject_holm.iloc[0])
           for j in ["wg", "hbnc", "wghnc"]}
    ara_changes = [f"{OUTCOME_SHORT[j]} {fterm(res[j]['ara_raw'])}" for j in ["hbnc", "wghnc"] if rej[j] != rej["wg"]]
    nchg = {}
    a0 = allt[(allt.judge == "wg") & (allt.subset == "all") & (allt.family == "H1") &
              ~allt.pair.str.startswith("main")].set_index("test_id")
    for j in ["hbnc", "wghnc"]:
        b0 = allt[(allt.judge == j) & (allt.subset == "all") & (allt.family == "H1") &
                  ~allt.pair.str.startswith("main")].set_index("test_id")
        cmn = a0.index.intersection(b0.index)
        dec_a = a0.loc[cmn].reject_holm.astype(str) + np.sign(a0.loc[cmn].estimate).astype(str)
        dec_b = b0.loc[cmn].reject_holm.astype(str) + np.sign(b0.loc[cmn].estimate).astype(str)
        ch = cmn[(dec_a != dec_b).values]
        nm = lambda t: t.replace(": new vs old", "")  # noqa: E731
        became_pos = [nm(t) for t in ch if b0.loc[t].reject_holm and b0.loc[t].estimate > 0]
        lost = [nm(t) for t in ch if a0.loc[t].reject_holm and not b0.loc[t].reject_holm]
        flipped_neg = [nm(t) for t in ch if b0.loc[t].reject_holm and b0.loc[t].estimate < 0]
        nchg[j] = (len(ch), int((b0.loc[ch].estimate > a0.loc[ch].estimate).sum()), len(cmn),
                   became_pos, lost, flipped_neg)
    car = {}
    for j in ["wg", "hbnc", "wghnc"]:
        x = hh[(hh.judge == j) & hh.pair.isin(["qwen", "gemma", "olmo", "mistral"])]
        car[j] = (tuple(x[(x.p_holm_H2 < 0.05) & (x.gee_interaction < 0)].pair),
                  tuple(x[(x.p_holm_H2 < 0.05) & (x.gee_interaction > 0)].pair))
    if len(set(car.values())) == 1:
        carriers = (f"per pair the interaction is significant in the predicted direction for {', '.join(car['wg'][0]) or 'no pair'}"
                    f" and opposite for {', '.join(car['wg'][1]) or 'no pair'} under every outcome")
    else:
        carriers = "per pair (predicted / opposite): " + "; ".join(
            f"{OUTCOME_SHORT[j]} {', '.join(car[j][0]) or 'none'} / {', '.join(car[j][1]) or 'none'}" for j in car)
    nr_h = pm.loc["hb_harmful_given_not_refused"]
    neither = pm.loc["not_refused_not_hb_harmful"]
    desc = (f"The construct point is right: of the ablated main-study responses that WildGuard does not call a refusal, "
            f"{f_pp(nr_h.rate, sign=False)}% [{f_pp(nr_h.lo, sign=False)}, {f_pp(nr_h.hi, sign=False)}] are harmful "
            f"per HarmBench and {f_pp(pm.loc['wg_harmful_given_not_refused', 'rate'], sign=False)}% per WildGuard, so "
            f"{f_pp(neither.rate, sign=False)}% [{f_pp(neither.lo, sign=False)}, {f_pp(neither.hi, sign=False)}] of all "
            "ablated main-study responses are neither refused nor harmful (HarmBench); residual refusal understates "
            "how often an edit fails to yield harmful content.")
    if pooled_same:
        verdict = ("contradicted" if (nr_h.lo > 0.95 and not pair_changes) else "partly supported")
        why = (desc + " The headline conclusions do not change with either harm label: baseline-adjusted direction "
               f"effect {fterm(res['hbnc']['dir_adj'])} (HarmBench) and {fterm(res['wghnc']['dir_adj'])} (WildGuard "
               f"harmful) vs {fterm(res['wg']['dir_adj'])} for refusal; H2 interaction {fterm(res['hbnc']['h2'])} and "
               f"{fterm(res['wghnc']['h2'])} vs {fterm(res['wg']['h2'])}; " + carriers + ". Some per-pair and raw "
               "results do change: "
               + ("; ".join(pair_changes + ([f"the raw pooled ARA generation effect becomes Holm-significant ("
                                             + ", ".join(ara_changes) + f"; refusal {fterm(res['wg']['ara_raw'])}, "
                                             "not significant after Holm)"] if ara_changes else [])) or "none")
               + "; of the " + str(nchg["hbnc"][2]) + " per-pair H1 McNemar tests, " + str(nchg["hbnc"][0])
               + f" (HarmBench; {nchg['hbnc'][1]} of them toward 'newer harder') and {nchg['wghnc'][0]} (WildGuard "
               f"harmful; {nchg['wghnc'][1]} toward 'newer harder') change Holm decision or sign (HarmBench: "
               f"significantly higher on the newer model now {', '.join(nchg['hbnc'][3]) or 'none'}; no longer "
               f"significant {', '.join(nchg['hbnc'][4]) or 'none'}; significantly lower now "
               f"{', '.join(nchg['hbnc'][5]) or 'none'})"
               + f"; Holm tally of the 16 main tests {res['hbnc']['tally'][0]}/{res['hbnc']['tally'][1]}/"
               f"{res['hbnc']['tally'][2]} (HarmBench) and {res['wghnc']['tally'][0]}/{res['wghnc']['tally'][1]}/"
               f"{res['wghnc']['tally'][2]} (WildGuard harmful) vs {res['wg']['tally'][0]}/{res['wg']['tally'][1]}/"
               f"{res['wg']['tally'][2]}. Adding harmful compliance as a secondary H1/H2 outcome answers the concern.")
    else:
        verdict = "supported"
        why = (desc + " A pooled H1/H2 conclusion changes with the harm label: " + "; ".join(
            f"{OUTCOME_SHORT[j]} adjusted direction {fterm(res[j]['dir_adj'])}, H2 {fterm(res[j]['h2'])}"
            for j in ["hbnc", "wghnc"]) + f" (refusal: {fterm(res['wg']['dir_adj'])}, {fterm(res['wg']['h2'])}).")
    return dict(lines=lines, verdict=verdict, why=why)


def c4_findings(pooled, cg, mods, df):
    lines = []

    def pg(group, variant, subset="harmful_all"):
        x = pooled[(pooled.group == group) & (pooled.variant == variant) & (pooled.subset == subset)]
        return x.iloc[0] if len(x) else None

    gs = lambda r: f"{f_pp(r.gap)} pp [{f_pp(r.gap_lo)}, {f_pp(r.gap_hi)}]"  # noqa: E731
    for gname, lab in (("ablated_primary", "all 42 ablated seed-1 cells"), ("ablated_primary:main", "main study (32)"),
                       ("ablated_primary:small", "small pair (10)"), ("original", "originals (10)")):
        a, b, c, h, k = (pg(gname, v) for v in ("kw_vs_wg", "kw100_vs_wg", "kw100_vs_wg256", "kw100_heretic_vs_wg",
                                                 "kw100_vs_kw"))
        lines.append(f"{lab}: keyword refusal {f_pp(a.rate1, sign=False)}% on <= 512 tokens, "
                     f"{f_pp(b.rate1, sign=False)}% on 100 tokens ({f_pp(k.gap)} pp); WildGuard "
                     f"{f_pp(a.rate2, sign=False)}% (full) / {f_pp(c.rate2, sign=False)}% (256 tokens). Gap "
                     f"keyword - WildGuard: {gs(a)} (paper) -> {gs(b)} at 100 tokens vs full-response WildGuard, "
                     f"{gs(c)} vs 256-token WildGuard; Heretic's own matcher {gs(h)}.")
    cap = pg("ablated_primary", "kw100_vs_wg_capped_only"); com = pg("ablated_primary", "kw100_vs_wg_complete_only")
    lines.append(f"By length (ablated cells, 100-token keyword vs full WildGuard): responses that hit the 512 cap "
                 f"{gs(cap)}, responses that ended earlier {gs(com)}.")
    x = mods[(mods.outcome == "kw100_vs_wg") & (mods.model == "GEE_linear_paired_gap (primary)") &
             (mods.subset == "harmful_all")]
    p = mods[(mods.outcome == "kw_vs_wg") & (mods.model == "GEE_linear_paired_gap (primary)") &
             (mods.subset == "harmful_all")]
    for study in ["main", "small"]:
        xs, ps = x[x.study == study], p[p.study == study]
        if not len(xs):
            continue
        lines.append(f"H3 GEE, {study}, 100-token keyword vs WildGuard: " + "; ".join(
            f"{r.term.split(' (')[0].split(',')[0]} {f_pp(r.estimate)} pp [{f_pp(r.lo)}, {f_pp(r.hi)}] (p_Holm "
            f"{f_p(r.p_holm)}; full text {f_pp(q.estimate)})" for r, q in zip(xs.itertuples(), ps.itertuples())) + ".")
    s = cg[(cg.set_group == "harmful_all") & (cg.role == "primary") & (cg.comparison == "kw100_vs_wg")]
    lines.append(f"Per ablated seed-1 cell (100-token keyword vs full WildGuard): keyword lower in "
                 f"{int((s.gap < 0).sum())}/{len(s)} cells ({int(((s.gap < 0) & (s.p_holm < 0.05)).sum())} "
                 f"Holm-significant), higher in {int((s.gap > 0).sum())} ({int(((s.gap > 0) & (s.p_holm < 0.05)).sum())}).")
    blo, bhi = pg("ablated_primary", "kw100_lo_vs_wg"), pg("ablated_primary", "kw100_hi_vs_wg")
    lines.append(f"Bounds on the true 100-token label (all ablated cells): gap vs full WildGuard between {gs(blo)} "
                 f"(lower bound) and {gs(bhi)} (upper bound).")
    # verdict (robust only if the point estimate and both bounds agree)
    b, c, a = pg("ablated_primary", "kw100_vs_wg"), pg("ablated_primary", "kw100_vs_wg256"), pg("ablated_primary",
                                                                                                "kw_vs_wg")
    bm, bs = pg("ablated_primary:main", "kw100_vs_wg"), pg("ablated_primary:small", "kw100_vs_wg")
    neg = (b.gap_hi < 0) and (c.gap_hi < 0) and (bhi.gap_hi < 0)
    pos = (b.gap_lo > 0) and (c.gap_lo > 0) and (blo.gap_lo > 0)
    gx = x[(x.study == "main") & x.term.str.startswith("generation x class")]
    gg = x[(x.study == "main") & x.term.str.startswith("generation effect")]
    cl = x[(x.study == "main") & x.term.str.startswith("class effect")]
    second = (f" The second part of H3 (the gap differs between method classes) is unchanged in the main study: at 100 "
              f"tokens the gap is still larger for ARA ({f_pp(cl.estimate.iloc[0])} pp, p_Holm "
              f"{f_p(cl.p_holm.iloc[0])}), grows with generation ({f_pp(gg.estimate.iloc[0])} pp) and shows the same "
              f"generation x class interaction ({f_pp(gx.estimate.iloc[0])} pp [{f_pp(gx.lo.iloc[0])}, "
              f"{f_pp(gx.hi.iloc[0])}], p_Holm {f_p(gx.p_holm.iloc[0])})." if len(gx) and len(cl) and len(gg) else "")
    n_pos, n_neg = int(((s.gap > 0) & (s.p_holm < 0.05)).sum()), int(((s.gap < 0) & (s.p_holm < 0.05)).sum())
    if neg:
        verdict = "supported"
        why = (f"At Heretic's 100-token window the keyword measure counts fewer refusals than WildGuard ({gs(b)} vs "
               f"the full-response label, {gs(c)} vs the 256-token label, all ablated cells), so the paper's "
               f"'H3 reversed' ({gs(a)}) is a property of the 512-token window." + second)
    elif pos:
        verdict = "partly supported"
        why = (f"The window matters for the size of the gap ({gs(a)} at 512 tokens -> {gs(b)} at 100 tokens vs full "
               f"WildGuard, {gs(c)} vs 256-token WildGuard), but the keyword measure still counts more refusals than "
               "WildGuard at the tool's own window, so 'H3 reversed' stands." + second)
    else:
        verdict = "supported"
        why = (f"At Heretic's own 100-token window the reversal disappears: keyword minus WildGuard over all 42 ablated "
               f"seed-1 cells is {gs(b)} against the full-response WildGuard label and {gs(c)} against the 256-token "
               f"label, versus {gs(a)} at 512 tokens in the paper (main study {gs(bm)}, small pair {gs(bs)}). The "
               "keyword count falls from "
               f"{f_pp(a.rate1, sign=False)}% to {f_pp(b.rate1, sign=False)}% while WildGuard stays at "
               f"{f_pp(a.rate2, sign=False)}%: most of the excess keyword refusals are markers that appear after the "
               "first 100 tokens. " + ("Neither the paper's 'keyword overcounts' nor H3's predicted 'keyword "
                                        "undercounts' holds pooled at the tool's window"
                                        if (b.gap_lo <= 0 <= b.gap_hi and c.gap_lo <= 0 <= c.gap_hi) else
                                        "The pooled sign depends on which WildGuard window is used") +
               f"; per cell the sign varies ({n_pos} cells significantly higher, {n_neg} "
               "lower, Holm). The bounds on the true 100-token label give the same result "
               f"({gs(blo)} to {gs(bhi)})." + second + " H3 should be reported per window, with the 100-token result "
               "as the one that bears on the tool.")
    return dict(lines=lines, verdict=verdict, why=why)


# ============================================================================================================ main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tokenize", action="store_true", help="run only the tokeniser stage (needs `tokenizers`)")
    ap.add_argument("--retokenize", action="store_true", help="force the tokeniser stage before the analysis")
    ap.add_argument("--boot", type=int, default=N_BOOT, help=f"item-bootstrap resamples for C2/C3 (default {N_BOOT})")
    args = ap.parse_args()
    if args.tokenize:
        stage_tokenize()
    else:
        main_analysis(args)


if __name__ == "__main__":
    main()
