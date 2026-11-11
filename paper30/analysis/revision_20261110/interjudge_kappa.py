"""WildGuard vs Qwen3Guard refusal-label agreement on held-out harmful prompts, per cell and pooled (Limitations,
section 5.2). Read-only on analysis/. Input: analysis/responses_p30.csv.gz. Output: CSV on stdout
(python interjudge_kappa.py > interjudge_kappa.csv)."""
import gzip, csv, collections, sys, pathlib
src = pathlib.Path(__file__).resolve().parents[1] / "responses_p30.csv.gz"   # analysis/responses_p30.csv.gz
cells = collections.defaultdict(list)
for x in csv.DictReader(gzip.open(src, "rt")):
    if x["kind"] != "harmful" or x["ref_wg"] not in ("True", "False") or x["ref_q3g"] not in ("True", "False"):
        continue
    p = (x["ref_wg"] == "True", x["ref_q3g"] == "True")
    cells[x["label"]].append(p); cells["__ALL__"].append(p)
w = csv.writer(sys.stdout); w.writerow(["label", "n", "raw_agreement", "cohen_kappa", "wg_rate_pct", "q3g_rate_pct"])
for k in sorted(cells):
    p = cells[k]; n = len(p); a = sum(u == v for u, v in p) / n
    px = sum(u for u, _ in p) / n; py = sum(v for _, v in p) / n; pe = px * py + (1 - px) * (1 - py)
    kap = (a - pe) / (1 - pe) if pe < 1 else float("nan")
    w.writerow([k, n, f"{a:.4f}", f"{kap:.4f}", f"{100*px:.1f}", f"{100*py:.1f}"])
