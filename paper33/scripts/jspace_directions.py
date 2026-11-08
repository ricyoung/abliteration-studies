#!/usr/bin/env python3
"""Export J-space (Jacobian-pullback) refusal directions for heretic's
PrecomputedDirections modifier (paper 33).

Run with the jspace-refusal venv (it has jlens + jrefusal):
  /data/heretic_2/external/jspace-refusal/.venv/bin/python -I jspace_directions.py \
      --model Qwen/Qwen3.5-4B \
      --lens-file qwen3.5-4b/jlens/Salesforce-wikitext/Qwen3.5-4B_jacobian_lens_n1000.pt \
      --lens-revision qwen-n1000 --out directions/qwen3.5-4b__jspace.pt

Method (Sekinal/jspace-refusal, refusal.py): d_l = J_l^T (g * (mean W[refusal] - mean W)),
unit-normalised, where W is the unembedding, g the final-norm gain and J_l the lens
Jacobian of layer l. No harmful/harmless contrast set is used.

Output: {"directions": Tensor[n_layers + 1, d_model] (row 0 = embeddings, row l+1 =
output of block l, heretic's convention), "meta": {...}}. Layers the lens does not
cover are filled from the nearest covered layer and listed in meta["filled_layers"].
"""
import argparse
import json
import sys

import jlens
import torch
import transformers
from jrefusal.refusal import REFUSAL_WORDS, refusal_token_ids

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--lens-repo", default="neuronpedia/jacobian-lens")
ap.add_argument("--lens-file", required=True)
ap.add_argument("--lens-revision", default=None)
ap.add_argument("--out", required=True)
ap.add_argument("--device", default="cpu")
ap.add_argument("--no-gain", action="store_true", help="skip final-norm gain (use for Gemma's (1+w) norm)")
a = ap.parse_args()

model = transformers.AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32 if a.device == "cpu" else torch.bfloat16).to(a.device).eval()
tok = transformers.AutoTokenizer.from_pretrained(a.model)
lens_model = jlens.from_hf(model, tok)
lens = jlens.JacobianLens.from_pretrained(a.lens_repo, filename=a.lens_file, revision=a.lens_revision)

n_layers = lens_model.n_layers
covered = sorted(int(l) for l in lens.jacobians)
ref_ids = refusal_token_ids(tok, REFUSAL_WORDS)

# Refusal covector in the final-residual basis: g * (mean W[refusal] - mean W).
# The final-norm gain g is PROBED (norm(ones) = effective per-dimension gain), which is correct for both
# plain-w RMSNorm and (1+w) RMSNorm variants (Qwen3.5/3.8, Gemma-2); the reference code reads norm.weight.
W = lens_model._lm_head.weight.detach().float()
w = W[ref_ids].mean(0) - W.mean(0)
norm = lens_model._final_norm
with torch.no_grad():
    ones = torch.ones(1, 1, W.shape[1], device=W.device, dtype=next(norm.parameters()).dtype)
    g = norm(ones).float().reshape(-1)
if not a.no_gain:
    w = w * g.to(w.device)

d_model = W.shape[1]
out = torch.zeros(n_layers + 1, d_model, dtype=torch.float32)
filled = []
for l in range(n_layers):
    if l in lens.jacobians:
        J = lens.jacobians[l].float().to(w.device)       # d(final residual)/d(residual after block l)
        d = J.t() @ w
    elif l == n_layers - 1:
        d = w.clone()                                     # J = I for the last block
    else:
        near = min(covered, key=lambda c: abs(c - l))
        d = lens.jacobians[near].float().to(w.device).t() @ w
        filled.append(l)
    out[l + 1] = (d / (d.norm() + 1e-8)).cpu()
out[0] = out[1]
filled.append(-1)  # embeddings row copied from block 0

meta = dict(model=a.model, lens_repo=a.lens_repo, lens_file=a.lens_file, lens_revision=a.lens_revision,
            n_layers=n_layers, d_model=d_model, covered_layers=covered, filled_layers=filled,
            refusal_words=list(REFUSAL_WORDS), n_refusal_token_ids=len(ref_ids), use_gain=not a.no_gain,
            gain_probe_mean=float(g.mean()), gain_probe_std=float(g.std()),
            method="jspace-pullback")
torch.save({"directions": out, "meta": meta}, a.out)
print(json.dumps({k: v for k, v in meta.items() if k not in ("refusal_words",)}, indent=1))
print("saved", a.out, tuple(out.shape), file=sys.stderr)
