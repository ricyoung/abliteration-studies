"""Upload the 4-bit MLX conversions of the paper-33 small-pair models (made on the M4 Max with mlx-lm 0.32.2) to Hugging Face,
with a model card, and add them to the user's 'Apple Silicon MLX Models' collection. Source dirs: /data/heretic_2/mlx_models/<name>."""
import sys, pathlib
from huggingface_hub import HfApi
api = HfApi(); ROOT = pathlib.Path("/data/heretic_2/mlx_models")
COLL = "richardyoung/apple-silicon-mlx-models-69267b28ac195e35285b951c"
CARD = """---
library_name: mlx
base_model: richardyoung/{src}
tags: [mlx, heretic, abliterated, uncensored, decensored, 4-bit]
license: apache-2.0
pipeline_tag: text-generation
---
# {name}

4-bit [MLX](https://github.com/ml-explore/mlx) conversion of [richardyoung/{src}](https://huggingface.co/richardyoung/{src}) for Apple Silicon.

- **Source model:** `richardyoung/{src}`, a refusal-removed ({method}) version of `{orig}` produced with Heretic 2.0, part of the study
  *Is Abliteration Getting Harder? Directional vs. Optimization-Based Refusal Removal Across Generations of Open-Weight LLMs* (Young, 2026, in preparation).
  See the source repository for the method, trial and evaluation details.
- **Quantization:** `mlx_lm.convert -q --q-bits 4 --q-group-size 64` (mlx-lm 0.32.2, affine 4-bit, group size 64). Only the quantization
  differs from the source weights; the paper's evaluations were run on the bf16 source model, not on this 4-bit version.
{think}
## Use
```bash
pip install mlx-lm
mlx_lm.generate --model richardyoung/{name} --prompt "Hello"
```

**Responsible use:** this model has had its refusal behaviour removed and may comply with harmful requests. It is released for research
on safety mechanisms and their robustness; the user is responsible for lawful and ethical use.
"""
ORIG = {"Qwen3.5-4B": "Qwen/Qwen3.5-4B", "Qwen2.5-7B-Instruct": "Qwen/Qwen2.5-7B-Instruct"}
done = []
for d in sorted(ROOT.glob("*-MLX-4bit")):
    name = d.name; src = name.replace("-MLX-4bit", "")
    base = "Qwen3.5-4B" if src.startswith("Qwen3.5-4B") else "Qwen2.5-7B-Instruct"
    method = src[len(base) + 1:].replace("-heretic", "")
    think = ("- **Thinking:** Qwen3.5 starts with a thinking section by default; the paper evaluated with thinking off "
             "(`enable_thinking=False` in the chat template).\n") if base == "Qwen3.5-4B" else ""
    (d / "README.md").write_text(CARD.format(name=name, src=src, method=method, orig=ORIG[base], think=think))
    repo = f"richardyoung/{name}"
    api.create_repo(repo, private=False, exist_ok=True)
    api.upload_folder(repo_id=repo, folder_path=str(d), ignore_patterns=["DONE", "*.test.txt"], commit_message="4-bit MLX conversion (mlx-lm 0.32.2, group 64)")
    try: api.add_collection_item(COLL, item_id=repo, item_type="model", exists_ok=True)
    except Exception as e: print("collection:", type(e).__name__, e)
    done.append(repo); print("uploaded", repo, flush=True)
print(len(done), "MLX repos published")
