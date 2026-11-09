"""Add the LoRA adapter (adapter/) to each heretic1x repo and note it in the card."""
import os, pandas as pd
from huggingface_hub import HfApi, CommitOperationAdd, hf_hub_download
api = HfApi(); c = pd.read_csv("/data/heretic_2/paper30/v3_2611/analysis/cells.csv").set_index("label")
B = {"qwen2.5-7b": "Qwen2.5-7B-Instruct", "qwen2.5-coder-7b": "Qwen2.5-Coder-7B-Instruct", "qwen2.5-14b": "Qwen2.5-14B-Instruct",
     "qwen2.5-coder-14b": "Qwen2.5-Coder-14B-Instruct", "qwen2.5-32b": "Qwen2.5-32B-Instruct", "qwen2.5-coder-32b": "Qwen2.5-Coder-32B-Instruct"}
NOTE = ("\n## LoRA adapter\n\n`adapter/` holds the rank-1 PEFT LoRA adapter that Heretic 1.x exported for the selected trial. Merging it "
        "into the base model on CPU (PEFT `merge_and_unload`, adapter weights in fp32) reproduces these weights tensor for tensor.\n")
for lab in [l for l in c.index if "heretic1x" in l]:
    m, v = lab.split("__heretic1x-"); repo = f"richardyoung/{B[m]}-heretic1x-{v}"
    ad = ("/data/heretic_2/paper30/v3_2611/reexport_1x/results_rtx/qwen2.5-32b__heretic1x-general-kl0.1/adapter"
          if v.endswith("kl0.1") else os.path.join(c.loc[lab, "h1x_dir"], "adapter"))
    files = [f for f in os.listdir(ad) if f in ("adapter_config.json", "adapter_model.safetensors")]
    assert len(files) == 2, (ad, files)
    card = open(hf_hub_download(repo, "README.md", force_download=True)).read()
    ops = [CommitOperationAdd(f"adapter/{f}", os.path.join(ad, f)) for f in files]
    if "## LoRA adapter" not in card:
        card = card.replace("\n## Integrity", NOTE + "\n## Integrity", 1); ops.append(CommitOperationAdd("README.md", card.encode()))
    api.create_commit(repo, operations=ops, commit_message="Add the Heretic 1.x LoRA adapter")
    print("ok", repo, ad, flush=True)
