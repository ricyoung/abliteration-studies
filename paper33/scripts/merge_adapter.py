"""
Merge a Heretic LoRA adapter into its base model to produce a standalone model
that vLLM can serve normally.

Needed because the adapters use PEFT `target_parameters: ["down_proj"]` (a newer
PEFT feature) alongside `target_modules: ["o_proj"]`, which vLLM's LoRA serving
does not support. Merging applies both edits into the weights.

CPU merge (no GPU contention with a running server; 177GB RAM handles 32B).

  python3 scripts/merge_adapter.py --base models/<base> --adapter <adapter_dir> --out <out_dir>
"""

import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.base)
    base = AutoModelForCausalLM.from_pretrained(
        args.base, dtype="bfloat16", device_map="cpu", low_cpu_mem_usage=True,
    )
    merged = PeftModel.from_pretrained(base, args.adapter).merge_and_unload()
    merged.save_pretrained(args.out, safe_serialization=True, max_shard_size="5GB")
    tok.save_pretrained(args.out)
    print(f"merged -> {args.out}")


if __name__ == "__main__":
    main()
