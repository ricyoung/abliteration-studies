"""Model card and checksum list for a <name>-GGUF repository (papers 30 and 33)."""
import re
from common import *

NOTE = {"IQ3_M": "smallest", "IQ4_XS": "small, close to Q4_K_M", "Q4_K_M": "default (`latest` on Ollama)",
        "Q5_K_M": "higher quality", "Q6_K": "very high quality", "Q8_0": "near-lossless"}
ORDER = ["IQ3_M", "IQ4_XS", "Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0"]

def source_license(name):
    """Licence fields from the source model card's front matter, carried over unchanged."""
    out = {}
    try:
        txt = open(f"{bdir(name)}/source_README.md").read()
        fm = txt.split("---", 2)[1] if txt.startswith("---") else ""
        for k in ("license", "license_name", "license_link"):
            mm = re.search(rf"^{k}:\s*(.+)$", fm, re.M)
            if mm: out[k] = mm.group(1).strip()
    except FileNotFoundError:
        pass
    return out or {"license": "apache-2.0"}

def sums(m, arts, imatrix):
    lines = [f"{arts[q]['sha256']}  {arts[q]['file']}" for q in ORDER if q in arts]
    if imatrix: lines.append(f"{imatrix['sha256']}  {m['name']}-imatrix.gguf")
    return "\n".join(lines) + "\n"

def make_card(m, arts, imatrix, src_rev, on_ollama):
    """arts: quant -> artifact record, for the files already in the repository."""
    name = m["name"]; lic = source_license(name)
    fm = ["---", f"base_model: {m['src_repo']}", "base_model_relation: quantized"]
    fm += [f"{k}: {v}" for k, v in lic.items()]
    fm += ["pipeline_tag: text-generation", "language:", "- en",
           "tags: [gguf, llama.cpp, ollama, heretic, abliterated, uncensored, imatrix, research]", "---", ""]
    L = fm + [f"# {name} GGUF", "",
        f"GGUF builds of [{m['src_repo']}](https://huggingface.co/{m['src_repo']}), an abliterated (refusal-removed) version of "
        f"[{m['base_repo']}](https://huggingface.co/{m['base_repo']}) made with {m['tool']}: {m['method']}. "
        "It is one of the models evaluated in:", "",
        f"> Richard J. Young. *{m['paper_title']}* (arXiv link to be added).", "",
        "These files are released for research on abliteration and its evaluation, not as a general-purpose assistant. "
        "The model complies with many requests that the original refuses.", ""]
    L += ["## Results for the full-precision model", "",
          "| | Original | This model |", "|---|---|---|",
          f"| Refusal on {m['heldout_n']} held-out harmful prompts (WildGuard judge) | {m['original_refusal_pct']}% | "
          f"{m['heldout_refusal_pct']}% ({m['heldout_k']}/{m['heldout_n']}) |"]
    if m.get("dev_refusals") is not None:
        L += [f"| Keyword refusals on the 100 tuning prompts, selected trial | | {m['dev_refusals']} |"]
    if m.get("dev_kl") is not None:
        L += [f"| First-token KL divergence from the original, selected trial | | {m['dev_kl']} |"]
    L += ["", f"Selection rule: {m['selection_rule']}. These numbers are from the paper and describe the full-precision "
          "(bf16) model. The quantized files here were not evaluated in the paper, and quantization can change refusal "
          "behaviour.", ""]
    L += ["## Files", "", "| File | Quantization | Size | Notes |", "|---|---|---|---|"]
    for q in ORDER:
        if q in arts:
            L.append(f"| `{arts[q]['file']}` | {q} | {arts[q]['size']/1e9:.1f} GB | {NOTE[q]} |")
    missing = [q for q in ORDER if q not in arts]
    if imatrix:
        L.append(f"| `{name}-imatrix.gguf` | | {imatrix['size']/1e6:.0f} MB | importance matrix |")
    L += [""]
    if missing:
        L += [f"More sizes are being added: {', '.join(missing)}.", ""]
    L += ["IQ3_M, IQ4_XS, Q4_K_M and Q5_K_M use the importance matrix (calibration_datav3, 128 chunks of 512 tokens, "
          "computed on the Q8_0 build); Q6_K and Q8_0 are static. SHA-256 sums are in `SHA256SUMS`.", ""]
    if m["image_text"]:
        L += ["Text only: the base model also accepts images, but the vision projector is not included here. The study "
              "used these models with text only.", ""]
    L += ["## Use", "", "```bash"]
    if on_ollama:
        L += [f"ollama run {m['ollama_name']}            # Q4_K_M; other sizes are tags, e.g. :Q8_0"]
    L += [f"ollama run hf.co/richardyoung/{name}-GGUF:Q4_K_M",
          f"llama-cli -hf richardyoung/{name}-GGUF:Q4_K_M", "```", ""]
    if m["gated"]:
        L += ["Access to this repository is gated with automatic approval, as for the source repository.", ""]
    L += ["## How the files were built", "",
          f"The published safetensors of `{m['src_repo']}` (revision `{src_rev[:12]}`, each weight file checked against its "
          f"SHA-256) were converted with llama.cpp (commit `{LLAMA_COMMIT}`): `convert_hf_to_gguf.py --outtype bf16`, then "
          "`llama-quantize`, on CPU. Each Q4_K_M file was loaded and answered a test prompt before upload.", "",
          "## Licence and intended use", "",
          f"Same licence and use restrictions as the source model ({lic.get('license_name') or lic.get('license')}); see the "
          f"[source model card](https://huggingface.co/{m['src_repo']}). Intended for research on refusal removal, its "
          "measurement and defences against it. Users are responsible for lawful use.", "",
          "## Citation", "", "```bibtex",
          "@misc{young2026" + ("harder" if m["paper"] == 33 else "selection") + ",",
          f"  title  = {{{m['paper_title']}}},", "  author = {Young, Richard J.},", "  year   = {2026},",
          "  note   = {arXiv link to be added}", "}", "```", ""]
    return "\n".join(L)
