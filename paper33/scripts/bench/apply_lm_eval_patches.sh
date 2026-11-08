#!/usr/bin/env bash
# Idempotent: patch lm-eval mbpp extractor in this bench venv (see LM_EVAL_PATCHES.md)
F=$(dirname "$0")/.venv/lib/python3.12/site-packages/lm_eval/tasks/mbpp/utils.py
sed -i -e 's#re.findall(pattern, r"```python\\n" + text, re.DOTALL)#re.findall(pattern, "```python\\n" + text, re.DOTALL)#' \
       -e 's#re.findall(pattern, r"```" + text, re.DOTALL)#re.findall(pattern, "```python\\n" + text, re.DOTALL)#' "$F"
grep -q 're.findall(pattern, "```python\\n" + text' "$F" && echo "patched $F" || { echo "PATCH FAILED"; exit 1; }
