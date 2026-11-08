# Local patches to lm-eval 0.4.13 (paper 33)

## mbpp_instruct: extract_code_blocks drops the first word of every answer
`lm_eval/tasks/mbpp/utils.py::extract_code_blocks` prepends a bare "```" to the generation (the gen_prefix
"\n```python\n" is not part of the response). The pattern ```` ```(?:\w+)?\n?(.*?)\n?``` ```` then consumes the
first word of the code ("def") as a language tag -> every prediction starts with " name(...):" -> pass@1 = 0.
Patch: prepend "```python" + a real newline (not a raw-string backslash-n) instead of "```". Applied with apply_lm_eval_patches.sh; MBPP scores produced before
the patch are recomputed from the logged samples with rescore_mbpp.py.
