# GGUF builds of the paper 33 / paper 30 models (started 2026-11-10)

Goal (user, 2026-11-10): GGUF builds of every public paper model, in the six sizes the account already uses
(IQ3_M, IQ4_XS, Q4_K_M, Q5_K_M, Q6_K, Q8_0), with model cards, on Hugging Face (`richardyoung/<name>-GGUF`) and on the
Ollama account (`richardyoung/<lowercase name>`, tags = sizes, `latest` = Q4_K_M).

- 54 models: 41 of paper 33 (the five Gemma-2-27B variants are withheld and excluded) + 13 of paper 30. List and card
  numbers: `models.json` (made by `scripts/gen_models.py`).
- OLMo-2-32B (5 models) is gated on Hugging Face; the Ollama registry cannot gate, so these are NOT pushed to Ollama
  (their GGUF repos on Hugging Face are gated with automatic approval). Awaiting the user's decision.
- CPU only (no GPU). llama.cpp commit 468fde9, CPU build in `/data/heretic_2/llama.cpp/build-cpu`.
- `scripts/builder.py`: phase A for every model (stage verified weights -> bf16 -> Q8_0 -> importance matrix -> Q4_K_M ->
  load test), then phase B (Q5_K_M, IQ4_XS, Q6_K, IQ3_M). `scripts/uploader.py`: publishes finished files, Q4_K_M first.
- State: `build/<name>/{build_state,artifacts,published}.json`. Logs: `logs/builder.log`, `logs/uploader.log`,
  `logs/failures.log`.
- Stop cleanly: `touch state/STOP`. Restart (resumes): run both scripts again with the bench venv python (see below).
  A model or file marked failed is skipped until its `failed_*` / `failed` entry is removed from the state file.
- PIDs: `state/builder.pid`, `state/uploader.pid` (kill by PID only).

Restart:
    cd /data/heretic_2/gguf && V=/data/heretic_2/33_2611_does_refusal_removal_get_harder/scripts/bench/.venv/bin/python
    rm -f state/STOP; setsid nohup $V scripts/builder.py > logs/builder.out 2>&1 < /dev/null &
    setsid nohup $V scripts/uploader.py > logs/uploader.out 2>&1 < /dev/null &

Measured on the pilot (7B, CPU): convert 156 s, Q8_0 46 s, importance matrix about 15 min, Q4_K_M 77 s.
Not done: the "Quant Quality" table (KL versus F16) that the account's other GGUF cards have; vision projectors.
