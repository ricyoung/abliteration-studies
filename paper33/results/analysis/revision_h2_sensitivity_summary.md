# Revision analyses REV-16 / REV-24 / REV-27 (held-out H2 sensitivity)

Generated 2026-11-09 00:26 UTC by scripts/analysis/revision_h2_sensitivity.py.

- qwen | all direction methods (dim+mpoa+som) | direction class: +10.7 pp, prompt + seed [-4.0, +25.5]; logit-level [-15.3, +36.7]
- qwen | all direction methods (dim+mpoa+som) | ARA: +34.4 pp, prompt + seed [+8.8, +59.9]; logit-level [-5.9, +74.6]
- qwen | all direction methods (dim+mpoa+som) | H2 contrast: +23.6 pp, prompt + seed [-5.9, +53.1]; logit-level [-24.3, +71.6]
- qwen | cap-bound direction units only (dim) | direction class: +26.9 pp, prompt + seed [+1.3, +52.4]; logit-level [-24.2, +77.9]
- qwen | cap-bound direction units only (dim) | ARA: +34.4 pp, prompt + seed [+8.8, +59.9]; logit-level [-5.9, +74.6]
- qwen | cap-bound direction units only (dim) | H2 contrast: +7.5 pp, prompt + seed [-28.6, +43.7]; logit-level [-57.5, +72.5]
- gemma | all direction methods (dim+mpoa+som) | direction class: +4.7 pp, prompt + seed [-10.2, +19.6]; logit-level [-23.2, +32.6]
- gemma | all direction methods (dim+mpoa+som) | ARA: -26.0 pp, prompt + seed [-51.6, -0.5]; logit-level [-57.9, +5.8]
- gemma | all direction methods (dim+mpoa+som) | H2 contrast: -30.8 pp, prompt + seed [-60.2, -1.3]; logit-level [-73.1, +11.5]
- gemma | cap-bound direction units only (dim+mpoa+som) | direction class: +4.7 pp, prompt + seed [-10.2, +19.6]; logit-level [-23.2, +32.6]
- gemma | cap-bound direction units only (dim+mpoa+som) | ARA: -26.0 pp, prompt + seed [-51.6, -0.5]; logit-level [-57.9, +5.8]
- gemma | cap-bound direction units only (dim+mpoa+som) | H2 contrast: -30.8 pp, prompt + seed [-60.2, -1.3]; logit-level [-73.1, +11.5]
- olmo | all direction methods (dim+mpoa+som) | direction class: +36.9 pp, prompt + seed [+22.1, +51.7]; logit-level [+9.8, +63.9]
- olmo | all direction methods (dim+mpoa+som) | ARA: -2.5 pp, prompt + seed [-27.9, +23.0]; logit-level [-26.9, +22.0]
- olmo | all direction methods (dim+mpoa+som) | H2 contrast: -39.3 pp, prompt + seed [-68.8, -9.9]; logit-level [-75.8, -2.9]
- olmo | cap-bound direction units only | H2 contrast: not estimable (no cap-bound direction unit)
- mistral | all direction methods (dim+mpoa+som) | direction class: -1.2 pp, prompt + seed [-16.0, +13.5]; logit-level [-18.5, +16.1]
- mistral | all direction methods (dim+mpoa+som) | ARA: -0.7 pp, prompt + seed [-26.1, +24.7]; logit-level [-10.8, +9.4]
- mistral | all direction methods (dim+mpoa+som) | H2 contrast: +0.5 pp, prompt + seed [-28.8, +29.9]; logit-level [-19.5, +20.5]
- mistral | cap-bound direction units only (dim) | direction class: -4.3 pp, prompt + seed [-29.9, +21.2]; logit-level [-53.1, +44.5]
- mistral | cap-bound direction units only (dim) | ARA: -0.7 pp, prompt + seed [-26.1, +24.7]; logit-level [-10.8, +9.4]
- mistral | cap-bound direction units only (dim) | H2 contrast: +3.6 pp, prompt + seed [-32.4, +39.6]; logit-level [-46.3, +53.5]
- gemma | Gemma, all direction methods (dim+mpoa+som) | direction class: +4.7 pp, prompt + seed [-10.2, +19.6]
- gemma | Gemma, all direction methods (dim+mpoa+som) | H2 contrast: -30.8 pp, prompt + seed [-60.2, -1.3]
- gemma | Gemma without the failed Gemma-4-31B DiM edit (mpoa+som) | direction class: -1.1 pp, prompt + seed [-19.3, +17.1]
- gemma | Gemma without the failed Gemma-4-31B DiM edit (mpoa+som) | H2 contrast: -24.9 pp, prompt + seed [-56.2, +6.3]
- gemma | gMPOA - MPOA, older model: -4.2 pp [-6.8, -1.6] (prompt only)
- gemma | gMPOA - MPOA, newer model: -19.9 pp [-22.7, -17.0] (prompt only)
- gemma | generation difference of the gamma-correction effect: -15.6 pp [-19.6, -11.7] (prompt only)
- olmo | gMPOA - MPOA, older model: -3.1 pp [-4.2, -2.0] (prompt only)
- olmo | gMPOA - MPOA, newer model: -7.5 pp [-9.4, -5.6] (prompt only)
- olmo | generation difference of the gamma-correction effect: -4.4 pp [-6.6, -2.2] (prompt only)

- pooled all direction methods (qwen+gemma+olmo+mistral), prompt sampling only: -11.5 pp HKSJ [-57.6, +34.6], Q 1006.7 (p 6.42e-218), I2 100%
- pooled all direction methods (qwen+gemma+olmo+mistral), prompt + seed (constant sigma): -11.5 pp HKSJ [-57.6, +34.7], Q 11.2 (p 0.0109), I2 73%
- pooled cap-bound direction units only (qwen+gemma+mistral), prompt sampling only: -6.6 pp HKSJ [-58.9, +45.8], Q 282.3 (p 4.91e-62), I2 99%
- pooled cap-bound direction units only (qwen+gemma+mistral), prompt + seed (constant sigma): -8.5 pp HKSJ [-62.5, +45.5], Q 3.4 (p 0.187), I2 40%
