# Revision analyses REV-14 / REV-15 (seed variance)

Generated 2026-11-09 00:25 UTC by scripts/analysis/revision_seed_variance.py. Read-only on its inputs.

## Run-to-run sigma
- constant pp: 9.15 pp [6.40, 16.07]; logit: 0.828 [0.578, 1.453]
- fit to the 10 seed-pair pp differences: log-likelihood constant 6.25, logit-level 6.01 (AIC difference -0.49); Spearman |d_pp| vs pbar(1-pbar) -0.01 (p 0.99).

## Per-method flags (16 main effects), number exceeding 2 x seed SE
- sigma lo: constant 5 raw / 7 adjusted; logit-level 3 raw / 5 adjusted
- sigma point: constant 5 raw / 3 adjusted; logit-level 1 raw / 4 adjusted
- sigma hi: constant 1 raw / 1 adjusted; logit-level 0 raw / 1 adjusted

Exceeding under logit-level sigma (point): olmo mpoa +61.4

## Per-pair class intervals excluding zero (pp)
- const_lo: 13 of 25: qwen direction raw; qwen ARA raw; qwen ARA adjusted; qwen H2 contrast (ARA - direction); gemma ARA raw; gemma ARA adjusted; gemma H2 contrast (ARA - direction); olmo direction raw; olmo direction adjusted; olmo H2 contrast (ARA - direction); mistral direction adjusted; mistral ARA adjusted; qwen-small direction raw
- const_point: 9 of 25: qwen ARA raw; gemma ARA raw; gemma ARA adjusted; gemma H2 contrast (ARA - direction); olmo direction raw; olmo direction adjusted; olmo H2 contrast (ARA - direction); mistral direction adjusted; qwen-small direction raw
- const_hi: 2 of 25: olmo direction raw; olmo direction adjusted
- level_lo: 10 of 25: qwen ARA raw; gemma ARA raw; gemma ARA adjusted; gemma H2 contrast (ARA - direction); olmo direction raw; olmo direction adjusted; olmo H2 contrast (ARA - direction); mistral direction adjusted; mistral ARA adjusted; qwen-small direction raw
- level_point: 5 of 25: olmo direction raw; olmo direction adjusted; olmo H2 contrast (ARA - direction); mistral direction adjusted; mistral ARA adjusted
- level_hi: 1 of 25: mistral ARA adjusted

## Heterogeneity (4 main pairs)
| estimand | scale | variance | Q | p | I2 | tau |
|---|---|---|---|---|---|---|
| direction raw | pp | prompt sampling only | 821.8 | 8.09e-178 | 100% | 0.168 |
| direction raw | pp | prompt + seed, constant sigma_pp (lo) | 29.8 | 1.54e-06 | 90% | 0.159 |
| direction raw | pp | prompt + seed, logit-level sigma (lo) | 11.3 | 0.0104 | 73% | 0.142 |
| direction raw | pp | prompt + seed, constant sigma_pp (point) | 14.8 | 0.00196 | 80% | 0.150 |
| direction raw | pp | prompt + seed, logit-level sigma (point) | 5.5 | 0.136 | 46% | 0.113 |
| direction raw | pp | prompt + seed, constant sigma_pp (hi) | 4.9 | 0.181 | 39% | 0.104 |
| direction raw | pp | prompt + seed, logit-level sigma (hi) | 1.8 | 0.613 | 0% | 0.000 |
| direction raw | log-odds | prompt sampling only | 590.8 | 9.93e-128 | 99% | 0.748 |
| direction raw | log-odds | prompt + seed, constant sigma_L (lo) | 7.6 | 0.0541 | 61% | 0.591 |
| direction raw | log-odds | prompt + seed, constant sigma_L (point) | 3.8 | 0.289 | 20% | 0.340 |
| direction raw | log-odds | prompt + seed, constant sigma_L (hi) | 1.2 | 0.747 | 0% | 0.000 |
| direction adjusted | pp | prompt sampling only | 849.4 | 8.52e-184 | 100% | 0.229 |
| direction adjusted | pp | prompt + seed, constant sigma_pp (lo) | 54.9 | 7.14e-12 | 95% | 0.225 |
| direction adjusted | pp | prompt + seed, logit-level sigma (lo) | 23.1 | 3.88e-05 | 87% | 0.224 |
| direction adjusted | pp | prompt + seed, constant sigma_pp (point) | 27.7 | 4.12e-06 | 89% | 0.218 |
| direction adjusted | pp | prompt + seed, logit-level sigma (point) | 11.5 | 0.0095 | 74% | 0.206 |
| direction adjusted | pp | prompt + seed, constant sigma_pp (hi) | 9.2 | 0.0266 | 67% | 0.190 |
| direction adjusted | pp | prompt + seed, logit-level sigma (hi) | 3.8 | 0.289 | 20% | 0.108 |
| direction adjusted | log-odds | prompt sampling only | 608.6 | 1.4e-131 | 100% | 1.238 |
| direction adjusted | log-odds | prompt + seed, constant sigma_L (lo) | 15.5 | 0.00143 | 81% | 0.982 |
| direction adjusted | log-odds | prompt + seed, constant sigma_L (point) | 7.7 | 0.0533 | 61% | 0.851 |
| direction adjusted | log-odds | prompt + seed, constant sigma_L (hi) | 2.5 | 0.473 | 0% | 0.000 |
| ARA raw | pp | prompt sampling only | 807.1 | 1.23e-174 | 100% | 0.179 |
| ARA raw | pp | prompt + seed, constant sigma_pp (lo) | 22.1 | 6.26e-05 | 86% | 0.230 |
| ARA raw | pp | prompt + seed, logit-level sigma (lo) | 10.9 | 0.0124 | 72% | 0.137 |
| ARA raw | pp | prompt + seed, constant sigma_pp (point) | 10.9 | 0.0121 | 73% | 0.212 |
| ARA raw | pp | prompt + seed, logit-level sigma (point) | 5.3 | 0.148 | 44% | 0.106 |
| ARA raw | pp | prompt + seed, constant sigma_pp (hi) | 3.6 | 0.31 | 16% | 0.100 |
| ARA raw | pp | prompt + seed, logit-level sigma (hi) | 1.7 | 0.627 | 0% | 0.000 |
| ARA raw | log-odds | prompt sampling only | 378.5 | 9.82e-82 | 99% | 1.693 |
| ARA raw | log-odds | prompt + seed, constant sigma_L (lo) | 27.3 | 5.21e-06 | 89% | 2.412 |
| ARA raw | log-odds | prompt + seed, constant sigma_L (point) | 14.3 | 0.00256 | 79% | 2.312 |
| ARA raw | log-odds | prompt + seed, constant sigma_L (hi) | 4.9 | 0.182 | 38% | 1.630 |
| ARA adjusted | pp | prompt sampling only | 381.8 | 1.94e-82 | 99% | 0.181 |
| ARA adjusted | pp | prompt + seed, constant sigma_pp (lo) | 15.8 | 0.00123 | 81% | 0.190 |
| ARA adjusted | pp | prompt + seed, logit-level sigma (lo) | 10.1 | 0.018 | 70% | 0.131 |
| ARA adjusted | pp | prompt + seed, constant sigma_pp (point) | 7.9 | 0.0481 | 62% | 0.167 |
| ARA adjusted | pp | prompt + seed, logit-level sigma (point) | 5.0 | 0.172 | 40% | 0.099 |
| ARA adjusted | pp | prompt + seed, constant sigma_pp (hi) | 2.6 | 0.457 | 0% | 0.000 |
| ARA adjusted | pp | prompt + seed, logit-level sigma (hi) | 1.6 | 0.65 | 0% | 0.000 |
| ARA adjusted | log-odds | prompt sampling only | 208.5 | 6.19e-45 | 99% | 1.397 |
| ARA adjusted | log-odds | prompt + seed, constant sigma_L (lo) | 20.5 | 0.000131 | 85% | 2.060 |
| ARA adjusted | log-odds | prompt + seed, constant sigma_L (point) | 10.8 | 0.0126 | 72% | 1.933 |
| ARA adjusted | log-odds | prompt + seed, constant sigma_L (hi) | 3.7 | 0.294 | 19% | 1.013 |
| H2 contrast (ARA - direction) | pp | prompt sampling only | 1006.7 | 6.42e-218 | 100% | 0.261 |
| H2 contrast (ARA - direction) | pp | prompt + seed, constant sigma_pp (lo) | 22.6 | 4.96e-05 | 87% | 0.270 |
| H2 contrast (ARA - direction) | pp | prompt + seed, logit-level sigma (lo) | 12.9 | 0.00496 | 77% | 0.224 |
| H2 contrast (ARA - direction) | pp | prompt + seed, constant sigma_pp (point) | 11.2 | 0.0109 | 73% | 0.248 |
| H2 contrast (ARA - direction) | pp | prompt + seed, logit-level sigma (point) | 6.3 | 0.0971 | 53% | 0.185 |
| H2 contrast (ARA - direction) | pp | prompt + seed, constant sigma_pp (hi) | 3.7 | 0.301 | 18% | 0.123 |
| H2 contrast (ARA - direction) | pp | prompt + seed, logit-level sigma (hi) | 2.1 | 0.56 | 0% | 0.000 |
| H2 contrast (ARA - direction) | log-odds | prompt sampling only | 539.7 | 1.19e-116 | 99% | 2.131 |
| H2 contrast (ARA - direction) | log-odds | prompt + seed, constant sigma_L (lo) | 21.0 | 0.000103 | 86% | 2.385 |
| H2 contrast (ARA - direction) | log-odds | prompt + seed, constant sigma_L (point) | 10.8 | 0.013 | 72% | 2.208 |
| H2 contrast (ARA - direction) | log-odds | prompt + seed, constant sigma_L (hi) | 3.6 | 0.307 | 17% | 1.077 |
