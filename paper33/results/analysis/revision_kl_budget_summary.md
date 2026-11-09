# Revision analysis REV-16 / REV-32 (KL budgets, dev set)

Generated 2026-11-09 00:26 UTC by scripts/analysis/revision_kl_budgets.py.

Dev keyword refusals (of 100) of the best trial at KL <= t; AUC = mean front value over log KL in [0.01, 1].

## Run-to-run sigma (small pair, 10 seed pairs)
- t0.05: sigma_dev 6.8 (median |d| 7.0, max 17.0)
- t0.1: sigma_dev 10.8 (median |d| 8.0, max 35.0)
- t0.2: sigma_dev 5.2 (median |d| 6.0, max 12.0)
- t0.5: sigma_dev 4.6 (median |d| 4.0, max 12.0)
- t1: sigma_dev 4.6 (median |d| 4.0, max 12.0)
- auc: sigma_dev 4.3 (median |d| 4.2, max 11.8)

Runs with the front carried forward over > 50% of the AUC grid: 7 of 42 seed-1 runs

## Class-level effects (newer - older), refusals of 100; * = beyond 1.96 x run-to-run SE

| pair | estimand | t0.05 | t0.1 | t0.2 | t0.5 | t1 | auc | held-out pp |
|---|---|---|---|---|---|---|---|---|
| qwen | direction class | -38.3* | -25.3* | -11.3* | -3.3 | -3.3 | -7.2* | +10.7 |
| qwen | ara | +86.0* | +53.0* | +13.0 | +14.0* | +1.0 | +37.1* | +34.4 |
| qwen | H2 contrast | +124.3* | +78.3* | +24.3* | +17.3* | +4.3 | +44.3* | +23.6 |
| gemma | direction class | +0.3 | +4.7 | +22.7* | +30.0* | +30.0* | +13.0* | +4.7 |
| gemma | ara | +23.0* | +12.0 | +4.0 | +1.0 | +1.0 | +5.1 | -26.0 |
| gemma | H2 contrast | +22.7* | +7.3 | -18.7* | -29.0* | -29.0* | -7.9 | -30.8 |
| olmo | direction class | +42.3* | +40.0* | +34.0* | +31.7* | +31.7* | +27.9* | +36.9 |
| olmo | ara | +55.0* | +14.0 | +24.0* | +31.0* | +4.0 | +21.9* | -2.5 |
| olmo | H2 contrast | +12.7 | -26.0 | -10.0 | -0.7 | -27.7* | -6.0 | -39.3 |
| mistral | direction class | +21.0* | -6.0 | -8.3 | -6.3 | -6.3 | +6.5 | -1.2 |
| mistral | ara | +17.0 | +2.0 | +2.0 | +2.0 | +2.0 | +23.6* | -0.7 |
| mistral | H2 contrast | -4.0 | +8.0 | +10.3 | +8.3 | +8.3 | +17.1* | +0.5 |
| qwen-small | direction class | -27.0* | -22.8* | -5.0 | -5.0 | -5.0 | -11.6* | +17.3 |
| qwen-small | ara | +6.0 | -2.0 | +3.0 | +1.0 | +1.0 | +4.0 | +14.2 |
| qwen-small | H2 contrast | +33.0* | +20.8 | +8.0 | +6.0 | +6.0 | +15.5* | -3.1 |

- direction class t0.05: sign as at 0.1 in 3 of 4 main pairs; sign as held-out in 2 of 4; beyond noise 3 of 4
- direction class t0.1: sign as at 0.1 in 4 of 4 main pairs; sign as held-out in 3 of 4; beyond noise 2 of 4
- direction class t0.2: sign as at 0.1 in 4 of 4 main pairs; sign as held-out in 3 of 4; beyond noise 3 of 4
- direction class t0.5: sign as at 0.1 in 4 of 4 main pairs; sign as held-out in 3 of 4; beyond noise 2 of 4
- direction class t1: sign as at 0.1 in 4 of 4 main pairs; sign as held-out in 3 of 4; beyond noise 2 of 4
- direction class auc: sign as at 0.1 in 3 of 4 main pairs; sign as held-out in 2 of 4; beyond noise 3 of 4
- ara t0.05: sign as at 0.1 in 4 of 4 main pairs; sign as held-out in 1 of 4; beyond noise 3 of 4
- ara t0.1: sign as at 0.1 in 4 of 4 main pairs; sign as held-out in 1 of 4; beyond noise 1 of 4
- ara t0.2: sign as at 0.1 in 4 of 4 main pairs; sign as held-out in 1 of 4; beyond noise 1 of 4
- ara t0.5: sign as at 0.1 in 4 of 4 main pairs; sign as held-out in 1 of 4; beyond noise 2 of 4
- ara t1: sign as at 0.1 in 4 of 4 main pairs; sign as held-out in 1 of 4; beyond noise 0 of 4
- ara auc: sign as at 0.1 in 4 of 4 main pairs; sign as held-out in 1 of 4; beyond noise 3 of 4
- H2 contrast t0.05: sign as at 0.1 in 2 of 4 main pairs; sign as held-out in 1 of 4; beyond noise 2 of 4
- H2 contrast t0.1: sign as at 0.1 in 4 of 4 main pairs; sign as held-out in 3 of 4; beyond noise 1 of 4
- H2 contrast t0.2: sign as at 0.1 in 3 of 4 main pairs; sign as held-out in 4 of 4; beyond noise 2 of 4
- H2 contrast t0.5: sign as at 0.1 in 3 of 4 main pairs; sign as held-out in 4 of 4; beyond noise 2 of 4
- H2 contrast t1: sign as at 0.1 in 3 of 4 main pairs; sign as held-out in 4 of 4; beyond noise 2 of 4
- H2 contrast auc: sign as at 0.1 in 3 of 4 main pairs; sign as held-out in 4 of 4; beyond noise 2 of 4
