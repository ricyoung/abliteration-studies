# Revision analyses REV-19 / REV-36 / REV-39

## (1) Conditional refusal retention, generation effect (newer - older), pp [bootstrap 95%]
- qwen ara: +36.9 [+33.4, +40.4]
- qwen direction class: +6.0 [+3.5, +8.3]
- qwen H2 contrast: +30.9 [+26.9, +34.9]
- gemma ara: -29.4 [-32.5, -26.4]
- gemma direction class: +4.0 [+1.5, +6.5]
- gemma H2 contrast: -33.3 [-36.8, -30.0]
- olmo ara: -3.2 [-5.8, -0.5]
- olmo direction class: +43.4 [+41.0, +45.7]
- olmo H2 contrast: -46.6 [-49.8, -43.3]
- mistral ara: -2.6 [-4.8, -0.5]
- mistral direction class: -6.6 [-9.0, -4.3]
- mistral H2 contrast: +4.0 [+1.2, +6.8]
- qwen-small ara: +15.0 [+12.0, +18.0]
- qwen-small direction class: +13.5 [+10.7, +15.9]
- qwen-small H2 contrast: +1.5 [-1.9, +5.1]

## (2) Keyword - WildGuard gap over the 42 primary ablated models
- kw100_vs_wg: mean +0.5 pp, mean |gap| 7.0 pp (median 5.6, max 25.0), n = 42
- kw100_vs_wg256: mean -0.1 pp, mean |gap| 7.0 pp (median 4.6, max 25.5), n = 42
- kw256_vs_wg256: mean +3.8 pp, mean |gap| 7.6 pp (median 5.7, max 24.1), n = 42
- kw_vs_wg: mean +8.4 pp, mean |gap| 10.5 pp (median 9.5, max 36.9), n = 42

## (3) OLMo generation effects, WildGuard vs Qwen3Guard (pp)
- wg original: +0.8
- wg dim: +34.8
- wg mpoa: +61.4
- wg som: +14.4
- wg ara: -2.5
- q3g original: +1.3
- q3g dim: +21.8
- q3g mpoa: +59.7
- q3g som: +15.3
- q3g ara: -2.8
- wg direction class: +36.9
- wg H2 contrast: -39.3
- q3g direction class: +32.3
- q3g H2 contrast: -35.0
