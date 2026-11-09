# Re-review revision analyses (revision_rr_extra.py)

## (1) Conditional retention: run-to-run SD sigma_ret = 10.96 pp [7.66, 19.24] (10 small-pair seed pairs)
- qwen ara: +36.9; prompt [+33.4, +40.4]; prompt + seed [+6.3, +67.5] *
- qwen direction class: +6.0; prompt [+3.5, +8.3]; prompt + seed [-11.7, +23.7]
- qwen H2 contrast: +30.9; prompt [+26.9, +34.9]; prompt + seed [-4.5, +66.2]
- gemma ara: -29.4; prompt [-32.5, -26.4]; prompt + seed [-59.9, +1.2]
- gemma direction class: +4.0; prompt [+1.5, +6.5]; prompt + seed [-13.8, +21.7]
- gemma H2 contrast: -33.3; prompt [-36.8, -30.0]; prompt + seed [-68.6, +1.9]
- olmo ara: -3.2; prompt [-5.8, -0.5]; prompt + seed [-33.7, +27.3]
- olmo direction class: +43.4; prompt [+41.0, +45.7]; prompt + seed [+25.7, +61.1] *
- olmo H2 contrast: -46.6; prompt [-49.8, -43.3]; prompt + seed [-81.8, -11.3] *
- mistral ara: -2.6; prompt [-4.8, -0.5]; prompt + seed [-33.0, +27.9]
- mistral direction class: -6.6; prompt [-9.0, -4.3]; prompt + seed [-24.3, +11.1]
- mistral H2 contrast: +4.0; prompt [+1.2, +6.8]; prompt + seed [-31.2, +39.2]
- qwen-small ara: +15.0; prompt [+12.0, +18.0]; prompt + seed [-15.6, +45.5]
- qwen-small direction class: +13.5; prompt [+10.7, +15.9]; prompt + seed [-1.9, +28.9]
- qwen-small H2 contrast: +1.5; prompt [-1.9, +5.1]; prompt + seed [-32.7, +35.6]

## (2) Wall-clock time per completed 100-trial main-study search (H200 box; meta.txt wall_seconds)
- ARA: n = 8, median 81.0 min, range 50.3-95.6 min, total 10.1 h
- direction: n = 23, median 18.1 min, range 10.2-61.5 min, total 8.0 h
- all 31 included searches: total 18.1 h; excluded: qwen3.8-27b__mpoa (resumed after an out-of-memory failure; search split over three processes)

## (3) tau^2: direction raw pp const. point 225 pp^2 (tau 15.0 pp)
- direction raw / pp / prompt + seed, constant sigma_pp (point): tau2 225, I2 0.798
- direction raw / pp / prompt + seed, logit-level sigma (point): tau2 127, I2 0.458
- direction raw / log-odds / prompt + seed, constant sigma_L (point): tau2 0.116, I2 0.201
- direction adjusted / pp / prompt + seed, constant sigma_pp (point): tau2 476, I2 0.892
- direction adjusted / pp / prompt + seed, logit-level sigma (point): tau2 425, I2 0.738
- direction adjusted / log-odds / prompt + seed, constant sigma_L (point): tau2 0.724, I2 0.609
- ARA raw / pp / prompt + seed, constant sigma_pp (point): tau2 447, I2 0.726
- ARA raw / pp / prompt + seed, logit-level sigma (point): tau2 113, I2 0.439
- ARA raw / log-odds / prompt + seed, constant sigma_L (point): tau2 5.34, I2 0.790
- ARA adjusted / pp / prompt + seed, constant sigma_pp (point): tau2 278, I2 0.620
- ARA adjusted / pp / prompt + seed, logit-level sigma (point): tau2 97.5, I2 0.400
- ARA adjusted / log-odds / prompt + seed, constant sigma_L (point): tau2 3.74, I2 0.723
- H2 contrast (ARA - direction) / pp / prompt + seed, constant sigma_pp (point): tau2 614, I2 0.731
- H2 contrast (ARA - direction) / pp / prompt + seed, logit-level sigma (point): tau2 343, I2 0.525
- H2 contrast (ARA - direction) / log-odds / prompt + seed, constant sigma_L (point): tau2 4.87, I2 0.721
