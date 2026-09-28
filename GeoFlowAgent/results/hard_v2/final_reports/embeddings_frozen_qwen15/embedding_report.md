# GeoFlowAgent embedding audit

## 읽는 법

`valid_hit@1`은 여러 정답 행동 중 하나를 맞히면 성공입니다. `single_trace`는 보조 지표이며, exact mask가 만든 이득과 dense geometry가 만든 이득을 반드시 분리해 보세요. 기본 실행은 test를 통계에 집계하지 않으며 `--include-test`가 명시된 최종 실행만 test를 보고합니다.

## Reported retrieval

| split | view | distance | mask | valid_hit@1 | valid_mrr | regret@1 | regret_label_coverage |
| --- | --- | --- | --- | --- | --- | --- | --- |
| dev | medcpt | cosine | no_mask | 0.027310 | 0.132683 | 0.972123 | 0.982644 |
| test | medcpt | cosine | no_mask | 0.037465 | 0.133772 | 0.961141 | 0.994375 |
| dev | medcpt | cosine | exact_mask | 0.306219 | 0.526516 | 0.413392 | 0.632357 |
| test | medcpt | cosine | exact_mask | 0.284916 | 0.505871 | 0.375947 | 0.562925 |
| dev | medcpt | euclidean | no_mask | 0.023737 | 0.126307 | 0.976043 | 0.990812 |
| test | medcpt | euclidean | no_mask | 0.021294 | 0.120850 | 0.978625 | 0.996183 |
| dev | medcpt | euclidean | exact_mask | 0.301722 | 0.523890 | 0.414793 | 0.631072 |
| test | medcpt | euclidean | exact_mask | 0.288370 | 0.508542 | 0.375005 | 0.561402 |
| dev | medcpt | train_whitened_euclidean | no_mask | 0.024502 | 0.118886 | 0.975422 | 0.996937 |
| test | medcpt | train_whitened_euclidean | no_mask | 0.020088 | 0.110185 | 0.979879 | 0.998393 |
| dev | medcpt | train_whitened_euclidean | exact_mask | 0.292984 | 0.511337 | 0.652684 | 0.586482 |
| test | medcpt | train_whitened_euclidean | exact_mask | 0.345962 | 0.535122 | 0.515236 | 0.521991 |
| dev | qwen_general | cosine | no_mask | 0.022205 | 0.174447 | 0.977701 | 0.995789 |
| test | qwen_general | cosine | no_mask | 0.020289 | 0.158920 | 0.983920 | 0.999397 |
| dev | qwen_general | cosine | exact_mask | 0.426369 | 0.597489 | 0.505616 | 0.576587 |
| test | qwen_general | cosine | exact_mask | 0.361402 | 0.543048 | 0.534470 | 0.499035 |
| dev | qwen_general | euclidean | no_mask | 0.019270 | 0.171608 | 0.980646 | 0.995661 |
| test | qwen_general | euclidean | no_mask | 0.020791 | 0.157718 | 0.980402 | 0.999397 |
| dev | qwen_general | euclidean | exact_mask | 0.428939 | 0.598944 | 0.496669 | 0.576715 |
| test | qwen_general | euclidean | exact_mask | 0.363738 | 0.544708 | 0.524471 | 0.498933 |
| dev | qwen_general | train_whitened_euclidean | no_mask | 0.026034 | 0.174305 | 0.973506 | 0.982644 |
| test | qwen_general | train_whitened_euclidean | no_mask | 0.023905 | 0.160519 | 0.975352 | 0.969867 |
| dev | qwen_general | train_whitened_euclidean | exact_mask | 0.379337 | 0.577353 | 0.588573 | 0.537522 |
| test | qwen_general | train_whitened_euclidean | exact_mask | 0.356526 | 0.544708 | 0.557337 | 0.494261 |

## Full JSON

세부 anisotropy, effective rank, CKA, minimal-pair, probe 결과는 `embedding_report.json`에 있습니다.
