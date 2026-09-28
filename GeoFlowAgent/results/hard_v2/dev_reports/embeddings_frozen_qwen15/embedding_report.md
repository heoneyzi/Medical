# GeoFlowAgent embedding audit

## 읽는 법

`valid_hit@1`은 여러 정답 행동 중 하나를 맞히면 성공입니다. `single_trace`는 보조 지표이며, exact mask가 만든 이득과 dense geometry가 만든 이득을 반드시 분리해 보세요. 기본 실행은 test를 통계에 집계하지 않으며 `--include-test`가 명시된 최종 실행만 test를 보고합니다.

## Reported retrieval

| split | view | distance | mask | valid_hit@1 | valid_mrr | regret@1 | regret_label_coverage |
| --- | --- | --- | --- | --- | --- | --- | --- |
| dev | medcpt | cosine | no_mask | 0.027310 | 0.132683 | 0.972123 | 0.982644 |
| dev | medcpt | cosine | exact_mask | 0.306219 | 0.526516 | 0.413392 | 0.632357 |
| dev | medcpt | euclidean | no_mask | 0.023737 | 0.126307 | 0.976043 | 0.990812 |
| dev | medcpt | euclidean | exact_mask | 0.301722 | 0.523890 | 0.414793 | 0.631072 |
| dev | medcpt | train_whitened_euclidean | no_mask | 0.024502 | 0.118887 | 0.975422 | 0.996937 |
| dev | medcpt | train_whitened_euclidean | exact_mask | 0.292984 | 0.511337 | 0.652684 | 0.586482 |
| dev | qwen_general | cosine | no_mask | 0.020929 | 0.174079 | 0.978982 | 0.995789 |
| dev | qwen_general | cosine | exact_mask | 0.426369 | 0.597522 | 0.505616 | 0.576587 |
| dev | qwen_general | euclidean | no_mask | 0.018377 | 0.171430 | 0.981543 | 0.995661 |
| dev | qwen_general | euclidean | exact_mask | 0.428939 | 0.598896 | 0.496669 | 0.576715 |
| dev | qwen_general | train_whitened_euclidean | no_mask | 0.026034 | 0.174148 | 0.973506 | 0.982644 |
| dev | qwen_general | train_whitened_euclidean | exact_mask | 0.374968 | 0.574226 | 0.604829 | 0.537522 |

## Full JSON

세부 anisotropy, effective rank, CKA, minimal-pair, probe 결과는 `embedding_report.json`에 있습니다.
