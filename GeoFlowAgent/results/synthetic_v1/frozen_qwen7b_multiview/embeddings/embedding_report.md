# GeoFlowAgent embedding audit

## 읽는 법

`valid_hit@1`은 여러 정답 행동 중 하나를 맞히면 성공입니다. `single_trace`는 보조 지표이며, exact mask가 만든 이득과 dense geometry가 만든 이득을 반드시 분리해 보세요. 기본 실행은 test를 통계에 집계하지 않으며 `--include-test`가 명시된 최종 실행만 test를 보고합니다.

## Reported retrieval

| split | view | distance | mask | valid_hit@1 | valid_mrr | regret@1 | regret_label_coverage |
| --- | --- | --- | --- | --- | --- | --- | --- |
| dev | medcpt | cosine | no_mask | 0.133333 | 0.343968 | 0.866667 | 1.000000 |
| test | medcpt | cosine | no_mask | 0.086957 | 0.352277 | 0.913043 | 1.000000 |
| dev | medcpt | cosine | exact_mask | 1.000000 | 1.000000 | 0.000000 | 1.000000 |
| test | medcpt | cosine | exact_mask | 1.000000 | 1.000000 | 0.000000 | 1.000000 |
| dev | medcpt | euclidean | no_mask | 0.133333 | 0.360635 | 0.866667 | 1.000000 |
| test | medcpt | euclidean | no_mask | 0.130435 | 0.350104 | 0.869565 | 1.000000 |
| dev | medcpt | euclidean | exact_mask | 1.000000 | 1.000000 | 0.000000 | 1.000000 |
| test | medcpt | euclidean | exact_mask | 1.000000 | 1.000000 | 0.000000 | 1.000000 |
| dev | medcpt | train_whitened_euclidean | no_mask | 0.133333 | 0.336190 | 0.866667 | 1.000000 |
| test | medcpt | train_whitened_euclidean | no_mask | 0.086957 | 0.339182 | 0.913043 | 1.000000 |
| dev | medcpt | train_whitened_euclidean | exact_mask | 1.000000 | 1.000000 | 0.000000 | 1.000000 |
| test | medcpt | train_whitened_euclidean | exact_mask | 1.000000 | 1.000000 | 0.000000 | 1.000000 |
| dev | qwen_general | cosine | no_mask | 0.133333 | 0.355635 | 0.866667 | 1.000000 |
| test | qwen_general | cosine | no_mask | 0.086957 | 0.351087 | 0.913043 | 1.000000 |
| dev | qwen_general | cosine | exact_mask | 1.000000 | 1.000000 | 0.000000 | 1.000000 |
| test | qwen_general | cosine | exact_mask | 1.000000 | 1.000000 | 0.000000 | 1.000000 |
| dev | qwen_general | euclidean | no_mask | 0.133333 | 0.357857 | 0.866667 | 1.000000 |
| test | qwen_general | euclidean | no_mask | 0.086957 | 0.354400 | 0.913043 | 1.000000 |
| dev | qwen_general | euclidean | exact_mask | 1.000000 | 1.000000 | 0.000000 | 1.000000 |
| test | qwen_general | euclidean | exact_mask | 1.000000 | 1.000000 | 0.000000 | 1.000000 |
| dev | qwen_general | train_whitened_euclidean | no_mask | 0.066667 | 0.304048 | 0.933333 | 1.000000 |
| test | qwen_general | train_whitened_euclidean | no_mask | 0.130435 | 0.365373 | 0.869565 | 1.000000 |
| dev | qwen_general | train_whitened_euclidean | exact_mask | 1.000000 | 1.000000 | 0.000000 | 1.000000 |
| test | qwen_general | train_whitened_euclidean | exact_mask | 1.000000 | 1.000000 | 0.000000 | 1.000000 |

## Full JSON

세부 anisotropy, effective rank, CKA, minimal-pair, probe 결과는 `embedding_report.json`에 있습니다.
