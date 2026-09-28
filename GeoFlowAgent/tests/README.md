<sub>[🏠 Portfolio](https://github.com/heoneyzi) › [🩺 Medical](../../README.md) › [GeoFlowAgent](../README.md) › **tests**</sub>

# 🧪 Tests

<details>
<summary><b>🇰🇷 한국어 요약</b></summary>

연구 코드 경로를 검사하는 34개 테스트 모듈입니다. 실행 결과는 295개 통과, 1개 선택 테스트 건너뜀입니다.

</details>


34 pytest modules (plus two `conftest.py`) covering the research code paths rather than only happy paths:

| Area | Examples |
|---|---|
| Data integrity | exact contracts, JSON schemas, preprocessing, data-quality gates, private-verifier isolation, serialization, MARRVEL import that never invents a plan |
| Search supervision | exact oracle (V\*, Q\*, regret), multi-path targets, search features, pipeline end to end |
| Models and training | distances, value geometry, flow, State Flow, DAgger, checkpoint chains and resumable training |
| Evaluation | agent episodes, closed-loop search agent, embedding metrics, cache immutability, Hugging Face adapter |
| GeoACMG (`geoacmg/`, `geoacmg_finish/`) | evidence scale, pipeline, experiments, analysis stages, batched-flow equivalence against the original evaluator |
| Reproducibility | run provenance binds the source tree, config and lockfile hashes |

```bash
pip install -e ".[dev,eval]"
PYTHONPATH=src pytest -q
```

Result: **295 passed, 1 skipped** — the skipped test runs when the optional generated search-smoke artifacts are present ([details](../verification/README.md)).
