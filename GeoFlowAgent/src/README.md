<sub>[🏠 Portfolio](https://github.com/heoneyzi) › [🩺 Medical](../../README.md) › [GeoFlowAgent](../README.md) › **src**</sub>

# 💻 `src/geoflowagent` — code map

<details>
<summary><b>🇰🇷 한국어 요약</b></summary>

GeoFlowAgent와 GeoACMG의 전체 라이브러리(83개 모듈)입니다. 데이터·계약, 정확 탐색, 동결 임베딩, 기하/State Flow 모델, 학습·평가, ClinGen 기반 GeoACMG 분석으로 나뉩니다. 신뢰구간이 0을 포함하면 UNRESOLVED로 판정하는 등 연구의 증거 규칙이 코드로 강제됩니다.

</details>


The library (83 modules) behind every stage: the GeoFlowAgent core plus the GeoACMG extension and its "finish"/"flowfast" analyses. Console entry point: `geoflow = geoflowagent.cli:main`; GeoACMG: `python -m geoflowagent.geoacmg`.

| Package | Responsibility | Used by stage |
|---|---|---|
| [`data/`](geoflowagent/data/) | Typed schemas, exact executable **contracts** (assembly/version/schema preconditions), synthetic and procedural task generators (`procedural_search.py`, `procedural_search_hard.py`), search supervision compiler, fail-closed data-quality audit, structured-state hasher, MARRVEL QA importer | ①–⑤ |
| [`search/`](geoflowagent/search/) | Weighted exact search over executable states → V\*, Q\*, regret and optimal-action sets | ②–⑤ |
| [`embeddings/`](geoflowagent/embeddings/) | Frozen Hugging Face encoder adapters (pinned revisions), field-wise immutable caches with checksums, hash-embedding control | ①–⑤ |
| [`models/`](geoflowagent/models/) | Distance families, whole-plan flow, **goal-conditioned value geometry** (cosine, Euclidean, Poincaré, directed quasimetric, pair MLP, …), conditional rectified **State Flow** | ①–⑤ |
| [`training/`](geoflowagent/training/) | Metric and flow training (stage ①), value-geometry training/comparison, Search-DAgger, State Flow training/evaluation with resumable, hash-checked checkpoints | ①–⑤ |
| [`evaluation/`](geoflowagent/evaluation/) | Snapshot environment, closed-loop agents (clean/perturbed), embedding diagnostics (anisotropy, effective rank, hubness, probes) | ①–③ |
| [`geoacmg/`](geoflowagent/geoacmg/) | ClinGen parsing, ACMG evidence/point scale, tool map, tasks, splits, minimal pairs, benchmark diagnostic, probes (RQ1), plan ladder (RQ2), estimators, **claims/verdicts**, **cited constants**, sealing, the R1–R7 stages, batched flow evaluation | ④–⑤ |
| [`utils/`](geoflowagent/utils/) | I/O, reporting, run provenance (source-tree, config and lockfile hashes) | all |

**Evidence rules in code.** [`geoacmg/claims.py`](geoflowagent/geoacmg/claims.py) turns findings into verdicts — an interval that straddles zero yields `UNRESOLVED`, never a "trend", and findings outside the pre-registered primary set are reported as exploratory. [`geoacmg/cited.py`](geoflowagent/geoacmg/cited.py) makes every decision constant carry its source, so ad-hoc thresholds are rejected. [`geoacmg/seal.py`](geoflowagent/geoacmg/seal.py) guards the test split; the hard-v2 CLI writes each test access to a hash-chained ledger.

Some later GeoACMG modules (`adapters`, `amend`, `arms`, `budget`, `control`, `durable`, `finish`, `flowfast*`, `paired`, `parallel`, `readout`, `seal`, `stages`, `strata`, `winnability`) have Korean docstrings; [docs/original/geoacmg/03_CODE_MAP.md](../docs/original/geoacmg/03_CODE_MAP.md) describes the layering.

```bash
pip install -e ".[dev,eval]"      # add the "hf" extra to re-embed with the pinned encoders
PYTHONPATH=src pytest             # 295 passed, 1 skipped
```
