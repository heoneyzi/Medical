# 재현 방법

[← 주장과 근거](04_CLAIMS_AND_EVIDENCE.md) · [연구 안내로 ↑](../README.md)

## 1. 재현 수준

| 수준 | 필요한 것 | 확인할 수 있는 것 |
|---|---|---|
| 결과 읽기 | 이 저장소의 문서와 `results/` | 어떤 결론을 어떤 근거로 내렸는가 |
| 수치 재계산 | 이 저장소 + Python · NumPy | 평균, 과제별 짝 차이, 시드 · 유전자 기준 신뢰구간 (`verification/verify_*.py`, 몇 초) |
| 저장 모델 재평가 | + checkpoint · 임베딩 캐시 · 전처리된 탐색 그래프 (로컬 연구 아카이브) | 같은 입력에 대한 저장 모델의 출력 |
| 전체 재학습 | + ClinGen Evidence Repository 원자료, 고정 revision의 Hugging Face 인코더, GPU | 임베딩 생성과 학습을 포함한 전체 절차 |

checkpoint의 위치 · 크기 · SHA-256은 [`provenance/checkpoint_catalog.csv`](../../provenance/checkpoint_catalog.csv)에 있습니다.

## 2. 환경

Python **3.10 이상 3.13 미만**을 씁니다. 의존성 버전은 [`uv.lock`](../../uv.lock)에 고정되어 있습니다.

| 구분 | 패키지 |
|---|---|
| 기본 | jsonschema, numpy, pyyaml, torch 2.x, tqdm |
| `dev` | pytest, pytest-cov, ruff |
| `eval` | matplotlib, scikit-learn, scipy |
| `hf` | accelerate, datasets, einops, safetensors, sentencepiece, transformers 4.x |

```bash
uv sync --frozen --extra dev --extra eval      # 인코더로 새로 임베딩할 때는 --extra hf 추가
uv run pytest                                  # 295 passed, 1 skipped
uv run ruff check src tests
uv run geoflow run-all --config configs/smoke.yaml --evaluation-split test          # ① 경로 CPU smoke
uv run geoflow run-search --config configs/search_smoke.yaml --evaluation-split dev # ②–③ 경로 CPU smoke
```

`uv` 대신 `pip install -e ".[dev,eval]"`와 `PYTHONPATH=src pytest`를 써도 됩니다. Windows 기본 인코딩(CP949)에서는 `python -X utf8 -m pytest`로 실행하면 UTF-8 보고서를 읽는 테스트까지 모두 통과합니다. 건너뛰는 1개 테스트는 생성된 smoke 산출물이 있을 때 실행되는 선택 테스트입니다.

## 3. 코드 지도

| 위치 | 책임 | 연결되는 연구 질문 |
|---|---|---|
| `src/geoflowagent/data` | 구조화 입력, 도구 계약, 합성 · 절차적 과제 생성, 탐색 감독, 데이터 품질 검사 | 누수 없는 데이터와 실행 가능성 |
| `src/geoflowagent/search` | 상태 그래프와 정확 탐색 oracle | V\*, Q\*, regret, 최적 행동 집합 |
| `src/geoflowagent/embeddings` | 동결 인코더, 다중 view, 캐시와 실행 기록 | 백본을 학습하지 않고 얻는 표현 |
| `src/geoflowagent/models` | 거리 · 에너지, value geometry, State Flow | 기하 선택과 계획 생성 |
| `src/geoflowagent/training` | value · State Flow · DAgger 학습, 체크포인트 저장과 재개 | 학습 감독과 체크포인트 계보 |
| `src/geoflowagent/evaluation` | 실행 환경, 폐루프 에이전트, 임베딩 지표 | 오프라인 지표와 실행 성능의 차이 |
| `src/geoflowagent/geoacmg` | ClinGen 파싱, ACMG 과제, probe, 짝 시드 · readout 분석, R1–R7, 배치 flow 평가 | 실제 전문가 기록으로의 확장 |
| `configs/` | 모델 revision, 시드, 분할 · 학습 설정 | 실험 조건 고정 |
| `tests/` | 계약 · oracle · 누수 · 캐시 · 체크포인트 · 배치 평가 검사 | 구현 검증 |
| `scripts/`, `scripts_run/` | 실험 실행 · 집계 스크립트 (①–③, ④–⑤) | 당시의 실행 순서와 명령 |

기본 명령은 `geoflow = geoflowagent.cli:main`이고, GeoACMG는 `python -m geoflowagent.geoacmg`로 실행합니다.

## 4. 고정한 모델 버전

| 모델 | revision |
|---|---|
| Qwen/Qwen2.5-1.5B-Instruct | `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` |
| Qwen/Qwen2.5-7B-Instruct | `a09a35458c702b33eeacc393d103063234e8bc28` |
| ncbi/MedCPT-Query-Encoder | `d83a36cc6b8e3a5c5e9d9d6ba156808c1643dcbc` |
| ncbi/MedCPT-Article-Encoder | `d05a736da4bb84ee4057b7f7999485be6ed85465` |
| cambridgeltl/SapBERT-from-PubMedBERT-fulltext | `090663c3ae57bf35ffe4d0d468a2a88d03051a4d` |
| zhihan1996/DNABERT-2-117M | `7bce263b15377fc15361f52cfab88f8b586abda0` |

`geoacmg.yaml`의 기본 view는 MedCPT입니다. 초기 다중 view 실험은 해당 단계의 config를 사용합니다([configs/README.md](../../configs/README.md)).

## 5. 경로 자리표시자

GPU 서버에서 쓰던 절대 경로는 다음 자리표시자로 바꿔 두었습니다. 실행 전에 자신의 경로로 지정합니다.

| 자리표시자 | 뜻 |
|---|---|
| `$GEOFLOW_RUNS` | 원자료 · 전처리 · 캐시 · 체크포인트 · 보고서가 있는 실행 루트 |
| `$GEOFLOW_RESULTS` | 결과 · 진행 파일 · test 접근 기록을 모아 두는 결과 폴더 |
| `$GEOFLOW_PROJECT_ROOT`, `$GEOACMG_WORK` | GPU 서버의 저장소 위치 (①–③, ④–⑤) |
| `$GEOWORK` | 작업 · 로그 폴더 |
| `$HF_CACHE_DIR`, `$GEOFLOW_RUNTIME` | Hugging Face 캐시, Python 실행 환경 |

예: `GEOFLOW_RUNS=/data/geoflow envsubst < configs/search_hard_v2_frozen.yaml > local.yaml`

## 6. 다시 실행할 때 지킬 점

1. **학습 재개는 지문을 확인합니다.** value · State Flow 재개 단계는 `source_tree_sha256`, `cache_content_sha256`, `training_config_sha256` 등을 비교합니다. 정확히 이어서 학습하려면 그 run을 만든 시점의 소스를 씁니다.
2. **config의 상대 경로는 config 파일 위치가 기준입니다.** `project_root: ..`는 `configs/`에서 저장소 루트를 가리킵니다.
3. **평가 프로토콜을 구분합니다.** 루트 전용 평가(`--root-only-evaluation`), 배치 flow 평가, verifier 보조 · 학습된 STOP · 폐루프 결과는 서로 다른 지표입니다. 결과를 인용할 때 프로토콜을 함께 적습니다.
4. **test 분할은 사전등록한 최종 평가에만 씁니다.** hard-v2의 test 접근은 hash-chain 기록([`test_access_ledger.jsonl`](../../results/hard_v2/final_reports/test_access_ledger.jsonl))에 남습니다.
5. **pickle 계열 파일은 출처를 확인한 것만 씁니다.** 일부 로더는 `torch.load(..., weights_only=False)`와 NPZ `allow_pickle=True`를 사용하므로, SHA-256이 확인된 파일만 불러옵니다.
6. **검증 스크립트는 한 폴더에 둡니다.** `verify_geoacmg_latest.py`와 `verify_portfolio.py`는 옆의 `verify_geoacmg_metrics.py`를 가져다 씁니다.

## 7. 체크포인트 파일

- `value_geometry.pt`, `search_state_flow.pt` — 평가에 쓰는 학습된 모델
- `training_resume.pt` — 학습 재개 상태 (후기 flow 실행기는 여기서 `best_state`를 읽습니다)

[`checkpoint_catalog.csv`](../../provenance/checkpoint_catalog.csv)의 723개 항목은 백업 사본과 재개 상태를 포함한 파일 항목 수이며, 서로 다른 모델의 수와는 다릅니다.
