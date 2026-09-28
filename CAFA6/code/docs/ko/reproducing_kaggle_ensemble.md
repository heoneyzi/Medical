# GOA + ProtT5 앙상블: 로컬 스크립트와 Kaggle 노트북

원본 Kaggle 노트북 `cafa-6-goa-prott5-ensemble-0-370.ipynb`은 모델을 학습하지 않습니다. 대신 다음을 수행합니다.

1. **사전 계산된** 두 개의 예측 파일을 로드합니다.
   - `goa_submission.tsv` (GOA 기반 예측)
   - `prott5_interpro_predictions.tsv` (ProtT5 + InterPro 예측)
2. 고정 가중치로 두 결과를 앙상블합니다.
3. 온톨로지 인지(post-processing) 후처리(“GOA+ 전파”)를 적용합니다.
4. `submission.tsv`를 작성합니다.

## 여기서 재현 가능한 것

이 저장소에는 위 로직을 로컬에서 재현한 구현이 포함되어 있습니다.

- 노트북: `notebooks/cafa-6-goa-prott5-ensemble-0-370.ipynb`
- 스크립트: `scripts/make_ensemble_submission.py`

**로컬 스크립트**의 기본 입력 경로는 다음과 같습니다.

- 대회 파일: `data/Train/go-basic.obo`, `data/Test/testsuperset.fasta`
- 예측 파일: `data/kaggle/goa_submission.tsv`, `data/kaggle/prott5_interpro_predictions.tsv`

노트북은 원본 Kaggle 경로인 `/kaggle/input/cafa-6-protein-function-prediction`과 `/kaggle/input/cafa6-goa-predictions`를 그대로 사용합니다. Kaggle 밖에서는 경로 셀을 수정해야 합니다. 로컬 스크립트는 `data/Test/testsuperset.fasta`에 존재하는 단백질 ID만 출력하지만, 원본 노트북에는 이 필터가 없습니다.

ID 필터링은 출력 대상만 제한합니다. 외부 GOA·모델 예측의 공개 시점, 라벨 출처, 대회 시간 기준 준수 여부를 검증하지는 않습니다. 해당 정보는 별도로 확인해야 합니다. 원본 파일명의 `0-370`은 이번 문서 정리에서 재현한 점수가 아닙니다.

## 추가 리소스 없이는 재현할 수 *없는* 것

다음의 상위 단계(업스트림) 생성 과정은 포함되어 있지 않습니다.

- GOA 예측 생성(외부 GOA/UniProt 리소스 및 특정 파이프라인 필요)
- ProtT5 + InterPro 예측 생성(모델 가중치/특징 파이프라인 필요, GPU 및 외부 리소스가 필요할 수 있음)

이들은 Kaggle 노트북에 포함된 것과는 별개의 학습/추론 전략입니다.

## 실행(스크립트)

[README](../../README.md)에 따라 Python 3.12+ 환경을 준비하고, 대회 데이터와 두 외부 예측 TSV를 확보한 후 프로젝트 인터프리터를 사용합니다.

`./.venv/bin/python scripts/make_ensemble_submission.py --out submissions/submission.tsv`

입력 루트는 `--competition-data`, `--prediction-data`로 변경할 수 있습니다. 예측 TSV와 그 생성에 사용된 가중치·외부 리소스는 저장소에 포함되지 않습니다. 이번 작업에서는 전체 앙상블 실행이나 리더보드 평가를 다시 수행하지 않았습니다.

주요 튜닝 파라미터(Kaggle 기본값과 일치):

- `--weight-goa 0.55`
- `--weight-prott5 0.45`
- `--top-k 200`
- `--min-score 0.001`
- `--neg-prop-alpha 0.7`
- `--scaling-power 0.8`
- `--max-score 0.95`
