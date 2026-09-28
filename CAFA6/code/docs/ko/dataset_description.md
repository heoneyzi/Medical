# 데이터셋 설명

이 데이터셋은 단백질에 대한 Gene Ontology (GO) 주석과, 기능 예측 모델을 학습/평가하는 데 필요한 지원 리소스를 제공합니다. 큐레이션된 단백질 서열, 실험적으로 지지되는 GO 용어 라벨, 분류학(taxonomy) 메타데이터, GO 그래프, 정보 누적(Information Accretion) 가중치, 그리고 추론을 위한 테스트 슈퍼셋을 포함합니다.

과제의 전체 정의(무엇을 예측하는지, 평가 방식, 제출 규칙)는 `docs/ko/task_specification.md`를 참고하세요.

## 배경

Gene Ontology (GO)는 기능 기술자(용어 또는 클래스)로 이루어진 방향 비순환 그래프(DAG)입니다. 용어들은 `is_a`, `part_of`와 같은 관계로 연결됩니다. 각 용어는 다음 루트 노드로 정의되는 세 가지 서브온톨로지 중 하나에 속합니다.

- 분자 기능 (MFO): `GO:0003674`
- 생물학적 과정 (BPO): `GO:0008150`
- 세포 구성요소 (CCO): `GO:0005575`

하나의 단백질은 서브온톨로지 내부 및 서로 다른 서브온톨로지에 걸쳐 여러 용어로 주석될 수 있습니다. 이 데이터셋에서 용어-단백질 할당은 실험적으로 지지되는 증거(및 관련된 고신뢰 증거)에서 유도되며, 정답(ground truth) 라벨로 취급됩니다. 주석의 부재는 기능의 부재를 의미하지 않습니다.

## 학습 셋

학습 셋은 다음으로 지지되는 주석을 가진 단백질을 포함합니다.

- 실험 또는 고처리량(high-throughput) 증거
- 추적 가능한 저자 진술(TAS)
- 큐레이터 추론(IC)

학습 서열은 UniProtKB (Swiss-Prot) 릴리스 2025_03(2025년 6월 18일)에서 가져왔습니다. 이 셋은 진핵생물과 소수의 비진핵생물(박테리아 13종, 고세균 1종)을 포함합니다. 라벨이 있는 단백질만 학습 서열 파일에 포함됩니다.

## 테스트 슈퍼셋과 테스트 셋

테스트 슈퍼셋에는 예측을 생성해야 하는 단백질 서열이 포함됩니다. 최종 테스트 셋은 제출 마감과 평가 사이에 실험 주석을 획득하는 슈퍼셋의 숨겨진 부분집합입니다. 새로운 실험 주석을 얻은 단백질만 채점에 사용됩니다.

## 파일

모든 파일은 저장소의 `data/` 아래에 있습니다.

### 온톨로지

- `data/Train/go-basic.obo`
  - GO DAG 구조(OBO 포맷), 릴리스 2025-06-01.
  - OBO를 파싱할 수 있는 라이브러리(예: Python의 obonet)로 읽을 수 있습니다.

### 학습 데이터

- `data/Train/train_sequences.fasta`
  - 학습 셋의 단백질 서열(FASTA 포맷).
  - 헤더에는 UniProt accession 및 식별자가 들어 있습니다. 예:
    - `sp|P9WHI7|RECN_MYCT`는 Swiss-Prot 엔트리 P9WHI7을 의미합니다.

- `data/Train/train_terms.tsv`
  - 학습 단백질의 정답 GO 용어 라벨.
  - 컬럼:
    1. UniProt accession ID
    2. GO 용어 ID
    3. 온톨로지 네임스페이스(`BPO`, `CCO`, `MFO`)

- `data/Train/train_taxonomy.tsv`
  - 학습 단백질에 대한 종(species) 메타데이터.
  - 컬럼:
    1. UniProt accession ID
    2. NCBI taxon ID

### 테스트 데이터

- `data/Test/testsuperset.fasta`
  - 예측 대상 단백질 서열(FASTA 포맷).
  - 헤더에 UniProt accession 및 taxon ID가 포함됩니다.

- `data/Test/testsuperset-taxon-list.tsv`
  - 테스트 슈퍼셋에 포함된 taxon ID 목록.

### 평가 가중치

- `data/IA.tsv`
  - 각 GO 용어에 대한 정보 누적 가중치.
  - 평가 시 가중 정밀도 및 재현율 계산에 사용됩니다.

### 제출 포맷

- `data/sample_submission.tsv`
  - 예측 제출 포맷 예시.

## 참고 및 규약

- 단백질은 서브온톨로지 내부 및 서로 다른 서브온톨로지에 걸쳐 여러 GO 용어를 가질 수 있습니다.
- 어떤 용어가 없다고 해서 해당 단백질에 그 기능이 없다는 뜻은 아닙니다(주석이 없을 수 있음).
- 모든 학습 서열은 Swiss-Prot 엔트리입니다.
- 테스트 셋은 시간이 지나며 새로운 실험 주석이 추가되는 단백질에 따라 테스트 슈퍼셋에서 파생됩니다.

