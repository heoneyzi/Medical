---
title: "Lab-in-the-Loop for Drug R&D with AI — 슬라이드별 해설"
language: ko
source: "컨퍼런스 필기 PDF와 공개 논문·기관 자료"
date_created: "2026-07-28"
---

# Lab-in-the-Loop for Drug R&D with AI  
## 컨퍼런스 슬라이드별 내용·의의·추가 조사 정리

> **작성 기준**
>
> - 원본 PDF는 여러 장의 발표 슬라이드를 한 페이지에 모아 놓은 콜라주 형태이다.
> - 따라서 PDF의 **페이지 수가 아니라, 제목·도식·경계가 독립적인 발표 단위**를 기준으로 총 **32개 슬라이드**로 구분하였다.
> - 각 항목은 `슬라이드가 말하는 내용 → 왜 중요한가 → 추가 조사와 해석 → 한 줄 핵심` 순서로 정리하였다.
> - **슬라이드가 말하는 내용**은 원본 발표 자료와 손글씨 메모를 바탕으로 하며, **추가 조사와 해석**은 공개 논문·학회 자료·기관 자료를 별도로 확인해 보완한 내용이다.
> - 발표 자료에만 등장하고 동일한 공개 논문을 확정하기 어려운 모델명이나 수치는 그 사실을 명시하였다.

---

# 전체 발표의 큰 흐름

이 발표는 단순히 “AI로 약물 후보를 예측한다”는 이야기가 아니다. 발표 전체는 다음과 같은 하나의 폐쇄형 연구 순환 구조를 설명한다.

```text
대규모 생물학·화학 데이터
        ↓
AI/ML 모델이 가설·표적·분자·실험을 제안
        ↓
실험실 또는 임상에서 실제 검증
        ↓
실험 결과가 새로운 학습 데이터가 됨
        ↓
다음 예측과 설계가 더 정교해짐
```

즉, **AI가 실험을 대체하는 것이 아니라 실험할 대상을 더 잘 고르고, 실험 결과를 다시 학습하여 다음 실험을 개선하는 구조**가 Lab-in-the-Loop의 핵심이다. 발표는 이 구조를 표적 발굴, 유전체 기능 예측, 가상 세포, 소분자 탐색, 항체 설계, 환자 선별, 맞춤형 암 백신까지 확장한다.

---

# Part 1. 임상 성공 사례와 Lab-in-the-Loop의 필요성

## 슬라이드 01. Inavolisib: HR+ 유방암의 PI3K 경로 표적화

### 슬라이드가 말하는 내용

이 슬라이드는 **PIK3CA 변이가 있는 호르몬수용체 양성(HR+), HER2 음성 유방암**에서 PI3Kα 억제제인 **inavolisib**가 어떤 의미를 갖는지를 보여준다.

왼쪽 그래프는 inavolisib가 기존 PI3Kα 억제제와 비교해 변이형 PI3Kα를 더 강하게 억제하면서도 다른 PI3K isoform에 대한 불필요한 작용을 줄이도록 설계되었다는 점을 강조한다. 즉, 단순히 강한 약물이 아니라 **원하는 변이 표적에 더 선택적으로 작동하는 약물**이라는 메시지다.

오른쪽 생존곡선은 inavolisib를 palbociclib·fulvestrant와 병용한 환자군이 위약 병용군보다 질병 진행 없이 지내는 기간이 길었다는 것을 나타낸다. 발표 슬라이드에서 강조된 위험비는 약 **0.43**으로, 관찰 기간 중 질병 진행 또는 사망의 상대적 위험이 크게 낮아졌음을 의미한다.

### 왜 중요한가

이 사례는 AI 기반 신약개발의 최종 목표가 단순히 예측 정확도를 높이는 것이 아니라, 다음 세 단계를 모두 연결하는 것임을 보여준다.

1. 질병을 일으키는 분자적 변이를 정확히 정의한다.
2. 그 변이에 선택적인 분자를 설계한다.
3. 실제 환자군에서 임상적 이득을 증명한다.

특히 PI3K 경로는 정상 세포에도 중요한 신호전달 경로이므로, 억제가 강하기만 해서는 독성이 문제가 될 수 있다. 따라서 **표적 선택성, 약효, 독성 사이의 균형**이 신약 설계의 핵심이다. Lab-in-the-Loop는 분자 설계·세포 실험·바이오마커 분석·임상 결과를 연결하여 이 균형을 반복적으로 개선하려는 접근이다.

### 추가 조사와 해석

INAVO120 3상 연구에서 inavolisib 병용군의 중앙 무진행생존기간은 15.0개월, 대조군은 7.3개월이었으며 질병 진행 또는 사망의 위험비는 0.43이었다. 이후 발표된 전체생존 분석에서도 생존 이득이 보고되었지만, 고혈당·구내염·위장관 및 안과 이상반응이 더 자주 나타났다는 점도 함께 보아야 한다.

- [NEJM: Inavolisib-Based Therapy in PIK3CA-Mutated Advanced Breast Cancer](https://www.nejm.org/doi/10.1056/NEJMoa2404625)
- [NEJM: Overall Survival with Inavolisib in PIK3CA-Mutated Advanced Breast Cancer](https://www.nejm.org/doi/10.1056/NEJMoa2501796)

> **한 줄 핵심:** 정확한 변이 환자군을 선택하고 선택성이 높은 억제제를 설계하면, 분자 수준의 정밀성이 실제 임상적 생존 이득으로 이어질 수 있다.

---

## 슬라이드 02. Divarasib: KRAS G12C 변이 종양 표적화

### 슬라이드가 말하는 내용

이 슬라이드는 오랫동안 “약물로 공략하기 어려운 표적”으로 여겨졌던 KRAS 중 <b>G12C 변이</b>를 억제하는 소분자 <b>divarasib(GDC-6036)</b>를 다룬다.

왼쪽은 divarasib가 KRAS G12C에 높은 potency와 selectivity를 보인다는 전임상 결과이고, 오른쪽 waterfall plot은 비소세포폐암 환자에서 종양 크기가 감소한 환자가 다수 존재했음을 보여준다. 슬라이드에는 단독요법에서 확인된 객관적 반응률이 약 55% 수준이라는 결과가 강조되어 있다.

Divarasib는 KRAS G12C 단백질의 변이 시 생성되는 cysteine에 공유결합하여, KRAS를 비활성 상태에 고정하는 방식으로 작동한다.

### 왜 중요한가

이 사례의 의미는 “AI가 새로운 표적을 찾았다”보다 더 넓다. KRAS G12C처럼 구조적으로 까다로운 표적에서는 다음 정보가 함께 필요하다.

- 단백질의 순간적인 3차원 구조와 결합 가능한 pocket
- 변이형과 정상형 사이의 선택성
- 세포 안에서의 실제 경로 억제 효과
- 내성 돌연변이와 우회 신호
- 환자 종양에서의 반응 차이

따라서 계산화학, 구조생물학, 약물동태, 세포 실험, 임상 바이오마커를 반복적으로 연결하는 Lab-in-the-Loop가 특히 유용한 영역이다.

### 추가 조사와 해석

초기 1상 연구는 137명의 KRAS G12C 변이 고형암 환자를 포함했으며, 연구의 주목적은 안전성과 초기 항종양 활성을 평가하는 것이었다. 비소세포폐암에서는 의미 있는 반응이 관찰되었으나, 암종별 반응 차이와 획득 내성은 계속 해결해야 할 문제다. 따라서 waterfall plot의 종양 감소는 유망한 신호이지만, 장기 생존 효과나 다른 치료제 대비 우월성을 곧바로 의미하지는 않는다.

- [NEJM: Single-Agent Divarasib in Solid Tumors with a KRAS G12C Mutation](https://www.nejm.org/doi/10.1056/NEJMoa2303810)

> **한 줄 핵심:** 구조적으로 어려운 표적도 구조·화학·세포·환자 데이터를 반복 연결하면 실제 약물 표적으로 전환할 수 있다.

---

## 슬라이드 03. Drug R&D의 네 가지 특성: Big Numbers, Multi-Scale, Multi-Modal, Varied

### 슬라이드가 말하는 내용

이 슬라이드는 신약개발이 왜 AI에 적합하면서도 동시에 어려운 문제인지 네 단어로 요약한다.

- **Big Numbers:** 가능한 화합물, DNA 서열, 단백질 조합, 세포 상태가 너무 많다.
- **Multi-Scale:** 원자에서 시작해 유전자·단백질·세포·조직·장기·환자까지 여러 수준을 연결해야 한다.
- **Multi-Modal:** 서열, 구조, 이미지, 전사체, 단백체, 임상기록 등 서로 다른 측정 방식이 존재한다.
- **Varied:** 질병·표적·실험·개발 단계마다 연구 경로가 달라 하나의 고정된 파이프라인으로 해결하기 어렵다.

### 왜 중요한가

신약개발에서 가장 큰 문제는 데이터가 단순히 “많다”는 것이 아니라 **서로 다른 공간과 척도에 흩어져 있다는 것**이다. 예를 들어 한 화합물의 구조가 좋은 결합을 예측하더라도, 세포막을 통과하지 못하거나 대사 독성이 있거나 환자 조직에서 표적이 발현되지 않으면 약물이 될 수 없다.

따라서 유능한 AI 시스템은 단일 데이터셋의 분류기가 아니라 다음을 수행해야 한다.

- 서로 다른 modality를 공통 표현 공간에 정렬한다.
- 원자 수준의 변화가 세포와 환자 수준에서 어떤 결과를 만드는지 연결한다.
- 새로운 질병·표적·실험 조건에 맞게 유연하게 적용된다.
- 예측 불확실성을 계산하고 필요한 실험을 선택한다.

### 추가 조사와 해석

이 네 특성은 최근 생명과학 foundation model이 단일 모델보다 **모델·도구·데이터·실험의 생태계**로 발전하는 이유를 설명한다. 특히 multimodal 모델은 데이터가 완전히 짝지어져 있지 않은 현실, 기관마다 다른 batch effect, 질병별 희소성까지 다뤄야 한다. 데이터 크기만 키우는 것으로는 충분하지 않으며, 인과적 perturbation 데이터와 실험 피드백이 필요하다.

- [Cell: Toward a Foundation Model of Causal Cell and Tissue Biology](https://doi.org/10.1016/j.cell.2024.07.035)

> **한 줄 핵심:** 신약개발은 거대하고 다척도·다중모달이며 매번 다른 문제이므로, AI는 단일 예측기가 아니라 통합적 의사결정 시스템이어야 한다.

---

## 슬라이드 04. Lab in the Loop: 데이터–AI–실험의 폐쇄형 순환

### 슬라이드가 말하는 내용

이 슬라이드는 발표 전체의 핵심 개념인 **Lab-in-the-Loop**를 가장 단순한 원형 구조로 보여준다.

1. 생물학·화학·임상 데이터를 수집한다.
2. AI/ML 모델이 패턴을 학습하고 새로운 가설이나 후보를 제안한다.
3. 인간 연구자와 자동화 실험실이 후보를 검증한다.
4. 실패와 성공을 모두 다시 데이터에 포함한다.
5. 업데이트된 모델이 다음 실험을 더 잘 선택한다.

손글씨 메모의 “모든 DNA 서열의 기능을 어떻게 알 수 있는가”라는 질문은 이 구조의 필요성을 잘 드러낸다. 모든 서열이나 화합물을 실제로 시험할 수 없으므로, 모델이 우선순위를 정하고 실험이 모델을 교정해야 한다.

### 왜 중요한가

기존의 일방향 AI 적용은 이미 만들어진 데이터에서 모델을 학습한 뒤, 한 번 후보를 내는 데 그치는 경우가 많다. 그러나 생물학 데이터는 편향되고 불완전하며, 모델이 가장 흥미롭다고 고른 후보는 학습 데이터와 가장 멀리 떨어져 있을 가능성이 높다.

Lab-in-the-Loop에서는 모델 성능을 단순 테스트셋 정확도가 아니라 다음과 같이 평가한다.

- 같은 실험 비용으로 더 많은 유효 후보를 찾았는가?
- 실패 가능성이 큰 후보를 일찍 제거했는가?
- 모델이 틀린 영역을 다음 실험이 보완했는가?
- 반복할수록 hit rate와 정보 효율이 증가했는가?

### 추가 조사와 해석

이 구조는 active learning, Bayesian optimization, design–make–test–analyze(DMTA), self-driving laboratory와 연결된다. 다만 완전 자동화보다 중요한 것은 **검증 가능한 기록과 인간의 판단**이다. 실험 조건, 음성 결과, batch, 샘플 품질이 기록되지 않으면 반복 루프가 오히려 오류를 확대할 수 있다.

- [Genentech: Redefining Drug Discovery with AI](https://www.gene.com/stories/redefining-drug-discovery-with-ai)
- [Cell: Perturbation Cell and Tissue Atlas](https://doi.org/10.1016/j.cell.2024.07.035)

> **한 줄 핵심:** Lab-in-the-Loop는 AI가 실험을 없애는 방식이 아니라, 가장 정보가 큰 실험을 반복적으로 선택하도록 만드는 방식이다.

---

## 슬라이드 05. Right Target for a Disease: 질병에 맞는 표적을 찾는 루프

### 슬라이드가 말하는 내용

이 슬라이드는 Lab-in-the-Loop를 <b>표적 발굴(target discovery)</b>에 적용한다. 순환 구조는 대략 다음과 같다.

```text
질병 샘플·세포 모델
→ 유전적/화학적 perturbation 실험
→ 다중오믹스·이미지 데이터
→ AI/ML 분석
→ 원인 후보 표적
→ 재검증 실험
```

단순히 질병 조직에서 많이 발현되는 유전자를 고르는 것이 아니라, 해당 유전자를 변화시켰을 때 질병 표현형이 실제로 개선되는지를 확인해야 한다는 메시지다.

### 왜 중요한가

신약 후보가 실패하는 가장 근본적인 이유 중 하나는 분자가 약해서가 아니라 **처음 고른 표적이 인간 질병의 원인이 아니기 때문**이다. 상관관계만으로 선정한 표적은 세포주에서는 작동해도 환자에서는 실패할 수 있다.

따라서 좋은 표적에는 다음 조건이 필요하다.

- 질병을 유발하거나 유지하는 인과적 역할
- 약물로 조절 가능한 분자적 특성
- 적절한 세포와 조직에서의 발현
- 환자 하위집단을 구분할 바이오마커
- 정상 조직에서 허용 가능한 안전성

### 추가 조사와 해석

최근 표적 발굴에서는 인간 유전학, CRISPR perturbation, single-cell atlas, 공간오믹스, 임상 데이터가 함께 사용된다. AI의 역할은 후보를 많이 나열하는 것이 아니라, 서로 다른 근거를 통합하여 **검증할 가치가 높은 표적과 실패 위험이 높은 표적을 분리**하는 것이다.

> **한 줄 핵심:** 올바른 표적을 찾는 것은 신약개발의 출발점이며, 상관관계를 perturbation 실험으로 인과관계로 바꾸는 과정이 핵심이다.

---

# Part 2. 유전체·세포 상태·인과관계를 학습하는 AI

## 슬라이드 06. Same Genes, Different Cells, Different Disease Outcomes

### 슬라이드가 말하는 내용

이 슬라이드는 같은 유전자 또는 같은 유전적 변이라도 **어떤 세포에서 작동하느냐에 따라 다른 질병으로 이어질 수 있다**는 점을 보여준다. 그림에서는 하나의 유전자 축이 면역세포, 근육세포, 신경세포, 암세포와 연결되며 류마티스관절염, 근이영양증, 다발성경화증, 암처럼 서로 다른 결과를 만든다.

유전자는 독립적으로 기능하지 않는다. 세포마다 열린 chromatin, 전사인자, enhancer, 신호전달 경로, 주변 세포와의 상호작용이 다르기 때문에 같은 변이의 기능적 효과도 달라진다.

### 왜 중요한가

전통적인 유전체 분석은 “변이 → 유전자 → 질병”처럼 단순한 연결을 가정하기 쉽다. 그러나 실제로는 다음과 같이 봐야 한다.

```text
변이
→ 특정 세포 상태의 조절요소 변화
→ 특정 유전자의 발현 변화
→ 세포 기능과 세포 간 상호작용 변화
→ 조직 병리
→ 환자 질병
```

따라서 정밀의학과 표적 발굴에는 **cell-type specificity**가 필수다. 질병과 관련된 변이가 어느 세포에서 기능하는지 모르면, 잘못된 세포 모델로 실험하거나 부작용이 큰 표적을 선택할 수 있다.

### 추가 조사와 해석

single-cell RNA-seq, single-cell ATAC-seq, 공간전사체는 세포별 질병 프로그램을 분리하는 데 도움을 준다. 다만 관찰 데이터만으로는 인과성을 확정하기 어렵기 때문에, cell-type-specific CRISPR나 organoid·co-culture 시스템과 연결해야 한다.

> **한 줄 핵심:** 유전자의 의미는 고정되어 있지 않으며, 세포 종류와 상태가 변이의 질병 효과를 결정한다.

---

## 슬라이드 07. Decima: 7,000개 이상의 세포 상태를 아우르는 Sequence-to-Function 모델

### 슬라이드가 말하는 내용

**Decima**는 DNA 서열을 입력받아 특정 세포 유형과 질병 상태에서 유전자 발현이 어떻게 나타날지를 예측하는 sequence-to-function 모델이다. 슬라이드는 하나의 공통 backbone이 다양한 cell state에 대한 출력을 생성하며 다음 응용을 지원한다고 설명한다.

- 비암호화 변이의 기능 효과 예측
- 인간 유전학 신호를 특정 세포 상태와 연결
- 질병 관련 유전자 프로그램 발견
- 유전자치료용 조절서열 설계

### 왜 중요한가

GWAS로 발견되는 질병 연관 변이의 다수는 단백질 서열이 아니라 enhancer·promoter 같은 비암호화 영역에 위치한다. 이 변이가 어느 유전자의 발현을, 어느 세포에서, 어느 방향으로 바꾸는지 해석하는 것이 어렵다.

Decima 같은 모델은 다음 질문을 연결한다.

- 이 DNA 변이가 기능을 가지는가?
- 어느 세포 상태에서 효과가 가장 큰가?
- 어떤 유전자의 발현이 달라지는가?
- 질병 상태와 정상 상태에서 효과가 다른가?
- 원하는 세포에서만 작동하는 조절서열을 설계할 수 있는가?

### 추가 조사와 해석

공개 preprint에 따르면 Decima는 2,200만 개가 넘는 single-cell 또는 single-nucleus RNA-seq 세포를 바탕으로 학습되었으며, 주변 DNA 서열만으로 보지 못한 유전자의 세포상태별 발현을 예측하고 비암호화 변이와 조절서열 설계에 적용했다. 발표의 “7,000 cell states”는 모델이 아우르는 세포 상태의 다양성을 강조한 표현으로 이해할 수 있다.

다만 sequence-to-function 모델은 관측된 데이터의 세포 구성과 assay bias를 학습할 수 있으며, 예측된 조절 효과는 reporter assay, CRISPR perturbation, 동물·환자 유래 모델에서 검증해야 한다.

- [bioRxiv: Decoding Sequence Determinants of Gene Expression in Diverse Cellular and Disease States](https://www.biorxiv.org/content/10.1101/2024.10.09.617507v1)

> **한 줄 핵심:** Decima는 DNA 서열과 세포 맥락 사이의 공백을 메워 비암호화 변이를 해석하고 세포 특이적 조절 DNA를 설계하려는 모델이다.

---

## 슬라이드 08. 인간 유전체 DNA를 유연하게 설계하는 플랫폼

### 슬라이드가 말하는 내용

이 슬라이드는 유전체 AI를 분석에서 **설계**로 확장한다. 중앙의 순환은 다음 세 단계다.

1. **Nominate:** 가능한 조절 DNA 후보를 생성한다.
2. **Score:** 모델로 기능·특이성·안전성을 평가하고 순위를 정한다.
3. **Design:** 상위 후보를 조합하거나 서열을 새롭게 설계한다.

오른쪽에는 세 가지 응용이 제시된다.

- **Gene therapy:** 목표 세포에서 유전자 발현을 회복시키는 promoter 설계
- **Cell engineering:** 세포 증식이나 기능을 켜고 끄는 조절 요소 설계
- **Cell-type-specific CRISPR delivery:** 특정 세포에서만 Cas9이 발현되도록 하는 mini-promoter 설계

### 왜 중요한가

유전자치료에서 치료 유전자의 내용만큼 중요한 것이 **어디서, 얼마나, 언제 발현되는가**이다. 너무 강한 promoter는 독성을 만들 수 있고, 잘못된 조직에서 발현되면 면역반응이나 부작용이 생길 수 있다.

AI 기반 조절 DNA 설계는 자연에 존재하는 promoter를 그대로 찾는 것을 넘어, 다음 조건을 동시에 만족하는 새로운 서열을 탐색한다.

- 목표 세포에서 높은 활성
- 비목표 세포에서 낮은 활성
- 작은 벡터 용량
- 장기 발현 안정성
- 제조와 전달 가능성

### 추가 조사와 해석

이 과정은 전형적인 design–build–test–learn 루프다. 모델이 후보를 만들고, 대규모 reporter assay나 MPRA로 기능을 측정한 뒤, 그 결과를 다시 학습한다. 중요한 점은 단일 예측 점수보다 **다목적 최적화**가 필요하다는 것이다. 활성만 최대화하면 특이성이나 안전성이 악화될 수 있다.

> **한 줄 핵심:** 유전체 AI의 최종 단계는 변이를 해석하는 데서 끝나지 않고, 원하는 세포에서 원하는 기능을 내는 DNA를 직접 설계하는 것이다.

---

## 슬라이드 09. Foundation Model을 위한 Foundational Datasets

### 슬라이드가 말하는 내용

이 슬라이드는 범용 생물학 foundation model을 만들기 위해 어떤 데이터가 필요한지 정리한다.

입력 데이터에는 다음이 포함된다.

- **Sequence to phenotype:** DNA 서열과 기능적 표현형
- **Perturbations:** 유전적 교란과 화합물 처리
- **Tissues:** 정상·질병 조직의 세포 지도
- **Cancer clinico-genomics:** 암 유전체와 환자 임상 결과

이 데이터는 세 종류의 모델로 이어진다.

- **Cell & tissue foundation model:** 세포 상태, 표적 특이성, 안전성
- **Genomic foundation model:** 인과적 질병 메커니즘, 병리 프로그램, 세포계통 치료
- **Perturbation foundation model:** 표적 인과성, 작용기전, 가상 표현형 스크리닝

손글씨의 “Virtual Cell”은 이 세 모델의 통합 목표를 나타낸다.

### 왜 중요한가

foundation model의 성능은 아키텍처보다 **어떤 생물학적 경험을 데이터로 제공했는가**에 크게 좌우된다. 자연 상태의 세포 관측만 학습하면 상관관계는 잘 포착할 수 있지만, 개입 후 어떤 일이 생기는지는 알기 어렵다.

따라서 “가상 세포”가 되려면 다음 축이 모두 필요하다.

- 정상과 질병 상태의 다양성
- 시간에 따른 변화
- 유전적·화학적 개입
- dose와 처리 시간
- 공간적 세포 상호작용
- 환자 수준의 결과

### 추가 조사와 해석

Perturbation Cell and Tissue Atlas 구상은 CRISPR와 고차원 readout을 반복적으로 결합하여 세포 회로의 인과 구조를 학습하는 것을 제안한다. 이는 단순한 대규모 사전학습보다 **실험 설계 자체를 모델 학습과 공동 최적화**한다는 점에서 중요하다.

- [Cell: Toward a Foundation Model of Causal Cell and Tissue Biology with a Perturbation Cell and Tissue Atlas](https://doi.org/10.1016/j.cell.2024.07.035)

> **한 줄 핵심:** 가상 세포를 만들려면 관찰 데이터뿐 아니라 “무엇을 바꾸었을 때 무엇이 달라졌는가”를 기록한 대규모 인과 데이터가 필요하다.

---

## 슬라이드 10. Perturbation Modeling: 인과관계 정의

### 슬라이드가 말하는 내용

이 슬라이드는 perturbation model이 해결해야 하는 두 방향의 문제를 구분한다.

- **Forward prediction:** 특정 유전자나 화합물을 교란하면 세포 상태가 어떻게 바뀌는가?
- **Inverse prediction:** 질병 세포를 건강한 상태로 바꾸려면 어떤 perturbation이 필요한가?

오른쪽은 이러한 예측이 신약개발에서 표적 발굴, 작용기전 규명, 표적 전환(target hopping), 안전성 예측에 활용될 수 있음을 보여준다.

### 왜 중요한가

질병 세포와 정상 세포가 다르다는 관찰만으로는 무엇을 치료해야 하는지 알 수 없다. perturbation은 특정 원인을 직접 조절하기 때문에 **상관관계를 인과관계 후보로 바꾸는 실험**이다.

특히 inverse prediction은 신약개발의 질문을 다음과 같이 재구성한다.

```text
“이 약물이 무엇을 하는가?”가 아니라
“원하는 세포 상태를 만들기 위해 무엇을 해야 하는가?”
```

이는 표현형 기반 신약개발, 약물 재창출, 조합치료 설계와 직접 연결된다.

### 추가 조사와 해석

현재 perturbation 모델의 대표적 한계는 보지 못한 cell type·dose·시간·복합 perturbation에 대한 일반화다. 평균 유전자 발현을 잘 재구성하는 모델이 반드시 중요한 pathway나 희귀 세포 반응을 잘 예측하는 것은 아니다. 따라서 평가에는 differential expression, pathway recovery, cell-state transition, 실제 hit enrichment가 함께 필요하다.

- [Cell: Perturbation Cell and Tissue Atlas](https://doi.org/10.1016/j.cell.2024.07.035)

> **한 줄 핵심:** perturbation model의 본질은 세포 상태를 설명하는 것을 넘어, 원하는 상태를 만들 개입을 역으로 찾는 데 있다.

---

## 슬라이드 11. 암 표적·약물 발굴에 적용된 Lab-in-the-Loop

### 슬라이드가 말하는 내용

이 슬라이드는 암 연구에서 여러 modality를 동시에 수집하고 연결하는 실제 workflow를 보여준다.

- Cell Painting과 같은 고내용량 이미지
- 유전자 perturbation을 측정하는 Perturb-seq
- transcriptomics map
- 화합물–유전자–경로 관계
- 유전자 기능과 회로의 재구성

같은 화합물에 의해 비슷하게 변하는 유전자, 또는 비슷한 세포 형태를 만드는 유전자와 화합물을 연결하여 작용기전을 추론한다.

### 왜 중요한가

단일 readout은 약물 효과의 일부만 보여준다.

- 세포 생존율만 보면 왜 죽었는지 알기 어렵다.
- 전사체만 보면 세포 구조와 조직 수준 변화를 놓칠 수 있다.
- 이미지만 보면 분자적 원인을 직접 알기 어렵다.

서로 다른 modality를 결합하면 약물의 phenotype, 표적, pathway, 독성 가능성을 함께 볼 수 있다. 특히 유전적 perturbation과 화학적 perturbation의 표현형이 유사하면, 화합물의 잠재적 작용기전을 추정할 수 있다.

### 추가 조사와 해석

이 방식은 “모든 데이터를 한 벡터에 넣는 것”보다 각 modality의 강점을 보존한 뒤 공통 잠재공간에서 정렬하는 것이 중요하다. 또한 같은 plate, 세포주, 시간, dose에서 얻어진 paired data가 이상적이지만 실제로는 불완전하게 짝지어진 데이터가 많아 missing modality와 batch correction이 주요 과제가 된다.

> **한 줄 핵심:** 암의 복잡한 작용기전을 이해하려면 이미지·전사체·유전적 교란·화합물 반응을 하나의 인과 지도에 연결해야 한다.

---

# Part 3. AI 에이전트, 질병 지도와 소분자 신약개발

## 슬라이드 12. Single-Cell 데이터 폭증과 Biomedical AI Agents

### 슬라이드가 말하는 내용

왼쪽 그래프는 공개된 single-cell 데이터의 세포 수가 시간에 따라 기하급수적으로 증가했음을 보여준다. 오른쪽에는 **BIOMNI**, **COCOA**, **SPATIAL AGENT** 같은 생의학 AI 에이전트가 배치되어 있다.

핵심 메시지는 데이터가 너무 커지고 분석 도구가 너무 많아져 인간 연구자가 모든 데이터셋·논문·도구를 수작업으로 연결하기 어렵다는 것이다. AI agent는 질문을 받아 데이터 검색, 코드 실행, 통계 분석, 시각화, 가설 제안까지 여러 작업을 조합한다.

### 왜 중요한가

foundation model이 “예측 모델”이라면 agent는 **연구 절차를 수행하는 실행 계층**이다. Lab-in-the-Loop에서 agent는 다음 역할을 할 수 있다.

- 사용 가능한 데이터와 프로토콜 검색
- 적절한 분석 도구 선택
- 여러 모델의 출력을 조합
- 다음 실험 후보와 대조군 제안
- 결과를 표준 형식으로 기록
- 실패 원인을 분석하고 루프를 재설계

### 추가 조사와 해석

Biomni는 LLM reasoning, 검색 기반 계획, 코드 실행, 생의학 도구를 결합한 범용 agent로 제안되었다. 그러나 agent가 최종 결론을 자동으로 신뢰할 수 있다는 뜻은 아니다. 최근 평가에서도 출처 검색은 잘하지만 방법 선택과 생물학적 해석에는 상당한 개선 여지가 보고된다. 즉, 에이전트는 연구자를 대체하기보다 **검증 가능한 copilot**으로 사용하는 것이 현실적이다.

슬라이드에 표시된 COCOA와 Spatial Agent는 발표자가 제시한 조직·공간오믹스 분석 agent 사례로 보이지만, 동일 명칭의 공개 논문과 버전을 슬라이드만으로 완전히 확정하기는 어렵다.

- [bioRxiv: Biomni — A General-Purpose Biomedical AI Agent](https://www.biorxiv.org/content/10.1101/2025.05.30.656746v1)
- [GitHub: SpatialAgent](https://github.com/Genentech/SpatialAgent)

> **한 줄 핵심:** 데이터가 커질수록 중요한 것은 더 큰 모델 하나가 아니라, 데이터·도구·실험을 올바른 순서로 연결하는 검증 가능한 AI agent다.

---

## 슬라이드 13. WHERE–WHICH–WHAT: 질병 세포를 찾고 정의하고 연결하기

### 슬라이드가 말하는 내용

이 슬라이드는 질병 세포를 이해하는 과정을 세 질문으로 나눈다.

- **WHERE are my disease cells?**  
  `scSimilarity`를 이용해 유사한 세포를 찾는다.
- **WHICH genes define them?**  
  `SIGnature`로 그 세포를 정의하는 중요 유전자를 찾는다.
- **WHAT drives disease in them?**  
  `PaSCient`로 환자, 세포, 유전자를 연결해 질병 원인을 탐색한다.

### 왜 중요한가

single-cell 데이터에는 수백만 개의 세포가 포함되지만, 질병에 실제로 기여하는 세포 상태는 일부일 수 있다. 단순 clustering만으로는 다음을 구분하기 어렵다.

- 정상 세포와 병적 세포 상태
- 서로 다른 연구에서 같은 세포 상태
- 많이 발현되는 유전자와 기능적으로 중요한 유전자
- 질병과 함께 나타난 유전자와 질병을 유도하는 유전자

세 질문을 순서대로 풀면 “질병 세포의 위치 → 핵심 프로그램 → 환자별 원인과 표적”으로 이어진다.

### 추가 조사와 해석

SIGnature는 single-cell foundation model의 attribution을 사용하여 단순 발현량과 다른 “세포 정체성에 대한 유전자 중요도”를 계산한다. 낮게 발현되는 전사인자처럼 기능적으로 중요하지만 발현량 기반 분석에서 놓치기 쉬운 유전자를 강조하는 것이 목표다.

PaSCient는 발표 자료상 환자·세포·유전자 연결을 통해 질병 메커니즘과 환자 stratification을 지원하는 도구로 제시된다. 다만 공개 자료의 정확한 버전과 세부 구현은 발표 슬라이드만으로 모두 확인하기 어렵기 때문에, 여기서는 발표의 개념적 역할을 중심으로 해석하였다.

- [Nature Biotechnology: Scoring Gene Importance by Interpreting Single-Cell Foundation Models](https://www.nature.com/articles/s41587-026-03112-5)

> **한 줄 핵심:** single-cell 분석의 목적은 세포를 예쁘게 군집화하는 것이 아니라, 질병을 만드는 세포와 유전자 프로그램을 환자 수준에서 찾는 것이다.

---

## 슬라이드 14. AI-Powered Pathobiology Maps

### 슬라이드가 말하는 내용

이 슬라이드는 암, 면역질환, 신경질환, 안과질환의 **pathobiology map**을 AI로 더 빠르게 구축하여 표적 발굴 포트폴리오를 확장한 사례를 제시한다.

슬라이드에는 첫 번째 염증성장질환(IBD) map을 만드는 데 약 1년이 걸렸지만, 이후 여러 신경질환 map은 수 주 안에 구축되었다는 비교가 나온다. 또한 2년 동안 discovery portfolio entry가 133% 증가했다는 회사 내부 성과 지표가 표시되어 있다.

### 왜 중요한가

pathobiology map은 질병에 관여하는 세포, 유전자, 경로, 조직 위치, 환자 하위집단을 연결한 지도다. 이 지도는 다음을 돕는다.

- 표적 후보의 우선순위 결정
- 서로 다른 질병의 공통 경로 발견
- 환자 하위집단별 표적 선택
- 기존 표적의 실패 원인 분석
- 새로운 바이오마커 설계

처음에는 데이터 표준화와 분석 방법을 구축하는 데 시간이 오래 걸리지만, 재사용 가능한 pipeline이 만들어지면 다음 질병으로 확장하는 속도가 빨라질 수 있다.

### 추가 조사와 해석

슬라이드의 “1년→수 주”와 “+133%”는 발표 기관이 제시한 운영 성과로 보이며, 공개 peer-reviewed 연구에서 독립적으로 검증된 일반적 수치로 해석해서는 안 된다. 중요한 것은 숫자 자체보다 **분석 자산과 데이터 표준을 재사용함으로써 질병 지도 생성의 한계비용을 낮춘다**는 운영 모델이다.

> **한 줄 핵심:** 질병별 분석을 매번 처음부터 시작하지 않고 재사용 가능한 AI 지도 제작 체계를 만들면 표적 발굴 속도를 크게 높일 수 있다.

---

## 슬라이드 15. 소분자 신약개발의 모든 단계에서 계산이 작동한다

### 슬라이드가 말하는 내용

이 슬라이드는 계산이 소분자 신약개발의 특정 단계만이 아니라 전체 과정에 관여한다고 설명한다.

1. **Druggability:** 표적 단백질에 실제 결합 가능한 부위가 있는지 판단
2. **Hit finding:** 초기 활성을 가진 분자 탐색
3. **Optimization:** potency, selectivity, ADME, 독성을 함께 개선

슬라이드의 weekly/monthly 표현은 모델 파라미터·아키텍처·데이터가 반복적으로 개선되며, 설계–합성–시험–분석 루프가 계속 진행된다는 뜻이다.

### 왜 중요한가

초기 후보 탐색과 후기 최적화는 서로 다른 문제다. hit를 찾는 모델이 좋은 약물을 자동으로 만들어 주지는 않는다. 후기 단계에서는 다음의 상충하는 목표를 동시에 다뤄야 한다.

- 표적 결합력 증가
- off-target 감소
- 용해도와 막 투과성 개선
- 대사 안정성 증가
- 독성 감소
- 합성 가능성 유지
- 특허 신규성 확보

따라서 하나의 점수로 분자를 정렬하기보다 단계별 모델과 실험을 연결하는 것이 중요하다.

### 추가 조사와 해석

현실적인 Lab-in-the-Loop에서는 모델 업데이트 주기와 실험 주기가 다르다. 계산 모델은 매주 개선될 수 있지만 합성·세포·동물 실험에는 더 긴 시간이 걸린다. 성공적인 시스템은 이 비동기성을 고려해 빠른 계산 proxy와 느리지만 신뢰도 높은 실험을 계층적으로 배치한다.

> **한 줄 핵심:** AI 신약개발은 한 번의 가상 스크리닝이 아니라 druggability부터 최적화까지 이어지는 다단계 의사결정 과정이다.

---

## 슬라이드 16. Druggability: 표적에서 분자로 넘어가는 다리

### 슬라이드가 말하는 내용

이 슬라이드는 표적을 찾았다고 해서 곧바로 약물을 만들 수 있는 것은 아니며, 그 사이에 **druggability** 평가가 필요하다고 강조한다.

그림은 단백질이 고정된 구조가 아니라 여러 conformation을 오가며, 그중 일부에서만 일시적으로 접근 가능한 **druggable pocket**이 나타날 수 있음을 보여준다. AI와 실험은 이러한 구조 ensemble을 분석해 가장 공략하기 좋은 결합 부위를 찾는다.

### 왜 중요한가

정적 단백질 구조 하나만 사용하면 다음을 놓칠 수 있다.

- 숨겨졌다가 열리는 cryptic pocket
- ligand 결합 후 유도되는 구조 변화
- allosteric site
- 변이로 달라지는 pocket
- 실제 세포 환경에서의 복합체 상태

반대로 구조 모델이 pocket을 예측해도 실제로 분자가 결합하고 기능을 조절한다는 보장은 없다. 따라서 구조 예측은 HDX-MS, NMR, cryo-EM, mutagenesis, biochemical assay와 연결되어야 한다.

### 추가 조사와 해석

최근 생성형 구조 모델은 단일 최저에너지 구조뿐 아니라 가능한 구조 ensemble과 단백질–리간드 공구조를 생성하려 한다. Lab-in-the-Loop의 핵심은 모델이 제안한 pocket에 대해 fragment screening이나 결합 실험을 수행하고, 결과를 다시 모델에 반영하는 것이다.

> **한 줄 핵심:** 좋은 질병 표적을 실제 약물로 바꾸려면 단백질의 동적 구조에서 실험적으로 공략 가능한 pocket을 찾아야 한다.

---

## 슬라이드 17. VoxBind와 SAGE를 이용한 Molecular Hit Finding

### 슬라이드가 말하는 내용

이 슬라이드는 두 종류의 생성적 hit-finding 접근을 보여준다.

- **VoxBind:** 단백질 pocket을 3차원 voxel grid로 표현하고, 노이즈를 제거하는 방식으로 pocket에 맞는 분자를 생성
- **SAGE:** 단백질–리간드 상호작용 점수와 진화적 탐색을 결합하여 후보 분자를 반복적으로 선택·변형

손글씨 메모는 “void를 채우듯” pocket 안의 공간과 상호작용 조건을 만족하는 분자를 만든다는 직관을 강조한다.

### 왜 중요한가

기존 virtual screening은 이미 존재하는 라이브러리에서 점수가 높은 분자를 찾는다. 생성 모델은 라이브러리 밖의 새로운 분자까지 제안할 수 있어 탐색 공간을 넓힌다.

그러나 생성 분자가 유용하려면 다음을 모두 만족해야 한다.

- pocket과 기하학적으로 잘 맞음
- 적절한 수소결합·전하·소수성 상호작용
- 비현실적 원자 충돌이 없음
- 화학적으로 유효하고 합성 가능
- 기존 분자와 충분히 다르면서도 약물성이 있음

### 추가 조사와 해석

VoxBind는 3D 원자 밀도 grid에 대한 score-based 생성 모델이며, 단순한 학습과 빠른 sampling, 다양한 생성 분자, 낮은 steric clash를 목표로 한다. 다만 in silico docking이나 생성 지표가 실제 결합 친화도를 대신하지는 않으므로 biochemical validation이 필수다.

슬라이드의 SAGE는 발표 내부의 분자 진화·선택형 hit-finding 시스템으로 보이지만, 동일한 도식과 명칭의 공개 원 논문을 확정하지 못했다. 따라서 구체적 성능 수치보다 “모델 생성 → 계산 평가 → 선택 → 반복” 구조로 이해하는 것이 안전하다.

- [ICML 2024: Structure-Based Drug Design by Denoising Voxel Grids](https://proceedings.mlr.press/v235/pinheiro24a.html)

> **한 줄 핵심:** 생성 모델은 기존 라이브러리 검색을 넘어 pocket 조건을 만족하는 새 분자를 만들 수 있지만, 실험 검증이 없는 생성 점수는 약효가 아니다.

---

## 슬라이드 18. Boltz Jump: 구조생물학과 생성 모델을 Lab-in-the-Loop로 통합

### 슬라이드가 말하는 내용

이 슬라이드는 **Boltz Jump**가 구조 생성 모델과 실험 구조생물학을 순환적으로 연결하는 방식을 보여준다.

- 생성 모델이 단백질 또는 단백질–리간드의 다양한 구조 ensemble을 생성
- 실험에서 얻은 HDX-MS, NMR 등의 정보를 이용해 ensemble을 평가·제약
- 실험에 부합하는 구조를 선택하거나 모델을 업데이트
- 개선된 구조 ensemble을 다시 분자 설계에 사용

핵심은 단일 구조 예측이 아니라 **Boltzmann-weighted conformational ensemble**을 다룬다는 점이다.

### 왜 중요한가

약물 결합과 단백질 기능은 하나의 정적인 구조보다 구조 분포에 의해 결정되는 경우가 많다. 특히 allosteric site와 cryptic pocket은 낮은 빈도로 나타나는 구조 상태에서만 보일 수 있다.

실험 데이터는 전체 원자 좌표를 직접 제공하지 않더라도, 특정 영역의 유연성이나 solvent exposure 같은 정보를 제공한다. 생성 모델은 이런 부분적인 정보를 통합하여 실험과 일치하는 ensemble을 제안할 수 있다.

### 추가 조사와 해석

발표 슬라이드와 일치하는 “Boltz Jump”의 공개 버전은 workshop 또는 초기 연구 형태로 보이며, 정식 저널 논문과 정확한 버전은 슬라이드만으로 확정하기 어렵다. 따라서 이 슬라이드는 특정 제품 성능보다 **생성 구조 ensemble과 희소한 실험 제약을 반복 결합하는 연구 패러다임**으로 이해하는 것이 적절하다.

관련 배경 연구인 walk–jump sampling은 고차원 에너지 지형에서 다양한 구조를 생성하고, 관측 데이터의 분포를 근사하는 데 사용된다.

> **한 줄 핵심:** 구조생물학의 미래는 하나의 예쁜 구조를 맞히는 것이 아니라, 실험에 맞는 구조 ensemble을 생성하고 계속 교정하는 데 있다.

---

## 슬라이드 19. Hit-Finding 모델 아키텍처를 개선하는 Lab-in-the-Loop

### 슬라이드가 말하는 내용

이 슬라이드는 표적 pocket과 알려진 결합체 정보에서 시작해 약물 후보를 생성하는 구조를 보여준다.

1. 표적 pocket과 known binder를 입력
2. 모델이 결합에 중요한 pharmacophore 또는 상호작용 패턴을 식별
3. autoregressive 방식으로 3D 분자를 생성
4. 생성된 분자를 물리적 적합성, 합성 가능성, 약물성으로 평가
5. 실험 결과를 다음 generation architecture에 반영

### 왜 중요한가

생성 모델은 reward function이나 training data의 허점을 이용해 높은 계산 점수를 얻지만 비현실적인 분자를 만들 수 있다. 따라서 아키텍처에는 다음의 inductive bias가 필요하다.

- 3D 회전·이동 대칭성
- 결합 부위의 화학적 제약
- 원자가와 결합 규칙
- 합성 가능 경로
- 불확실성 또는 ensemble 평가

실험 실패 데이터까지 학습하면 모델이 단순히 성공 예시를 모방하는 것을 넘어 실패 영역을 피할 수 있다.

### 추가 조사와 해석

이 슬라이드는 “모델이 분자를 만든 뒤 실험하는 것”보다 **실험 정보를 다음 모델 설계에 직접 반영**한다는 점에 초점이 있다. 예를 들어 특정 기능기가 반복적으로 대사 불안정을 일으킨다면, 단순 데이터 추가뿐 아니라 생성 과정의 constraint나 scoring head 자체를 바꿀 수 있다.

> **한 줄 핵심:** 좋은 Lab-in-the-Loop는 데이터만 추가하는 것이 아니라, 반복된 실험 실패를 바탕으로 생성 모델의 구조와 목적함수까지 바꾼다.

---

## 슬라이드 20. GNEprop: 항균 화합물 Virtual Hit의 실험 검증

### 슬라이드가 말하는 내용

이 슬라이드는 GNEprop 기반 항균제 탐색의 전체 funnel을 보여준다.

- 약 200만 개 화합물의 high-throughput screen
- 학습된 모델로 약 14억 개의 합성 가능한 화합물 가상 스크리닝
- 44,437개 virtual hit 선정
- 345개 합성 또는 확보 후 실험
- 165개 dose–response 평가
- 82개 validated hit

슬라이드가 강조하는 것은 거대한 화학 공간을 AI가 압축하고, 최종적으로 wet-lab에서 실제 활성을 확인했다는 점이다.

### 왜 중요한가

virtual screening의 진짜 성과는 AUROC나 docking score가 아니라 다음 지표로 평가해야 한다.

- 실험 hit rate가 얼마나 개선되었는가?
- 기존 scaffold와 다른 신규성이 있는가?
- out-of-distribution 영역에서도 작동하는가?
- 표적과 작용기전이 확인되는가?
- 합성·구매·후속 최적화가 가능한가?

GNEprop 사례는 모델의 예측을 실제 대규모 실험으로 검증하고, 탐색 공간을 확장했다는 점에서 Lab-in-the-Loop의 대표적 예다.

### 추가 조사와 해석

Nature Biotechnology 논문은 약 14억 개의 합성 가능한 분자를 스크리닝해 82개의 활성을 확인했으며, 초기 HTS 대비 약 90배 높은 hit rate를 보고했다. 일부 후보는 알려진 항생제와 구조적으로 달랐고, 후속 생물학적 검증에서 구체적인 표적도 확인되었다.

다만 82개 hit가 곧 82개의 신약을 뜻하지는 않는다. 이후 potency, 독성, 선택성, 내성, 약동학, 동물 효능, 임상 개발의 긴 최적화 과정이 남는다.

- [Nature Biotechnology: Deep-Learning-Based Virtual Screening of Antibacterial Compounds](https://www.nature.com/articles/s41587-025-02814-6)

> **한 줄 핵심:** GNEprop의 의미는 14억 개를 계산했다는 데 있지 않고, AI가 고른 후보가 실제 실험에서 훨씬 높은 비율로 적중했다는 데 있다.

---

# Part 4. 표현형 가상 스크리닝과 단백질·항체 설계

## 슬라이드 21. PhenoCompass: Virtual High-Content Phenotypic Screening

### 슬라이드가 말하는 내용

PhenoCompass는 화합물 구조와 Cell Painting 기반 세포 형태를 같은 표현 공간에 정렬하여, 실제로 모두 실험할 수 없는 거대 화합물 라이브러리에서 **원하는 세포 표현형을 만들 가능성이 높은 분자**를 가상으로 찾는 모델이다.

발표 슬라이드의 흐름은 다음과 같다.

1. JUMP Cell Painting 데이터로 화합물 구조–세포 형태 관계를 학습
2. 실험실에서 원하는 anchor perturbation의 표현형 측정
3. 모델이 대규모 라이브러리를 질의
4. anchor와 유사한 phenotype을 낼 것으로 예측된 화합물 선택
5. 실제 실험으로 작용기전과 활성을 검증

슬라이드에는 약 38억 개의 Enamine 화합물에 대한 virtual screen이 표시되어 있다.

### 왜 중요한가

target-based screening은 특정 단백질에 결합하는 분자를 찾는 데 강하지만, 복잡한 질병 프로그램 전체를 되돌리는 분자를 놓칠 수 있다. 반대로 phenotypic screening은 세포 전체의 반응을 볼 수 있지만 고내용량 이미징 비용 때문에 수십억 개를 직접 시험할 수 없다.

PhenoCompass는 두 접근의 장점을 결합한다.

- 실험은 생물학적으로 풍부한 phenotype을 제공
- 구조 encoder는 거대한 화학 공간으로 확장 가능
- 공통 embedding은 “이 구조가 이 phenotype을 낼 가능성”을 계산

### 추가 조사와 해석

2026년 공개 preprint에서 PhenoCompass는 10만 개가 넘는 JUMP Cell Painting compound profile로 학습되고, 38억 개 Enamine REAL 화합물에서 PI3K/mTOR pathway inhibitor를 탐색했다. 11개의 새로운 후보가 pathway-consistent morphology를 보였고, orthogonal assay에서 7개의 구조적으로 새로운 억제제가 확인되었다고 보고한다.

이 결과는 매우 유망하지만 현재 preprint이며, 다른 세포주·질병 phenotype·batch에서의 일반화와 prospective replication이 더 필요하다.

- [bioRxiv: Virtual Phenotypic Screening Discovers Novel Scaffolds Inhibiting the PI3K/mTOR Pathway](https://www.biorxiv.org/content/10.64898/2026.06.10.731476v1)
- [JUMP Cell Painting Consortium](https://jump-cellpainting.broadinstitute.org/)

> **한 줄 핵심:** PhenoCompass는 값비싼 세포 이미지 실험을 제한된 anchor로 사용하면서, 그 생물학적 신호를 수십억 화합물의 가상 스크리닝으로 확장한다.

---

## 슬라이드 22. 화학 구조와 풍부한 세포 표현형의 Co-Embedding

### 슬라이드가 말하는 내용

이 슬라이드는 PhenoCompass의 모델 구조를 더 구체적으로 설명한다.

- **Morphology encoder:** Cell Painting 이미지를 세포 형태 표현으로 변환
- **Compound encoder:** molecular graph 또는 fingerprint를 화학 구조 표현으로 변환
- **Joint encoder / contrastive objective:** 같은 화합물에서 얻은 구조와 phenotype은 가깝게, 관계없는 쌍은 멀게 정렬

이렇게 학습한 공통 잠재공간에서는 실험 이미지 하나를 query로 사용해 유사 phenotype을 유도할 구조를 검색할 수 있다.

### 왜 중요한가

화학적 유사성과 생물학적 유사성은 같지 않다.

- 구조가 비슷해도 세포 효과가 다를 수 있다.
- 구조가 매우 달라도 같은 경로를 조절해 비슷한 phenotype을 만들 수 있다.

공통 embedding은 후자의 **scaffold hopping**을 가능하게 한다. 즉, 기존 hit와 다른 구조를 가지면서도 유사한 생물학적 효과를 내는 후보를 찾을 수 있다.

### 추가 조사와 해석

contrastive learning에서 가장 중요한 것은 positive와 negative pair의 정의다. plate batch, 세포 수, imaging artifact가 phenotype embedding에 들어가면 모델이 생물학보다 실험 환경을 맞힐 수 있다. 따라서 다음이 중요하다.

- replicate-aware sampling
- plate와 site batch correction
- control normalization
- hard-negative 설계
- 구조 유사성에만 의존하지 않는 prospective 평가

PhenoCompass와 연관된 연구에서는 morphology-guided GFlowNet을 이용해 target image와 유사한 표현형을 만들 신규 분자를 생성하는 방향도 제시됐다.

- [arXiv: Cell Morphology-Guided Small Molecule Generation with GFlowNets](https://arxiv.org/abs/2408.05196)

> **한 줄 핵심:** co-embedding은 분자의 모양이 아니라 분자가 만들어 내는 세포 상태를 기준으로 화학 공간을 탐색하게 한다.

---

## 슬라이드 23. Large Molecule 설계·최적화를 위한 Lab-in-the-Loop

### 슬라이드가 말하는 내용

이 슬라이드는 항체와 같은 large molecule에 Lab-in-the-Loop를 적용하는 일반 구조를 보여준다.

```text
기존 항체/단백질 서열
→ AI/ML 모델
→ 새로운 설계
→ 후보 선택
→ 최적화·합성
→ 결합/기능 실험
→ 결과를 모델에 재학습
```

소분자와 달리 large molecule은 서열 공간이 매우 크고, 구조·결합·발현·응집·면역원성 등 여러 특성을 함께 최적화해야 한다.

### 왜 중요한가

항체 설계에서 affinity만 높이는 것은 충분하지 않다. 후보는 다음을 동시에 만족해야 한다.

- 표적 epitope에 대한 높은 결합력
- 관련 단백질에 대한 선택성
- 세포 기반 기능 활성
- 안정성·용해도·낮은 응집
- 생산 세포에서 높은 발현
- 낮은 면역원성
- 원하는 species cross-reactivity

각 평가에는 다른 assay가 필요하므로, 실험 결과를 단계적으로 모델에 반영하는 것이 필수다.

### 추가 조사와 해석

large molecule LiTL에서 중요한 것은 실험의 양보다 **어떤 sequence를 다음 round에 측정할지**다. 모델은 가장 높은 점수만 고르는 대신, 높은 성능 후보와 불확실성이 큰 후보를 섞어 실험해야 한다. 그래야 exploitation과 exploration의 균형을 잡고 local optimum에 갇히지 않는다.

> **한 줄 핵심:** 항체 AI는 한 번에 완성된 항체를 만드는 기술보다, 각 실험 round에서 더 좋은 서열을 선택해 진화시키는 기술에 가깝다.

---

## 슬라이드 24. De Novo Antibody–Antigen Design

### 슬라이드가 말하는 내용

이 슬라이드는 항원 구조를 조건으로 새로운 항체 결합부위, 특히 **CDR H3 loop**를 생성하는 구조 기반 diffusion 모델을 보여준다.

도식은 다음 과정을 나타낸다.

1. 표적 항원의 구조와 epitope 정의
2. 무작위 초기 구조 또는 noise에서 시작
3. latent diffusion으로 항체 결합부위를 반복적으로 denoise
4. 항원 표면과 상보적인 CDR 구조 생성
5. 구조 예측과 실험으로 결합 확인

### 왜 중요한가

기존 항체 발굴은 동물 면역, phage display, B-cell cloning에 크게 의존한다. de novo design이 가능해지면 자연 면역 레퍼토리에 드물거나 접근하기 어려운 epitope도 탐색할 수 있다.

특히 CDR H3는 항체 결합 특이성에 큰 영향을 주지만 길이와 구조 다양성이 커서 예측과 설계가 어렵다. 생성 모델은 항원 표면의 형태와 화학적 특성을 조건으로 이 공간을 탐색한다.

### 추가 조사와 해석

공개 연구에서는 SE(3)-equivariant diffusion, backbone generation, sequence design, co-folding을 결합한 항체 설계가 빠르게 발전하고 있다. 그러나 구조 모델의 높은 confidence가 실제 결합을 보장하지 않으며, de novo 항체의 실험 hit rate는 표적과 평가 기준에 크게 달라진다.

슬라이드에 적힌 저자·학회 표기를 공개 자료와 완전히 일치하는 하나의 논문으로 확정하지 못했으므로, 이 부분은 특정 논문의 성능을 단정하기보다 **항원 조건부 CDR 생성이라는 기술 방향**을 설명한 것으로 정리한다.

- [arXiv: De Novo Antibody Design with SE(3) Diffusion](https://arxiv.org/abs/2405.07622)
- [NeurIPS 2024: Antigen-Specific Antibody Design](https://proceedings.neurips.cc/paper_files/paper/2024/hash/daef77101ba5711084a57442c8cf2709-Abstract-Conference.html)

> **한 줄 핵심:** de novo 항체 설계는 항원의 3차원 표면을 조건으로 결합 가능한 CDR 구조와 서열을 처음부터 생성하려는 접근이다.

---

## 슬라이드 25. 이중·다중특이 항체를 위한 Robust Building Blocks

### 슬라이드가 말하는 내용

이 슬라이드는 여러 표적에 동시에 결합하는 bi-specific 또는 multi-specific 항체를 만들기 위해, 먼저 **재사용 가능한 고품질 항체 arm**을 구축하는 과정을 보여준다.

- rat, mouse, rabbit, llama 등 다양한 면역 레퍼토리 사용
- 여러 종에서 교차반응하는 높은 affinity binder 선택
- 기능적 potency가 있는 후보 확인
- lead와 backup arm 확보
- 고처리량 실험과 계산을 통한 sequence optimization
- developability risk를 고려한 최종 후보 선택

### 왜 중요한가

다중특이 항체는 arm 하나가 좋다고 전체 분자가 좋은 것이 아니다. arm을 결합하면 다음 문제가 새롭게 발생한다.

- 서로 다른 arm 간 간섭
- 비대칭 조립 오류
- avidity와 geometry 변화
- 응집과 점도 증가
- 예상치 못한 cytokine activation
- 생산 수율 저하

따라서 각 arm이 affinity뿐 아니라 구조적 안정성과 조합 가능성을 가져야 한다.

### 추가 조사와 해석

이 슬라이드는 AI의 역할을 sequence ranking에 한정하지 않는다. 다양한 동물·실험 플랫폼에서 생성된 서로 다른 데이터 분포를 통합하고, 여러 목표를 동시에 최적화하며, 후속 조합에서 실패할 가능성을 조기에 예측해야 한다.

실제 모델에서는 affinity, specificity, expression, melting temperature, hydrophobicity, immunogenicity 등의 multi-task prediction과 Pareto optimization이 유용하다.

> **한 줄 핵심:** 다중특이 항체의 성공은 최종 조립 단계보다, 처음부터 조합 가능하고 개발성이 높은 항체 arm을 확보하는 데 달려 있다.

---

## 슬라이드 26. Novelty 탐색에서 통계적 보장이 유용한가?

### 슬라이드가 말하는 내용

이 슬라이드는 AI가 학습 데이터보다 우수하고 새로운 분자나 단백질을 제안할 때, 예측 신뢰도를 어떻게 보장할 수 있는지 묻는다.

오른쪽에는 세 가지 개념이 나온다.

- **Conformal prediction for design**
- **Reliable design algorithm selection**
- **Conformal policy control**

왼쪽 도식은 모델이 높은 점수로 예측한 새로운 영역이 실제로는 학습 분포에서 멀어 오류가 커질 수 있음을 보여준다.

### 왜 중요한가

신약 설계에서는 모델이 바로 그 모델의 예측값을 이용해 다음 데이터를 선택한다. 따라서 일반적인 독립·동일분포 테스트 가정이 깨진다.

```text
모델이 높은 점수라고 예측
→ 그 후보만 선택
→ 선택된 후보는 원래 데이터 분포와 다름
→ 예측 오차가 과소평가될 수 있음
```

특히 novelty를 높일수록 분포 이동이 커지므로, 높은 예측 점수와 높은 실제 성공률 사이의 간극이 커질 수 있다.

### 추가 조사와 해석

Conformal prediction은 교정 데이터에 기반해 예측 집합이나 신뢰 구간을 만들지만, 설계 문제에서는 테스트 후보가 모델에 의해 적응적으로 선택된다는 점을 고려해야 한다. 관련 연구는 이러한 의존성을 반영한 불확실성 추정과, 성공 기준을 만족할 가능성이 높은 설계 알고리즘 선택 방법을 제안한다.

통계적 보장은 모델의 생물학적 타당성을 증명하는 것이 아니라, **정해진 가정 아래 예측 실패 위험을 정량화하는 도구**다. 분포 이동이 크고 calibration data가 부적절하면 보장도 약해진다.

- [arXiv: Conformal Prediction for the Design Problem](https://arxiv.org/abs/2202.03613)
- [ICML 2025: Reliable Algorithm Selection for Machine Learning-Guided Design](https://proceedings.mlr.press/v267/fannjiang25a.html)

> **한 줄 핵심:** 가장 새롭고 높은 점수의 후보일수록 모델이 가장 자신 없어야 할 수 있으므로, novelty 탐색에는 명시적 불확실성 관리가 필요하다.

---

## 슬라이드 27. Patient Selection, Biomarkers, Covariates, Personalized Medicine

### 슬라이드가 말하는 내용

이 슬라이드는 Lab-in-the-Loop의 종착점을 환자 수준으로 확장한다.

- **Patient selection:** 치료가 효과를 낼 환자를 선택
- **Biomarkers:** 치료 반응과 약물 작용을 측정·예측
- **Covariates:** 환자 간 변동을 설명하고 통제
- **Personalized medicine:** 환자별 치료를 최적화

### 왜 중요한가

같은 약물도 모든 환자에게 같은 효과를 내지 않는다. 표적 변이, 세포 상태, 면역환경, 장기 기능, 병용약, 질병 단계가 모두 반응에 영향을 준다.

따라서 약물과 환자를 함께 모델링해야 한다. 임상시험에서 올바른 환자군을 선택하면 효과 신호를 더 명확하게 볼 수 있고, 불필요한 부작용 노출을 줄일 수 있다.

### 추가 조사와 해석

바이오마커에는 서로 다른 역할이 있다.

- **Predictive biomarker:** 어떤 환자가 특정 치료에 더 반응하는지 예측
- **Prognostic biomarker:** 치료와 무관하게 질병 경과를 예측
- **Pharmacodynamic biomarker:** 약물이 표적을 실제로 조절했는지 측정
- **Safety biomarker:** 독성 위험을 조기에 감지

AI가 covariate adjustment를 수행할 때는 사전 규정, 외부 검증, 결측 처리, subgroup fairness가 중요하다. 사후적으로 유리한 환자군만 찾으면 결과가 과대평가될 수 있다.

> **한 줄 핵심:** 정밀의학은 좋은 약을 만드는 것에서 끝나지 않고, 그 약을 받을 올바른 환자를 신뢰성 있게 찾는 과정이다.

---

# Part 5. 디지털 병리, 임상 예측과 개인맞춤 치료

## 슬라이드 28. SCHAF: H&E 이미지에서 Single-Cell Transcriptome 생성

### 슬라이드가 말하는 내용

SCHAF(Single Cell omics from Histology Analysis Framework)는 일상적으로 얻는 H&E 조직 이미지를 입력으로 받아, 공간적으로 배치된 single-cell transcriptomic profile을 생성하려는 모델이다.

슬라이드는 paired와 unpaired 학습을 함께 사용한다.

- H&E 이미지
- sc/snRNA-seq
- 공간전사체
- 세포 segmentation과 cell type 정보
- adversarial 또는 supervised spatial pretraining

이 데이터를 연결해 이미지의 조직 형태로부터 세포별 유전자 발현을 추론한다.

### 왜 중요한가

H&E는 임상에서 널리 사용되고 저렴하지만 분자 정보를 직접 측정하지 않는다. 반면 single-cell RNA-seq와 spatial transcriptomics는 풍부한 분자 정보를 제공하지만 비용이 높고 조직을 소모하며 모든 환자에게 적용하기 어렵다.

이미지에서 분자 상태를 신뢰성 있게 추론할 수 있다면 다음이 가능하다.

- 과거 병리 슬라이드의 재분석
- 공간적 세포 상태와 미세환경 추정
- 환자별 표적과 바이오마커 탐색
- 비싼 오믹스 실험의 우선순위 결정

### 추가 조사와 해석

공개 연구에서 SCHAF는 vision transformer와 adversarial learning을 이용해 H&E에서 공간적으로 해석된 whole-transcriptome single-cell omics를 생성하고, 폐암·전이성 유방암·태반·마우스 조직 등에 적용했다. 다만 생성된 transcriptome은 직접 측정치가 아니라 모델 추정값이며, 희귀 세포·낮은 발현 유전자·새로운 병리 조건에서 오류 가능성이 있다.

따라서 SCHAF는 spatial transcriptomics를 완전히 대체하기보다, 광범위한 H&E를 선별하고 일부 샘플을 실제 오믹스로 검증하는 Lab-in-the-Loop에 적합하다.

- [PubMed: Inference of Single Cell Profiles from Histology Stains with SCHAF](https://pubmed.ncbi.nlm.nih.gov/36993643/)

> **한 줄 핵심:** SCHAF는 값싼 병리 이미지와 비싼 single-cell omics 사이를 연결해, 더 많은 환자 조직에서 분자 정보를 추정하려는 모델이다.

---

## 슬라이드 29. Multimodal Generative Modeling Suite

### 슬라이드가 말하는 내용

이 슬라이드는 EHR, WES, WSI 같은 서로 다른 환자 데이터로부터 RNA-seq를 예측하거나 생성하는 통합 multimodal framework를 보여준다.

- EHR: 임상 정보
- WES: 유전체 변이
- WSI: 병리 whole-slide image
- RNA: 전사체

각 modality는 encoder로 잠재표현으로 변환되고, conditional flow matching 또는 transformer 기반 모델이 이를 조건으로 RNA latent를 생성한다. 일부 modality가 없어도 작동하도록 학습하는 것이 특징으로 제시된다.

### 왜 중요한가

실제 임상 데이터는 모든 환자에게 모든 modality가 존재하지 않는다. 완전한 paired dataset만 요구하면 사용할 수 있는 환자가 급격히 줄고 selection bias가 생긴다.

결측 modality를 허용하는 모델은 다음을 지원할 수 있다.

- 값비싼 RNA-seq가 없는 환자의 분자 상태 추정
- 서로 다른 코호트의 데이터 통합
- 환자 표현의 보완
- 예측에 가장 유용한 다음 검사의 선택

### 추가 조사와 해석

이 슬라이드와 완전히 일치하는 공개 논문·모델명을 확정하지 못했으므로, 구체적 성능을 단정하지 않는다. 기술적으로는 modality dropout, cross-modal attention, product-of-experts, conditional flow matching 등이 결측 modality 대응에 사용될 수 있다.

중요한 한계는 생성된 RNA가 실제 환자 측정값이 아니라는 점이다. 다운스트림 바이오마커나 치료 결정에 사용하려면 외부 기관 검증, calibration, 불확실성 표시가 필요하다.

> **한 줄 핵심:** 임상 멀티모달 AI의 핵심은 모든 데이터를 가진 이상적 환자가 아니라, 일부 데이터가 빠진 현실의 환자에서도 신뢰성 있게 작동하는 것이다.

---

## 슬라이드 30. AI를 이용한 병변 성장 예측과 임상시험 Covariate Adjustment

### 슬라이드가 말하는 내용

이 슬라이드는 안과 영상에서 병변의 향후 성장률을 AI로 예측하고, 이 예측값을 임상시험의 baseline covariate로 사용하는 사례를 보여준다.

흐름은 다음과 같다.

1. baseline image 입력
2. AI가 geographic atrophy lesion growth rate 예측
3. 실제 관측 결과와 비교
4. 예측 성장률로 환자 간 초기 위험 차이를 보정
5. 치료 효과 추정의 정밀도 향상

손글씨에는 AI 보정이 표본 수를 크게 늘린 것과 유사한 통계적 효율을 줄 수 있다는 해석이 적혀 있다.

### 왜 중요한가

임상시험에서는 치료군과 대조군이 무작위 배정되더라도 환자별 질병 진행 속도가 매우 다를 수 있다. baseline 영상에서 진행 속도를 미리 예측하면 이 변동을 설명하여 치료 효과의 표준오차를 줄일 수 있다.

이는 다음과 같은 이점이 있다.

- 같은 환자 수에서 더 높은 검정력
- 더 작은 임상시험 가능성
- 치료 효과 추정의 정밀도 향상
- 환자별 예후와 trial stratification 개선

### 추가 조사와 해석

관련 연구는 fundus autofluorescence와 OCT에서 연간 geographic atrophy 성장률을 예측하는 multimodal deep learning 모델을 개발하고, prognostic covariate adjustment로 임상시험 검정력을 높일 수 있음을 제안했다.

다만 “표본 수가 두 배가 된 것과 동일하다”는 효과는 특정 데이터와 모델 성능에 의존한다. 임상시험에 적용하려면 모델을 trial 시작 전에 고정하고, 독립 데이터에서 검증하며, missingness와 site shift를 관리해야 한다.

- [PubMed: Deep Learning to Predict Geographic Atrophy Area and Growth Rate from Multimodal Imaging](https://pubmed.ncbi.nlm.nih.gov/36038116/)

> **한 줄 핵심:** AI 예후 모델은 치료를 직접 결정하는 것뿐 아니라, 임상시험의 환자 변동을 줄여 더 정밀하게 약효를 측정하는 데도 사용할 수 있다.

---

## 슬라이드 31. Individualized Cancer Neoantigen Vaccine Workflow

### 슬라이드가 말하는 내용

이 슬라이드는 개인맞춤형 암 neoantigen vaccine을 만드는 전체 순환을 보여준다.

1. 환자의 종양 조직과 정상 샘플 확보
2. 종양 돌연변이 분석
3. AI/ML로 MHC에 제시될 가능성이 높은 돌연변이 peptide 예측
4. 면역원성·종양 특이성·발현을 고려해 neoantigen 선택
5. 환자 맞춤형 mRNA vaccine 제조
6. 투여 후 T-cell 반응과 재발 여부 측정
7. 결과를 다음 예측 모델과 백신 설계에 반영

### 왜 중요한가

암세포의 돌연변이는 환자마다 다르므로 모든 환자에게 같은 항원을 사용하는 백신으로는 충분하지 않을 수 있다. neoantigen은 정상 세포에는 없고 종양에만 존재할 가능성이 있어 높은 특이성을 기대할 수 있다.

AI가 필요한 이유는 후보 돌연변이가 많고, 그중 실제로 다음 조건을 만족하는 것은 일부이기 때문이다.

- 종양에서 충분히 발현
- 단백질로 번역
- 환자의 HLA에 결합
- 세포 표면에 presentation
- T-cell이 인식
- 종양 클론의 충분한 비율에 존재

### 추가 조사와 해석

MHC binding prediction은 중요한 단계지만 전체 면역원성의 한 부분일 뿐이다. antigen processing, T-cell receptor repertoire, 종양 미세환경, 면역억제, 종양 heterogeneity가 모두 반응을 좌우한다. 따라서 백신 후보 선정 모델은 실제 면역반응 결과를 지속적으로 학습해야 한다.

> **한 줄 핵심:** 개인맞춤 암 백신은 환자의 종양 서열을 분석해 가장 면역원성이 높은 돌연변이를 골라 제조하는, 임상 수준의 Lab-in-the-Loop다.

---

## 슬라이드 32. 췌장암 Individualized Neoantigen Vaccine 임상 결과

### 슬라이드가 말하는 내용

마지막 슬라이드는 절제 가능한 췌장관선암 환자를 대상으로 한 개인맞춤형 mRNA neoantigen vaccine 연구 결과를 보여준다.

슬라이드의 Kaplan–Meier 곡선은 백신 후 neoantigen-specific T-cell 반응이 확인된 **responders**와 그렇지 않은 **non-responders**의 재발 없는 생존을 비교한다. 발표 자료에서 responders의 중앙 재발무진행생존기간은 도달하지 않았고, non-responders는 13.4개월이었다.

### 왜 중요한가

췌장암은 면역반응이 약하고 종양 돌연변이 부담이 상대적으로 낮아 백신 개발이 어려운 암으로 여겨져 왔다. 그럼에도 환자별 neoantigen을 실시간으로 설계·제조하고, 일부 환자에서 강하고 지속적인 T-cell 반응을 유도했다는 것은 다음을 보여준다.

- 계산적 neoantigen 선택이 실제 제조와 임상 투여로 연결될 수 있음
- 매우 개인화된 치료도 운영적으로 가능할 수 있음
- 면역반응을 바이오마커로 삼아 다음 설계를 개선할 수 있음

### 추가 조사와 해석

Nature의 1상 연구에서는 16명 중 8명에게서 백신 유도 T-cell 반응이 확인되었다. 18개월 중앙 추적에서 responder의 중앙 재발 없는 생존은 도달하지 않았고, non-responder는 13.4개월이었다. 후속 장기 추적에서도 responder에서 지속적인 T-cell clone과 더 긴 재발 없는 생존의 연관성이 보고되었다.

그러나 이 연구는 소규모 1상 시험이며, 환자들은 수술·atezolizumab·mFOLFIRINOX를 함께 받았다. responder 여부는 무작위 배정된 치료군이 아니므로, 곡선 차이가 백신의 인과적 효능을 확정하는 것은 아니다. 더 큰 무작위 임상시험이 필요하다.

- [Nature 2023: Personalized RNA Neoantigen Vaccines Stimulate T Cells in Pancreatic Cancer](https://www.nature.com/articles/s41586-023-06063-y)
- [Nature: RNA Neoantigen Vaccines Prime Long-Lived CD8+ T Cells in Pancreatic Cancer](https://www.nature.com/articles/s41586-024-08508-4)

> **한 줄 핵심:** 개인맞춤 백신은 실현 가능성과 면역원성을 보여주었지만, 임상 효능을 확정하려면 더 큰 통제 임상시험이 필요하다.

---

# 발표 전체에서 도출되는 Lab-in-the-Loop의 성공 조건

## 1. 상관관계 데이터보다 인과적 perturbation 데이터가 중요하다

질병 상태를 관찰하는 데이터는 “무엇이 함께 변했는가”를 보여주지만, 치료 가능한 원인을 확정하지는 못한다. CRISPR, 화합물, cytokine, dose, time-course를 체계적으로 변화시킨 데이터가 필요하다.

## 2. 모델의 성능은 실험 hit rate로 평가해야 한다

AUROC, embedding similarity, docking score는 중간 지표다. 실제로는 같은 예산에서 얼마나 많은 검증 hit를 얻었는지, 얼마나 새로운 scaffold를 찾았는지, 후기 실패를 얼마나 줄였는지가 중요하다.

## 3. 멀티모달 통합은 단순 concatenation이 아니다

서열·구조·이미지·전사체·임상기록은 노이즈와 해상도가 다르다. 각 modality의 고유 정보와 batch를 보존하면서 공통 생물학적 축을 정렬해야 한다.

## 4. 가장 새로운 후보일수록 불확실성이 커진다

AI가 설계한 후보는 의도적으로 학습 데이터 밖으로 이동한다. 따라서 OOD detection, conformal uncertainty, ensemble, diversity-aware selection이 필요하다.

## 5. 음성 결과와 실험 메타데이터가 학습 자산이다

실패한 합성, 결합하지 않은 분자, 독성이 나타난 조건, batch와 protocol 차이를 저장해야 모델이 같은 실수를 반복하지 않는다.

## 6. AI Agent에는 검증 가능성과 책임 구조가 필요하다

도구 선택, 데이터 provenance, 코드 버전, 통계 가정, 사람이 승인한 단계가 모두 추적되어야 한다. 에이전트의 자연스러운 설명보다 재현 가능한 기록이 중요하다.

## 7. 최종 목적은 환자 수준의 의사결정이다

표적·분자·세포 모델에서 좋은 결과가 나와도 환자 선택, 바이오마커, 안전성, 임상 endpoint와 연결되지 않으면 신약개발로 완성되지 않는다.

---

# 핵심 참고자료

1. [Inavolisib-Based Therapy in PIK3CA-Mutated Advanced Breast Cancer](https://www.nejm.org/doi/10.1056/NEJMoa2404625)
2. [Single-Agent Divarasib in KRAS G12C-Mutated Solid Tumors](https://www.nejm.org/doi/10.1056/NEJMoa2303810)
3. [Decima: Decoding Sequence Determinants of Gene Expression](https://www.biorxiv.org/content/10.1101/2024.10.09.617507v1)
4. [Perturbation Cell and Tissue Atlas](https://doi.org/10.1016/j.cell.2024.07.035)
5. [Biomni: A General-Purpose Biomedical AI Agent](https://www.biorxiv.org/content/10.1101/2025.05.30.656746v1)
6. [SIGnature: Scoring Gene Importance with Single-Cell Foundation Models](https://www.nature.com/articles/s41587-026-03112-5)
7. [VoxBind: Structure-Based Drug Design by Denoising Voxel Grids](https://proceedings.mlr.press/v235/pinheiro24a.html)
8. [GNEprop: Deep-Learning-Based Virtual Screening of Antibacterial Compounds](https://www.nature.com/articles/s41587-025-02814-6)
9. [PhenoCompass: Virtual Phenotypic Screening](https://www.biorxiv.org/content/10.64898/2026.06.10.731476v1)
10. [Conformal Prediction for the Design Problem](https://arxiv.org/abs/2202.03613)
11. [SCHAF: Single-Cell Profiles from Histology](https://pubmed.ncbi.nlm.nih.gov/36993643/)
12. [Deep Learning Prediction of Geographic Atrophy Growth](https://pubmed.ncbi.nlm.nih.gov/36038116/)
13. [Personalized RNA Neoantigen Vaccines in Pancreatic Cancer](https://www.nature.com/articles/s41586-023-06063-y)

---

# 최종 요약

이 컨퍼런스의 핵심은 <b>“AI가 신약을 단번에 만들어 낸다”가 아니라 “AI와 실험이 서로의 약점을 반복적으로 보완한다”</b>는 것이다.

- AI는 거대한 탐색 공간을 압축한다.
- 실험은 AI의 예측을 현실에 맞게 교정한다.
- 멀티모달 데이터는 분자에서 환자까지의 연결을 만든다.
- 반복 루프는 표적, 분자, 항체, 바이오마커와 환자 선택을 점점 개선한다.
- 성공 여부는 모델 점수보다 실제 실험과 임상 결과로 판단한다.

따라서 Lab-in-the-Loop for Drug R&D with AI는 특정 모델 하나의 이름이 아니라, **데이터–모델–실험–환자를 하나의 학습 시스템으로 만드는 연구개발 운영 방식**이라고 이해하는 것이 가장 정확하다.
