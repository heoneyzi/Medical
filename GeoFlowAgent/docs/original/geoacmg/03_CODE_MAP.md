> 📜 **Original document** — written during the experiments and kept as written (machine paths replaced by `$GEOFLOW_*` placeholders). Final numbers and conclusions: [연구 안내](../../README.md) · [주장과 근거](../../research/04_CLAIMS_AND_EVIDENCE.md).

# 코드 구조와 확장 지점

## 의존 방향이 한쪽으로만 흐릅니다

```
┌─ 도메인 층 ────────────────────────────────────────────┐
│  cited        출처를 가진 상수            ← 아무것도 모름 │
│  evidence     ACMG 기준·강도·Tavtigian 점수 ← claim 모름  │
│  clingen      Evidence Repository 파싱 (fail-loud)     │
│  toolmap      기준 ↔ 도구, 판정 불가능한 기준의 선언     │
│  tasks        curated variant → task, 충분성 규칙       │
│  pairs        최소쌍 (표기 불변 / 반대 분류 / 대조)      │
│  splits       3축 분할 (변이 · 유전자 · 시간)           │
│  diagnostics  벤치마크에 계획 문제가 있는가              │
└────────────────────────────────────────────────────────┘
┌─ 실험 층 ──────────────────────────────────────────────┐
│  probes       RQ1 — 정보 · 정렬 · 생물학                │
│  ladder       RQ2 — k-커밋, blind 통제, 비용 회계        │
│  runners      산출물 ↔ 실험 배선                        │
│  pipeline     11단계 오케스트레이터                      │
└────────────────────────────────────────────────────────┘
┌─ 분석 층 ──────────────────────────────────────────────┐
│  estimators   구간·검정 (행이 아니라 유전자를 재표집)     │
│  inference    estimator → Finding                      │
│  claims       Claim / Finding / Verdict / Prereg ← 유전학 모름 │
│  report       판정 → REPORT.md                         │
└────────────────────────────────────────────────────────┘
```

**도메인 층과 분석 층이 서로를 모릅니다.** `evidence.py`에 "claim"이라는 단어가 없고,
`claims.py`에 유전자가 없습니다. 둘을 잇는 건 `inference.py` 한 층뿐입니다. 그래서 ACMG 점수 규칙이
틀렸는지와 신뢰구간 계산이 틀렸는지를 따로 테스트할 수 있습니다.

---

## 논리를 타입이 강제합니다

세 가지가 코드 수준에서 막힙니다.

```python
>>> assert_decision_safe(convention(0.5, "적당해 보여서"), "게이트")
ValueError: 게이트 would let a convention decide an outcome: '적당해 보여서'=0.5.
            Use a guideline citation or a measured estimate, or compare against
            a null distribution instead.

>>> prereg.enforce_roles([Finding(..., name="나중에_찾은_효과", role=Role.PRIMARY)])
ValueError: finding '나중에_찾은_효과' claims primary role but was not pre-registered

>>> paired_contrast(left=[1,2,3], right=[0,1,2], clusters=["g1"])
ValueError: paired contrast needs one value per side per unit and a cluster label
            for each; got 3, 3, 1
```

판정기가 내리는 결론은 넷뿐입니다.

| 관측 | 결론 |
|---|---|
| CI가 0을 배제, 예상 방향 | **SUPPORTED** |
| CI가 0을 배제, 반대 방향 | **REFUTED** |
| CI가 0을 포함 | **UNRESOLVED** — "경향성"으로 격하 불가 |
| 주검정 finding 없음 | **NOT_TESTED** |

---

## 벤치마크가 잘 정의됐다는 네 가지 성질

전부 테스트로 고정되어 있습니다.

| 성질 | 왜 중요한가 | 어디서 |
|---|---|---|
| **증거 없이 커밋하면 검증기 실패** | 정답 라벨만 찍는 퇴화 정책 차단 | `tasks.private_verifier` |
| **PVS1은 호출 두 번** | consequence만으론 0점, gene mechanism까지 알아야 8점 | `toolmap.CriterionBinding.requires` |
| **호출 3번 중 1번은 0점** | 어느 도구가 값을 할지 미리 모름 (`none_met` 34.1%) | `tasks.snapshots_for` |
| **필요 도구군 ⊊ 사용 가능 도구군** | 버릴 줄 아는 것이 결정 | `diagnostics.decision_freedom` |

---

## 확장 지점

새 실험을 붙일 때 건드릴 곳입니다.

| 하고 싶은 것 | 어디에 | 무엇을 반환 |
|---|---|---|
| 새 RQ1 측정 | `probes.py` | `Finding` (role=EXPLORATORY 기본) |
| 새 planner rung | `ladder.py` — `PlanSource` 프로토콜 구현 | `propose(state, candidates, ctx) → [tool_id]` |
| 학습된 모델을 사다리에 | `runners.ladder_findings`의 `rungs` dict | torch는 `runners`에만 들어감 |
| 새 단계 | `pipeline.STAGES`에 `Stage(...)` 추가 | `outputs`를 반드시 선언 (재개에 필요) |
| 새 주장 | `cli._claims()` + `cli._primary_findings()` | 사전등록 fingerprint가 바뀜 |

**torch는 `runners.py`에만 import 됩니다.** `probes`와 `ladder`는 GPU 없이 테스트됩니다.

---

## PlanSource — 사다리에 모델을 붙이는 방법

```python
class MyFlowPlanner:
    name = "flow"
    evaluations_per_call = 16        # NFE. 예산에 기록되지만 도구 비용엔 안 들어감

    def propose(self, state, candidates, context) -> list[str]:
        anchors = self.model.sample(encode(state), nfe=16)
        return [nearest_tool(a, candidates) for a in anchors]
```

이게 전부입니다. `run_episode`가 커밋 길이·blind 통제·예산 회계를 전부 처리합니다.
탐욕도 같은 인터페이스로, 길이 1짜리 계획을 내놓는 planner일 뿐입니다 — **그래서 비교가 공정합니다.**

---

## 비용 회계

세 통화를 따로 기록합니다. 섞이지 않습니다.

```python
Budget(tool_cost=...,          # ← 주 축. 이 도메인에서 지배적
       calls=...,
       plan_evaluations=...)   # ← NFE / forward pass
```

모든 rung을 **같은 `tool_cost` 예산에서 잘라** 비교하고 곡선 전체를 보고합니다.
단일 점 비교는 하지 않습니다.
