# GeoFlowAgent hard-v2 최종 결론

- 최종 평가 상태: 완료
- 독립 test task: 48
- test 재학습·사후 조정: 없음
- 범위: synthetic exact-snapshot benchmark에 한정

## 최종 결과

### 1. Frozen representation

최종 선택인 `MedCPT-only + cosine + 2× head`의 3-seed test 평균은 다음과 같다.

| 지표 | MedCPT-only 3-seed 평균 | 원 사전등록 full+DAGGER | 차이와 95% task CI |
|---|---:|---:|---:|
| optimal-set policy accuracy | 0.8010 | 0.6875 | +0.1198 [+0.0708,+0.1682] |
| joint STOP/action accuracy | 0.7364 | 0.6702 | +0.0691 [+0.0153,+0.1231] |
| regret@1 | 0.1403 | 0.3848 | -0.2489 [-0.3263,-0.1754] |

결론: frozen representation의 유효성은 지지된다. 다만 모든 view의 단순 결합이 아니라
MedCPT 좌표가 핵심이며, naive multiview fusion은 negative transfer를 일으킨다.

### 2. Closed-loop agent

MedCPT 최종 정책의 성공률은 clean 0.6667, perturbed 0.6667이었다. Clean random은
0.5000으로, 대응 차이는 +0.1667, 95% CI [+0.0417,+0.2917]이다. Perturbed random도
0.6667이어서 차이는 0.0000, CI [-0.1042,+0.1042]다.

원 사전등록 full+DAGGER 정책과의 성공률 차이는 clean +0.0208, perturbed -0.0208이며
두 CI 모두 0을 포함한다. 따라서 MedCPT 모델의 state-level ranking 우위는 명확하지만,
closed-loop 성공률에서 원 모델보다 우월하다고 결론낼 수는 없다. 특히 perturbation
recovery에서 learned policy의 우위는 입증되지 않았다.

### 3. State Flow feedback

3-seed test 평균 goal completion은 다음과 같다.

| 조건 | completion |
|---|---:|
| one-shot | 0.1233 |
| execute-observe-replan | 0.4028 |
| compute-matched blind replan | 0.0000 |
| external-verifier STOP guard | 0.4375 |

Observed replanning의 평균 이득은 one-shot 대비 +0.2795, blind 대비 +0.4028이다.
세 seed 모두 같은 방향이며 각 seed의 task-bootstrap CI가 0을 벗어난다.

결론: 반복 호출 자체가 아니라 successor observation을 받아 재계획하는 feedback이
성능 향상의 원인이라는 가설은 강하게 지지된다.

## 최종 판정

1. **채택:** MedCPT frozen coordinate에는 실행 가능한 workflow 행동 정보가 있다.
2. **수정 채택:** 작은 learned geometry는 유효하지만 복잡한 비대칭·hyperbolic 거리는
   필요하지 않았다. 단순 cosine이 최종 선택이다.
3. **채택:** execute-observe-replan은 one-shot과 compute-matched blind replanning보다 낫다.
4. **기각/미입증:** naive multiview fusion의 이점, DAgger의 명확한 이점, perturbation
   recovery에서 learned policy의 우위는 입증되지 않았다.
5. **범위 제한:** 이 결과는 synthetic algorithmic evidence다. 실제 biomedical/live-API
   일반화는 아직 결론낼 수 없다.

최종적으로 GeoFlowAgent의 가장 강한 근거는 “복잡한 geometry”가 아니라
`선택된 frozen biomedical representation + 단순 cosine + 관측 기반 재계획`이다.
