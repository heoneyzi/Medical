# EXP3 configuration

`exp3.example.json`은 `exp3.config.Exp3Config`가 직접 읽을 수 있는 예시다.

권장 절차는 다음과 같다.

1. 예시를 `exp3.run.json`으로 복사한다.
2. discovery 데이터를 보기 전에 후보 grid와 seed를 정한다.
3. discovery/개발 뒤 허용된 선택을 마치고, locked chromosome을 열기 전에 `Exp3Config.save()`로 봉인본을 만든다.
4. 산출물마다 `config_sha256`을 함께 기록한다.

기본 설정은 단순 예시가 아니라 Phase 1–2의 최소 confirmatory panel이다.
`alpha_doses`는 정확히 21개의 `1e-5..1` log grid여야 하고,
`required_scale_sites`는 7B의 `b28_g`, `b29_g`, `b30_m`과 비교 모델의
사전 지정 대응 site를 모두 포함해야 한다. effect, equivalence, transition,
plateau, mediation, reciprocal transfer margin은 locked 결과를 열기 전에
봉인한다.

Phase 3–8의 method-specific 선택은 `Exp3Config`에 억지로 합치지 않는다.
README의 `ConfirmatoryAnalysisPlan` 절차로 가능한 모든 adaptive branch의
estimand, decision rule, threshold, hyperparameter, control plan, split 역할을
별도 JSON에 봉인한다. Scientific `exp3.run`은 config hash와 locked chromosome이
일치하고 graph의 Phase 3–8 node를 정확히 모두 포함하는 plan만 받는다.
`confirmatory_designs.example.json`은 이 12개 node를 모두 채운 실행 가능한
출발점이다. 이를 `confirmatory_designs.json`으로 복사한 뒤 연구 설계에 맞게
수정하고, locked data를 열기 전에 봉인한다. 예시 수치를 사후적으로 그대로
채택하는 것은 사전등록을 대신하지 않는다.

역할은 기본적으로 다음과 같다.

- `chr22`: discovery. U28 안정성, 후보 방향, 후보 motif 생성.
- `chr21`: development. rank, dose 부근의 해상도, predictor/SAE 복잡도 선택.
- `chr17`: locked evaluation. 선택이나 threshold 조정에 사용하지 않음.

이미 chr17을 보며 선택을 했다면 chr17은 locked가 아니다. 그 경우 새로운 untouched chromosome을 `locked_chromosome`에 지정해야 한다.

`continue_after_scientific_non_support`는 반드시 `true`다. 이는 분석을 억지로 계속한다는 뜻이 아니라, 유효한 부정·혼합·동등·미결 결과를 각각 명시된 대안 분석으로 연결한다는 뜻이다.
