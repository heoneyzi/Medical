# EXP3 실행 Runbook

이 문서는 실제 실행 순서와 산출물 규칙을 정리한다. EXP2의 Step 1–10은 여기서 실행하지 않는다.

## 1. 환경 확인

```bash
cd '/path/to/EXP3'
export PYTHONPATH="/path/to/exp2:/path/to/EXP3:${PYTHONPATH:-}"
python - <<'PY'
import exp2
import exp3
from exp3.config import Exp3Config

print("exp3", exp3.__version__)
print("config", Exp3Config().config_sha256)
PY
```

현재 실험 중인 EXP2 import 경로와 Evo 2/Vortex 환경은 바꾸지 않는다.

## 2. model-free 검증

```bash
cd '/path/to/EXP3'
export PYTHONPATH="/path/to/exp2:/path/to/EXP3:${PYTHONPATH:-}"
for test_file in tests/test_*.py; do
  python "$test_file" || exit 1
done
```

여기서 실패하면 real checkpoint를 열지 않는다. 다만 과학적 비지지와 코드 실패를 혼동하지 않는다. 테스트 실패는 구현 문제이고, 실험 effect가 작은 것은 과학적 결과다.

## 3. 설정 봉인

```bash
cp configs/exp3.example.json configs/exp3.run.json
python - <<'PY'
from exp3.config import Exp3Config

cfg = Exp3Config.load("configs/exp3.run.json")
cfg.save("configs/exp3.run.sealed.json")
print("sealed config:", cfg.config_sha256)
PY
```

확인할 값:

- 21개의 `1e-5..1` alpha dose
- discovery/development/locked chromosome이 서로 다름
- direction/transport rank grid
- required candidate methods ≥ 2
- specificity control families ≥ 3
- `continue_after_scientific_non_support = true`

### 3.1 Phase 3–8 분석 계획 봉인

locked data를 열기 전에 README의 `ConfirmatoryAnalysisPlan` 절차를 실행한다. 다음 검사는 필수다.

- `required_analysis_experiment_ids(build_exp3_research_router())`가 반환하는 모든 node가 정확히 한 번 선언됨
- 각 node의 estimand와 decision rule이 문장으로 고정됨
- threshold와 hyperparameter가 값으로 고정되거나, 정말 없으면 이유가 명시됨
- control family와 동시 실행 방법이 고정됨
- discovery/development/locked 역할 및 이미 존재하는 입력 artifact hash가 고정됨
- 생성된 `plan_sha256`을 locked access 전에 기록함

어떤 branch가 선택될지 모르기 때문에 dormant branch도 모두 미리 선언한다. 이 절차는 음성·mixed·unresolved 결과의 후속 분석을 막지 않는다. 결과가 고르는 것은 미리 선언된 branch이지, 새로운 threshold가 아니다.

## 4. EXP2 handoff 검증

Handoff 생성법은 README의 예시를 따른다. 이미 만든 handoff는 다음처럼 읽는다.

```bash
python - <<'PY'
from exp3.inputs import Exp2Handoff

path = "/absolute/path/to/exp2_handoff.json"
handoff = Exp2Handoff.load(path, verify_files=True)
print("handoff:", handoff.handoff_id)
print("sha256:", handoff.handoff_sha256)
print("roles:", handoff.layer_roles)
print("fixed foundations:", handoff.accepted_exp2_foundations)
PY
```

중단해야 하는 경우:

- artifact 또는 checkpoint hash가 다름
- 다섯 foundation 중 하나라도 누락되거나 `false`
- layer role이 누락되거나 중복
- locked/external artifact가 selection input으로 들어옴
- margin이 음수·비유한값

이는 EXP2 가설을 다시 묻는 것이 아니라, 다른 실행의 결과를 잘못 섞는 것을 막는 무결성 검사다.

### 4.1 EXP3 데이터 panel 검증

README 절차로 discovery(chr22), development(chr21), locked(chr17) token panel을
각각 schema v2로 저장한다. 실행 전 빠른 검증은 다음과 같다.

```bash
python - <<'PY'
from exp3.data import load_token_panel_refs

paths = (
    "/absolute/path/to/panels/chr22-discovery",
    "/absolute/path/to/panels/chr21-development",
    "/absolute/path/to/panels/chr17-locked",
)
for ref in load_token_panel_refs(paths):
    ref.verify()
    print(ref.source_role, ref.source_split, ref.panel_metadata_sha256,
          ref.tokens_sha256)
PY
```

Scientific run에는 세 역할이 모두 필요하다. 좌표·dependency key·unit ID가 split을
가로질러 겹치거나 tokenizer/metadata/token bytes가 바뀌면 시작 또는 callback 전후
검증에서 거부된다.

## 5. run 디렉터리

권장 구조:

```text
runs/<run_id>/
├── run_contract.json
├── adaptive_state.json
├── report.json
├── artifacts/
│   ├── phase1/
│   ├── phase2/
│   └── ...
├── tables/
├── figures/
└── logs/
```

큰 tensor와 table은 router JSON에 직접 넣지 않는다. `ArtifactStore`에 저장하고 hash reference만 decision에 기록한다.

```python
from exp3.artifacts import ArtifactStore

store = ArtifactStore("runs/exp3-main/artifacts")
ref = store.put_tensor(
    "phase1/u28",
    tensor,
    source_split="chr22",
    source_role="discovery",
    created_by="freeze_u28",
    metadata={"rank": int(tensor.shape[0])},
)
print(ref.sha256)
```

## 6. adaptive router 시작

```python
from exp3.adaptive import build_exp3_research_router

router = build_exp3_research_router("exp3-main")
router.save("runs/exp3-main/adaptive_state.json")
print(router.ready_experiments())
```

처음에는 다음 세 개만 ready다.

- `b29_decomposition`
- `m28_preconditioning`
- `scale_plateau`

Router는 실제 GPU 함수를 대신하지 않는다. 각 experiment runner가 scientific
module로 row/table을 만들고 사전 margin으로 판정한 뒤, 명시적
`ValidityReport`를 포함한 `ExperimentOutcome` 또는 기술적 재시도를 위한
`RetryDirective`만 반환해야 한다. Bare `ScientificDecision`은 production에서
허용되지 않는다.

실제 장시간 실행은 기존 loader를 provider로 감싼 뒤 다음처럼 시작한다.

먼저 README의 절차로 provider 코드와 실제 checkpoint를 묶은 `runtime_manifest.json`을 만든다. `callback_contract(configured_callback)`은 `EXP3_PROVIDER_FACTORY`가 가리키는 provider source까지 hash한다.

```bash
export EXP3_PROVIDER_FACTORY=my_exp3_provider:build_provider
python -m exp3.run \
  --handoff /absolute/path/to/exp2_handoff.json \
  --config /absolute/path/to/exp3.run.sealed.json \
  --runtime-manifest /absolute/path/to/runtime_manifest.json \
  --analysis-plan /absolute/path/to/exp3.analysis-plan.sealed.json \
  --token-panel /absolute/path/to/panels/chr22-discovery \
  --token-panel /absolute/path/to/panels/chr21-development \
  --token-panel /absolute/path/to/panels/chr17-locked \
  --output-dir /absolute/path/to/runs \
  --run-id exp3-main
```

실행 시 `run_contract.json`, `adaptive_state.json`, `report.json`이 원자적으로
저장된다. `run_contract.json`은 handoff·checkpoint·architecture·EXP2 code·입력
artifact·provider/runtime·EXP3 config·confirmatory plan과 token-panel
metadata/token hash를 하나로 묶는다. 재개 시 이 계약이 같아야 한다.

```bash
python -m exp3.run \
  --handoff /absolute/path/to/exp2_handoff.json \
  --config /absolute/path/to/exp3.run.sealed.json \
  --runtime-manifest /absolute/path/to/runtime_manifest.json \
  --analysis-plan /absolute/path/to/exp3.analysis-plan.sealed.json \
  --token-panel /absolute/path/to/panels/chr22-discovery \
  --token-panel /absolute/path/to/panels/chr21-development \
  --token-panel /absolute/path/to/panels/chr17-locked \
  --output-dir /absolute/path/to/runs \
  --run-id exp3-main \
  --resume /absolute/path/to/runs/exp3-main/adaptive_state.json
```

같은 `run-id`로 새 실행을 덮어쓰는 것은 금지된다. 이어서 할 때는 `--resume`, 독립 반복이면 새 `run-id`를 사용한다.

## 7. 결과 기록 형식

Provider는 router를 직접 시작하거나 저장하지 않는다. `execute_program`이 전달한 `Exp3RunContext`에서 panel/verdict를 실행하고 명시적 `ValidityReport`가 있는 `ExperimentOutcome`만 반환한다.

```python
from exp3.adaptive import ValidityReport
from exp3.decisions import outcome_from_verdict

def run_experiment(context):
    declaration = context.expected_confirmatory_analysis
    if declaration is not None:
        # 이 값을 실제 분석 함수에 전달한다. 복사만 하고 다른 값을
        # 사용하는 것은 contract 위반이다.
        thresholds = declaration["thresholds_used"]
        hyperparameters = declaration["hyperparameters_used"]
        controls = declaration["control_families_used"]

    verdict, artifact_refs = run_declared_panel(
        context=context,
        thresholds=None if declaration is None else thresholds,
        hyperparameters=None if declaration is None else hyperparameters,
        control_families=None if declaration is None else controls,
    )
    if declaration is not None:
        verdict["confirmatory_analysis"] = dict(declaration)
    return outcome_from_verdict(
        context.experiment_id,
        verdict,
        validity=ValidityReport(),
        config=context.config,
        runtime_provenance=context.expected_runtime_provenance,
        artifact_refs=artifact_refs,
    )
```

Phase 1–2의 module verdict는 `margins_used`와 필요한 dose/site panel을 함께 반환해야 한다. Phase 3–8은 `confirmatory_analysis`가 sealed declaration과 정확히 같아야 한다. `diagnostics`에는 작은 JSON 값만 넣고, tensor나 큰 배열은 `ArtifactStore`에 저장한 뒤 실제 `ArtifactRef` 객체를 넘긴다. 문자열 경로는 `artifact_refs`가 아니다.

### 공식 conclusion code

b29 결과는 가능한 한 다음 code를 사용한다.

- `b29_amplifier`
- `b29_second_writer`
- `b29_editor`
- `b29_mixed_role`
- bounded-null을 나타내는 명확한 lowercase slug
- unresolved 원인을 나타내는 명확한 lowercase slug

scale 결과:

- `shared_normalized_transition`
- `site_specific_transition`
- `model_specific_transition`
- `no_shared_plateau`
- `wide_interval`

synthesis 결과:

- `common_b29_scale_target`
- `branch_specific_b29_scale`
- `mixed_b29_scale`
- `no_common_b29_scale`

Router는 status와 conclusion code를 함께 사용해 더 구체적인 branch를 연다.

## 8. Phase 1 실제 실행 체크리스트

### 8.0 m28 preconditioning

- EXP2의 b28 writer 결론을 다시 판정하지 않음
- `m28=0`에서 b28 factor/product, g28, output을 함께 측정
- exact natural-g28 clamp는 path-isolation 진단으로만 표시
- primary sufficiency는 development에서 만든 held-out predicted g28 delta rescue
- mixed/equivalent/unresolved 결과도 각각 conditional input, bounded null, identifiability 분석으로 기록

### 8.1 b29 decomposition

- [ ] U28 path/hash/rank가 handoff와 일치
- [ ] discovery/development에서만 U28 선택
- [ ] `g29_parallel + g29_perp == g29_delta` 수치 검증
- [ ] parallel/perp remove, only, predicted rescue 모두 존재
- [ ] independent event perpendicular 조건 존재
- [ ] dependency cluster ID 존재
- [ ] primary rescue가 cached actual delta가 아님
- [ ] effect와 equivalence CI를 함께 계산
- [ ] amplifier, writer/editor, mixed, null, unresolved 중 bounded conclusion 기록

관련 API:

- `freeze_subspace`
- `b29_parallel_perp_panel`
- `b29_parallel_plateau_gate`
- `b29_role_verdict`

### 8.2 scale plateau

- [ ] `standard_7b_scale_sites()`의 b28_g, b29_g, b30_m
- [ ] 비교 model 대응 site를 architecture manifest로 확인
- [ ] 21개 alpha dose 누락 없음
- [ ] radial과 equal-norm angular control의 norm 일치
- [ ] `alpha* = ||host|| / ||update||`
- [ ] `q = alpha / alpha*` 계산 검증
- [ ] site별 plateau와 전체 collapse를 분리 보고
- [ ] 공통, site/model-specific, nonlinear, unresolved 중 결론 기록

관련 API:

- `multi_model_log_scale_scan`
- `scale_curve_collapse_gate`

## 9. Phase 1 후 자동 분기 확인

세 초기 결과가 기록되면 각 결과에 맞는 refinement branch가 하나씩 ready여야
한다. 그 세 refinement가 유효한 bounded conclusion을 낸 뒤
`m28_target_spec`, `b29_target_spec`, `scale_target_spec`이 차례로 완성되고,
세 target spec이 모두 기록되어야 `exp3_mechanism_synthesis`가 ready가 된다.

```python
from exp3.adaptive import AdaptiveRouter

router = AdaptiveRouter.load("runs/exp3-main/adaptive_state.json")
print(router.report())
```

예상 분기:

- amplifier → `amplifier_replication`
- writer/editor/mixed/refuted amplifier → `second_writer_mapping`
- b29 equivalent → `b29_null_or_equivalence`
- b29 unresolved → `b29_identifiability`
- shared q → `normalized_scale_transfer`
- heterogeneous/equivalent → `site_specific_scale`
- shared plateau refuted → `nonlinear_scale_mechanism`
- scale unresolved → `scale_identifiability`

여기서 alternative branch를 연 것은 결과에 맞춰 estimand를 바꾼 것이다. locked 결과를 본 뒤 같은 가설의 threshold를 유리하게 바꾼 것이 아니다.

## 10. Phase 2–8 실행 순서

### Phase 2

1. 세 branch-specific target spec을 `exp3_mechanism_synthesis`에서 하나의 operational specification으로 봉인
2. `scale_content_carrier_cube` 실행
3. `cube_factorial_effects`와 `cube_b30_mediation` 계산
4. `bidirectional_direction_transfer`와 control 비교
5. 세 판정을 `causal_factorization_synthesis`에서 공통, 조건부, bounded-null 또는 대안 transport architecture로 봉인
6. 이 합성 판정이 끝난 뒤에만 Phase 3–8 활용 node를 연다

### Phase 3

1. carrier-removed intervention delta로 direction bank 생성
2. rank는 discovery/dev에서만 선택
3. held-out output intervention 확인
4. exact b28 contribution, path-restricted ISM, hierarchical scan 중 최소 두 방법 실행
5. candidate consensus 및 reverse-chain bottleneck 보고

### Phase 4

1. motif/variant와 matched controls 봉인
2. natural, edit, block, predicted rescue panel 실행
3. b28 block→held-out b29 rescue, b29 block→held-out m30 rescue와 역순 실패를
   포함한 `ordered_serial_path_verdict`로 `g28→b29→m30`의 즉시 순서를 검증
4. W1-W2 pairing과 2×2 bilinear grammar 검사
5. context-window saturation 검사

### Phase 5

1. `SplitRecord`마다 ordered `unit_ids`, `source_sha256`, `target_sha256`을 넣어 `TransportProvenance`의 train/dev/locked를 봉인
2. batch 종류에 맞는 commitment helper 사용: intervention은 `intervention_source_sha256`/`intervention_target_sha256`, native factor는 `native_factor_source_sha256`/`native_factor_target_sha256`, lag는 `lag_source_sha256`/`lag_target_sha256`
3. randomized delta design과 native-factor ladder를 구성하고 reduced-rank, spherical, lag, sparse delta model 비교
4. `predict_locked_delta`로 locked prediction 생성. HCL이 아닌 model은 `locked_input`뿐 아니라 source commitment에 포함된 `base`도 반드시 전달
5. 실제 suffix에서 predicted-delta rescue와 dose/rank-matched random, shuffled, wrong-layer, wrong-pair control 실행
6. `RescueRuntimeContract`에 checkpoint, architecture, code, handoff, config, tap, edit, intervention SHA-256을 모두 넣고 `seal_rescue_record`에 전달
7. `predicted_delta_rescue_verdict`에 같은 `expected_runtime_contract`와 `locked_model_base`를 전달하여 prediction 재생성과 receipt를 함께 검증

### Phase 6

1. mechanism fingerprint table 고정
2. leave-family-out benchmark 예측
3. EXP1 case를 stage ontology로 재분석
4. stage debugger와 matched repair
5. evidence가 있는 component에만 compression plan 작성
6. carrier calibration과 sequence steering
7. 외부 assay가 있으면 prediction을 label 공개 전에 hash-seal

### Phase 7

1. untouched chromosome 평가
2. leave-task/family-out 평가
3. 7B↔1B/base alignment를 discovery anchor로 고정
4. wrong-layer/random control과 predicted rescue를 포함한 transfer
5. 가능하면 training checkpoint별 acquisition order 측정

### Phase 8

1. raw causal target이 식별되었는지 확인
2. SAE가 아니라 delta transcoder를 우선 비교
3. one-standard-error rule로 development 복잡도 선택
4. locked delta prediction, feature ablation, decoded rescue
5. random feature, wrong layer, cross-chromosome stability

Phase 8이 열리지 않아도 Phase 1–7은 유효하다.

## 11. 기술 오류와 과학 결과 처리

### 과학 결과

효과 없음, mixed, equivalence, refutation, 넓은 CI는 `ScientificDecision`이다. 이를 예외로 던지거나 run을 종료하지 않는다.

### 유효성 오류

잘못된 측정, provenance, leakage는 `ValidityReport`에서 false로 기록한다. 해당 노드와 그 결과에 의존하는 활성 descendant만 차단한다. 독립 branch는 계속 가능하다.

### 실행 오류

CUDA OOM, temporary I/O, callback 오류는 `RetryDirective`다.

```python
from exp3.adaptive import RetryDirective

retry = RetryDirective(
    reason="CUDA OOM during b29 panel",
    adjustments={
        "microbatch": 1,
        "gradient_checkpointing": False,
        "estimand_changed": False,
    },
)
router.record_retry("b29_decomposition", retry)
router.save("runs/exp3-main/adaptive_state.json")
```

수정 후 명시적으로 재개한다.

```python
router.approve_retry(
    "b29_decomposition",
    note="microbatch만 줄였고 intervention과 estimand는 동일함",
)
```

## 12. 실행 후 감사

각 wave가 끝날 때 다음을 남긴다.

- sealed config SHA-256
- confirmatory analysis plan SHA-256 및 외부 registration reference(있다면)
- runtime manifest/provider source SHA-256
- token-panel metadata/index/tokens SHA-256 및 discovery/development/locked 역할
- EXP2 handoff SHA-256
- checkpoint/architecture/code SHA-256
- router ledger head SHA-256
- artifact SHA-256
- split과 dependency cluster 수
- 사전 margin과 CI
- 모든 조건의 row count와 누락 cell
- scientific status와 conclusion code
- alternative explanation
- 다음 branch가 열린 이유

Router snapshot은 ledger chain과 snapshot hash를 검증한다.

```python
from exp3.adaptive import AdaptiveRouter

router = AdaptiveRouter.load("runs/exp3-main/adaptive_state.json")
router.verify_ledger()
print(router.report()["ledger_head_sha256"])
```

## 13. 논문용 최소 표와 그림

1. EXP2 fixed premise와 EXP3 question boundary 표
2. b29 parallel/perp remove-only-rescue forest plot
3. raw alpha와 normalized q dose curve, angular control 포함
4. scale×content×carrier cube main/interaction effect
5. output→b30→b29→b28→sequence reverse-chain diagram
6. motif/variant block-rescue와 ordered bypass
7. learned delta prediction과 actual predicted rescue
8. benchmark family별 mechanism fingerprint 및 leave-family-out 성능
9. chr17, 1B/base, checkpoint robustness
10. optional SAE는 main claim이 아니라 해석 보조 figure
