# EXP3 — Evo 2 late-stack causal use

EXP3는 EXP2의 **후속 활용 연구**다. EXP2에서 확정한 late-stack 구조를 고정된 출발점으로 받아, 그 관점을 실제 예측·역추적·motif/variant 규명·benchmark 설명·오류 진단·선택적 복구·압축·steering에 쓰는 방법을 구축하고 검증한다. 따라서 EXP3의 성공 기준은 EXP2를 재증명하는 것이 아니라, EXP2의 관점이 새로운 표본과 과제에서 유용한 개입과 사전 예측을 만들어 내는지 보이는 것이다.

EXP2에서 받아들이는 전제는 다음 다섯 가지다.

1. b28은 출력에 필요한 방향성 causal content를 쓰는 writer/transform 단계다.
2. b30은 그 content를 출력 형식으로 재부호화·통합하는 re-encoder다.
3. x31은 late-stack 결과가 최종 readout에 노출되는 readout gate다.
4. raw norm 자체보다 방향과 carrier에 대한 상대적 기하가 중요하다.
5. 이 해석은 GDTR, TDIG, After_GDTR, Handoff와 모순되지 않는 하나의 관점이다.

이 다섯 명제는 EXP3의 결과에 따라 뒤집히는 가설이 아니다. `Exp2Handoff`에 `true`로 명시되고, 해시가 봉인된 입력으로 전달된다. EXP3가 새로 묻는 것은 다음이다.

- m28의 어떤 입력 조건이 이미 확정된 b28 writer의 causal write를 준비하는가?
- b29는 b28이 쓴 방향을 키우는 amplifier인가, 새로운 정보를 쓰는 second writer/editor인가?
- scale 효과는 `q = alpha ||g|| / ||host||`, 즉 `alpha / alpha*`로 정규화하면 공통 경계로 정렬되는가?
- scale, content, carrier는 late stack에서 서로 다른 인과적 역할을 갖는가?
- 출력에서 b30, b29, b28, 입력 서열로 역추적한 후보를 motif/variant intervention으로 재현할 수 있는가?
- 이 구조가 benchmark 차이, 오류 진단, 선택적 복구, 압축, steering, 외부 assay 예측에 실제로 유용한가?

## 디렉터리 경계

EXP3는 `/path/to/EXP3`에 독립적으로 존재한다. EXP2의 Step 1–10 구현을 복사하거나 다시 실행하지 않는다. EXP2에서는 아래의 일반 API와 봉인된 산출물만 읽는다.

- `exp2.naming`
- `exp2.endpoints`
- `exp2.manifest`
- `exp2.stats`
- `exp2.taps`

EXP3 내부의 유일한 EXP2 import 경계는 `exp3/exp2_api.py`다. `exp2.step*`를 직접 import하는 것은 허용하지 않는다.

## 연구 흐름

| Phase | 새로 답할 질문 | 핵심 모듈 |
|---|---|---|
| 1 | m28 preparation, b29 subtype, scale 경계는 무엇인가? | `mechanism.py`, `bilinear_pairing.py` |
| 2 | scale·content·carrier의 인과적 역할은 어떻게 분리되는가? | `causal_use.py` |
| 3 | 출력 효과를 입력 서열까지 역추적할 수 있는가? | `reverse_trace.py` |
| 4 | 후보 motif/variant가 실제 causal chain을 재현하는가? | `motifs.py` |
| 5 | 관찰된 causal delta를 학습 모델이 예측하고 rescue할 수 있는가? | `learned_transport.py` |
| 6 | 이 기전을 benchmark 설명과 실용 작업에 쓸 수 있는가? | `applications.py` |
| 7 | chromosome·task·checkpoint·training 시점이 바뀌어도 유지되는가? | `generalization.py` |
| 8 | 원시 인과 결과가 확보된 뒤 SAE/transcoder가 해석을 압축하는가? | `optional_features.py` |

상세한 가설, 비교 조건, 판정과 대안 분기는 [RESEARCH_PROGRAM.md](RESEARCH_PROGRAM.md)에 있다. 실제 실행 순서는 [RUNBOOK.md](RUNBOOK.md)를 따른다.

## 설치와 import

현재 진행 중인 EXP2/Evo 2 환경을 바꾸지 않는다. EXP3의 최소 추가 의존성만 설치한다.

```bash
cd '/path/to/EXP3'
python -m pip install -r requirements-core.txt
export PYTHONPATH="/path/to/exp2:/path/to/EXP3:${PYTHONPATH:-}"
```

실제 checkpoint 실행은 기존 Vortex/Evo 2 환경을 그대로 사용한다. `requirements-core.txt`는 그 환경을 재설치하거나 버전을 덮어쓰지 않는다.

## 설정 파일 만들기

예시 설정을 복사한 뒤 연구별 메모만 수정한다. alpha dose, chromosome 역할, rank grid처럼 결과에 따라 바뀌면 안 되는 값은 locked 결과를 보기 전에 고정한다.

```bash
cd '/path/to/EXP3'
cp configs/exp3.example.json configs/exp3.run.json
PYTHONPATH="/path/to/exp2:/path/to/EXP3" \
python - <<'PY'
from exp3.config import Exp3Config

cfg = Exp3Config.load("configs/exp3.run.json")
cfg.save("configs/exp3.run.sealed.json")
print(cfg.config_sha256)
PY
```

`Exp3Config`는 내용의 SHA-256을 계산한다. 저장 후 값을 몰래 바꾸면 다시 load할 때 실패한다.

## Phase 3–8 confirmatory analysis plan 봉인

적응형 분기는 결과에 따라 **어느 분석을 실행할지**만 고른다. locked split을 본 뒤 그 분석의 threshold, hyperparameter, control 또는 decision rule까지 고르는 것은 허용하지 않는다. 따라서 scientific run 전에 가능한 Phase 3–8 node 전부를 `ConfirmatoryAnalysisPlan`에 선언한다. 실행되지 않은 branch의 선언은 그대로 남는다.

먼저 아래 명령으로 필요한 experiment id를 확인한다.

```bash
python - <<'PY'
from exp3.adaptive import build_exp3_research_router
from exp3.analysis_plan import required_analysis_experiment_ids

for name in required_analysis_experiment_ids(build_exp3_research_router()):
    print(name)
PY
```

실행 가능한 전체 예시는 다음처럼 복사해 시작한다.

```bash
cp configs/confirmatory_designs.example.json configs/confirmatory_designs.json
```

그 뒤 각 id를 key로 하는 `configs/confirmatory_designs.json`을 locked data를 열기 전에 연구 설계에 맞게 수정·확정한다. 각 값에는 `estimand`, `decision_rule`, `thresholds`, `hyperparameters`, `control_families`, `control_plan`, `data_roles`, 선택적 `input_artifact_sha256`을 넣는다. 진짜 parameter-free 분석이면 빈 mapping을 숨기지 말고 `no_thresholds_reason` 또는 `no_hyperparameters_reason`을 명시한다. 예시의 수치와 설명은 시작점일 뿐이며 실제 데이터를 보기 전에 확정해야 한다. 다음 코드는 모든 branch가 빠짐없이 선언되었을 때만 plan을 봉인한다.

```bash
python - <<'PY'
import json
from exp3.adaptive import build_exp3_research_router
from exp3.analysis_plan import (
    AnalysisDeclaration,
    ConfirmatoryAnalysisPlan,
    required_analysis_experiment_ids,
)
from exp3.config import Exp3Config

cfg = Exp3Config.load("configs/exp3.run.sealed.json")
with open("configs/confirmatory_designs.json", encoding="utf-8") as stream:
    designs = json.load(stream)
required = required_analysis_experiment_ids(build_exp3_research_router())
if set(designs) != set(required):
    raise RuntimeError(
        f"design keys differ: missing={sorted(set(required) - set(designs))}, "
        f"extra={sorted(set(designs) - set(required))}"
    )
plan = ConfirmatoryAnalysisPlan(
    plan_id="exp3-main-confirmatory-v1",
    config_sha256=cfg.config_sha256,
    locked_split=cfg.locked_chromosome,
    declarations=tuple(
        AnalysisDeclaration(experiment_id=name, **designs[name])
        for name in required
    ),
    sealed_before_locked_data=True,
    registration_reference="<timestamped registry/commit, if used>",
)
plan.save("configs/exp3.analysis-plan.sealed.json")
print(plan.plan_sha256)
PY
```

이 SHA-256은 무결성과 run binding을 제공한다. `sealed_before_locked_data=True` 자체는 독립 timestamp 증명이 아니므로, 강한 preregistration이 필요하면 출력 plan hash를 외부 timestamp registry나 immutable commit에도 남긴다.

## EXP2 handoff 만들기

EXP3를 시작하려면 EXP2 결과 전체를 복사하지 말고 필요한 basis, carrier axis, margin, architecture identity만 `FrozenInput`으로 등록한다. 아래 예시는 형식 예시이며 실제 경로와 해시는 완료된 EXP2 산출물로 채워야 한다.

```python
from exp3.inputs import (
    FrozenInput,
    REQUIRED_EXP2_FOUNDATIONS,
    create_handoff,
    sha256_file,
)

def frozen_tensor(name, path, semantic_role):
    return FrozenInput(
        name=name,
        path=path,
        sha256=sha256_file(path),
        kind="tensor",
        source_split="chr22",
        source_role="discovery",
        metadata={
            "semantic_role": semantic_role,
            "selected_before_locked": True,
        },
    )

inputs = {
    "u28": frozen_tensor(
        "u28", "/absolute/path/to/frozen_u28.pt",
        "b28_causal_subspace"),
    "carrier_axis_x31": frozen_tensor(
        "carrier_axis_x31", "/absolute/path/to/carrier_axis_x31.pt",
        "x31_carrier_axis"),
    "b30_subspace": frozen_tensor(
        "b30_subspace", "/absolute/path/to/b30_subspace.pt",
        "b30_mediation_subspace"),
    "reference_directions": frozen_tensor(
        "reference_directions", "/absolute/path/to/reference_directions.pt",
        "late_stack_reference_directions"),
}

handoff = create_handoff(
    handoff_id="exp2-to-exp3-v1",
    exp2_run_id="exp2-final",
    checkpoint="evo2_7b",
    checkpoint_sha256="<64-hex>",
    architecture_sha256="<64-hex>",
    code_sha256="<64-hex>",
    layer_roles={
        "writer": 28,
        "b29_candidate": 29,
        "reencoder": 30,
        "readout_gate": 31,
    },
    accepted_exp2_foundations={name: True for name in REQUIRED_EXP2_FOUNDATIONS},
    inputs=inputs,
    scientific_margins={
        "effect": 0.05,
        "equivalence": 0.02,
        "rescue": 0.10,
        "specificity": 0.05,
    },
)
handoff.save("/absolute/path/to/exp2_handoff.json")
```

locked/external 결과로 선택한 입력은 handoff에 넣을 수 없다. `FrozenInput.verify()`는 파일 존재 여부와 SHA-256을 매번 확인한다.

## EXP3 token panel 봉인

Scientific run은 EXP3 데이터 자체도 고정한다. 현재 Evo 2 tokenizer로 만든
discovery(`chr22`), development(`chr21`), locked(`chr17`) panel을 각각
`TokenPanel.save()`로 schema-v2 디렉터리에 저장한다. `panel.json`은 좌표,
family, dependency key, matched pair, tokenizer fingerprint와 provenance를
봉인하고, `tokens.pt`의 SHA-256도 함께 묶는다.

```python
from exp3.data import TokenPanel

# records는 (TokenUnit, sequence), pairs는 사전에 정한 TokenPair 목록이다.
panel = TokenPanel.from_sequences(
    records,
    tokenizer=currently_loaded_evo2_tokenizer,
    pairs=pairs,
)
panel.save(
    "/absolute/path/to/panels/chr22-discovery",
    panel_name="chr22-discovery",
    source_split="chr22-discovery",
    source_role="discovery",
    created_by="build_exp3_panels_v1",
)
```

같은 방식으로 `development`, `locked` panel을 만든다. 세 panel의 genomic
interval, unit ID, dependency key가 겹치면 실행이 시작되기 전에 거부된다.
기존 디렉터리에 다른 bytes를 덮어쓰는 것도 허용되지 않는다.

## 적응형 실행의 핵심

`build_exp3_research_router()`는 세 개의 Phase 1 실험을 먼저 연다.

```python
from exp3.adaptive import build_exp3_research_router

router = build_exp3_research_router("exp3-main")
print(router.ready_experiments())
# ('b29_decomposition', 'm28_preconditioning', 'scale_plateau')
```

실험 결과는 성공/실패 한 비트가 아니라 다섯 상태 중 하나다.

- `supported`: 사전 정의한 양성 기준을 만족
- `equivalent`: 사전 정의한 동등성 범위 안의 bounded null
- `mixed`: 조건·branch·task에 따라 효과가 다름
- `unresolved`: 신뢰구간이나 식별성이 아직 부족함
- `refuted`: 검증한 가설과 반대되거나 양립하지 않는 결과

이 다섯 상태는 모두 정상적인 과학적 결론이며 해당 노드는 `completed`가 된다. 예를 들어 b29 amplifier가 지지되지 않으면 전체 실험을 멈추지 않고 `second_writer_mapping`으로 간다. 정규화 plateau가 공통으로 맞지 않으면 `site_specific_scale` 또는 `nonlinear_scale_mechanism`으로 간다.

오직 다음만 과학적 판정을 사용할 수 없게 만든다.

- measurement invalidity: NaN, 잘못된 tap, intervention 미적용 등
- provenance invalidity: checkpoint/config/artifact hash 불일치
- leakage: discovery/development/locked 또는 dependency group 누수
- 구현·인프라 오류: 과학 결론으로 바꾸지 않고 `retry_required`로 기록

즉 “예상과 다른 결과”는 실패가 아니다. “그 결과를 믿을 수 없는 상태”만 해당 분석 경로를 중단한다.

## 실제 실행과 재개

기존 Evo 2 loader를 바꾸지 말고, EXP3 experiment id를 받아 해당 panel/verdict를 실행하는 provider factory만 연결한다. factory는 mapping, `run_experiment(context)` 객체, experiment-id별 method 객체, 또는 callable을 반환할 수 있다. Scientific run은 provider 코드까지 묶은 `RuntimeManifest`와 위 confirmatory plan을 둘 다 요구한다.

```bash
export EXP3_PROVIDER_FACTORY=my_exp3_provider:build_provider
python - <<'PY'
from exp3.callbacks import configured_callback
from exp3.inputs import Exp2Handoff
from exp3.run import callback_contract
from exp3.runtime import create_runtime_manifest

handoff = Exp2Handoff.load("/absolute/path/to/exp2_handoff.json")
identity = callback_contract(configured_callback)
manifest = create_runtime_manifest(
    provider_id=str(identity["provider_id"]),
    checkpoint_sha256=handoff.checkpoint_sha256,
    architecture_sha256=handoff.architecture_sha256,
    exp2_code_sha256=handoff.code_sha256,
    provider_source_sha256=str(
        identity.get("provider_source_sha256", identity["source_sha256"]) 
    ),
)
manifest.save("/absolute/path/to/runtime_manifest.json")
print(manifest.manifest_sha256)
PY
```

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
  --run-id exp3-main
```

중단된 실행은 같은 handoff와 config로만 재개된다.

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

`run_contract.json`은 EXP2 run/checkpoint/architecture/code/input hash,
provider/runtime hash, EXP3 config, confirmatory plan, 세 token panel의 metadata와
token bytes hash를 함께 봉인한다. 재개할 때 하나라도 달라지면 이전 결과와 섞지
않고 즉시 거부한다. Provider는 `context.token_panel_refs`에서 exact panel을 읽는다.
Phase 3–8 provider는 실제 사용한 선언을
`context.expected_confirmatory_analysis`에서 읽고, 동일 mapping을
`decision.diagnostics["confirmatory_analysis"]`에 기록해야 한다. 값 하나라도
달라지거나 누락되면 해당 결과는 provenance invalid로 처리된다.

## 테스트

각 파일을 독립적으로 실행한다. 이 방식은 현재의 script형 검증과 `unittest`형 검증을 모두 포함한다.

```bash
cd '/path/to/EXP3'
export PYTHONPATH="/path/to/exp2:/path/to/EXP3:${PYTHONPATH:-}"
for test_file in tests/test_*.py; do
  python "$test_file" || exit 1
done
```

테스트는 toy/model-free 수학 검증이다. 실제 Evo 2 checkpoint의 생물학적 결론을 대신하지 않는다.

## 가장 중요한 해석 규칙

1. EXP3 결과로 EXP2의 b28/b30 구조를 다시 pass/fail하지 않는다.
2. output effect와 block/rescue가 없는 attribution은 후보 생성이다.
3. donor swap 하나만으로 motif를 주장하지 않는다. 역추적, ISM, bilinear contribution 등 독립 후보 생성법을 합의시킨다.
4. chromosome은 단순 데이터 라벨이 아니라 selection leakage를 막는 역할로 나눈다.
5. SAE 재구성률은 기전 증거가 아니다. held-out delta prediction, feature ablation, decoded rescue까지 필요하다.
6. benchmark 차이는 nuisance가 아니라 scale·content·carrier 사용 비율이 다른 현상으로 직접 설명한다.
