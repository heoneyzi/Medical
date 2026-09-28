> 📜 **Original document** — written during the experiments and kept as written (machine paths replaced by `$GEOFLOW_*` placeholders). Final numbers and conclusions: [연구 안내](../../README.md) · [주장과 근거](../../research/04_CLAIMS_AND_EVIDENCE.md).

# 실제 medical/genomics 데이터를 연결하는 방법

이 문서는 QA 한 행을 GeoFlowAgent의 학습 가능한 trajectory 한 건으로 바꾸는 최소 절차를 설명한다. 핵심 원칙은 세 가지다.

1. 모델이 보는 목표와 정답 판정 조건을 분리한다.
2. 도구 설명이 아니라 실제 실행 결과를 versioned snapshot으로 보존한다.
3. assembly, accession version, database release처럼 결과를 바꾸는 값은 모두 typed state와 tool argument에 명시한다.

아래 예시는 교육용 가상 데이터다. 실제 환자 정보나 임상 판단에 사용하지 않는다.

## 1. 한 task를 구성하는 파일

### `tools.jsonl`: 실행 계약

```json
{
  "tool_id": "lookup_variant_annotation",
  "name": "Lookup pinned variant annotation",
  "description": "Retrieve a normalized annotation from a reviewed database snapshot.",
  "execution_mode": "snapshot",
  "input_schema": {
    "type": "object",
    "required": ["assembly", "variant_id", "database_release"],
    "properties": {
      "assembly": {"const": "GRCh38"},
      "variant_id": {"type": "string", "minLength": 1},
      "database_release": {"type": "string", "minLength": 1}
    },
    "additionalProperties": false
  },
  "argument_bindings": {
    "assembly": "assembly",
    "variant_id": "selected_variant.normalized_id",
    "database_release": "annotation_database_release"
  },
  "output_schema": {
    "type": "object",
    "required": ["gene_symbol", "clinical_significance"],
    "properties": {
      "gene_symbol": {"type": "string"},
      "clinical_significance": {"type": "string"}
    },
    "additionalProperties": false
  },
  "output_bindings": {
    "gene_symbol": "annotation.gene_symbol",
    "clinical_significance": "annotation.clinical_significance"
  },
  "preconditions": [
    {"field": "assembly", "op": "eq", "value": "GRCh38"},
    {"field": "selected_variant.normalized_id", "op": "nonempty"}
  ],
  "effects": [
    {"field": "annotation_complete", "op": "set", "value": true}
  ],
  "provenance": {
    "wrapper_revision": "YOUR_WRAPPER_COMMIT"
  }
}
```

`argument_bindings`는 `API 인자 이름 → typed-state 경로`다. `output_bindings`는 반대 방향인 `snapshot output 경로 → 다음 typed-state 경로`다. 이 방향은 코드와 JSON Schema에서 고정되어 있다.

`GRCh37`과 `GRCh38`이 텍스트 임베딩에서 가깝더라도 위 계약은 둘을 같다고 취급하지 않는다. 도구 결과를 바꿀 수 있는 build, coordinate convention, transcript/accession version, database release가 하나라도 빠지면 동일한 `tool_id + arguments`가 서로 다른 결과를 뜻하게 되므로 snapshot을 만들기 전에 계약부터 고쳐야 한다.

### `tasks.jsonl`: 모델이 볼 수 있는 문제와 공개 목표

```json
{
  "task_id": "case-0001",
  "query": "Review the normalized variant using the pinned annotation source.",
  "category": "variant_annotation",
  "initial_state": {
    "assembly": "GRCh38",
    "coordinate_system": "one_based_closed",
    "selected_variant": {"normalized_id": "SYNTH_VAR_0001"},
    "annotation_database_release": "synthetic-2026-09",
    "annotation_complete": false
  },
  "goal": {"annotation_complete": true},
  "available_tools": ["lookup_variant_annotation"],
  "background_ids": ["annotation-policy-2026-09"],
  "split": "test",
  "split_group": "synthetic-case-family-0001",
  "provenance": {"source_record": "synthetic-case-0001"}
}
```

공개 `goal`은 “annotation 단계가 끝났는가”처럼 정답을 노출하지 않는 완료 조건이다. `clinical_significance=...`와 같은 정답값을 여기에 쓰면 Qwen/MedCPT goal embedding이 답을 미리 보게 된다.

위 한 행은 schema 설명용이다. 실제 학습 corpus에는 nonterminal train/dev와 최종 test
task가 모두 있어야 하며, 같은 환자·variant·template에서 나온 행에는 같은 `split_group`을
붙인다. 비어 있는 dev나 요청한 test split은 trainer/evaluator가 조용히 0점 처리하지 않고
중단한다.

### `verifiers.private.jsonl`: 정답 판정 전용 조건

```json
{
  "task_id": "case-0001",
  "private_verifier": {
    "conditions": [
      {
        "field": "annotation.clinical_significance",
        "op": "eq",
        "value": "SYNTHETIC_LABEL"
      }
    ]
  },
  "provenance": {"review_status": "double_checked"}
}
```

전처리 replay와 최종 평가는 `public goal AND private verifier`를 사용한다. 반면 `tasks.jsonl`, processed example의 `goal`, runtime context key, frozen encoder 입력에는 private verifier가 들어가지 않는다. 파일명 패턴은 기본 `.gitignore`에 포함되어 있지만, 접근 통제와 보관 정책은 별도로 적용해야 한다.

### `snapshots.jsonl`: 정규화된 실제 실행 결과

```json
{
  "snapshot_id": "case-0001-annotation-v1",
  "tool_id": "lookup_variant_annotation",
  "arguments": {
    "assembly": "GRCh38",
    "variant_id": "SYNTH_VAR_0001",
    "database_release": "synthetic-2026-09"
  },
  "status": "success",
  "output": {
    "gene_symbol": "SYNTH1",
    "clinical_significance": "SYNTHETIC_LABEL"
  },
  "summary": "A reviewed normalized fixture was found.",
  "source_revision": "synthetic-2026-09",
  "provenance": {"normalizer_revision": "YOUR_NORMALIZER_COMMIT"}
}
```

허용 status는 `success`, `empty`, `domain_error`, `transport_error`, `parse_error`다. 성공한 결과만 `output_schema`, `output_bindings`, 정적 `effects`를 통해 상태를 갱신한다. 나머지는 상태를 성공한 것처럼 바꾸지 않고 typed observation으로 history에 남는다.

snapshot lookup은 canonicalized `tool_id + resolved arguments`의 완전 일치다. 동일 key가 두 번 나오면 전처리를 거절한다. 같은 variant라도 source release가 다르다면 반드시 arguments가 달라야 한다. 원본 API response를 그대로 저장하지 말고, 필요한 필드만 정규화하고 PHI·토큰·요청 헤더를 제거한다.

### `trajectories.jsonl`: 실행 검증된 경로

```json
{"task_id":"case-0001","tool_ids":["lookup_variant_annotation"]}
```

경로는 expected answer에서 추측하지 않는다. pinned wrapper와 snapshot으로 실제 replay한 뒤 reviewer가 승인한 순서를 쓴다. 길이 `m`인 경로는 전처리에서 `m+1`개 prefix가 되고 마지막 행은 명시적인 STOP 학습 예제가 된다.

### 선택 파일

`background.jsonl`에는 모델이 봐도 되는 source/build 정책을 넣는다.

```json
{
  "card_id": "annotation-policy-2026-09",
  "title": "Synthetic annotation policy",
  "text": "Use GRCh38 normalized identifiers with the synthetic-2026-09 snapshot.",
  "source": "internal-reviewed-policy",
  "redistribution_status": "allowed",
  "provenance": {"data_kind": "synthetic"}
}
```

`minimal_pairs.jsonl`에는 한 필드만 통제해 바꾼 쌍을 넣는다. 예를 들어 assembly, transcript version, coordinate direction을 바꾼 쌍은 `functional_sensitive`, 의미를 보존한 문장 바꾸기는 `invariant`로 표시한다.

## 2. observed-only 전처리

실제 데이터에서 실행하지 않은 모든 대안 도구의 결과를 알고 있다고 가정하면 안 된다. config를 다음처럼 둔다.

```yaml
preprocessing:
  max_search_depth: 16
  require_optimal_demo: false
  counterfactual_policy: observed_only
```

그 후 실행한다.

```bash
uv run geoflow prepare --config configs/geomarrvel_mvp.yaml
```

이 모드의 label 의미는 다음과 같다.

| 행동 | status | regret | loss mask |
|---|---|---:|---|
| 실제 실행한 gold edge | `observed` | `0` | 사용 |
| exact contract 자체가 불가능 | `contract_invalid` | `1` | 사용 |
| 적용 가능하지만 실행하지 않음 | `unobserved` | `null` | 제외 |

즉 “snapshot이 없다”는 사실을 “도구가 실패했다”는 label로 바꾸지 않는다. 반대로 `symbolic` 모드는 모든 효과가 완전히 정의된 합성 simulator나, 필요한 모든 counterfactual snapshot이 확보된 경우에만 사용한다.

## 3. MARRVEL을 seed로 쓰는 절차

```bash
uv sync --extra hf
uv run geoflow import-marrvel \
  --output-dir data/curation/marrvel \
  --revision 8e895924b1b19dc5ede19669bbf66c2d8d2eb5f0
```

생성된 queue에서 각 행에 typed initial state, 공개 goal, private verifier, available tools, 실제 실행한 gold tool IDs, split을 채우고 `curation_status`를 `validated`로 바꾼다. 그 뒤 다음을 실행한다.

```bash
uv run geoflow finalize-marrvel \
  --queue data/curation/marrvel/marrvel_curation_queue.jsonl \
  --raw-dir data/raw/geomarrvel
```

finalizer는 validated 행마다 비어 있지 않은 private verifier를 요구하고, 공개 task에서 expected answer를 제거한다. 다음 파일은 사람이 별도로 준비해야 한다.

- `tools.jsonl`: 실제 wrapper와 대조한 계약
- `snapshots.jsonl`: 실행해 얻은 정규화·검토 결과
- `background.jsonl`: 선택 사항인 공개 가능한 정책 카드
- `minimal_pairs.jsonl`: 선택 사항인 통제쌍

MARRVEL 외의 medical QA도 같은 변환 경계를 쓴다. QA는 `query` seed일 뿐이며, trajectory는 chart/database 작업 절차를 실행 검증해 새로 붙인다. EHR/FHIR 데이터라면 live endpoint 대신 비식별·접근 통제된 offline fixture를 사용하고, snapshot 재배포 가능성을 manifest와 별도 data card에 명시한다.

## 4. frozen embedding부터 closed loop까지

```bash
uv run geoflow embed --config configs/geomarrvel_mvp.yaml
uv run geoflow evaluate-embeddings --config configs/geomarrvel_mvp.yaml
uv run geoflow compare-distances \
  --config configs/geomarrvel_mvp.yaml \
  --distances euclidean cosine diagonal_mahalanobis lowrank_mahalanobis bilinear poincare
uv run geoflow train-metric \
  --config configs/geomarrvel_mvp.yaml \
  --distance lowrank_mahalanobis
uv run geoflow train-flow --config configs/geomarrvel_mvp.yaml
uv run geoflow evaluate-agent --config configs/geomarrvel_mvp.yaml
```

위 명령은 기본적으로 dev까지만 본다. `lowrank_mahalanobis`는 예시일 뿐이다. family와 threshold를 dev에서 고정한 후, 최종 한 번만 아래처럼 test를 연다.

```bash
selected_distance=cosine  # replace with the dev-selected family
uv run geoflow evaluate-embeddings --config configs/geomarrvel_mvp.yaml --include-test
uv run geoflow train-metric \
  --config configs/geomarrvel_mvp.yaml \
  --distance "$selected_distance" \
  --include-test
uv run geoflow train-flow --config configs/geomarrvel_mvp.yaml --include-test
uv run geoflow evaluate-agent --config configs/geomarrvel_mvp.yaml --split test
```

cache manifest, metric checkpoint, flow checkpoint는 processed manifest, encoder config, feature serialization version, tool order, 모든 cache array checksum을 포함한 content digest의 SHA chain으로 묶인다. 하나라도 다르면 평가를 중단한다.

## 5. 결과를 믿기 전 확인할 것

- private verifier 값이 initial model text나 goal embedding에 나타나지 않는가?
- snapshot의 모든 결과 결정 인자가 input schema와 argument binding에 들어 있는가?
- real wrapper output이 output schema를 실제로 통과하는가?
- unknown counterfactual의 regret mask가 false인가?
- no-mask 성능과 exact-contract-mask 성능을 분리했는가?
- raw embedding 개선이 learned geometry와 closed-loop recovery로 이어지는가?
- STOP recall과 premature STOP을 따로 보았는가?
- flow의 plan-slot 경로 길이/turn angle과 ODE transport 길이/turn angle을 혼동하지 않았는가?
- perturbation 대상, 실제 trigger, 회복 분모를 따로 보고했는가?

이 경계를 지키면 frozen encoder가 genomic exactness를 “알고 있다”고 가정하지 않으면서도, domain/background embedding이 기능적 경로 선택에 실제로 기여하는지를 검증할 수 있다.
