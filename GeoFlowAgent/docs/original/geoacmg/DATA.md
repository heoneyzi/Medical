> 📜 **Original document** — written during the experiments and kept as written (machine paths replaced by `$GEOFLOW_*` placeholders). Final numbers and conclusions: [연구 안내](../../README.md) · [주장과 근거](../../research/04_CLAIMS_AND_EVIDENCE.md).

# Data design and curation

## Why QA is not enough

An agent trajectory dataset needs more than a question and final answer. The minimal training unit is a state graph edge:

```json
{
  "state_before": {},
  "tool_id": "...",
  "applicable": true,
  "outcome_status": "observed",
  "observation": {},
  "state_after": {},
  "regret": 0.0
}
```

The public MARRVEL-MCP benchmark supplies QA seeds only. `geoflow import-marrvel` preserves those rows and creates empty curation fields. It never guesses tools from the expected text.

During finalization, the expected text is written to `reference_answers.private.jsonl`, not to model-facing `tasks.jsonl`. Exact terminal predicates live separately in `verifiers.private.jsonl`. Keep both files evaluation-only and outside every model-input serializer.

## Raw files expected by the compiler

### `tools.jsonl`

Required keys:

```json
{
  "tool_id": "liftover_grch37_to_grch38",
  "name": "Lift GRCh37 coordinates to GRCh38",
  "description": "...",
  "execution_mode": "snapshot",
  "input_schema": {},
  "argument_bindings": {
    "chromosome": "selected_variant.chromosome",
    "position": "selected_variant.position"
  },
  "output_schema": {"type": "object"},
  "output_bindings": {
    "lifted.position": "selected_variant.position"
  },
  "preconditions": [
    {"field": "assembly", "op": "eq", "value": "GRCh37"}
  ],
  "effects": [
    {"field": "assembly", "op": "set", "value": "GRCh38"},
    {"field": "ref_alt_verified", "op": "set", "value": false}
  ]
}
```

The DSL supports `eq`, `neq`, `exists`, `missing`, `nonempty`, `in`, `not_in`, `contains`, `gte`, and `lte`; effects support `set`, `copy`, `delete`, `append`, and `increment`. It never calls `eval()`.

The flow/metric models select a tool ID. Before the contract regards it as applicable, `argument_bindings` resolves each input name from a dotted typed-state path (the input name itself is the default path), then performs full Draft 2020-12 JSON Schema validation. For snapshot tools, `output_bindings` has the deliberately fixed direction `output path → typed next-state path`; successful output is schema-validated again before any binding or effect is applied.

### `tasks.jsonl`

```json
{
  "task_id": "case-001",
  "query": "...",
  "initial_state": {
    "assembly": "GRCh37",
    "sequence_reference": "NC_000017.10",
    "coordinate_system": "one_based_closed"
  },
  "goal": {"report_ready": true},
  "available_tools": ["..."],
  "background_ids": ["..."],
  "split": "train",
  "split_group": "same-patient-or-variant-family",
  "provenance": {}
}
```

Keep the following exact when known:

- assembly family and reference release;
- contig/transcript accession including version;
- coordinate convention;
- start/end and ref/alt;
- normalization representation/version;
- database and tool snapshot.

If the source supplies only `hg38`, do not invent an exact GRCh38 patch or accession. Store the known precision explicitly in a production extension.

The public `goal` must describe completion without exposing the answer. For example, use
`{"annotation_complete": true}` rather than embedding the expected pathogenicity label in the
task.

### `verifiers.private.jsonl`

```json
{
  "task_id": "case-001",
  "private_verifier": {
    "conditions": [
      {"field": "annotation.label", "op": "eq", "value": "PRIVATE_EXPECTED_VALUE"}
    ]
  },
  "provenance": {}
}
```

Replay, action labels, terminal validation, and strict closed-loop success use the conjunction of
the public goal and this private verifier. Processed `examples.jsonl` retains only the public goal.
The private file is nevertheless checksum-bound by the processed manifest so evaluation cannot
silently swap labels after embeddings or checkpoints are produced.

### `snapshots.jsonl`

```json
{
  "snapshot_id": "liftover-case-001-v1",
  "tool_id": "liftover_grch37_to_grch38",
  "arguments": {"chromosome": "17", "position": 43071077},
  "status": "success",
  "output": {"lifted": {"position": 43045629}},
  "source_revision": "PINNED_SOURCE_OR_DATABASE_RELEASE",
  "provenance": {}
}
```

Snapshot lookup is an exact canonical match over `tool_id + resolved arguments`; duplicates are
rejected. The available statuses are `success`, `empty`, `domain_error`, `transport_error`, and
`parse_error`. Non-success outcomes remain typed observations and do not apply successful state
updates. Every result-determining input—including build, direction, database release, and accession
version—must therefore be part of the bound arguments.

### `trajectories.jsonl`

```json
{"task_id":"case-001","tool_ids":["parse","liftover","verify","annotate"]}
```

The compiler replays every step. Unknown tools, unavailable tools, failed preconditions, and a final unsatisfied goal are hard errors.
An empty `tool_ids` list is permitted only in the meaningful case where the initial typed state already satisfies the goal; preprocessing then emits the single terminal STOP example.

The processed manifest binds every generated JSONL file by SHA-256. Cache construction and every `FeatureStore` load recheck those hashes, then verify that the embedding cache was created from that exact manifest. During a rebuild, recognized optional outputs whose raw source disappeared are removed. If an optional file is later introduced manually without a manifest entry, verification rejects the unbound file.

### `background.jsonl`

```json
{
  "card_id": "grch-policy-v1",
  "title": "Reference policy",
  "text": "...",
  "source": "...",
  "redistribution_status": "allowed"
}
```

Background is encoded separately and pooled into context; it is never baked into the static tool prototype.

### `minimal_pairs.jsonl`

```json
{
  "pair_id": "assembly-swap-001",
  "relation": "functional_sensitive",
  "changed_fields": ["state.assembly"],
  "left_text": "assembly=GRCh37",
  "right_text": "assembly=GRCh38",
  "split": "test",
  "source_task_id": "case-001"
}
```

`split` is required. `source_task_id` is optional, but when present the compiler requires the pair and source task to have the same assigned split. Use `invariant` for verified paraphrases/serialization changes and `functional_sensitive` for build, direction, accession-version, coordinate-convention, level-of-analysis, or schema changes.

## Processed example schema

Each row of `examples.jsonl` contains:

```text
identity       example_id, task_id, split, phase, record_sha256
visible input  query, goal, state, history, candidate_tools, background_ids
supervision    next_state, gold_suffix_tool_ids, gold_next_tool
set labels     contract_valid_tools, valid_next_tools
utility        action_regret, action_regret_mask, action_outcome_status
audit          terminal, provenance
```

`gold_suffix_tool_ids` never contains STOP or PAD. The flow loader appends one STOP and pads in memory.

## Counterfactual policy

`symbolic` is valid only when every candidate's effect is truly defined by the deterministic research environment. It labels non-gold applicable actions as `simulated`.

`observed_only` is the safe default for real MARRVEL curation:

```text
gold executed edge      outcome_status=observed, regret_mask=true
failed exact contract   outcome_status=contract_invalid, regret_mask=true
applicable, not run     outcome_status=unobserved, regret=null, regret_mask=false
```

Never convert “snapshot absent” into “tool failed.”

At evaluation time, zero-regret is reported only when the current prefix has a trusted label.
Novel/perturbed states without a counterfactual label report `regret=null` and reduce the explicit
regret-label coverage instead of being assigned a fabricated penalty.

## Split rules

Split at the base-task/case level before prefix expansion. Every paraphrase, prefix, terminal state, counterfactual branch, and minimal pair derived from one base case must remain in the same split. Tasks that share `split_group` are hashed as one unit when no split is supplied; contradictory explicit splits within a group are a hard error. A pair linked with `source_task_id` is checked against the resulting task assignment.

For a real study, publish separate protocols:

- grouped random split;
- entity/variant-family holdout;
- template holdout;
- tool-family holdout;
- long-trajectory holdout.

The test tool description/schema may be encoded by a stateless frozen encoder, but no projector, whitening matrix, threshold, or model parameter may be fitted on that test tool.

## MARRVEL contract curation warning

Do not treat Python docstrings as executable truth. A tool's description, decorator metadata, actual return fields, reference build, and a pinned fixture response must agree. Store the evidence and review result alongside the registry. This repository intentionally does not vendor MARRVEL-MCP code.

## Leakage checklist

The model-visible serializer must exclude:

- expected answer text;
- private verifier values;
- gold next tool and suffix;
- future observations;
- `stop_target` or gold remaining length;
- labels derived by comparing a prediction with the gold trace.

The compiler writes a manifest stating that future observations are excluded. Add project-specific tests before using a new importer.

For a complete linked example, see [REAL_DATA_WALKTHROUGH.md](REAL_DATA_WALKTHROUGH.md).
