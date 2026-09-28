> 📜 **Original document** — written during the experiments and kept as written (machine paths replaced by `$GEOFLOW_*` placeholders). Final numbers and conclusions: [연구 안내](../../README.md) · [주장과 근거](../../research/04_CLAIMS_AND_EVIDENCE.md).

# Whole-plan flow and closed-loop evaluation

## Target construction

For prefix phase `h`, the supervised suffix is converted to static tool prototypes and one STOP anchor:

```text
Y_h = [p(t_h), ..., p(t_m), p(STOP)]
```

The collator appends `p(PAD)` to `L_max`. Real tool/STOP slots receive weight 1; PAD receives a smaller positive weight. The terminal prefix target is exactly `[p(STOP)]`.

## Rectified flow

The prior and target have shape `[B, L_max, d]`. The velocity Transformer attends across every plan slot. A separate sinusoidal embedding represents ODE time, and a learned table represents plan position.

The training endpoint estimate is:

```text
x1_hat = x_tau + (1 - tau) * v_theta(x_tau, tau, context)
```

Loss terms:

- weighted flow velocity MSE;
- STOP binary cross entropy on non-PAD slots;
- tool codebook cross entropy on tool slots;
- STOP/tool/PAD separation regularization.

Tool-codebook CE is normalized only over each row's visible candidate catalogue. Special-anchor
separation uses tools visible in the current training batch, so a registry entry held out from every
training task does not enter the loss merely by existing.

Checkpoint selection is lexicographic on dev: valid-set first-action accuracy, then balanced STOP
accuracy, then lower normalized suffix edit distance. Test is not evaluated by default and never
participates in selection; `--include-test` is an explicit post-selection reporting action.

## Why this differs from the public FlowPlan trainer

The audited public trainer models one endpoint vector. GeoFlowAgent deliberately models a joint plan tensor because the research question is whether embedding geometry helps complete future-path generation. This is a FlowAgent-inspired correction/extension, not a claim of faithful reproduction.

## Decoding

The generated slot is normalized and compared with static tool prototypes. Candidate tools come from the visible catalogue. For the first executable slot only, exact current-state contract validity is enforced. Later provisional slots cannot use the unobserved future state unless an explicit learned state rollout is enabled; the MVP therefore treats them as provisional and executes only slot zero under closed-loop evaluation.

This last codebook lookup is cosine-based for every learned energy family. Euclidean, cosine,
Mahalanobis, bilinear, and Poincaré choices act upstream: they train the state/tool projectors,
state-conditioned gate, compatibility energy, and the prototype space that the flow sees. Keeping a
common final decoder makes the family comparison controlled, but it means the result must not be
described as swapping the flow decoder's metric itself.

Decoding chooses a tool, after which reviewed `argument_bindings` resolve API-shaped values from typed state and validate the input contract. Free-form argument generation is intentionally outside this path experiment.

## Offline metrics

- valid-set first-action accuracy;
- single-demonstration first-action accuracy;
- terminal STOP recall, premature-STOP rate, and balanced STOP accuracy;
- full suffix exact match;
- normalized edit distance;
- active/tool/STOP/PAD-separated relative endpoint error;
- normalized plan-slot path length and mean turn angle;
- mean ODE transport length and ODE update turn angle;
- generated plan length at fixed NFE.

Plan-position path length/curvature and ODE transport length describe different geometries and should not be merged into one number.

The MVP evaluator batches a complete split on one device. This is intentional for the small curated
benchmark; larger studies should stream task batches while preserving deterministic per-task noise,
perturbation assignment, and planner-call accounting.

## Closed-loop metrics

The evaluator separates `goal_reached` from `correct_stop`. Strict `task_success` requires both, preventing a planner that reaches a goal but never halts from being counted as complete. It also separates exact contract-valid actions, execution-successful actions, and known zero-regret actions, then logs regret-label coverage, invalid/redundant calls, mean valid-candidate count, planner calls, total NFE, and latency.

Public completion predicates and private exact-answer verifiers are conjoined only inside replay and
evaluation. The planner sees the public goal. Snapshot-backed tools resolve exact arguments, replay
a normalized versioned output, validate it, and update typed state; an absent snapshot is
`unobserved`, not a fabricated failure.

Known reviewed prefixes reuse immutable frozen embeddings by exact context key. Reports include cache-hit rate and runtime embedding-miss planning calls; novel failure/perturbation histories fall back according to `runtime_embedding_policy`. Runtime encodings apply the cache storage dtype before pooling, preventing a hidden float16/float32 representation shift. A miss call is a planner context that required runtime encoding, not a count of internal Transformer forwards. `prewarm_runtime_encoders=false` preserves lazy, fully cached small-model execution; use a separately declared `true` config when comparing warm planner latency without cold-load order effects.

### Fair feedback comparison

Run both:

1. same provisional plan with `commit-k` varied;
2. repeated compute-matched planning without exposing new observations.

Only the second comparison approximates the marginal effect of feedback at comparable planning compute. The blind condition receives the same per-task call budget and per-call noise sequence as commit-1; early termination is padded with non-executed solves for compute accounting. Open loop is hard-limited to one solve.

### Contract and stopping controls

The default report adds random-within-contract and learned metric baselines, plus no-contract-mask
metric/flow ablations. If masked performance is high while no-mask and random-contract controls are
weak, exact formalization—not embedding geometry—is doing most of the work.

The metric/random baselines do not own a learned STOP head and therefore stop using the public goal
predicate. Reports mark this as `public_goal_oracle`; compare their action selection metrics, not
their STOP accuracy, directly with the flow conditions marked `learned`.

### Perturbations

The synthetic evaluator supports deterministic failures, empty results, and state overrides. Real experiments should include snapshot-backed versions of:

- empty lookup;
- alias ambiguity;
- unmapped liftover interval;
- missing response field;
- database release mismatch;
- unexpected assembly/build output.

Every perturbation must be applied equally across compared planners.

A tool-result perturbation is consumed only after that attempted action passes the exact current
contract. Merely naming the target tool in an invalid call cannot inflate the triggered/recovery
denominator or consume a once-only failure fixture.

The report fixes `assigned_perturbation_tasks` before running a policy, then separately records
`triggered_perturbation_tasks`. `assigned_task_success_rate` avoids dropping policies that never
selected the targeted tool; `triggered_recovery_rate` remains a conditional diagnostic rather than
the sole robustness number.
