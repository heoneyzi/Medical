> 📜 **Original document** — written during the experiments and kept as written (machine paths replaced by `$GEOFLOW_*` placeholders). Final numbers and conclusions: [연구 안내](../../README.md) · [주장과 근거](../../research/04_CLAIMS_AND_EVIDENCE.md).

# FlowPlan reference audit

Audited source: [ssy166/FlowPlan](https://github.com/ssy166/FlowPlan), commit `c65155b85931e5d38c38573c8e4f84d31b056c45`.

The official snapshot is useful for understanding its benchmark and experimental plumbing, but it is not a drop-in implementation for this repository's experiment.

Observed differences relevant to GeoFlowAgent:

- the public main FM trainer learns a single endpoint vector rather than a joint `[L,d]` plan tensor;
- a public nearest-path decoder retrieves a training endpoint and copies its suffix rather than decoding each anchor against semantic tool prototypes;
- the snapshot omits the paper checkpoint, real Qwen cache/extractor, projection, STOP/utility stack, and some final artifacts;
- several public branches use different meanings for `future_tools` (full suffix versus next action/STOP);
- the public executable evaluation largely evaluates one predicted next action or runs a previously predicted action list; it is not the receding-horizon loop implemented here;
- some expanded-data scripts expose target-derived state such as `stop_target`, so GeoFlowAgent uses an independently specified model-visible schema.

Useful primary references:

- [FlowPlan README](https://github.com/ssy166/FlowPlan/blob/c65155b85931e5d38c38573c8e4f84d31b056c45/README.md)
- [single-endpoint FM trainer](https://github.com/ssy166/FlowPlan/blob/c65155b85931e5d38c38573c8e4f84d31b056c45/scripts/train_fm_model.py)
- [workflow record builder](https://github.com/ssy166/FlowPlan/blob/c65155b85931e5d38c38573c8e4f84d31b056c45/scripts/build_workflow_records.py)
- [candidate-aware pointer decoder](https://github.com/ssy166/FlowPlan/blob/c65155b85931e5d38c38573c8e4f84d31b056c45/scripts/train_fm_pointer_decoder.py)
- [one-action execution evaluator](https://github.com/ssy166/FlowPlan/blob/c65155b85931e5d38c38573c8e4f84d31b056c45/scripts/evaluate_replan_execution.py)
- [reproducibility notes](https://github.com/ssy166/FlowPlan/blob/c65155b85931e5d38c38573c8e4f84d31b056c45/docs/BASELINE_REPRODUCIBILITY.md)

No source file from FlowPlan was copied into GeoFlowAgent.

