<sub>[🏠 Portfolio](https://github.com/heoneyzi) › [🩺 Medical](../../README.md) › [GeoFlowAgent](../README.md) › **data**</sub>

# 🗃️ Data — synthetic fixtures and schemas

Everything here is small, invented and used by the tests and smoke runs; real records are fetched at run time from their public sources.

| Path | Content |
|---|---|
| [`raw/smoke/`](raw/smoke/) | Invented `variant_to_report` fixture (tasks, tools, trajectories, background cards, minimal pairs) with synthetic gene/transcript/contig/HPO identifiers — exercises the stage-01 pipeline on CPU |
| [`raw/search_smoke/`](raw/search_smoke/) | Invented search-distillation fixture: 21 tasks (7/7/7), 19 tools, 172 versioned snapshots and private verifiers, with alternate sources, contract-valid detours and verifier-detectable dead ends |
| [`schemas/`](schemas/) | JSON Schemas for tasks, tools, snapshots, trajectories, verifiers, background cards, minimal pairs and processed examples |
| [`curation/marrvel/marrvel_import_manifest.json`](curation/marrvel/marrvel_import_manifest.json) | Pin (dataset id, revision, content hash, 100 rows) of the public MARRVEL-MCP QA benchmark that `geoflow import-marrvel` turns into a local curation queue |

`verifiers.private.jsonl` holds evaluation-only terminal predicates that the serializers keep away from the model. The MARRVEL curation queue is generated locally from the pinned dataset (terms in [THIRD_PARTY.md](../THIRD_PARTY.md)), and GeoACMG downloads the ClinGen Evidence Repository export at run time ([`clingen.py`](../src/geoflowagent/geoacmg/clingen.py)).
