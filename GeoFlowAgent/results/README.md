<sub>[🏠 Portfolio](https://github.com/heoneyzi) › [🩺 Medical](../../README.md) › [GeoFlowAgent](../README.md) › **results**</sub>

# 📊 Saved results

Every number and figure in this folder's pages comes from these files; [`verification/`](../verification/README.md) recomputes the headline ones. Machine paths inside some JSON/Markdown files are `$GEOFLOW_*` placeholders.

| Folder | Stage | Contents |
|---|---|---|
| [`synthetic_v1/`](synthetic_v1/) | ① | Run log ([`GeoFlowAgent_RUN_STATE.md`](synthetic_v1/GeoFlowAgent_RUN_STATE.md)), embedding audits, per-seed dev/test agent summaries for the 1.5B and 7B studies |
| [`hard_v2/`](hard_v2/) | ②–③ | [Final conclusion](hard_v2/GeoFlowAgent_A_FINAL_EXPERIMENT_CONCLUSION.md), [development report](hard_v2/GeoFlowAgent_A_EXPERIMENT_REPORT_CURRENT.md), pre-registration v1 + amendment v2; [`final_reports/`](hard_v2/final_reports/) — one-time test: per-seed value metrics, State Flow, closed loop, data quality, test-access ledger; [`dev_reports/`](hard_v2/dev_reports/) — decisive dev comparisons |
| [`geoacmg_final/`](geoacmg_final/) | ④ | [`findings/`](geoacmg_final/findings/) — B0, RQ1 probes, R0–R7, pre-registration and its amendment, mechanism/capacity/strata analyses, the auto-generated `REPORT.md` |
| [`geoacmg_working_findings/`](geoacmg_working_findings/) | ⑤ | R2b–R2f, R3b–R3c, R8–R11 |
| [`geoacmg_latest_checkpoints/`](geoacmg_latest_checkpoints/README.md) | ④–⑤ | Per-run development metrics (`value_metrics.json` with per-task values) and `comparison.json` aggregates — the inputs from which the planning-axis statistics are recomputed |

**Which file to read.** For R3 and R4, use `geoacmg_final/findings/` (5 seeds and 19 runs); the `R3_horizon_strata.json` and `R4_crossmodel_control.json` in `geoacmg_working_findings/` are an earlier 15-run subset analysis. The Markdown reports in `synthetic_v1/` and `hard_v2/` were written during the experiments; the final values are in [docs/research/03_RESULTS_AND_VERIFICATION.md](../docs/research/03_RESULTS_AND_VERIFICATION.md).

**Large artifacts** — checkpoints (`.pt`), embedding caches, processed search graphs, the ClinGen export and the readout-margin arrays — are kept in the local research archive; checkpoint hashes are listed in [`provenance/checkpoint_catalog.csv`](../provenance/checkpoint_catalog.csv).

---
<sub>[← GeoFlowAgent](../README.md) · [🏠 Portfolio](https://github.com/heoneyzi)</sub>
