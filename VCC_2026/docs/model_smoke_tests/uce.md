# UCE-4L real-input smoke

- Smoke status: passed (`model_provenance.csv` status remains `verified_broad` because pretraining overlap is unknown)
- Official repository commit: `9c416007be15ad6753dc84af4468c1dc10421ab9`
- Checkpoint SHA-256: `acb28f3f0a1d803e4a4ffe891b9bab38bf93c84762dc06b2452f0d515da91560`
- Input: 128 raw-count Replogle K562 control pseudobulks, 4,017 genes
- Result: 128/128 cells retained; finite `X_uce` matrix with 1,280 columns
- Elapsed inference time on this machine: 18.66 seconds

The official UCE entry point was used. Preflight also checks the eight official species protein
embedding files, token file, species chromosome table, species offsets, exact Git commit, executable,
and checkpoint hash. The authoritative machine-readable record is
`data_experiments/model_smoke/uce/uce/k562_replogle_smoke/metadata.json`.
