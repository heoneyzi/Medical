# Documentation update scope

This working copy was cloned from `heoneyzi/CAFA6`, branch `main`, at commit `481881c`. GitHub identifies it as a fork of [SOL1archive/CAFA6](https://github.com/SOL1archive/CAFA6). Original source files, job scripts, notebooks, and the dependency lockfile are retained.

This update corrects the notebook map, environment requirements, existing script names, module/configuration paths, CLI versus SLURM defaults, metric signatures, and external-resource requirements. It adds Jiheon Kang's CV-reported team-member and bronze-medal context, with a link to the [research portfolio](https://heoneyzi.github.io/).

## Validation scope

- Checked notebook names against tracked files and inspected their source cells.
- Checked documented command flags/defaults against Python parsers and shell scripts.
- Checked local Markdown links and module paths against the repository tree.
- Ran the existing ensemble script's `--help` entrypoint successfully without loading data or models.
- Scanned notebook source and outputs for recognizable credential patterns; none were detected. Notebooks remain unchanged.
- Preserved the lockfile and research implementation.

No training, GPU inference, competition-data download, external prediction generation, or leaderboard evaluation was performed. Medal, dataset-scale figures, and participation role are recorded from the supplied CV. The inherited `0-370` notebook filename is not a newly validated result.

## Resources still needed

Competition data and its original ontology snapshot, foundation-model downloads, trained checkpoints, ESM-C helper artifacts, and the external GOA/ProtT5+InterPro TSVs are not bundled. Their provenance and time cutoffs remain necessary for assessing competition compliance and reproducibility. Filtering predictions to test-superset IDs alone does not establish absence of temporal leakage.
