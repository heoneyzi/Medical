#!/usr/bin/env bash
# Helper notes for obtaining Replogle GWPS (the only DB source used by this repo's
# matching methods). The processed AnnData live behind the GWPS portal / Figshare;
# exact URLs change, so this script documents the steps rather than hard-coding links.
set -euo pipefail
DEST="${1:-data/gwps}"
mkdir -p "$DEST"

cat <<'EOF'
Replogle et al. 2022 genome-scale Perturb-seq (CRISPRi):
  Portal  : https://gwps.wi.mit.edu/
  Figshare: https://plus.figshare.com/articles/dataset/_Mapping_information-rich_genotype-phenotype_landscapes_with_genome-scale_Perturb-seq_Replogle_et_al_2022_processed_Perturb-seq_datasets/20029387
  Paper   : Replogle et al., Cell 2022 (PMID 35688146)

Download the processed AnnData you want (e.g. K562 genome-wide, RPE1 essential) and
place them in the destination folder, then set gwps.sources in your config, e.g.:

  gwps:
    sources:
      k562: data/gwps/ReplogleWeissman2022_K562_gwps.h5ad
      rpe1: data/gwps/ReplogleWeissman2022_RPE1_essential.h5ad
    pert_col: gene            # confirm the real perturbation column name
    control_label: non-targeting

Then verify the obs schema:
  python - <<'PY'
import anndata as ad
a = ad.read_h5ad("data/gwps/ReplogleWeissman2022_K562_gwps.h5ad")
print(a); print(a.obs.columns.tolist()); print(a.obs.head())
PY
EOF
echo "Destination: $DEST"
