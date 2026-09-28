#!/usr/bin/env bash
set -eEuo pipefail

cd "$(dirname "$0")/.."

MODEL="${1:-}"
MODEL_STORAGE_ROOT="${MODEL_STORAGE_ROOT:-}"
if [[ -n "$MODEL_STORAGE_ROOT" && -z "${UV_CACHE_DIR:-}" ]]; then
  export UV_CACHE_DIR="$MODEL_STORAGE_ROOT/uv-cache"
  echo "placed uv package cache on model storage: $UV_CACHE_DIR"
fi
if [[ -z "$MODEL" ]]; then
  echo "usage: $0 gene-map|replogle-data|uce|transcriptformer|scgpt|scfoundation|scprint2"
  exit 2
fi

clone_pinned() {
  local url="$1" dir="$2" commit="$3"
  local clone_dir
  clone_dir="$(readlink -f "$dir")"
  if [[ ! -d "$dir/.git" ]]; then
    git clone "$url" "$clone_dir"
  fi
  git -C "$dir" fetch --depth 1 origin "$commit"
  git -C "$dir" checkout --detach "$commit"
}

ensure_venv() {
  local directory="$1"
  local resolved
  resolved="$(readlink -f "$directory")"
  if [[ ! -x "$directory/bin/python" ]]; then
    uv venv "$resolved" --python 3.11
  else
    echo "reusing virtual environment: $directory"
  fi
}

place_on_model_storage() {
  local relative="$1" target
  [[ -n "$MODEL_STORAGE_ROOT" ]] || return
  if [[ -e "$relative" || -L "$relative" ]]; then
    echo "storage placement unchanged for existing path: $relative"
    return
  fi
  target="$(readlink -m "$MODEL_STORAGE_ROOT/$relative")"
  mkdir -p "$target" "$(dirname "$relative")"
  ln -s "$target" "$relative"
  echo "placed $relative on model storage: $target"
}

download_verified() {
  local file_id="$1" destination="$2" expected_size="$3" expected_md5="$4"
  local partial="${destination}.part"
  local observed_size observed_md5

  mkdir -p "$(dirname "$destination")"
  if [[ -f "$destination" ]]; then
    observed_size="$(stat -c %s "$destination")"
    observed_md5="$(md5sum "$destination" | awk '{print $1}')"
    if [[ "$observed_size" == "$expected_size" && "$observed_md5" == "$expected_md5" ]]; then
      echo "verified existing artifact: $destination"
      return
    fi
    echo "invalid existing artifact will be replaced after a verified download: $destination" >&2
  fi

  echo "downloading Figshare file $file_id -> $destination"
  curl -fL --retry 5 --retry-all-errors --continue-at - \
    "https://api.figshare.com/v2/file/download/${file_id}" -o "$partial"
  observed_size="$(stat -c %s "$partial")"
  observed_md5="$(md5sum "$partial" | awk '{print $1}')"
  if [[ "$observed_size" != "$expected_size" || "$observed_md5" != "$expected_md5" ]]; then
    echo "download verification failed for $destination" >&2
    echo "expected size=$expected_size md5=$expected_md5" >&2
    echo "observed size=$observed_size md5=$observed_md5" >&2
    return 1
  fi
  mv -f "$partial" "$destination"
  echo "verified artifact: $destination (size=$observed_size md5=$observed_md5)"
}

case "$MODEL" in
  gene-map)
    mkdir -p data/mappings
    curl -fL --retry 3 \
      https://storage.googleapis.com/public-download-files/hgnc/tsv/tsv/hgnc_complete_set.txt \
      -o data/mappings/hgnc_complete_set.txt
    .venv/bin/python scripts/build_gene_map.py \
      --config data_shadow/jiang24_ifng_bxpc3_loco/shadow.yaml \
      --config data_shadow/gse270828_rep3_lobo_vcc2026/shadow.yaml
    ;;
  replogle-data)
    download_verified 35774443 data/replogle/K562_gwps_raw_bulk_01.h5ad \
      374587922 4570b53c9d62ff6df281e622f0350060
    download_verified 35775581 data/replogle/rpe1_raw_bulk_01.h5ad \
      95350546 74765fa87635467a869ea972356ae0e7
    ;;
  uce)
    place_on_model_storage external/UCE
    place_on_model_storage envs/uce
    clone_pinned https://github.com/snap-stanford/UCE.git external/UCE \
      9c416007be15ad6753dc84af4468c1dc10421ab9
    ensure_venv envs/uce
    uv pip install --python envs/uce/bin/python -r external/UCE/requirements.txt
    mkdir -p external/UCE/model_files
    # The github.com/figshare redirect can return a zero-byte WAF challenge.
    # Use the official API endpoint and verify every upstream artifact before use.
    download_verified 42706576 external/UCE/model_files/4layer_model.torch \
      3403514339 3e2f59d6da6eaa5396aa297edfcec08f
    download_verified 42706558 external/UCE/model_files/species_chrom.csv \
      4097781 42a9b5871b1ed955fd05414b3ab89081
    download_verified 42706555 external/UCE/model_files/species_offsets.pkl \
      139 ddf9143508857e62607f323b338ff383
    download_verified 42706585 external/UCE/model_files/all_tokens.torch \
      2979205876 3523ea431c403f95701b6eeb0c30b997
    UCE_PROTEINS=(
      Homo_sapiens.GRCh38.gene_symbol_to_embedding_ESM2.pt
      Mus_musculus.GRCm39.gene_symbol_to_embedding_ESM2.pt
      Xenopus_tropicalis.Xenopus_tropicalis_v9.1.gene_symbol_to_embedding_ESM2.pt
      Danio_rerio.GRCz11.gene_symbol_to_embedding_ESM2.pt
      Microcebus_murinus.Mmur_3.0.gene_symbol_to_embedding_ESM2.pt
      Sus_scrofa.Sscrofa11.1.gene_symbol_to_embedding_ESM2.pt
      Macaca_fascicularis.Macaca_fascicularis_6.0.gene_symbol_to_embedding_ESM2.pt
      Macaca_mulatta.Mmul_10.gene_symbol_to_embedding_ESM2.pt
    )
    proteins_ready=1
    for filename in "${UCE_PROTEINS[@]}"; do
      [[ -s "external/UCE/model_files/protein_embeddings/$filename" ]] || proteins_ready=0
    done
    if [[ "$proteins_ready" != 1 ]]; then
      download_verified 42715213 external/UCE/model_files/protein_embeddings.tar.gz \
        2735410523 6756d7fa8348e0f70b5e5b11e324f023
      if tar -tzf external/UCE/model_files/protein_embeddings.tar.gz | \
          awk 'substr($0,1,1)=="/" || $0 ~ /(^|\/)\.\.($|\/)/ {bad=1} END {exit bad ? 0 : 1}'; then
        echo "unsafe path in UCE protein archive" >&2
        exit 1
      fi
      tar -xzf external/UCE/model_files/protein_embeddings.tar.gz \
        -C external/UCE/model_files
      for filename in "${UCE_PROTEINS[@]}"; do
        [[ -s "external/UCE/model_files/protein_embeddings/$filename" ]] || {
          echo "missing extracted UCE runtime file: $filename" >&2
          exit 1
        }
      done
    fi
    if [[ -f external/UCE/model_files/protein_embeddings.tar.gz ]]; then
      rm -f external/UCE/model_files/protein_embeddings.tar.gz
      echo "removed verified UCE archive after all eight extracted files were checked"
    fi
    ;;
  transcriptformer)
    place_on_model_storage external/transcriptformer
    place_on_model_storage envs/transcriptformer
    place_on_model_storage models/transcriptformer
    clone_pinned https://github.com/czi-ai/transcriptformer.git external/transcriptformer \
      c943a89a4de9511a8ab1010715bf1cfa8827cc99
    ensure_venv envs/transcriptformer
    uv pip install --python envs/transcriptformer/bin/python ./external/transcriptformer
    TF_CHECKPOINT="models/transcriptformer/tf_sapiens"
    TF_MEMBERS=(
      config.json
      model_weights.pt
      vocabs/assay_vocab.json
      vocabs/homo_sapiens_gene.h5
    )
    tf_ready=1
    for filename in "${TF_MEMBERS[@]}"; do
      [[ -s "$TF_CHECKPOINT/$filename" ]] || tf_ready=0
    done
    if [[ "$tf_ready" == 1 ]]; then
      echo "reusing complete TranscriptFormer checkpoint: $TF_CHECKPOINT"
    else
      envs/transcriptformer/bin/transcriptformer download tf-sapiens \
        --checkpoint-dir models/transcriptformer
    fi
    for filename in "${TF_MEMBERS[@]}"; do
      [[ -s "$TF_CHECKPOINT/$filename" ]] || {
        echo "missing TranscriptFormer checkpoint member: $TF_CHECKPOINT/$filename" >&2
        exit 1
      }
    done
    ;;
  scgpt)
    place_on_model_storage external/scGPT
    place_on_model_storage envs/scgpt
    place_on_model_storage models/scGPT
    clone_pinned https://github.com/bowang-lab/scGPT.git external/scGPT \
      cebd6fae655b9c585a4807daa3ac31bb764f06b4
    ensure_venv envs/scgpt
    uv pip install --python envs/scgpt/bin/python ./external/scGPT
    echo "AUDIT_REQUIRED: download the official whole-human checkpoint folder"
    echo "to models/scGPT/whole-human (args.json, vocab.json, best_model.pt)."
    ;;
  scfoundation)
    place_on_model_storage external/scFoundation
    place_on_model_storage envs/scfoundation
    clone_pinned https://github.com/biomap-research/scFoundation.git external/scFoundation \
      397631c495eddf9ad6644fc00c6ea8139e651245
    ensure_venv envs/scfoundation
    uv pip install --python envs/scfoundation/bin/python \
      torch scanpy einops local-attention performer-pytorch
    echo "AUDIT_REQUIRED: obtain the official models.ckpt from the URL in"
    echo "external/scFoundation/model/models/download.txt."
    ;;
  scprint2)
    place_on_model_storage external/scPRINT-2
    place_on_model_storage envs/scprint2
    place_on_model_storage models/scPRINT-2
    clone_pinned https://github.com/cantinilab/scPRINT-2.git external/scPRINT-2 \
      9dbc57acb66ec45cf0a8676740473e0328887188
    ensure_venv envs/scprint2
    uv pip install --python envs/scprint2/bin/python ./external/scPRINT-2
    mkdir -p models/scPRINT-2
    envs/scprint2/bin/hf download jkobject/scPRINT small-v2.ckpt \
      --local-dir models/scPRINT-2
    SCPRINT_LAMIN_SETTINGS="$PWD/models/scPRINT-2/lamin-settings"
    SCPRINT_LAMIN_CACHE="$PWD/models/scPRINT-2/lamin-cache"
    SCPRINT_LAMIN_STORAGE="$PWD/models/scPRINT-2/lamin-storage"
    mkdir -p "$SCPRINT_LAMIN_SETTINGS" "$SCPRINT_LAMIN_CACHE"
    if env LAMIN_SETTINGS_DIR="$SCPRINT_LAMIN_SETTINGS" \
        LAMIN_CACHE_DIR="$SCPRINT_LAMIN_CACHE" \
        envs/scprint2/bin/python -c \
        'from scdataloader.utils import load_genes; assert len(load_genes("NCBITaxon:9606")) > 0' \
        >/dev/null 2>&1; then
      echo "reusing populated scPRINT-2 LaminDB instance"
    else
      env LAMIN_SETTINGS_DIR="$SCPRINT_LAMIN_SETTINGS" \
        LAMIN_CACHE_DIR="$SCPRINT_LAMIN_CACHE" \
        envs/scprint2/bin/lamin init --storage "$SCPRINT_LAMIN_STORAGE" \
        --name vcc-scprint2 --modules bionty
      env LAMIN_SETTINGS_DIR="$SCPRINT_LAMIN_SETTINGS" \
        LAMIN_CACHE_DIR="$SCPRINT_LAMIN_CACHE" \
        envs/scprint2/bin/python -c \
        'import bionty as bt; bt.core.sync_all_sources_to_latest()'
      env LAMIN_SETTINGS_DIR="$SCPRINT_LAMIN_SETTINGS" \
        LAMIN_CACHE_DIR="$SCPRINT_LAMIN_CACHE" \
        envs/scprint2/bin/python -c \
        'from scdataloader.utils import populate_my_ontology, _adding_scbasecamp_genes; populate_my_ontology(organisms_clade=["vertebrates"], sex=["PATO:0000384", "PATO:0000383"], organisms=["NCBITaxon:10090", "NCBITaxon:9606"]); _adding_scbasecamp_genes()'
    fi
    ;;
  *)
    echo "unknown model: $MODEL"
    exit 2
    ;;
esac
