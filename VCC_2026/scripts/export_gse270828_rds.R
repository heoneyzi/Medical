#!/usr/bin/env Rscript

# Export the official, double-gzipped GSE270828 Seurat RDS without expanding the
# 19,698 x 87,896 count matrix to dense memory or a huge MatrixMarket text file.
# The binary sparse arrays are consumed by build_gse270828_h5ad.py.

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2) {
  stop("usage: export_gse270828_rds.R INPUT.rds.gz OUTPUT_DIR")
}

input <- normalizePath(args[[1]], mustWork = TRUE)
out_dir <- args[[2]]
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

suppressPackageStartupMessages(library(Matrix))
suppressPackageStartupMessages(library(SeuratObject))

message("[gse270828] reading double-gzipped RDS: ", input)
outer <- gzfile(input, open = "rb")
inner <- gzcon(outer)
on.exit(try(close(inner), silent = TRUE), add = TRUE)
obj <- readRDS(inner)

counts <- GetAssayData(obj, assay = "RNA", slot = "counts")
if (!inherits(counts, "dgCMatrix")) {
  counts <- as(counts, "dgCMatrix")
}
if (any(!is.finite(counts@x)) || any(counts@x < 0) ||
    any(abs(counts@x - round(counts@x)) > 1e-6)) {
  stop("RNA counts are not finite non-negative integer-like values")
}

meta <- obj@meta.data
required <- c("rep", "guide", "har", "num_guide")
missing <- setdiff(required, colnames(meta))
if (length(missing)) {
  stop("missing metadata columns: ", paste(missing, collapse = ", "))
}
if (!identical(rownames(meta), colnames(counts))) {
  meta <- meta[colnames(counts), , drop = FALSE]
}
meta$cell_id <- rownames(meta)
meta <- meta[, c("cell_id", required,
                 setdiff(colnames(meta), c("cell_id", required))), drop = FALSE]

write_bin <- function(values, filename, size) {
  con <- file(file.path(out_dir, filename), open = "wb")
  on.exit(close(con))
  writeBin(values, con, size = size, endian = "little")
}

message("[gse270828] exporting CSC arrays (nnz=", length(counts@x), ")")
write_bin(as.integer(counts@i), "indices.int32.bin", 4L)
write_bin(as.integer(counts@p), "indptr.int32.bin", 4L)
write_bin(as.numeric(counts@x), "data.float32.bin", 4L)
writeLines(rownames(counts), file.path(out_dir, "genes.txt"), useBytes = TRUE)
write.csv(meta, file.path(out_dir, "metadata.csv"), row.names = FALSE,
          quote = TRUE, na = "")

manifest <- c(
  paste0("n_genes=", nrow(counts)),
  paste0("n_cells=", ncol(counts)),
  paste0("nnz=", length(counts@x)),
  "matrix_layout=CSC_genes_by_cells",
  "matrix_origin=RNA_counts",
  paste0("source=", input)
)
writeLines(manifest, file.path(out_dir, "matrix.properties"))
message("[gse270828] exported ", nrow(counts), " genes x ", ncol(counts),
        " cells -> ", normalizePath(out_dir))
