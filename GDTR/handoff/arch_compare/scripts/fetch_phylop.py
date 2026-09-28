"""T4 prerequisite: phyloP100way (hg38) for exactly the windows used by the
Evo2 extraction, in the same row order as *_scalars.npz.  Independent label
source -- comparative genomics, not GENCODE -- so it cannot be dismissed as
recycling the annotation the profiles were discovered on.
"""
import numpy as np, pandas as pd, pyBigWig, sys

ROOT = "/path/to/TDiG"; AC = f"{ROOT}/arch_compare"
URL = "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/phyloP100way/hg38.phyloP100way.bw"
bw = pyBigWig.open(URL)
print("remote bigwig open |", "chr22", bw.chroms().get("chr22"), "chr17", bw.chroms().get("chr17"))

for chrom, sf in (("chr22", "evo2_7b_scalars.npz"), ("chr17", "chr17_evo2_7b_scalars.npz")):
    z = np.load(f"{AC}/results/{sf}")
    win = z["window"]
    order = win[np.concatenate(([True], win[1:] != win[:-1]))]      # windows in row order
    meta = pd.read_parquet(f"{ROOT}/tdig_integration/data_cache_minimal/{chrom}_metadata.parquet")
    st = meta.set_index("window_idx")["start"].to_dict()
    en = meta.set_index("window_idx")["end"].to_dict()
    out = []
    for w in order:
        w = int(w); a, b = int(st[w]), int(en[w])
        v = np.array(bw.values(chrom, a, b), dtype=np.float32)
        out.append(v)
    P = np.concatenate(out)
    assert len(P) == len(win), f"length mismatch {len(P)} vs {len(win)}"
    fin = np.isfinite(P)
    np.savez(f"{AC}/results/{chrom}_phylop100.npz", phylop=P)
    print(f"{chrom}: n={len(P):,} windows={len(order)} finite={fin.mean():.3f} "
          f"mean={np.nanmean(P):+.3f} sd={np.nanstd(P):.3f} "
          f"q[1,50,99]={np.nanpercentile(P,[1,50,99]).round(2)}")
bw.close()
