"""Build a chr19 window list matching the chr22/chr17 panels: 6 kb windows,
low N content, and enough splice-site positions to populate every region class.
Out-of-sample test bed for the 'read 4 blocks before the cascade' rule.
"""
import gzip, numpy as np, pandas as pd
ROOT = "/path/to/TDiG"; AC = f"{ROOT}/arch_compare"
CHROM = "chr19"; W = 6000; NWIN = 100; MAXN = 0.01

seq = []
with gzip.open(f"{AC}/{CHROM}.fa.gz", "rt") as fh:
    for line in fh:
        if not line.startswith(">"): seq.append(line.strip())
S = "".join(seq).upper()
lab = np.load(f"{ROOT}/paper/data_local/{CHROM}_position_labels.npy")
assert len(S) == len(lab), (len(S), len(lab))
print(f"{CHROM}: {len(S):,} bp | label counts {dict(zip(*[x.tolist() for x in np.unique(lab, return_counts=True)]))}")

starts = np.arange(0, len(S) - W, W)
rows = []
for a in starts:
    sub = lab[a:a + W]
    nsp = int(((sub == 5) | (sub == 6)).sum())
    if nsp == 0: continue
    seg = S[a:a + W]
    nfrac = seg.count("N") / W
    if nfrac > MAXN: continue
    rows.append((a, a + W, nsp, int((sub == 2).sum()), int((sub == 0).sum()), int((sub == 1).sum()), nfrac))
df = pd.DataFrame(rows, columns=["start", "end", "n_splice", "n_coding_exon", "n_intergenic", "n_intron", "n_fraction"])
print(f"candidate windows with splice sites and low N: {len(df)}")

rng = np.random.default_rng(20260916)
pick = df.iloc[rng.choice(len(df), min(NWIN, len(df)), replace=False)].sort_values("start").reset_index(drop=True)
pick.insert(0, "window_idx", np.arange(len(pick)))
pick.insert(1, "chrom", CHROM)
pick.to_parquet(f"{ROOT}/tdig_integration/data_cache_minimal/{CHROM}_metadata.parquet")
print(pick[["window_idx", "start", "end", "n_splice", "n_coding_exon", "n_intergenic"]].head().to_string())
print(f"wrote metadata for {len(pick)} windows | total positions {len(pick) * W:,}")
