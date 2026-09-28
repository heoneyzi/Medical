"""Map BRCA1 SGE (MaveDB urn:mavedb:00000097-0-2) c. positions to GRCh38 chr17.

BRCA1 is on the minus strand, so c.1 is the highest genomic coordinate of the
CDS and the c. reference base is the complement of the genomic base.  Every
converted variant is checked against the chr17 reference; the script aborts if
a single base disagrees, because a silent strand or off-by-one error here would
produce a plausible but wrong result.

Exon map cross-checked across ncbiRefSeqSelect, ncbiRefSeqCurated and GENCODE
ENST00000357654.9, which agree exactly (CDS 5592 nt = 1863 aa + stop).
"""
import gzip, io, csv, re, json, urllib.request
import numpy as np

AC = "/path/to/TDiG/arch_compare"
URN = "urn:mavedb:00000097-0-2"
CDS_S, CDS_E = 43045677, 43124096          # 0-based half-open
EXON_S = [43044294,43047642,43049120,43051062,43057051,43063332,43063873,43067607,
          43070927,43074330,43076487,43082403,43090943,43091434,43095845,43097243,
          43099774,43104121,43104867,43106455,43115725,43124016,43125270]
EXON_E = [43045802,43047703,43049194,43051117,43057135,43063373,43063951,43067695,
          43071238,43074521,43076614,43082575,43091032,43094860,43095922,43097289,
          43099880,43104261,43104956,43106533,43115779,43124115,43125364]
COMP = {"A":"T","C":"G","G":"C","T":"A"}

# ---- CDS walk in transcript order (minus strand: descending genomic) --------
segs = []
for s, e in zip(EXON_S, EXON_E):
    cs, ce = max(s, CDS_S), min(e, CDS_E)
    if ce > cs:
        segs.append((cs, ce))
segs.sort(key=lambda x: -x[0])
cds_to_g = []
for cs, ce in segs:
    cds_to_g.extend(range(ce - 1, cs - 1, -1))
print(f"CDS length {len(cds_to_g)} nt | %3 == {len(cds_to_g) % 3} | "
      f"protein {len(cds_to_g)//3 - 1} aa + stop")
assert len(cds_to_g) == 5592, "CDS length does not match the cross-checked map"

# ---- chr17 reference --------------------------------------------------------
seq = []
with gzip.open(f"{AC}/chr17.fa.gz", "rt") as fh:
    for line in fh:
        if not line.startswith(">"):
            seq.append(line.strip())
S = "".join(seq).upper()
print(f"chr17 length {len(S)}")

# ---- MaveDB scores ----------------------------------------------------------
url = f"https://api.mavedb.org/api/v1/score-sets/{URN}/scores"
with urllib.request.urlopen(url, timeout=90) as r:
    txt = r.read().decode("utf-8", "replace")
rdr = csv.DictReader(io.StringIO(txt))
rows = list(rdr)
print(f"MaveDB rows {len(rows)}")

pat = re.compile(r"^NM_007294\.\d+:c\.(\d+)([ACGT])>([ACGT])$")
out, skipped, mism = [], 0, []
for r in rows:
    m = pat.match((r.get("hgvs_nt") or "").strip())
    sc = r.get("score")
    if not m or sc in (None, "", "NA"):
        skipped += 1
        continue
    cpos, cref, calt = int(m.group(1)), m.group(2), m.group(3)
    if not (1 <= cpos <= len(cds_to_g)):
        skipped += 1
        continue
    g = cds_to_g[cpos - 1]                       # 0-based genomic index
    gref = S[g]
    if COMP.get(gref) != cref:                   # minus strand
        mism.append((cpos, cref, gref, g + 1))
        continue
    out.append((g + 1, gref, COMP[calt], float(sc)))   # 1-based VCF-style

print(f"converted {len(out)} | skipped {skipped} | reference mismatches {len(mism)}")
if mism:
    print("  first mismatches (c.pos, c.ref, genomic_ref, 1-based pos):")
    for x in mism[:10]:
        print("   ", x)
    raise SystemExit("ABORT: reference bases disagree; the coordinate map is wrong.")

arr = np.array(out, dtype=[("pos", "i8"), ("ref", "U1"), ("alt", "U1"), ("score", "f8")])
np.save(f"{AC}/results/brca1_sge_variants.npy", arr)
print(f"saved {AC}/results/brca1_sge_variants.npy")
print(f"  pos range {arr['pos'].min()}-{arr['pos'].max()}")
print(f"  score: min {arr['score'].min():.3f} max {arr['score'].max():.3f} "
      f"mean {arr['score'].mean():.3f}")
q = np.percentile(arr["score"], [1, 5, 25, 50, 75, 95, 99])
print("  score percentiles 1/5/25/50/75/95/99:", np.round(q, 3).tolist())
