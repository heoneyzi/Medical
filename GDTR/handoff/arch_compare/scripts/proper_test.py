"""Correct test: are attention blocks over-represented among local maxima?
Permutation over layer positions, done separately for context-dependent and
motif-driven regions (they behave oppositely, so pooling them is meaningless).
"""
import numpy as np, json
R = "/path/to/TDiG/arch_compare/results"
res = json.load(open(f"{R}/layer_select.json"))
ATTN = set([3, 10, 17, 24])
GROUP = {"coding_exon": "context", "intergenic": "context",
         "splice_donor": "motif", "splice_acceptor": "motif"}
RNG = np.random.default_rng(0)

cells = []
for r in res:
    for nm, c in r["auroc"].items():
        c = np.array(c)
        loc = set(l for l in range(1, 28) if c[l] >= c[l-1] and c[l] >= c[l+1])
        cells.append((r["chrom"], nm, GROUP[nm], loc, len(ATTN & loc)))

for grp in ("context", "motif"):
    sub = [x for x in cells if x[2] == grp]
    obs = sum(x[4] for x in sub)
    exp = sum(4 * len(x[3]) / 27 for x in sub)
    # permutation: reassign 4 "attention" positions uniformly among layers 1..27 per cell
    null = np.array([sum(len(set(RNG.choice(np.arange(1, 28), 4, replace=False)) & x[3])
                         for x in sub) for _ in range(200000)])
    p = (null >= obs).mean()
    print(f"{grp:<9} attn hits {obs}/{4*len(sub)}  expected {exp:.2f}  "
          f"permutation p = {p:.2e}  (null mean {null.mean():.2f})")

print("\nper-cell local-maximum sets:")
for ch, nm, g, loc, h in cells:
    print(f"  {ch:<6}{nm:<17}{g:<9} local maxima at {sorted(loc)}  attn hits {h}/4")
