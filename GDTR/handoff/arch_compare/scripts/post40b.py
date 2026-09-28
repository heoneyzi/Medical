"""Post-hoc evaluation of the 40B probe curves, side by side with 7B.

Fixes a defect in analyze_40b.py: post-cascade blocks are flat, so `>=`
counted ties as local maxima.  Here attention hits use STRICT comparison and are
restricted to interior PRE-ONSET blocks only.
"""
import json, os, numpy as np

R = "/path/to/TDiG/arch_compare/results"
ATTN40 = [3, 10, 17, 24, 31, 35, 42, 49]
GROUP = {"coding_exon": "context", "intergenic": "context",
         "splice_donor": "motif", "splice_acceptor": "motif"}
rng = np.random.default_rng(0)

# 7B reference (fp32 late eval with error bars + full fp16 sweep)
late7 = json.load(open(f"{R}/late_eval_reps.json"))
sweep7 = {r["chrom"]: r["auroc"] for r in json.load(open(f"{R}/layer_select.json"))}

rows, attn_cells = [], []
for chrom in ("chr22", "chr17"):
    p = f"{R}/{chrom}_evo2_40b/analysis.json"
    if not os.path.exists(p):
        print(f"{chrom}: analysis.json missing"); continue
    a = json.load(open(p)); onset = a["onset"]; nb = len(a["ratio"]) + 1
    for nm, curve in a["auroc"].items():
        c = np.array(curve[:nb]); hn = curve[nb]; b = int(c.argmax())
        band = c[onset - 4:onset]; bb = onset - 4 + int(band.argmax())
        r1 = c[onset - 1]
        v7 = late7[chrom][nm]; best7 = max(v7[k][0] for k in ("h24", "h25", "h26", "h27"))
        rows.append((chrom, nm, b, c[b], bb, band.max(), onset - 1, r1, hn, c[-1], best7, v7["hnorm"][0]))
        pre = list(range(1, onset - 1))                              # interior, strictly before onset
        loc = set(l for l in pre if c[l] > c[l - 1] and c[l] > c[l + 1])
        at = [l for l in ATTN40 if l in pre]
        attn_cells.append((chrom, nm, GROUP[nm], loc, at, len(pre)))

print("=" * 118)
print("40B vs 7B: reading position")
print("=" * 118)
print(f"  {'':<6}{'region':<17}{'40B best':>14}{'band best':>14}{'onset-1':>13}{'h_norm':>9}{'last':>8}"
      f"{'40B gain':>10}{'7B gain':>10}")
for ch, nm, b, cb, bb, bm, o1, r1, hn, last, b7, hn7 in rows:
    print(f"  {ch:<6}{nm:<17}{f'{cb:.3f}(b{b})':>14}{f'{bm:.3f}(b{bb})':>14}{f'{r1:.3f}(b{o1})':>13}"
          f"{hn:>9.3f}{last:>8.3f}{bm-hn:>+10.3f}{b7-hn7:>+10.3f}")
if rows:
    g40 = np.array([r[5] - r[8] for r in rows]); reg = np.array([r[3] - r[5] for r in rows])
    reg1 = np.array([r[3] - r[7] for r in rows])
    print(f"\n  40B mean gain of pre-onset band over h_norm {g40.mean():+.3f} (min {g40.min():+.3f}) | "
          f"band regret vs oracle mean {reg.mean():.3f} max {reg.max():.3f} | "
          f"onset-1 regret mean {reg1.mean():.3f} max {reg1.max():.3f}")

print("\n" + "=" * 118)
print("attention blocks as STRICT local maxima, pre-onset interior blocks only")
print("=" * 118)
for grp in ("context", "motif"):
    sub = [x for x in attn_cells if x[2] == grp]
    if not sub: continue
    obs = sum(len(set(x[4]) & x[3]) for x in sub); tot = sum(len(x[4]) for x in sub)
    null = np.array([sum(len(set(rng.choice(x[4] and np.arange(1, x[5] + 1), len(x[4]), replace=False)) & x[3])
                         for x in sub) for _ in range(100000)])
    print(f"  {grp:<8} hits {obs}/{tot} | null mean {null.mean():.2f} | permutation p = {(null >= obs).mean():.4f}")
for ch, nm, g, loc, at, n in attn_cells:
    print(f"    {ch} {nm:<16} attention {at} -> hits {sorted(set(at) & loc)} | local maxima {len(loc)}/{n}")

json.dump({"rows": [list(map(lambda v: v if isinstance(v, str) else float(v), r)) for r in rows]},
          open(f"{R}/post40b.json", "w"), indent=1)
