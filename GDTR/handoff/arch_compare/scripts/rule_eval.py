"""Evaluate label-free layer-selection rules against the oracle."""
import numpy as np, json, itertools
from scipy.stats import spearmanr

R = "/path/to/TDiG/arch_compare/results"
res = json.load(open(f"{R}/layer_select.json"))
ATTN = [3, 10, 17, 24]; LAST_ATTN = 24; CONV = 28
BT = {}
for n, ls in [("hcs",[0,4,7,11,14,18,21,25,28]),("hcm",[1,5,8,12,15,19,22,26,29]),
              ("hcl",[2,6,9,13,16,20,23,27,30]),("attn",[3,10,17,24,31])]:
    for l in ls: BT[l] = n

# ---- 1. are attention blocks local maxima? ----
print("="*88); print("1. attention blocks as local maxima (interior layers 1..27)"); print("="*88)
tot_a = tot_hit = 0; pvals = []
for r in res:
    for nm, c in r["auroc"].items():
        c = np.array(c)
        loc = [l for l in range(1, 28) if c[l] >= c[l-1] and c[l] >= c[l+1]]
        hit = [l for l in ATTN if l in loc]
        # exact prob that all 4 attn layers land in the observed local-max set
        k, n = len(loc), 27
        p = np.prod([(k - i) / (n - i) for i in range(4)]) if k >= 4 else 0.0
        tot_a += 4; tot_hit += len(hit); pvals.append(p)
        print(f"  {r['chrom']:<6}{nm:<17} attn local-max {len(hit)}/4 {str(hit):<22}"
              f" (all local maxima: {k}/27, P[all 4]={p:.4f})")
print(f"\n  overall {tot_hit}/{tot_a} attention layers are local maxima"
      f" | Fisher combined p = {np.exp(np.sum(np.log(np.clip(pvals,1e-12,1)))):.2e}")

# ---- 2. rule regret ----
print("\n"+"="*88); print("2. regret of label-free rules vs the oracle"); print("="*88)
rows = []
for r in res:
    st = {int(k): v for k, v in r["stats"].items()}
    aniso_min = min(range(9, 29), key=lambda l: st[l]["aniso"])      # geometry rule
    rank_rec  = max(range(9, 29), key=lambda l: st[l]["eff_rank"])   # rank-recovery rule
    for nm, c in r["auroc"].items():
        c = np.array(c); b = int(c.argmax())
        rows.append((r["chrom"], nm, b, c[b], c[LAST_ATTN], c[aniso_min], c[rank_rec], c[CONV],
                     aniso_min, rank_rec))
hdr = f"  {'':<6}{'':<17}{'oracle':>16}{'last-attn L24':>15}{'aniso-min':>13}{'rank-rec':>12}{'conv L28':>11}"
print(hdr)
for ch, nm, b, o, la, am, rr, cv, ai, ri in rows:
    print(f"  {ch:<6}{nm:<17}{f'{o:.3f} (L{b})':>16}{la:>15.3f}{am:>13.3f}{rr:>12.3f}{cv:>11.3f}")
A = np.array([[o, la, am, rr, cv] for *_, o, la, am, rr, cv, _, _ in
              [(0,0,0,*r[3:8],r[8],r[9]) for r in rows]])
names = ["last-attn L24", "aniso-min", "rank-recovery", "conventional L28"]
print(f"\n  {'rule':<20}{'mean regret':>13}{'max regret':>12}{'layer picked':>28}")
for j, nmr in enumerate(names, start=1):
    reg = A[:, 0] - A[:, j]
    pick = ("L24 (fixed)" if j == 1 else
            f"chr22 L{rows[0][8] if j==2 else rows[0][9]}, chr17 L{rows[4][8] if j==2 else rows[4][9]}"
            if j in (2, 3) else "L28 (fixed)")
    print(f"  {nmr:<20}{reg.mean():>13.4f}{reg.max():>12.4f}{pick:>28}")

# ---- 3. do label-free statistics track the supervised curve? ----
print("\n"+"="*88); print("3. Spearman(label-free statistic, AUROC) across layers 1..28"); print("="*88)
keys = ["norm", "pr", "eff_rank", "top1_var", "aniso", "vel", "curv"]
print(f"  {'':<6}{'':<17}" + "".join(f"{k:>11}" for k in keys))
for r in res:
    st = {int(k): v for k, v in r["stats"].items()}
    for nm, c in r["auroc"].items():
        line = ""
        for k in keys:
            x = np.array([st[l].get(k, np.nan) for l in range(1, 29)])
            y = np.array(c[1:29]); m = np.isfinite(x) & np.isfinite(y)
            line += f"{spearmanr(x[m], y[m]).correlation:>11.3f}"
        print(f"  {r['chrom']:<6}{nm:<17}" + line)
