"""(1) Is the cascade a huge SHARED offset that swamps the position-to-position
signal?  (2) At what depth does each region type become decodable?"""
import numpy as np, json
R = "/path/to/TDiG/arch_compare/results"

print("="*104)
print("1. shared offset vs informative variation")
print("="*104)
for chrom in ("chr22", "chr17"):
    d = np.load(f"{R}/{chrom}_late_fp32.npz")
    print(f"\n  {chrom}")
    print(f"    {'layer':<8}{'mean||h||':>12}{'||mean h||':>12}{'mean||h-mu||':>14}"
          f"{'signal share':>14}{'SNR':>10}")
    for l in list(range(24, 32)) + ["norm"]:
        H = d[f"h{l}" if l != "norm" else "hnorm"].astype(np.float64)
        mu = H.mean(0)
        nh = np.linalg.norm(H, axis=1).mean()
        nmu = np.linalg.norm(mu)
        nc = np.linalg.norm(H - mu, axis=1).mean()
        print(f"    L{str(l):<7}{nh:>12.4g}{nmu:>12.4g}{nc:>14.4g}"
              f"{nc/nh:>14.4f}{nc/nmu:>10.4f}")

print("\n" + "="*104)
print("2. resolution depth: first layer reaching 90% of the achievable gain")
print("="*104)
sel = json.load(open(f"{R}/layer_select.json"))
print(f"\n  {'':<6}{'region':<17}{'L0':>7}{'max':>8}{'@L':>5}{'90% 도달층':>12}{'유형':>10}")
BT = {}
for n, ls in [("hcs",[0,4,7,11,14,18,21,25,28]),("hcm",[1,5,8,12,15,19,22,26,29]),
              ("hcl",[2,6,9,13,16,20,23,27,30]),("attn",[3,10,17,24,31])]:
    for l in ls: BT[l] = n
rows = {}
for r in sel:
    for nm, c in r["auroc"].items():
        c = np.array(c); base = c[0]; top = c.max()
        thr = base + 0.90 * (top - base)
        first = int(np.argmax(c >= thr))
        rows.setdefault(nm, []).append(first)
        print(f"  {r['chrom']:<6}{nm:<17}{base:>7.3f}{top:>8.3f}{int(c.argmax()):>5}"
              f"{f'L{first}':>12}{BT[first]:>10}")
print()
for nm, v in rows.items():
    print(f"  {nm:<17} 두 염색체 도달층 {v}")
