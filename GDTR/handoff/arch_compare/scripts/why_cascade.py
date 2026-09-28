"""What actually blows up at L28?  A few positions, a few channels, or everything?"""
import numpy as np
R = "/path/to/TDiG/arch_compare/results"

for chrom in ("chr22", "chr17"):
    d = np.load(f"{R}/{chrom}_late_fp32.npz")
    lab, win = d["label"], d["window"]
    idx = np.arange(len(lab))
    # position offset within its window (rows are stored in window order)
    off = idx - np.concatenate(([0], np.maximum.accumulate(
        np.where(win[1:] != win[:-1], idx[1:], 0))))[:len(idx)]
    print(f"\n{'='*104}\n{chrom}\n{'='*104}")
    print(f"  {'layer':<7}{'mean':>11}{'median':>11}{'p99':>11}{'max':>11}"
          f"{'mean/med':>10}{'top1 ch':>9}{'ch for 90%':>12}")
    prev_top = None
    for l in list(range(24, 32)) + ["norm"]:
        H = d[f"h{l}" if l != "norm" else "hnorm"]
        n = np.linalg.norm(H, axis=1)
        sq = H.astype(np.float64) ** 2
        chan = sq.mean(0)                          # mean energy per channel
        chan_sorted = np.sort(chan)[::-1]
        share = chan_sorted / chan_sorted.sum()
        k90 = int(np.searchsorted(np.cumsum(share), 0.90) + 1)
        top = np.argsort(chan)[::-1][:8]
        print(f"  L{str(l):<6}{n.mean():>11.4g}{np.median(n):>11.4g}"
              f"{np.percentile(n,99):>11.4g}{n.max():>11.4g}"
              f"{n.mean()/np.median(n):>10.2f}{share[0]:>9.3f}{k90:>12}")
        if l == 28:
            print(f"      top channels @L28: {list(top)}")
            prev_top = set(top.tolist())
        if l == 27:
            print(f"      top channels @L27: {list(top)}")
            prev27 = set(top.tolist())
    print(f"      L27 vs L28 top-8 channel overlap: {len(prev27 & prev_top)}/8")

    # is the blow-up carried by particular positions?
    n27 = np.linalg.norm(d["h27"], axis=1); n28 = np.linalg.norm(d["h28"], axis=1)
    r = n28 / n27
    print(f"\n  per-position ratio L28/L27: min {r.min():.1f}  p1 {np.percentile(r,1):.1f}  "
          f"median {np.median(r):.1f}  p99 {np.percentile(r,99):.1f}  max {r.max():.1f}")
    print(f"  fraction of positions with ratio > 100: {(r > 100).mean():.4f}")
    print(f"  corr(window offset, ratio) = {np.corrcoef(off, r)[0,1]:+.3f}"
          f" | ratio at offset<10: {r[off < 10].mean():.1f} vs rest {r[off >= 10].mean():.1f}")
