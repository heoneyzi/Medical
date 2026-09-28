"""Within the pre-onset band, can a label-free statistic pick the best layer?
7B band L24-27 (oracle from fp32 5-rep means); 40B band b17-20 (single run)."""
import json, numpy as np
R = "/path/to/TDiG/arch_compare/results"
late7 = json.load(open(f"{R}/late_eval_reps.json"))
st7 = {r["chrom"]: {int(k): v for k, v in r["stats"].items()} for r in json.load(open(f"{R}/layer_select.json"))}
ATTN7, ATTN40 = {3, 10, 17, 24, 31}, {3, 10, 17, 24, 31, 35, 42, 49}
cells = []
for ch in ("chr22", "chr17"):
    band = [24, 25, 26, 27]; st = st7[ch]
    for nm, v in late7[ch].items():
        auc = {l: v[f"h{l}"][0] for l in band}
        feats = {"eff_rank": {l: st[l]["eff_rank"] for l in band}, "aniso": {l: -st[l]["aniso"] for l in band},
                 "norm": {l: -st[l]["norm"] for l in band}}
        cells.append(("7B", ch, nm, band, auc, feats, ATTN7))
    a = json.load(open(f"{R}/{ch}_evo2_40b/analysis.json")); on = a["onset"]
    band = list(range(on - 4, on)); g = a["geometry"]
    for nm, c in a["auroc"].items():
        auc = {l: c[l] for l in band}
        feats = {"eff_rank": {l: g[f"b{l}"]["eff_rank"] for l in band}, "share": {l: g[f"b{l}"]["share"] for l in band}}
        cells.append(("40B", ch, nm, band, auc, feats, ATTN40))
rules = {"band start (onset-4)": lambda b, f, at: b[0], "onset-2": lambda b, f, at: b[2],
         "onset-1": lambda b, f, at: b[-1],
         "last attention in band else onset-4": lambda b, f, at: max([l for l in b if l in at], default=b[0])}
for key in ("eff_rank", "share", "aniso", "norm"):
    rules[f"argmax {key}" + (" (-)" if key in ("aniso", "norm") else "")] = (lambda k: lambda b, f, at: max(b, key=lambda l: f[k][l]) if k in f else None)(key)
print(f"{'rule':<40}{'model':>6}{'n':>4}{'mean regret':>13}{'max regret':>12}{'hits':>6}")
for rn, fn in rules.items():
    for model in ("7B", "40B"):
        reg = []; hit = 0
        for m, ch, nm, b, auc, f, at in cells:
            if m != model: continue
            pick = fn(b, f, at)
            if pick is None: continue
            best = max(auc.values()); reg.append(best - auc[pick]); hit += auc[pick] == best
        if reg: print(f"{rn:<40}{model:>6}{len(reg):>4}{np.mean(reg):>13.4f}{np.max(reg):>12.4f}{hit:>6}")
print("\nper-cell oracle layer:")
for m, ch, nm, b, auc, f, at in cells:
    print(f"  {m:<4}{ch:<6}{nm:<17} best {max(auc, key=auc.get)} | spread {max(auc.values())-min(auc.values()):.3f}")
