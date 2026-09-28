"""The reviewer's strongest objection: is the norm-ratio rule just a constant
fraction of depth in disguise?

Compare, on every model x chromosome x region we have: our rule (4 blocks before
the first adjacent-norm-ratio >= 10) against fixed relative depths, scored as
regret against the oracle layer of that cell.
"""
import json, numpy as np
R = "/path/to/TDiG/arch_compare/results"
CELLS = []
for r in json.load(open(f"{R}/layer_select.json")):          # 7B chr22/chr17, layers 0..28
    CELLS.append(("7B", r["chrom"], 32, 28, {k: np.array(v) for k, v in r["auroc"].items()}))
a = json.load(open(f"{R}/chr19_evo2_7b_all/analysis.json"))  # 7B chr19, layers 0..31(+norm)
CELLS.append(("7B", "chr19", 32, a["onset"], {k: np.array(v[:32]) for k, v in a["auroc"].items()}))
for ch in ("chr22", "chr17"):
    a = json.load(open(f"{R}/{ch}_evo2_40b/analysis.json"))
    CELLS.append(("40B", ch, 50, a["onset"], {k: np.array(v[:50]) for k, v in a["auroc"].items()}))

FRACS = [0.50, 0.60, 0.70, 0.75, 0.80, 0.85]
rows = {f"fixed {int(f*100)}% depth": [] for f in FRACS}
rows["our rule (onset-4)"] = []
picks = {}
for model, ch, NB, onset, auroc in CELLS:
    for f in FRACS:
        L = min(int(round(f * NB)), max(auroc[list(auroc)[0]].shape[0] - 1, 0))
        for nm, c in auroc.items():
            if L < len(c): rows[f"fixed {int(f*100)}% depth"].append(c.max() - c[L])
        picks.setdefault(f"fixed {int(f*100)}% depth", {})[f"{model} {ch}"] = L
    L = max(0, onset - 4)
    for nm, c in auroc.items():
        if L < len(c): rows["our rule (onset-4)"].append(c.max() - c[L])
    picks.setdefault("our rule (onset-4)", {})[f"{model} {ch}"] = L

print(f"{'rule':<24}{'n':>4}{'mean regret':>13}{'median':>9}{'max':>9}   picked layer per cell")
for k in ["our rule (onset-4)"] + [f"fixed {int(f*100)}% depth" for f in FRACS]:
    v = np.array(rows[k])
    pk = " ".join(f"{c.split()[0]}/{c.split()[1][3:]}:{l}" for c, l in picks[k].items())
    print(f"  {k:<22}{len(v):>4}{v.mean():>13.4f}{np.median(v):>9.4f}{v.max():>9.4f}   {pk}")

print("\nrelative depth of our rule's pick, per model:")
for model, ch, NB, onset, _ in CELLS:
    print(f"  {model:<4}{ch:<7}blocks {NB:<3} onset {onset:<3} pick {onset-4:<3} = {100*(onset-4)/NB:.0f}% depth")
