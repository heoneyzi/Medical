"""E3: does the rule's block beat the final embedding on a standard benchmark?

Features come from bend_extract.py: Evo 2 7B hidden states at 26,174 positions on
chr17/19/22, with BEND's own nine-class gene-finding labels and BEND's own
train/valid/test split, so neither the labels nor the split are ours.

One caveat governs how these numbers may be read.  Storing every position was not
possible, so the four rare classes were taken whole and the common ones capped.
The evaluation set is therefore NOT BEND's natural class distribution, and these
MCC values are not comparable with published BEND leaderboard numbers.  They are a
within-row comparison between blocks of the same model, which is the only thing the
paper claims.
"""
import numpy as np, json, os, sys, collections, warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.metrics import matthews_corrcoef, f1_score, accuracy_score

D="/path/to/TDiG/arch_compare/results/bend_gf"
RULE=24
for p in (f"{D}/H.npy", f"{D}/meta.npz"):
    if not os.path.exists(p): print("ABORT missing", p); sys.exit(1)

m=np.load(f"{D}/meta.npz", allow_pickle=True)
y=m["y"].astype(int); split=np.array([str(s) for s in m["split"]])
tid=np.array([str(s) for s in m["tid"]]); keys=[str(k) for k in m["keys"]]
H=np.load(f"{D}/H.npy", mmap_mode="r")
print(f"rows {len(y):,} | H {H.shape} | blocks {len(keys)}", flush=True)
print("  class counts:", dict(sorted(collections.Counter(y.tolist()).items())), flush=True)
print("  split counts:", dict(collections.Counter(split.tolist())), flush=True)
print(f"  transcripts: {len(set(tid.tolist()))}", flush=True)

tr=np.where(split=="train")[0]; va=np.where(split=="valid")[0]; te=np.where(split=="test")[0]
if len(te)==0: print("ABORT: no test rows"); sys.exit(1)
# BEND splits by transcript; verify no transcript straddles train and test
assert not (set(tid[tr]) & set(tid[te])), "a transcript appears in both train and test"
print(f"  train {len(tr):,} | valid {len(va):,} | test {len(te):,} | no transcript overlap", flush=True)

def evaluate(j):
    X=np.nan_to_num(np.asarray(H[:,j,:],dtype=np.float32),nan=0.,posinf=0.,neginf=0.)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", ConvergenceWarning)
        p=make_pipeline(StandardScaler(),PCA(256,random_state=0),
                        LogisticRegression(max_iter=20000,n_jobs=-1)).fit(X[tr],y[tr])
        nconv=sum(1 for x in w if issubclass(x.category,ConvergenceWarning))
    out={}
    for nm,idx in (("valid",va),("test",te)):
        if len(idx)==0: continue
        pr=p.predict(X[idx])
        out[nm]={"mcc":float(matthews_corrcoef(y[idx],pr)),
                 "macro_f1":float(f1_score(y[idx],pr,average="macro")),
                 "acc":float(accuracy_score(y[idx],pr))}
    out["unconverged"]=int(nconv)
    return out

OUT={"n":int(len(y)),"blocks":keys,"note":"rare classes taken whole, common classes capped; "
     "not comparable with published BEND numbers"}
curve=[]
for j,k in enumerate(keys):
    r=evaluate(j); curve.append(r.get("test",{}).get("mcc",float("nan")))
    OUT[k]=r
    tag=""
    if k==f"b{RULE}": tag="   <- the rule's block"
    if k==keys[-1]:   tag="   <- the final embedding"
    print(f"  {k:>5}: test MCC {r['test']['mcc']:.4f} | macro-F1 {r['test']['macro_f1']:.4f} "
          f"| acc {r['test']['acc']:.4f}{tag}", flush=True)

c=np.array(curve[:-1]); b=int(np.nanargmax(c))
rule_mcc=curve[RULE]; fin_mcc=curve[-1]
OUT["summary"]={"rule_block":RULE,"rule_mcc":float(rule_mcc),"final_mcc":float(fin_mcc),
                "oracle_block":b,"oracle_mcc":float(c[b]),
                "gain_over_final":float(rule_mcc-fin_mcc),"regret":float(c[b]-rule_mcc)}
print(f"\n  rule b{RULE} MCC {rule_mcc:.4f} | final embedding {fin_mcc:.4f} "
      f"| gain {rule_mcc-fin_mcc:+.4f}", flush=True)
print(f"  oracle block b{b} MCC {c[b]:.4f} | rule regret {c[b]-rule_mcc:.4f}", flush=True)
json.dump(OUT,open(f"{D}/bend_eval.json","w"),indent=1)
print("\nwrote bend_eval.json", flush=True)
