"""E2: Evo 2 read at the rule's block against GPN-MSA on GPN-MSA's own chr17 panel.

The panel ships published scores for every variant (GPN-MSA, CADD, phyloP, ESM-1b,
SpliceAI), so the comparison is on identical variants rather than on an approximate
join, which an earlier attempt showed was hopeless: the overlap with our own panels
was 69 BRCA1 variants and 44 ClinVar negatives.

The panel's Gene column is empty for every row, so genes are assigned from GENCODE
v44 by position.  That matters: without leave-genes-out folds a variant's neighbours
in the same gene sit on both sides of the split and the readout scores itself.
"""
import numpy as np, pandas as pd, gzip, json, collections, os, sys
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

AC="/path/to/TDiG/arch_compare"; R=f"{AC}/results"
GTF=f"{AC}/refs/gencode.v44.annotation.gtf.gz"
RULE=24
need=[f"{R}/gpnmsa17_meta.npz",f"{R}/gpnmsa17_dh_tok.npy",f"{R}/gpnmsa17_dh_pool.npy",
      f"{R}/gpnmsa17_panel.parquet",GTF]
miss=[p for p in need if not os.path.exists(p)]
if miss: print("ABORT, missing:",miss); sys.exit(1)

m=np.load(f"{R}/gpnmsa17_meta.npz",allow_pickle=True)
y=m["y"].astype(int); pos=m["pos"].astype(np.int64); keys=[str(k) for k in m["keys"]]
panel=pd.read_parquet(f"{R}/gpnmsa17_panel.parquet")
print(f"variants {len(y)} | positive {y.sum()} | blocks {len(keys)}", flush=True)

# panel rows must line up with the extraction, or every comparison below is wrong
assert len(panel)==len(y), f"panel {len(panel)} vs meta {len(y)}"
assert np.array_equal(panel["pos"].to_numpy(np.int64), pos), "panel/meta position order differs"
assert np.array_equal(panel["y"].to_numpy(int), y), "panel/meta label mismatch"
print("panel and extraction agree on position and label, row for row", flush=True)

# ---- genes from GENCODE, by position -------------------------------------
iv=[]
with gzip.open(GTF,"rt") as fh:
    for line in fh:
        if line.startswith("#"): continue
        f=line.split("\t")
        if len(f)<9 or f[0]!="chr17" or f[2]!="gene": continue
        nm=None
        for part in f[8].split(";"):
            part=part.strip()
            if part.startswith("gene_name"): nm=part.split('"')[1]; break
        if nm: iv.append((int(f[3])-1,int(f[4]),nm))
iv.sort()
starts=np.array([a for a,b,n in iv]); ends=np.array([b for a,b,n in iv])
names=[n for a,b,n in iv]
print(f"GENCODE chr17 genes: {len(iv)}", flush=True)
gene=np.empty(len(pos),dtype=object)
for i,p in enumerate(pos):
    j=np.searchsorted(starts,p,side="right")-1
    hit=None
    for k in range(max(0,j-40), min(len(iv), j+2)):
        if starts[k]<=p<ends[k]: hit=names[k]; break
    gene[i]= hit if hit else f"intergenic_{p//1000000}Mb"
ng=len(set(gene.tolist())); nint=sum(1 for g in gene if str(g).startswith("intergenic_"))
print(f"assigned {ng} distinct groups | {nint} variants fell outside any gene", flush=True)
cnt=collections.Counter(gene.tolist())
print("  largest groups:", cnt.most_common(5), flush=True)

def grouped_auroc(X):
    per=[]
    for tr,te in GroupKFold(5).split(X,y,gene):
        p=make_pipeline(StandardScaler(),PCA(128,random_state=0),
                        LogisticRegression(max_iter=3000)).fit(X[tr],y[tr])
        per.append(roc_auc_score(y[te],p.predict_proba(X[te])[:,1]))
    return float(np.mean(per)), float(np.std(per,ddof=1)), [float(v) for v in per]

OUT={}
# ---- published scores on the same variants, no training ------------------
print("\n=== published scores, evaluated directly on these variants ===", flush=True)
for col in ("GPN-MSA","CADD","phyloP","phyloP-241-mammals","ESM-1b","SpliceAI","NT"):
    if col not in panel.columns: continue
    v=panel[col].to_numpy(float); ok=np.isfinite(v)
    if ok.sum()<len(v)*0.5: print(f"  {col:<20} too many missing ({ok.sum()}/{len(v)})"); continue
    a=roc_auc_score(y[ok],v[ok]); a=max(a,1-a)
    OUT[f"published|{col}"]={"auroc":float(a),"n":int(ok.sum())}
    print(f"  {col:<20} AUROC {a:.3f}  (n={ok.sum()})", flush=True)

# ---- Evo 2 readout, same folds -------------------------------------------
print("\n=== Evo 2 7B readout, GroupKFold(5) by GENCODE gene, PCA(128) ===", flush=True)
for style in ("tok","pool"):
    A=np.load(f"{R}/gpnmsa17_dh_{style}.npy",mmap_mode="r")
    curve=[]
    for j,k in enumerate(keys):
        X=np.nan_to_num(np.asarray(A[:,j,:],dtype=np.float64),nan=0.,posinf=0.,neginf=0.)
        mu,sd,per=grouped_auroc(X); curve.append(mu)
        if k in (f"b{RULE}", keys[-1]):
            tag="rule_b24" if k==f"b{RULE}" else "final_norm"
            OUT[f"evo2|{style}|{tag}"]={"auroc":mu,"sd":sd,"per_fold":per}
            print(f"  {style:<5} {tag:<11} AUROC {mu:.3f} +- {sd:.3f}  folds {[round(v,3) for v in per]}", flush=True)
    c=np.array(curve[:len(keys)-1]); b=int(c.argmax())
    OUT[f"evo2|{style}|oracle"]={"block":b,"auroc":float(c.max())}
    OUT[f"evo2|{style}|curve"]=[float(v) for v in curve]
    print(f"  {style:<5} oracle block b{b} = {c.max():.3f} | rule b{RULE} = {curve[RULE]:.3f} "
          f"| final = {curve[-1]:.3f} | regret {c.max()-curve[RULE]:+.4f}", flush=True)

json.dump(OUT,open(f"{R}/gpnmsa_eval.json","w"),indent=1)
print("\nwrote gpnmsa_eval.json", flush=True)
