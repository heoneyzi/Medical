"""C5: fold-level spread for the three downstream tasks, at the rule's block.

The published numbers are pooled: cross_val_score(...).mean() for the AUROCs and
one Spearman over pooled out-of-fold predictions.  Neither carries a spread, which
is the reviewer's point.  This re-runs the same readouts and keeps the per-fold
values, so the paper can quote mean +- sd across folds alongside the pooled figure.

Everything here mirrors clinvar_eval.py / brca1_eval.py / phylop_layers.py exactly,
including PCA width (128 for the variant tasks, 256 for phyloP), the estimator, the
grouping variable, and phyloP's subsample seed.  Only the rule's block and the final
embedding are evaluated, which is all the paper contrasts.
"""
import numpy as np, json, os, sys
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.mixture import GaussianMixture

ROOT="/path/to/TDiG"; AC=f"{ROOT}/arch_compare"; R=f"{AC}/results"
RULE=24                       # 7B: onset 28, rule reads block onset-4
need=[f"{R}/clinvar17_meta.npz",f"{R}/clinvar17_dh_tok.npy",f"{R}/clinvar17_dh_pool.npy",
      f"{R}/brca1_meta.npz",f"{R}/brca1_dh_tok.npy",f"{R}/brca1_dh_pool.npy",
      f"{AC}/tracks/hg38.phyloP100way.bw"]
miss=[p for p in need if not os.path.exists(p)]
if miss:
    print("ABORT, missing inputs:"); [print("   ",p) for p in miss]; sys.exit(1)
print("all inputs present", flush=True)
OUT={}

def summarise(per):
    per=[float(v) for v in per]
    return {"per_fold":per,"mean":float(np.mean(per)),"sd":float(np.std(per,ddof=1))}

# ----------------------------------------------------------- ClinVar (by gene)
print("\n=== ClinVar missense, GroupKFold(5) by gene, PCA(128) ===", flush=True)
m=np.load(f"{R}/clinvar17_meta.npz",allow_pickle=True)
y=m["y"].astype(int); gene=m["gene"]; keys=[str(k) for k in m["keys"]]
print(f"variants {len(y)} | pathogenic {y.sum()} | genes {len(set(gene.tolist()))}", flush=True)
for style in ("tok","pool"):
    A=np.load(f"{R}/clinvar17_dh_{style}.npy",mmap_mode="r")
    for tag,j in (("rule_b24",RULE),("final_norm",len(keys)-1)):
        X=np.nan_to_num(np.asarray(A[:,j,:],dtype=np.float64),nan=0.,posinf=0.,neginf=0.)
        per=[]
        for tr,te in GroupKFold(5).split(X,y,gene):
            p=make_pipeline(StandardScaler(),PCA(128,random_state=0),
                            LogisticRegression(max_iter=3000)).fit(X[tr],y[tr])
            per.append(roc_auc_score(y[te],p.predict_proba(X[te])[:,1]))
        s=summarise(per); OUT[f"clinvar|{style}|{tag}"]=s
        print(f"  {style:<5} {tag:<11} AUROC {s['mean']:.3f} +- {s['sd']:.3f}   "
              f"folds {[round(v,3) for v in s['per_fold']]}", flush=True)

# ------------------------------------------------------------ BRCA1 (by exon)
print("\n=== BRCA1 SGE, GroupKFold(5) by exon, PCA(128) ===", flush=True)
m=np.load(f"{R}/brca1_meta.npz",allow_pickle=True)
score=m["score"].astype(float); exon=m["exon"].astype(int); keys=[str(k) for k in m["keys"]]
gm=GaussianMixture(2,random_state=0).fit(score.reshape(-1,1))
lo=int(np.argmin(gm.means_.ravel())); yb=(gm.predict(score.reshape(-1,1))==lo).astype(int)
print(f"variants {len(score)} | exons {len(set(exon.tolist()))} | abnormal {yb.sum()}", flush=True)
for style in ("tok","pool"):
    A=np.load(f"{R}/brca1_dh_{style}.npy",mmap_mode="r")
    for tag,j in (("rule_b24",RULE),("final_norm",len(keys)-1)):
        X=np.nan_to_num(np.asarray(A[:,j,:],dtype=np.float64),nan=0.,posinf=0.,neginf=0.)
        rp=[]; ap=[]; pooled=np.zeros(len(score))
        for tr,te in GroupKFold(5).split(X,score,exon):
            r=make_pipeline(StandardScaler(),PCA(128,random_state=0),
                            Ridge(alpha=1.0)).fit(X[tr],score[tr])
            pr=r.predict(X[te]); pooled[te]=pr
            rp.append(spearmanr(pr,score[te]).statistic)
            c=make_pipeline(StandardScaler(),PCA(128,random_state=0),
                            LogisticRegression(max_iter=3000)).fit(X[tr],yb[tr])
            ap.append(roc_auc_score(yb[te],c.predict_proba(X[te])[:,1]))
        sr=summarise(rp); sa=summarise(ap)
        sr["pooled"]=float(spearmanr(pooled,score).statistic)
        OUT[f"brca1|{style}|{tag}|spearman"]=sr; OUT[f"brca1|{style}|{tag}|auroc"]=sa
        print(f"  {style:<5} {tag:<11} rho pooled {sr['pooled']:+.3f} | folds {sr['mean']:+.3f} +- {sr['sd']:.3f}"
              f" | AUROC {sa['mean']:.3f} +- {sa['sd']:.3f}", flush=True)

# --------------------------------------------------------- phyloP (by window)
print("\n=== phyloP, GroupKFold(5) by window, PCA(256), Ridge(10.0) ===", flush=True)
import pandas as pd, pyBigWig
BW=pyBigWig.open(f"{AC}/tracks/hg38.phyloP100way.bw"); NSUB=6000
def windows(ch):
    scal={"chr22":"evo2_7b_scalars.npz","chr17":"chr17_evo2_7b_scalars.npz"}[ch]
    w=np.load(f"{R}/{scal}")["window"]
    meta=pd.read_parquet(f"{ROOT}/tdig_integration/data_cache_minimal/{ch}_metadata.parquet")
    st=meta.set_index("window_idx")["start"].to_dict(); en=meta.set_index("window_idx")["end"].to_dict()
    labels=np.load(f"{ROOT}/paper/data_local/{ch}_position_labels.npy")
    use=[int(x) for x in w[np.concatenate(([True],w[1:]!=w[:-1]))]]
    for x in use:
        a,b=int(st[x]),int(en[x]); lab=np.zeros(b-a,np.uint8); gp=a+np.arange(b-a)
        ok=(gp>=0)&(gp<len(labels)); lab[ok]=labels[gp[ok]]
        yield x,a,lab
def replay_7b(ch):
    rng=np.random.default_rng(20260903); pos=[]; labs=[]
    for w,a,lab in windows(ch):
        sel=np.isin(lab,[2,5,6])
        for code,frac in {1:0.035,0:0.035}.items():
            idx=np.where(lab==code)[0]
            if len(idx): sel[rng.choice(idx,max(1,int(len(idx)*frac)),replace=False)]=True
        o=np.where(sel)[0]; pos.append(a+o); labs.append(lab[o])
    return np.concatenate(pos),np.concatenate(labs)
def phylop(ch,pos):
    out=np.full(len(pos),np.nan,np.float32); order=np.argsort(pos); p=pos[order]
    for i in range(0,len(p),20000):
        chunk=p[i:i+20000]; lo,hi=int(chunk.min()),int(chunk.max())+1
        v=np.array(BW.values(ch,lo,hi),np.float32); out[order[i:i+20000]]=v[chunk-lo]
    return out
for ch in ("chr22","chr17"):
    late=np.load(f"{R}/{ch}_late_fp32.npz"); pos,lab=replay_7b(ch)
    assert len(pos)==len(late["label"]) and np.array_equal(lab,late["label"]), f"{ch} replay mismatch"
    print(f"  7B {ch}: replay matches {len(pos)} rows", flush=True)
    y=phylop(ch,pos); g=late["window"]; rng=np.random.default_rng(3)
    for sname,mask in {"all":np.ones(len(y),bool),"noncoding":np.isin(lab,[0,1]),
                       "coding_exon":lab==2}.items():
        idx=np.where(mask&np.isfinite(y))[0]
        if len(idx)>NSUB: idx=np.sort(rng.choice(idx,NSUB,replace=False))
        yy,gg=y[idx],g[idx]
        for tag,l in (("rule_b24",RULE),("final_norm","norm")):
            X=np.nan_to_num(late["hnorm" if l=="norm" else f"h{l}"][idx].astype(np.float64),
                            nan=0.,posinf=0.,neginf=0.)
            per=[]; pooled=np.zeros(len(idx))
            for tr,te in GroupKFold(5).split(X,yy,gg):
                mdl=make_pipeline(StandardScaler(),PCA(256,random_state=0),
                                  Ridge(alpha=10.0)).fit(X[tr],yy[tr])
                pr=mdl.predict(X[te]); pooled[te]=pr
                per.append(spearmanr(pr,yy[te]).correlation)
            s=summarise(per); s["pooled"]=float(spearmanr(pooled,yy).correlation); s["n"]=int(len(idx))
            OUT[f"phylop|{ch}|{sname}|{tag}"]=s
            print(f"  {ch} {sname:<12} {tag:<11} rho pooled {s['pooled']:+.3f} | "
                  f"folds {s['mean']:+.3f} +- {s['sd']:.3f}  n={len(idx)}", flush=True)

json.dump(OUT,open(f"{R}/fold_sd.json","w"),indent=1)
print("\nwrote fold_sd.json", flush=True)
