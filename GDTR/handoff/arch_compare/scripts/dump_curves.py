import json, numpy as np
R="/path/to/TDiG/arch_compare/results"
CTX={2:"coding_exon",5:"splice_donor",6:"splice_acceptor",0:"intergenic"}
LMAX=27
def auroc(x,y):
    o=np.argsort(x,kind="mergesort"); r=np.empty(len(x)); r[o]=np.arange(1,len(x)+1)
    xs=x[o]; i=0
    while i<len(xs):
        j=i
        while j+1<len(xs) and xs[j+1]==xs[i]: j+=1
        if j>i: r[o[i:j+1]]=(i+1+j+1)/2
        i=j+1
    n1=y.sum(); n0=len(y)-n1
    return float((r[y==1].sum()-n1*(n1+1)/2)/(n1*n0)) if n1>1 and n0>1 else None
out={}
for chrom,f in (("chr22","evo2_7b_scalars.npz"),("chr17","chr17_evo2_7b_scalars.npz")):
    d=np.load(f"{R}/{f}"); lab=d["label"]
    out[chrom]={}
    for gauge in ("cos","vel"):
        X=d[gauge].astype(np.float32)[:,:LMAX]
        g={}
        for code,nm in CTX.items():
            if (lab==code).sum()<2000: continue
            m=(lab==code)|(lab==1); y=(lab[m]==code).astype(int)
            g[nm]=[round(auroc(X[m][:,l],y),4) for l in range(LMAX)]
            g[nm+"_n"]=int((lab==code).sum())
        out[chrom][gauge]=g
p=json.load(open(f"{R}/evo2_7b_layer_profile.json"))
out["profile"]={"norm":[round(x,6) for x in p["norm"]],
                "cos":[round(x,4) for x in p["cos_norm"]],
                "vel":[round(x,4) for x in p["vel"]]}
json.dump(out, open(f"{R}/curves_for_report.json","w"), separators=(",",":"))
print("ok", sum(len(v) for c in out if c!="profile" for v in out[c].values()))
