"""chr17 replication of the per-layer AUROC profiles, side by side with chr22."""
import numpy as np
R="/path/to/TDiG/arch_compare/results"
ATTN=[3,10,17,24,31]; HCS=[0,4,7,11,14,18,21,25,28]
HCM=[1,5,8,12,15,19,22,26,29]; HCL=[2,6,9,13,16,20,23,27,30]
BT={}
for l in ATTN: BT[l]="attn"
for l in HCS: BT[l]="hcs"
for l in HCM: BT[l]="hcm"
for l in HCL: BT[l]="hcl"
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
    return (r[y==1].sum()-n1*(n1+1)/2)/(n1*n0) if n1>1 and n0>1 else np.nan

def curves(chrom, gauge):
    f = f"{R}/evo2_7b_scalars.npz" if chrom=="chr22" else f"{R}/chr17_evo2_7b_scalars.npz"
    d=np.load(f); X=d[gauge].astype(np.float32)[:,:LMAX]
    lab=d["label"]
    out={}
    for code,nm in CTX.items():
        if (lab==code).sum()<2000: continue
        m=(lab==code)|(lab==1); y=(lab[m]==code).astype(int)
        out[nm]=np.array([auroc(X[m][:,l],y) for l in range(LMAX)])
    return out

for gauge,gn in (("cos","cos(h_l, h_norm)"),("vel","relative velocity")):
    c22=curves("chr22",gauge); c17=curves("chr17",gauge)
    print("\n"+"="*100)
    print(f"chr22 vs chr17 — per-layer AUROC vs intron — {gn}")
    print("="*100)
    for nm in c22:
        if nm not in c17: continue
        a,b=c22[nm],c17[nm]
        r=np.corrcoef(a,b)[0,1]
        pa,pb=int(np.nanargmax(np.abs(a-0.5))),int(np.nanargmax(np.abs(b-0.5)))
        agree=np.mean(np.sign(a-0.5)==np.sign(b-0.5))
        print(f"\n  {nm}")
        print(f"    chr22  " + "".join(f"{v:>6.3f}" for v in a[::2]))
        print(f"    chr17  " + "".join(f"{v:>6.3f}" for v in b[::2]))
        print(f"    프로파일 상관 r={r:+.3f} | 부호 일치 {agree:.0%} | "
              f"정점 chr22 L{pa}({BT[pa]}) {a[pa]:.3f} / chr17 L{pb}({BT[pb]}) {b[pb]:.3f}")
