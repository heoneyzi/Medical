"""Two checks the block sawtooth forces us to make.

(A) Is the d DENOMINATOR block-structured? d = dmean / pooled_sd, and pooled_sd
    is layer-specific. If within-layer variance dips at hcs, d is inflated there
    for reasons that have nothing to do with biology.

(B) Does the peak location survive a COMMON denominator? Recompute the per-layer
    effect with one pooled sd shared across layers; if the peak moves, the
    original peak was a variance artefact.

Also: is hcs systematically a SMALLER update? That would explain the sawtooth
mechanically — small update leaves alignment intact, big updates perturb it.
"""
import numpy as np

R = "/path/to/TDiG/arch_compare/results"
d = np.load(f"{R}/evo2_7b_scalars.npz")
cos, vel = d["cos"].astype(np.float32), d["vel"].astype(np.float32)
lab, win = d["label"], d["window"]
ATTN=[3,10,17,24,31]; HCS=[0,4,7,11,14,18,21,25,28]
HCM=[1,5,8,12,15,19,22,26,29]; HCL=[2,6,9,13,16,20,23,27,30]
BT={}
for l in ATTN: BT[l]="attn"
for l in HCS: BT[l]="hcs"
for l in HCM: BT[l]="hcm"
for l in HCL: BT[l]="hcl"
CTX={2:"coding_exon",5:"splice_donor",6:"splice_acceptor",0:"intergenic"}
LMAX=27

print("="*88)
print("(A) per-layer WITHIN-layer spread — does the d denominator follow block type?")
print("="*88)
intr = lab==1
sd_cos = cos[intr,:LMAX].std(0); sd_vel = vel[intr,:LMAX].std(0)
mu_vel = vel[intr,:LMAX].mean(0)
print(f"  {'L':>3}{'type':>6}{'sd(cos)':>10}{'sd(vel)':>10}{'mean(vel)':>11}")
for l in range(5, LMAX):
    star = " <<<" if BT[l]=="hcs" else ""
    print(f"  {l:>3}{BT[l]:>6}{sd_cos[l]:>10.4f}{sd_vel[l]:>10.4f}{mu_vel[l]:>11.4f}{star}")

print()
print("  블록 종류별 평균 (L5-26):")
print(f"  {'type':<6}{'sd(cos)':>10}{'sd(vel)':>10}{'mean(vel)':>11}")
for t in ("hcs","hcm","hcl","attn"):
    ls=[l for l in range(5,LMAX) if BT[l]==t]
    if ls: print(f"  {t:<6}{sd_cos[ls].mean():>10.4f}{sd_vel[ls].mean():>10.4f}{mu_vel[ls].mean():>11.4f}")

def cohen(a,b):
    na,nb=len(a),len(b)
    va,vb=a.var(ddof=1),b.var(ddof=1)
    p=np.sqrt(((na-1)*va+(nb-1)*vb)/(na+nb-2))
    return (a.mean()-b.mean())/p if p>0 else np.nan

print()
print("="*88)
print("(B) does the peak survive a COMMON denominator?")
print("="*88)
print(f"  {'region':<17}{'gauge':>6}{'per-layer sd':>26}{'common sd':>26}")
for code,nm in CTX.items():
    if (lab==code).sum()<2000: continue
    for gname, X in (("cos",cos),("vel",vel)):
        A=X[lab==code,:LMAX]; B=X[intr,:LMAX]
        d_lay=np.array([cohen(A[:,l],B[:,l]) for l in range(LMAX)])
        # common denominator: one pooled sd over all layers of the intron baseline
        sd_common=B.std()
        d_com=(A.mean(0)-B.mean(0))/sd_common
        pl=int(np.nanargmax(np.abs(d_lay))); pc=int(np.nanargmax(np.abs(d_com)))
        print(f"  {nm:<17}{gname:>6}"
              f"{f'|d|={abs(d_lay[pl]):.3f} @L{pl}({BT[pl]})':>26}"
              f"{f'|d|={abs(d_com[pc]):.3f} @L{pc}({BT[pc]})':>26}")
