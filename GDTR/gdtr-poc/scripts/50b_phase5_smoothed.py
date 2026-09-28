"""Phase 5b: re-run quadrant analysis using 100bp-smoothed c & phyloP.

Settling depth c is integer-quantized; raw thresholds produce 1-2bp noisy
runs. A 100bp box-car average yields biologically meaningful contiguous
regions while preserving the high-gDTR + low-conservation signal.
"""
import json, gzip, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, '/path/to/gDTR/scripts')
import _runner_utils as ru

PHASE='phase5'
OUT = ru.GDTR_ROOT/'results'/'phase5'
LOG = ru.setup_logging(PHASE)

CHR_LEN=50_818_468; CHR='chr22'; SMOOTH=100
GTOP=0.25; PBOT=0.25; MIN=100
CTX_NAME={0:'intergenic',1:'intron',2:'coding_exon',3:'5utr',4:'3utr',5:'splice_donor',6:'splice_acceptor',7:'repeat'}
CD = ru.GDTR_ROOT/'data'/'conservation'

def smooth_box(arr, w):
    finite = np.isfinite(arr)
    a = np.where(finite, arr, 0.0).astype(np.float64)
    cs = np.concatenate([[0.0], np.cumsum(a)])
    cn = np.concatenate([[0],   np.cumsum(finite.astype(np.int64))])
    half = w//2
    n = arr.size
    s = np.maximum(0, np.arange(n)-half)
    e = np.minimum(n, np.arange(n)+(w-half))
    sums = cs[e]-cs[s]
    cnts = cn[e]-cn[s]
    out = np.full(n, np.nan, dtype=np.float32)
    valid = cnts >= max(1, w//4)
    out[valid] = (sums[valid]/cnts[valid]).astype(np.float32)
    return out

def load_phylop():
    import pyBigWig
    bw = pyBigWig.open(str(CD/'hg38.phyloP100way.bw'))
    n = min(bw.chroms()[CHR], CHR_LEN)
    out = np.full(CHR_LEN, np.nan, dtype=np.float32)
    chunk=5_000_000
    for s in range(0,n,chunk):
        e = min(s+chunk, n)
        out[s:e] = np.asarray(bw.values(CHR, s, e, numpy=True), dtype=np.float32)
    bw.close()
    return out

def load_bed(bed):
    s_,e_=[],[]
    with open(bed) as f:
        for ln in f:
            if not ln or ln[0]=='#': continue
            p = ln.rstrip('\n').split('\t')
            if len(p)<3 or p[0]!=CHR: continue
            try: s_.append(int(p[1])); e_.append(int(p[2]))
            except ValueError: pass
    if not s_: return np.zeros((0,2),dtype=np.int64)
    a=np.empty((len(s_),2),dtype=np.int64); a[:,0]=s_; a[:,1]=e_
    return a[np.argsort(a[:,0])]

def load_rmsk():
    by={}
    with gzip.open(CD/'rmsk.txt.gz','rt') as f:
        for ln in f:
            p = ln.rstrip('\n').split('\t')
            if len(p)<13 or p[5]!=CHR: continue
            try: s=int(p[6]); e=int(p[7])
            except ValueError: continue
            by.setdefault(p[11],[]).append((s,e))
    out={}
    for k,iv in by.items():
        a=np.array(iv,dtype=np.int64); out[k]=a[np.argsort(a[:,0])]
    return out

def to_mask(iv, n):
    m = np.zeros(n,dtype=bool)
    for s,e in iv:
        s=max(0,int(s)); e=min(n,int(e))
        if e>s: m[s:e]=True
    return m

def runs(mask):
    if mask.size==0: return np.zeros((0,2),dtype=np.int64)
    d=np.diff(mask.astype(np.int8), prepend=0, append=0)
    s=np.where(d==1)[0]; e=np.where(d==-1)[0]
    return np.stack([s,e],axis=1).astype(np.int64)

def hg(k,K,n,N):
    from scipy.stats import hypergeom
    if min(K,n,N)<=0 or k<=0: return 1.0
    return float(hypergeom.sf(k-1,N,K,n))

def main():
    np.random.seed(42)
    LOG.info('load arrays')
    c_arr = np.load(ru.GDTR_ROOT/'results/phase1.6_sub/chr22_position_c.npy')
    labels = np.load(ru.GDTR_ROOT/'data/annotation/chr22_position_labels.npy')
    LOG.info('load phylop')
    phylop = load_phylop()
    nan_p_raw = float(np.isnan(phylop).mean())
    nan_c_raw = float(np.isnan(c_arr).mean())
    LOG.info('raw NaN: c=%.2f%% phylop=%.2f%%', 100*nan_c_raw, 100*nan_p_raw)

    LOG.info('smoothing %dbp', SMOOTH)
    c_s = smooth_box(c_arr, SMOOTH)
    p_s = smooth_box(phylop, SMOOTH)
    valid = ~(np.isnan(c_s) | np.isnan(p_s))
    N = int(valid.sum())
    LOG.info('valid (smoothed) %d (%.2f%%)', N, 100*N/CHR_LEN)

    cv = c_s[valid]; pv = p_s[valid]
    c_thr = float(np.quantile(cv, 1.0-GTOP))
    p_thr = float(np.quantile(pv, PBOT))
    LOG.info('c_thr top25=%.4f, phylop_thr bot25=%.4f', c_thr, p_thr)

    high_g = c_s >= c_thr
    low_c  = p_s <= p_thr
    q = np.zeros(CHR_LEN, dtype=np.uint8)
    q[valid & high_g & ~low_c] = 1
    q[valid & high_g &  low_c] = 2
    q[valid & ~high_g & ~low_c] = 3
    q[valid & ~high_g &  low_c] = 4
    sizes={f'Q{i}':int((q==i).sum()) for i in (1,2,3,4)}
    sizes['NA']=int((q==0).sum())
    sizes_pct={k:100.0*v/CHR_LEN for k,v in sizes.items()}
    LOG.info('sizes: %s', sizes)
    np.save(OUT/'chr22_quadrants.npy', q)

    rs = runs(q==2); L=rs[:,1]-rs[:,0]
    rg = rs[L>=MIN]
    LOG.info('Q2 runs total=%d  >=%dbp=%d', int(rs.shape[0]), MIN, int(rg.shape[0]))

    rows=[]
    for s,e in rg:
        cseg = c_arr[s:e]; pseg = phylop[s:e]; lab = labels[s:e]
        mc = float(np.nanmean(cseg)) if cseg.size else float('nan')
        mp = float(np.nanmean(pseg)) if pseg.size else float('nan')
        if lab.size:
            u,cn = np.unique(lab,return_counts=True)
            dn = CTX_NAME.get(int(u[cn.argmax()]),'NA')
        else: dn='NA'
        rows.append((CHR,int(s),int(e),int(e-s),mc,mp,dn))
    bed = OUT/'q2_regions.bed'
    with open(bed,'w') as f:
        f.write('#chrom\tstart\tend\tlength\tmean_c\tmean_phylop\tdom_context\n')
        for r in rows:
            f.write('\t'.join(str(x) if not isinstance(x,float) else f'{x:.4f}' for x in r) + '\n')
    LOG.info('wrote %s (%d regions)', bed, len(rows))

    LOG.info('annotations')
    ccre = to_mask(load_bed(CD/'GRCh38-cCREs.bed'), CHR_LEN)
    rdhs = to_mask(load_bed(CD/'GRCh38-rDHSs.bed'), CHR_LEN)
    rmsk = load_rmsk()
    ann = {'ENCODE_cCRE':ccre,'ENCODE_rDHS':rdhs}
    for cls in ('SINE','LINE','LTR','DNA','Simple_repeat','Low_complexity','Satellite','Retroposon'):
        if cls in rmsk: ann[f'rmsk_{cls}'] = to_mask(rmsk[cls], CHR_LEN)

    q2 = (q==2); n=int(q2.sum())
    LOG.info('N=%d Q2=%d', N, n)

    enr={}
    for nm,mk in ann.items():
        K = int((mk & valid).sum())
        k = int((mk & q2).sum())
        if N==0 or n==0 or K==0:
            f_=float('nan'); pv=1.0
        else:
            f_ = (k/n)/(K/N) if K else float('nan')
            pv = hg(k,K,n,N)
        enr[nm]={'n_overlap_q2':k,'n_overlap_total_valid':K,'n_q2':n,'n_total_valid':N,
                 'obs_frac_in_q2':k/n if n else 0.0,'exp_frac_genome':K/N if N else 0.0,
                 'fold_enrichment':f_,'hypergeom_pval_oneSided_overrep':pv}
        LOG.info('  %-22s k=%-7d K=%-9d fold=%.3f p=%.3e', nm,k,K,f_,pv)

    ctx={}
    for cid,nm in CTX_NAME.items():
        m=(labels==cid)
        K=int((m & valid).sum()); k=int((m & q2).sum())
        if N and n and K:
            f_=(k/n)/(K/N); pv=hg(k,K,n,N)
        else:
            f_,pv=float('nan'),1.0
        ctx[nm]={'n_overlap_q2':k,'n_overlap_total_valid':K,
                 'fold_enrichment':f_,'hypergeom_pval_oneSided_overrep':pv}

    out = {'phase':'phase5','seed':42,'chrom':CHR,'chrom_len':CHR_LEN,
           'smoothing_bp':SMOOTH,
           'n_valid_positions':N,'frac_valid':N/CHR_LEN,
           'frac_phylop_nan_raw':nan_p_raw,'frac_c_nan_raw':nan_c_raw,
           'c_threshold_top25pct_smoothed':c_thr,
           'phylop_threshold_bot25pct_smoothed':p_thr,
           'quadrant_sizes_bp':sizes,'quadrant_pct_of_chr22':sizes_pct,
           'q2_n_runs_total':int(rs.shape[0]),
           'q2_n_regions_min100bp':int(rg.shape[0]),
           'annotation_enrichment':enr,
           'context_enrichment_in_q2':ctx}
    def _safe(x):
        if isinstance(x,float) and not np.isfinite(x): return None
        return x
    with open(OUT/'q2_enrichment.json','w') as f:
        json.dump(out,f,indent=2,default=_safe)
    LOG.info('wrote q2_enrichment.json')

    try:
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        rng=np.random.default_rng(42)
        idx=np.flatnonzero(valid)
        sn=min(100_000, idx.size)
        sub=rng.choice(idx, size=sn, replace=False)
        cs=c_s[sub]; ps=p_s[sub]; sq=q[sub]
        col={1:'#1f77b4',2:'#d62728',3:'#7f7f7f',4:'#cccccc'}
        lbl={1:'Q1 hi-c hi-cons',2:'Q2 hi-c lo-cons (recently evolved)',3:'Q3 lo-c hi-cons',4:'Q4 lo-c lo-cons'}
        fig,ax=plt.subplots(figsize=(7,6))
        for qi in (4,3,1,2):
            mk=sq==qi
            if mk.any(): ax.scatter(cs[mk], ps[mk], s=1, alpha=0.35, c=col[qi], label=lbl[qi])
        ax.axvline(c_thr,color='k',lw=0.6,ls='--')
        ax.axhline(p_thr,color='k',lw=0.6,ls='--')
        ax.set_xlabel('settling depth c (100bp smoothed)')
        ax.set_ylabel('PhyloP 100-way (100bp smoothed)')
        ax.set_title(f'chr22 gDTR vs PhyloP  Q2 regions>=100bp = {len(rows)}')
        ax.legend(loc='lower right', fontsize=8, markerscale=4)
        fig.tight_layout()
        for ext in ('pdf','png'): fig.savefig(OUT/f'F_quadrant_scatter.{ext}', dpi=150)
        plt.close(fig)

        names=list(enr.keys())
        folds=[enr[k]['fold_enrichment'] if np.isfinite(enr[k]['fold_enrichment']) else 0.0 for k in names]
        pvals=[enr[k]['hypergeom_pval_oneSided_overrep'] for k in names]
        fig,ax=plt.subplots(figsize=(8, 0.4*len(names)+1.5))
        yp=np.arange(len(names))
        ax.barh(yp, folds, color=['#d62728' if f>=1 else '#1f77b4' for f in folds])
        ax.axvline(1.0, color='k', lw=0.7, ls='--', label='genome-wide expectation')
        ax.set_yticks(yp); ax.set_yticklabels(names)
        ax.set_xlabel('Fold enrichment in Q2')
        ax.set_title('Q2 enrichment, chr22 [100bp smoothed]')
        for i,(f,p) in enumerate(zip(folds,pvals)):
            txt = f'  {f:.2f}x  p={p:.2e}' if (np.isfinite(p) and p>0) else f'  {f:.2f}x'
            ax.text(max(f,1.0), i, txt, va='center', fontsize=8)
        ax.legend(loc='lower right', fontsize=8)
        fig.tight_layout()
        for ext in ('pdf','png'): fig.savefig(OUT/f'F_q2_enrichment.{ext}', dpi=150)
        plt.close(fig)
    except Exception as e:
        LOG.warning('plot fail: %s', e)

    LOG.info('phase 5 done (smoothed %dbp)', SMOOTH)

if __name__=='__main__':
    main()
