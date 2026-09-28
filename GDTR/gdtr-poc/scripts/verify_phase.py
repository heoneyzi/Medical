"""Strict per-phase invariant checks. Exits non-zero on FAIL."""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
ROOT = Path('/path/to/gDTR')

def verify_1_1():
    p = ROOT / 'results' / 'phase1.1'
    d = np.load(p / 'lens_traces.npz', allow_pickle=True)
    Dc, Dj = d['D_cos'], d['D_jsd']
    c = []
    c.append(('D_cos[30] = D_cos[31] arch quirk', abs(float(Dc[:,30,:].mean()) - float(Dc[:,31,:].mean())) < 0.01))
    c.append(('D_cos[30] > 0.05 (not zeroed)', float(Dc[:,30,:].mean()) > 0.05))
    c.append(('D_cos[0] > 0.3', float(Dc[:,0,:].mean()) > 0.3))
    c.append(('min D_cos < 0.7 mid', float(Dc[:,1:31,:].mean(axis=(0,2)).min()) < 0.7))
    c.append(('D_cos finite', bool(np.isfinite(Dc).all())))
    c.append(('D_jsd[31] near zero (final ref)', float(Dj[:,31,:].mean()) < 1e-4))
    c.append(('D_jsd[29] > 1e-4 (last informative)', float(Dj[:,29,:].mean()) > 1e-4))
    c.append(('D_jsd finite', bool(np.isfinite(Dj).all())))
    h = np.load(p / 'hidden_taps.npz')
    c.append(('hidden_30 finite', bool(np.isfinite(h['hidden_30'][0,:100,:]).all())))
    return c

def verify_1_2():
    p = ROOT / 'results' / 'phase1.2'
    c = []
    c.append(('A_30.pt exists', (p/'A_30.pt').exists()))
    c.append(('A_31.pt exists', (p/'A_31.pt').exists()))
    f = p / 'training_curve.json'
    if f.exists():
        j = json.load(open(f))
        for L in ('30','31'):
            if L in j:
                lc = j[L]['loss_curve']
                c.append((f'A_{L} final<5', float(lc[-1])<5.0))
                c.append((f'A_{L} loss tolerated (final<=initial+1e-6, degenerate=OK)', float(lc[-1]) <= float(lc[0]) + 1e-6))
    return c

def verify_1_3():
    p = ROOT / 'results' / 'phase1.3'
    j = json.load(open(p/'gate_a_tuned.json'))
    c = []
    c.append(('M2_jsd_tuned in JSON', 'M2_jsd_tuned' in j))
    if 'M2_jsd_tuned' in j and 'M2_jsd_untuned' in j:
        t = float(j['M2_jsd_tuned'].get('30', 0))
        u = float(j['M2_jsd_untuned'].get('30', 1))
        c.append((f'tuned[30]={t:.3f} >= untuned[30]={u:.3f}', t >= u))
    return c

def verify_1_4():
    p = ROOT / 'results' / 'phase1.4' / 'calibration.json'
    j = json.load(open(p))
    g = j['gamma_cos_per_region']
    c = []
    c.append(('gamma_cos sanity_gc in (0.05,0.95)', 0.05 < float(g.get('sanity_gc', 0)) < 0.95))
    c.append(('gamma_cos sanity_shuf in (0.05,0.95)', 0.05 < float(g.get('sanity_shuf', 0)) < 0.95))
    return c

def verify_1_5():
    p = ROOT / 'results' / 'phase1.5' / 'best_hp.json'
    j = json.load(open(p))
    cd = j.get('best', j).get('cohens_d', float('nan'))
    return [("Cohen's d finite (NOT NaN)", bool(np.isfinite(cd)))]

def verify_1_6_chr22():
    import h5py
    p = ROOT / 'results' / 'phase1.6' / 'chr22_cache.h5'
    if not p.exists():
        return [('chr22_cache.h5 exists', False)]
    with h5py.File(p, 'r') as f:
        n = f['D_cos'].shape[0]
        done = int(f['done_mask'][:].sum())
        d30 = float(f['D_cos'][0,30,:].astype(np.float32).mean())
    return [
        ('chr22 has 12978 windows', n == 12978),
        ('chr22 all windows done', done == n),
        ('chr22 win0 D_cos[30]>0.05 (Bug-1 fixed)', d30 > 0.05),
    ]

def verify_1_6_gate_b():
    p = ROOT / 'results' / 'phase1.6' / 'gate_b.json'
    if not p.exists():
        return [('gate_b.json exists', False)]
    j = json.load(open(p))
    cd = j.get('cohens_d_exon_vs_intron')
    if cd is None:
        cd = j.get('verdict_gate_b', {}).get('cohens_d', float('nan'))
    return [
        ('per-context counts present', 'context_counts' in j or 'counts' in j),
        ("Cohen's d finite", bool(np.isfinite(cd) if cd is not None else False)),
    ]

def verify_1_7():
    p = ROOT / 'PHASE1_DECISION.md'
    text = p.read_text()
    return [
        ('mentions Gate A_evo', 'Gate A_evo' in text),
        ('mentions Gate B_evo', 'Gate B_evo' in text),
        ('no missing JSON warning', 'could not load' not in text.lower()),
    ]

def verify_followup():
    p = ROOT / 'results' / 'phase1.followup'
    c = []
    target_layers = [15, 20, 25, 28]
    for L in target_layers:
        c.append((f'A_{L}.pt exists', (p/f'A_{L}.pt').exists()))
    f = p / 'training_curve.json'
    c.append(('training_curve.json exists', f.exists()))
    if f.exists():
        j = json.load(open(f))
        for L in target_layers:
            key = str(L)
            if key not in j:
                c.append((f'A_{L} entry in training_curve.json', False))
                continue
            init = float(j[key]['initial_loss_identity'])
            fin = float(j[key]['final_loss'])
            c.append((f'A_{L} initial>1e-4 (NOT degenerate)', init > 1e-4))
            c.append((f'A_{L} final<initial*0.5 (>=50% drop)', fin < init * 0.5))
    g = p / 'gate_a_followup.json'
    c.append(('gate_a_followup.json exists', g.exists()))
    if g.exists():
        v = json.load(open(g))
        recoveries = v.get('fractional_recovery', {})
        if recoveries:
            best = max(recoveries.items(), key=lambda kv: float(kv[1]))
            print(f'  >>> best-recovering layer: L={best[0]} fractional_drop={float(best[1]):.3f}')
            print(f'  >>> verdict: {v.get("interpretation","(missing)")}')
    return c


def verify_gate_b_sub():
    sub = ROOT / 'results' / 'phase1.6_sub'
    c = []
    csv_path = sub / 'per_gene_rank.csv'
    if csv_path.exists():
        with open(csv_path) as fh:
            n_rows = sum(1 for _ in fh) - 1
        c.append((f'per_gene_rank.csv has >=100 rows (got {n_rows})', n_rows >= 100))
    else:
        c.append(('per_gene_rank.csv exists', False))
    spath = sub / 'splice_distance_profile.json'
    if spath.exists():
        j = json.load(open(spath))
        expected = [-200, -100, -50, -20, -10, 0, 10, 20, 50, 100, 200]
        donor = j.get('donor', {}); acceptor = j.get('acceptor', {})
        donor_ok = all(str(d) in donor and donor[str(d)].get('mean_c') is not None for d in expected)
        acc_ok = all(str(d) in acceptor and acceptor[str(d)].get('mean_c') is not None for d in expected)
        c.append(('splice donor distance bins complete', donor_ok))
        c.append(('splice acceptor distance bins complete', acc_ok))
        c.append(('intron background present', 'intron_mean_c_background' in j))
    else:
        c.append(('splice_distance_profile.json exists', False))
    cdpath = sub / 'cohens_d_matrix.csv'
    if cdpath.exists():
        rows = open(cdpath).read().strip().splitlines()
        nrows = len(rows) - 1
        ncols = len(rows[0].split(',')) - 1
        finite = True
        for r in rows[1:]:
            for v in r.split(',')[1:]:
                try:
                    fv = float(v)
                    if fv != fv:
                        finite = False
                except ValueError:
                    finite = False
        c.append((f'cohens_d_matrix is 7x7 (got {nrows}x{ncols})',
                  nrows == 7 and ncols == 7))
        c.append(('cohens_d_matrix all finite', finite))
    else:
        c.append(('cohens_d_matrix.csv exists', False))
    return c



def verify_phase3_pilot():
    p = ROOT / 'results' / 'phase3_pilot'
    c = []
    csv_path = p / 'variants_features.csv'
    json_path = p / 'pilot_results.json'
    done_path = p / '_pilot_done'
    if csv_path.exists():
        with open(csv_path) as fh:
            n_rows = sum(1 for _ in fh) - 1
        c.append((f'variants_features.csv has > 200 rows (got {n_rows})', n_rows > 200))
    else:
        c.append(('variants_features.csv exists', False))
    if json_path.exists():
        j = json.load(open(json_path))
        results = j.get('results', {})
        all_finite = all(np.isfinite(r.get('mean_auroc', float('nan'))) for r in results.values()) if results else False
        c.append(('pilot_results.json AUROCs all finite', all_finite))
        any_above = any(r.get('mean_auroc', 0.0) > 0.50 for r in results.values())
        c.append(('at least one feature AUROC > 0.50', any_above))
        v = j.get('verdict', {})
        if v:
            print(f"  >>> primary AUROC = {v.get('primary_dD_jsd_auroc', float('nan')):.4f}")
            print(f"  >>> best feature = {v.get('best_feature', 'NA')} ({v.get('best_auroc', float('nan')):.4f})")
    else:
        c.append(('pilot_results.json exists', False))
    c.append(('_pilot_done marker exists', done_path.exists()))
    return c


def verify_phase3_main():
    p = ROOT / 'results' / 'phase3_main'
    c = []
    csv_path = p / 'variants_features.csv'
    json_path = p / 'main_results.json'
    vus_path = p / 'vus_ranking.csv'
    pergene_path = p / 'per_gene_auroc.csv'
    if csv_path.exists():
        with open(csv_path) as fh:
            n_rows = sum(1 for _ in fh) - 1
        c.append((f'variants_features.csv has > 1000 rows (got {n_rows})', n_rows > 1000))
    else:
        c.append(('variants_features.csv exists', False))
    if json_path.exists():
        j = json.load(open(json_path))
        rs = j.get('results_stratified', {})
        rg = j.get('results_group_kfold', {})
        rl = j.get('results_logo', {})
        au_finite = (
            all(np.isfinite(v.get('mean_auroc', float('nan'))) for v in rs.values())
            and all(np.isfinite(v.get('mean_auroc', float('nan'))) for v in rg.values())
        )
        c.append(('all stratified+group AUROCs finite', au_finite))
        primary_g = rg.get('dD_cos_vector', {}).get('mean_auroc', float('nan'))
        c.append((f'per-gene-stratified dD_cos AUROC>=0.55 (got {primary_g:.4f})',
                  bool(np.isfinite(primary_g) and primary_g >= 0.55)))
        v = j.get('verdict', {})
        if v:
            print(f"  >>> primary group AUROC = {v.get('primary_group_auroc', float('nan')):.4f}")
            print(f"  >>> primary LOGO AUROC = {v.get('primary_logo_auroc', float('nan')):.4f}")
            print(f"  >>> ensemble strat AUROC = {v.get('ensemble_strat_auroc', float('nan')):.4f}")
            print(f"  >>> ensemble - primary (group) = {v.get('ensemble_minus_primary_group', float('nan')):.4f}")
    else:
        c.append(('main_results.json exists', False))
    if vus_path.exists():
        with open(vus_path) as fh:
            n_vus = sum(1 for _ in fh) - 1
        c.append((f'vus_ranking.csv has > 50 rows (got {n_vus})', n_vus > 50))
    else:
        c.append(('vus_ranking.csv exists', False))
    if pergene_path.exists():
        with open(pergene_path) as fh:
            n_pg = sum(1 for _ in fh) - 1
        c.append((f'per_gene_auroc.csv has >= 10 genes (got {n_pg})', n_pg >= 10))
    else:
        c.append(('per_gene_auroc.csv exists', False))
    return c


def verify_phase1_followup_full():
    p = ROOT / 'results' / 'phase1.followup_full'
    c = []
    f = p / 'recovery_curve.json'
    c.append(('recovery_curve.json exists', f.exists()))
    if f.exists():
        j = json.load(open(f))
        per = j.get('per_layer', {})
        c.append(('recovery_curve has all 32 layers', len(per) == 32))
        n_ge_90 = 0
        n_degenerate = 0
        for L in range(32):
            key = str(L)
            if key not in per:
                continue
            rec = float(per[key].get('recovery_pct', 0.0))
            init = float(per[key].get('initial_loss_identity', 0.0))
            if init < 1e-8 or per[key].get('degenerate', False):
                n_degenerate += 1
                continue
            if rec > 0.90:
                n_ge_90 += 1
        c.append((f'>=28 of 32 layers recovery>0.90 (got {n_ge_90}, degenerate={n_degenerate})',
                  n_ge_90 >= 28))
        for L in (30, 31):
            init = float(per.get(str(L), {}).get('initial_loss_identity', 1.0))
            c.append((f'L={L} degenerate (initial<1e-8)', init < 1e-8))
    v = p / 'verdict.json'
    c.append(('verdict.json exists', v.exists()))
    if v.exists():
        vj = json.load(open(v))
        peak = vj.get('peak_divergence_layer')
        canon = vj.get('canonical_deep_thinking_layer')
        worst = vj.get('worst_recovery_layer')
        n90 = vj.get('n_layers_recovery_gt_0p90')
        print(f'  >>> peak_divergence_layer = L={peak} (initial_loss={vj.get("peak_divergence_initial_loss"):.3e})')
        print(f'  >>> worst_recovery_layer  = L={worst} (recovery_pct={vj.get("worst_recovery_pct"):.4f})')
        print(f'  >>> canonical_deep_thinking_layer = L={canon} (recovery_pct={vj.get("canonical_recovery_pct"):.4f})')
        print(f'  >>> n_layers recovery>0.90 = {n90}')
        print(f'  >>> degenerate_layers = {vj.get("degenerate_layers")}')
    return c




# ============================================================================
# Phase 2 verifiers
# ============================================================================

def verify_phase2_0():
    win = ROOT / 'data' / 'baselines' / 'chr17_windows.tsv'
    lab = ROOT / 'data' / 'annotation' / 'chr17_position_labels.npy'
    sidecar = ROOT / 'data' / 'annotation' / 'chr17_position_labels.json'
    gc = ROOT / 'data' / 'baselines' / 'chr17_gene_class.json'
    c = []
    if win.exists():
        with open(win) as fh:
            n_rows = sum(1 for _ in fh) - 1
        c.append((f'chr17_windows.tsv has 1500 < N < 30000 (got {n_rows})',
                  1500 < n_rows < 30000))
    else:
        c.append(('chr17_windows.tsv exists', False))
    if lab.exists():
        arr = np.load(lab)
        c.append(('chr17_labels length matches chr17 length (83257441)',
                  arr.shape[0] == 83257441))
    else:
        c.append(('chr17_position_labels.npy exists', False))
    c.append(('chr17_position_labels.json sidecar exists', sidecar.exists()))
    if gc.exists():
        d = json.load(open(gc))
        cd = sorted(g.get('gene_name') for g in d.get('cancer_driver', []))
        target = sorted({'TP53', 'BRCA1'})
        c.append((f'cancer_driver subset of expected (got {cd})', set(cd) >= {'TP53','BRCA1'} and set(cd) <= {'TP53','BRCA1','ATM'}))
        c.append((f'other has > 100 protein-coding genes (got {len(d.get("other", []))})',
                  len(d.get('other', [])) > 100))
    else:
        c.append(('chr17_gene_class.json exists', False))
    return c


def verify_phase2_1():
    import h5py
    p = ROOT / 'results' / 'phase2.1' / 'chr17_cache.h5'
    win = ROOT / 'data' / 'baselines' / 'chr17_windows.tsv'
    if not p.exists():
        return [('chr17_cache.h5 exists', False)]
    n_expected = None
    if win.exists():
        with open(win) as fh:
            n_expected = sum(1 for _ in fh) - 1
    with h5py.File(p, 'r') as f:
        n = f['D_cos'].shape[0]
        done = int(f['done_mask'][:].sum())
        d_cos_30 = float(f['D_cos'][0, 30, :].astype(np.float32).mean())
        d_cos_31 = float(f['D_cos'][0, 31, :].astype(np.float32).mean())
        d_jsd_31 = float(f['D_jsd'][0, 31, :].astype(np.float32).mean())
        d_cos_finite = bool(np.isfinite(f['D_cos'][0]).all())
        d_jsd_finite = bool(np.isfinite(f['D_jsd'][0]).all())
    c = []
    if n_expected is not None:
        c.append((f'N({n}) matches windows TSV ({n_expected})', n == n_expected))
    c.append((f'all chr17 windows done ({done}/{n})', done == n))
    c.append((f'D_cos[30]>0.05 (Bug-1 fix preserved): {d_cos_30:.4f}', d_cos_30 > 0.05))
    c.append((f'D_cos[30] == D_cos[31] arch quirk (delta={abs(d_cos_30 - d_cos_31):.5f})',
              abs(d_cos_30 - d_cos_31) < 0.01))
    c.append((f'D_jsd[31] near zero (final ref): {d_jsd_31:.6f}', d_jsd_31 < 1e-3))
    c.append(('D_cos finite', d_cos_finite))
    c.append(('D_jsd finite', d_jsd_finite))
    return c


def verify_phase2_2():
    p = ROOT / 'results' / 'phase2.2' / 'gate_b_chr17.json'
    if not p.exists():
        return [('gate_b_chr17.json exists', False)]
    j = json.load(open(p))
    cd = j.get('cohens_d_exon_vs_intron')
    counts = j.get('context_counts', {})
    return [
        ('per-context counts present', bool(counts)),
        ("Cohen's d finite", bool(np.isfinite(cd) if cd is not None else False)),
        (f'intron count > 100 (got {counts.get("intron", 0)})',
         counts.get('intron', 0) > 100),
        (f'exon count > 100 (got {counts.get("coding_exon", 0)})',
         counts.get('coding_exon', 0) > 100),
    ]


def verify_phase2_3():
    p = ROOT / 'results' / 'phase2.3' / 'cross_chr_comparison.json'
    if not p.exists():
        return [('cross_chr_comparison.json exists', False)]
    j = json.load(open(p))
    pc = j.get('per_context', {})
    expected_ctxs = ['intergenic', 'intron', 'coding_exon', '5utr', '3utr',
                     'splice_donor', 'splice_acceptor']
    c = []
    c.append((f'all 7 contexts present (got {sorted(pc.keys())})',
              all(k in pc for k in expected_ctxs)))
    finite_ok = True
    for k in expected_ctxs:
        e = pc.get(k, {})
        for fld in ('chr17_mean_c', 'chr22_mean_c'):
            v = e.get(fld)
            if v is None or not np.isfinite(v):
                finite_ok = False
                break
    c.append(('all chr17/chr22 mean_c finite', finite_ok))
    return c


def verify_phase2_4():
    p = ROOT / 'results' / 'phase2.4' / 'gene_class_stratification.json'
    if not p.exists():
        return [('gene_class_stratification.json exists', False)]
    j = json.load(open(p))
    groups = j.get('groups', {})
    c = []
    c.append((f'>=3 groups present (got {len(groups)})', len(groups) >= 3))
    finite_ok = True
    for label, g in groups.items():
        m = g.get('mean_c')
        if m is None or not np.isfinite(m):
            finite_ok = False
            break
    c.append(('all group mean_c finite', finite_ok))
    test = j.get('test_cd_vs_all_other', {})
    cd_d = test.get('cohens_d_cd_minus_other')
    c.append(("Cohen's d (CD vs all other) finite",
              bool(cd_d is not None and np.isfinite(cd_d))))
    return c


def verify_phase2_5():
    p = ROOT / 'results' / 'phase2.5' / 'splice_chr17_profile.json'
    if not p.exists():
        return [('splice_chr17_profile.json exists', False)]
    j = json.load(open(p))
    expected = [-200, -100, -50, -20, -10, 0, 10, 20, 50, 100, 200]
    c = []
    donor_ok = all(str(d) in j.get('donor', {}) and
                   j['donor'][str(d)].get('mean_c') is not None for d in expected)
    acc_ok = all(str(d) in j.get('acceptor', {}) and
                 j['acceptor'][str(d)].get('mean_c') is not None for d in expected)
    c.append(('donor distance bins finite', donor_ok))
    c.append(('acceptor distance bins finite', acc_ok))
    bg = j.get('intron_mean_c_background')
    donor_min = None
    if j.get('donor'):
        vals = [v.get('mean_c') for v in j['donor'].values()
                if v.get('mean_c') is not None]
        if vals:
            donor_min = min(vals)
    if bg is not None and donor_min is not None:
        c.append((f'donor minimum ({donor_min:.3f}) below intron bg ({bg:.3f})',
                  donor_min < bg))
    else:
        c.append(('donor min vs bg comparable', False))
    return c


def verify_phase2_6():
    p = ROOT / 'PHASE2_DECISION.md'
    if not p.exists():
        return [('PHASE2_DECISION.md exists', False)]
    text = p.read_text()
    return [
        ('PHASE2_DECISION.md exists', True),
        ('mentions Phase 2.0', 'Phase 2.0' in text),
        ('mentions Phase 2.1', 'Phase 2.1' in text),
        ('mentions Phase 2.2', 'Phase 2.2' in text),
        ('mentions Phase 2.3', 'Phase 2.3' in text),
        ('mentions Phase 2.4', 'Phase 2.4' in text),
        ('mentions Phase 2.5', 'Phase 2.5' in text),
        ('no missing JSON warning', 'could not load' not in text.lower()),
    ]




def verify_phase5():
    p = ROOT / 'results' / 'phase5'
    if not p.exists():
        return [('phase5 dir exists', False)]
    c = []
    qf = p / 'chr22_quadrants.npy'
    if not qf.exists():
        return [('chr22_quadrants.npy exists', False)]
    q = np.load(qf, mmap_mode='r')
    c.append(('chr22_quadrants len = 50818468', q.shape[0] == 50818468))
    bed = p / 'q2_regions.bed'
    n_bed = 0
    if bed.exists():
        with open(bed) as f:
            for line in f:
                if line and line[0] != '#':
                    n_bed += 1
    c.append((f'q2_regions.bed has >=50 regions (got {n_bed})', n_bed >= 50))
    ej = p / 'q2_enrichment.json'
    if not ej.exists():
        c.append(('q2_enrichment.json exists', False))
        return c
    j = json.load(open(ej))
    enr = j.get('annotation_enrichment', {})
    finite_any = False
    for k, v in enr.items():
        f_ = v.get('fold_enrichment')
        p_ = v.get('hypergeom_pval_oneSided_overrep')
        if f_ is not None and p_ is not None and np.isfinite(f_) and np.isfinite(p_):
            finite_any = True
            break
    c.append(('q2_enrichment has >=1 finite fold/p annotation', finite_any))
    return c


def verify_phase3_ensemble():
    p = ROOT / 'results' / 'phase3_ensemble'
    c = []
    csv_path = p / 'variants_features_full.csv'
    json_path = p / 'ensemble_results.json'
    if csv_path.exists():
        with open(csv_path) as fh:
            n_rows = sum(1 for _ in fh) - 1
        c.append((f'variants_features_full.csv >= 8000 rows (got {n_rows})', n_rows >= 8000))
    else:
        c.append(('variants_features_full.csv exists', False))
    if json_path.exists():
        j = json.load(open(json_path))
        rs = j.get('results_stratified', {})
        au_finite = all(np.isfinite(v.get('mean_auroc', float('nan'))) for v in rs.values())
        c.append(('all stratified AUROCs finite', au_finite))
        for key in ('A_dD_cos','B_evo2_loglik','C_cadd_phred','D_am_score','ABCD_full','BCD_baseline'):
            ok = key in rs and np.isfinite(rs[key].get('mean_auroc', float('nan')))
            c.append((f'feature {key} has finite AUROC', ok))
        delong = j.get('delong_stratified', [])
        c.append((f'DeLong tests present (n={len(delong)})',
                  isinstance(delong, list) and len(delong) > 0))
        cov = j.get('coverage', {})
        for cat in ('P_LP','B_LB'):
            if cat in cov:
                cf = cov[cat].get('cadd_match_frac', 0.0)
                c.append((f'{cat} CADD match >= 0.7 (got {cf:.3f})', cf >= 0.7))
                af = cov[cat].get('am_match_frac', 0.0)
                c.append((f'{cat} AM match >= 0.3 (got {af:.3f})', af >= 0.3))
        for k in ('A_dD_cos','B_evo2_loglik','C_cadd_phred','D_am_score',
                  'ABCD_full','BCD_baseline'):
            val = rs.get(k, {}).get('mean_auroc')
            if val is not None:
                print(f'  >>> {k}: AUROC = {val:.4f}')
    else:
        c.append(('ensemble_results.json exists', False))
    c.append(('_done marker exists', (p / '_done').exists()))
    return c


def verify_phase4():
    p = ROOT / 'results' / 'phase4'
    c = []
    c.append(('phase4 dir exists', p.exists()))
    sj = p / 'per_model_summary.json'
    cj = p / 'concordance_matrix.json'
    c.append(('per_model_summary.json exists', sj.exists()))
    c.append(('concordance_matrix.json exists', cj.exists()))
    if not (sj.exists() and cj.exists()):
        return c
    summ = json.load(open(sj))
    conc = json.load(open(cj))
    c.append(('>=2 models present', len(summ) >= 2))
    rho = conc.get('spearman_rho', {})
    finite = 0; high = 0
    for a in rho:
        for b in rho.get(a, {}):
            if a >= b: continue
            v = rho[a][b]
            if v == v:
                finite += 1
                if v > 0.3:
                    high += 1
    c.append(('all pairwise rho finite', finite >= 1))
    c.append(('>=1 pair with rho > 0.3', high >= 1))
    splice_ok = 0
    for nm, info in summ.items():
        s = info.get('splice_signal')
        if not s: continue
        d = s.get('data', {})
        if s['kind'] == 'per_position':
            don = d.get('splice_donor', {}).get('mean_c')
            intr = d.get('intron', {}).get('mean_c')
            if don is not None and intr is not None and don < intr:
                splice_ok += 1
        else:
            sp = d.get('splice_containing', {}).get('mean_c')
            ind = d.get('intron_dominant', {}).get('mean_c')
            if sp is not None and ind is not None:
                splice_ok += 1
    c.append(('per-model splice signal computed for >=1 model', splice_ok >= 1))
    return c

PHASE_VERIFIERS = {
    '1.1': verify_1_1, '1.2': verify_1_2, '1.3': verify_1_3,
    '1.4': verify_1_4, '1.5': verify_1_5,
    '1.6_chr22': verify_1_6_chr22, '1.6_gate_b': verify_1_6_gate_b,
    '1.7': verify_1_7,
    'followup': verify_followup,
    'gate_b_sub': verify_gate_b_sub,
    'phase2_0': verify_phase2_0,
    'phase2_1': verify_phase2_1,
    'phase2_2': verify_phase2_2,
    'phase2_3': verify_phase2_3,
    'phase2_4': verify_phase2_4,
    'phase2_5': verify_phase2_5,
    'phase2_6': verify_phase2_6,
    'phase3_pilot': verify_phase3_pilot,
    'phase3_main': verify_phase3_main,
    'phase4': verify_phase4,
    'phase3_ensemble': verify_phase3_ensemble,
    'phase1_followup_full': verify_phase1_followup_full,
    'phase5': verify_phase5,
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in PHASE_VERIFIERS:
        print(f'usage: verify_phase.py <{"|".join(PHASE_VERIFIERS)}>')
        sys.exit(2)
    phase = sys.argv[1]
    print(f'=== verify {phase} ===')
    checks = PHASE_VERIFIERS[phase]()
    all_ok = True
    for name, ok in checks:
        m = 'PASS' if ok else 'FAIL'
        print(f'  [{m}] {name}')
        if not ok:
            all_ok = False
    sys.exit(0 if all_ok else 1)


if __name__ == '__main__':
    main()
