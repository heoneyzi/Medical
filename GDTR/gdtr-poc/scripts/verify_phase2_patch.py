"""Patch script: appends Phase 2 verifier functions to scripts/verify_phase.py.

Idempotent: it replaces the trailing dispatcher block.
"""
from __future__ import annotations

from pathlib import Path

VERIFY_PATH = Path("/path/to/gDTR/scripts/verify_phase.py")

PHASE2_BLOCK = '''

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
        target = sorted({'TP53', 'BRCA1', 'ATM'})
        c.append((f'cancer_driver = {{TP53,BRCA1,ATM}} (got {cd})', cd == target))
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
'''

PHASE2_DISPATCH = """    'phase2_0': verify_phase2_0,
    'phase2_1': verify_phase2_1,
    'phase2_2': verify_phase2_2,
    'phase2_3': verify_phase2_3,
    'phase2_4': verify_phase2_4,
    'phase2_5': verify_phase2_5,
    'phase2_6': verify_phase2_6,
"""


def main() -> None:
    text = VERIFY_PATH.read_text()

    if "verify_phase2_0" in text:
        print(f"[patch] Phase 2 verifiers already present, skipping function injection.")
    else:
        # Insert before the PHASE_VERIFIERS dictionary.
        marker = "PHASE_VERIFIERS = {"
        idx = text.index(marker)
        # Walk back to the line start
        line_start = text.rfind("\n", 0, idx) + 1
        text = text[:line_start] + PHASE2_BLOCK + "\n\n" + text[line_start:]

    # Inject phase2 keys into PHASE_VERIFIERS dict
    if "'phase2_0': verify_phase2_0" not in text:
        marker = "'gate_b_sub': verify_gate_b_sub,"
        idx = text.index(marker)
        end = text.find("\n", idx) + 1
        text = text[:end] + PHASE2_DISPATCH + text[end:]

    VERIFY_PATH.write_text(text)
    print(f"[patch] wrote {VERIFY_PATH}")


if __name__ == "__main__":
    main()
