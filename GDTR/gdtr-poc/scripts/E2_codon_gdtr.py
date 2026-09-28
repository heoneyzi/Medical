"""E2_codon_gdtr.py - Codon-position stratification of gDTR in TP53 CDS.

Extension to Phase 0 Finding 3. Reuses cached D_cos from
results/runs/02_gene_structure_cache.npz (no new forward).
"""
from __future__ import annotations
import json, logging, sys, time, platform, socket
from pathlib import Path
from itertools import combinations
import numpy as np
import pandas as pd
import torch
from scipy.stats import kruskal, mannwhitneyu

ROOT = Path('/path/to/gDTR-PoC')
sys.path.insert(0, str(ROOT))
from src.gdtr import settling_depth_interp
from src.viz import setup_publication_style, WONG_PALETTE

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(name)s %(levelname)s %(message)s')
log = logging.getLogger('E2_codon_gdtr')

TABLES = ROOT / 'results' / 'tables'
FIGURES = ROOT / 'results' / 'figures'
RUNS = ROOT / 'results' / 'runs'
ANNO_DB = ROOT / 'data' / 'annotation' / 'gencode.v44.chr17_chr13.gtf.db'

TP53_CHROM = 'chr17'
TP53_START = 7_668_402
TP53_END = 7_687_550
TP53_TRANSCRIPT = 'ENST00000269305'
WINDOW = 6000
STRIDE = 500
EDGE_WARMUP = 5
CENTRAL_WIDTH = 1000
SEED = 42
RHO = 0.85


def load_region_offset():
    return max(0, TP53_START - 1 - WINDOW // 2)


def codon_positions_for_cds(region_start_0, region_len):
    import gffutils
    db = gffutils.FeatureDB(str(ANNO_DB))
    tx = None
    for t in db.features_of_type('transcript',
            limit=(TP53_CHROM, TP53_START, TP53_END)):
        tid = t.attributes.get('transcript_id', [''])[0]
        if TP53_TRANSCRIPT in tid:
            tx = t; break
    assert tx is not None and tx.strand == '-'
    cp_arr = np.zeros(region_len, dtype=np.int8)
    cds_list = []
    for cds in db.children(tx, featuretype='CDS', order_by='start'):
        F = int(cds.frame); s, e = cds.start, cds.end
        cds_list.append({'s_1b': s, 'e_1b': e, 'frame': F})
        for p in range(s, e + 1):
            i = (p - 1) - region_start_0
            if 0 <= i < region_len:
                cp_arr[i] = ((e - p - F) % 3) + 1
    return cp_arr, cds_list


def stitch_profile(window_results, region_len):
    """Mean-overlap stitch list of (start_0, profile) into region-length array."""
    s = np.zeros(region_len, dtype=np.float64)
    c = np.zeros(region_len, dtype=np.int32)
    for st, prof in window_results:
        en = st + len(prof)
        lo = max(0, st); hi = min(region_len, en)
        if hi <= lo: continue
        off = lo - st
        s[lo:hi] += prof[off:off + (hi - lo)]
        c[lo:hi] += 1
    out = np.full(region_len, np.nan)
    m = c > 0
    out[m] = s[m] / c[m]
    return out


def main():
    t0 = time.time()
    setup_publication_style()
    cache = np.load(RUNS / '02_gene_structure_cache.npz', allow_pickle=True)
    D_cos = cache['D_cos']  # [n_win, L, T]
    starts = list(cache['starts'])
    n_win, L, T = D_cos.shape
    log.info('cache: n_win=%d L=%d T=%d', n_win, L, T)

    pen = D_cos[:, L - 2, EDGE_WARMUP:].ravel()
    gamma_cos = float(np.quantile(pen, 0.70))
    log.info('gamma_cos (q70 penultimate) = %.4f', gamma_cos)

    # Region geometry
    region_start_0 = load_region_offset()
    pad = WINDOW // 2
    region_len = (TP53_END + pad) - region_start_0
    log.info('region_start_0=%d region_len=%d', region_start_0, region_len)

    # Compute c_interp per window (using full D_cos[L,T])
    half = (WINDOW - CENTRAL_WIDTH) // 2
    win_results = []
    for i, st in enumerate(starts):
        D_win = torch.from_numpy(D_cos[i])  # [L, T]
        c_int, _ = settling_depth_interp(D_win, gamma=gamma_cos)
        central = c_int.numpy()[half:half + CENTRAL_WIDTH].astype(np.float32)
        win_results.append((int(st) + half, central))

    profile = stitch_profile(win_results, region_len)
    log.info('profile coverage: %d / %d positions', int((~np.isnan(profile)).sum()), region_len)

    # Codon positions per genomic position
    cp_arr, cds_list = codon_positions_for_cds(region_start_0, region_len)
    in_cds = cp_arr > 0
    valid = in_cds & ~np.isnan(profile)
    log.info('CDS positions (in region): %d, with profile: %d', int(in_cds.sum()), int(valid.sum()))
    for cp_val in (1, 2, 3):
        n = int(((cp_arr == cp_val) & valid).sum())
        log.info('  codon pos %d: n=%d', cp_val, n)

    # Build per-position dataframe
    rows = []
    idxs = np.where(valid)[0]
    for i in idxs:
        rows.append({'genome_pos_0b': region_start_0 + int(i),
                     'codon_pos': int(cp_arr[i]),
                     'gdtr_c_interp': float(profile[i])})
    df = pd.DataFrame(rows)
    log.info('built df: %d rows', len(df))

    # gDTR is the binary deep-thinking flag aggregate; per-position c_interp is
    # the underlying continuous quantity. Report both per-codon-pos.
    df['deep_thinking'] = (df['gdtr_c_interp'] > RHO * L).astype(int)

    # Per-codon-position summary
    from src.stats import bootstrap_ci
    summary_rows = []
    for cp_val in (1, 2, 3):
        sub = df[df['codon_pos'] == cp_val]['gdtr_c_interp'].values
        if len(sub) == 0: continue
        mean_v, lo, hi = bootstrap_ci(sub, n_boot=1000, seed=SEED)
        summary_rows.append({
            'codon_pos': cp_val,
            'n': int(len(sub)),
            'mean_c_interp': float(np.mean(sub)),
            'ci95_low': lo, 'ci95_high': hi,
            'median_c_interp': float(np.median(sub)),
            'std_c_interp': float(np.std(sub)),
            'gdtr_frac_deep': float((sub > RHO * L).mean()),
        })
    df_sum = pd.DataFrame(summary_rows)

    # Kruskal-Wallis 3-way
    g1 = df[df['codon_pos'] == 1]['gdtr_c_interp'].values
    g2 = df[df['codon_pos'] == 2]['gdtr_c_interp'].values
    g3 = df[df['codon_pos'] == 3]['gdtr_c_interp'].values
    H, p_kw = kruskal(g1, g2, g3)
    log.info('Kruskal-Wallis: H=%.4f p=%.3e', H, p_kw)

    # Pairwise MWU + Bonferroni
    pairs = list(combinations([1, 2, 3], 2))
    posthoc = []
    bonf = len(pairs)
    for a, b in pairs:
        ga = df[df['codon_pos'] == a]['gdtr_c_interp'].values
        gb = df[df['codon_pos'] == b]['gdtr_c_interp'].values
        U, p = mannwhitneyu(ga, gb, alternative='two-sided')
        n1, n2 = len(ga), len(gb)
        rbr = 1 - 2 * U / (n1 * n2)
        posthoc.append({'pair': f'{a}_vs_{b}',
            'U': float(U), 'p': float(p),
            'p_bonf': float(min(1.0, p * bonf)),
            'rank_biserial_r': float(rbr),
            'mean_diff': float(np.mean(ga) - np.mean(gb))})

    for r in posthoc:
        log.info('  %s: U=%.0f p=%.3e p_bonf=%.3e rbr=%+.3f Δmean=%+.3f',
                 r['pair'], r['U'], r['p'], r['p_bonf'],
                 r['rank_biserial_r'], r['mean_diff'])

    # CSV outputs
    df_sum.to_csv(TABLES / 'E2_codon_gdtr.csv', index=False)
    log.info('wrote %s', TABLES / 'E2_codon_gdtr.csv')

    # Figure: boxplot + violin overlay
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    data_lists = [g1, g2, g3]
    positions = [1, 2, 3]
    palette = [WONG_PALETTE['blue'], WONG_PALETTE['orange'], WONG_PALETTE['vermillion']]
    parts = ax.violinplot(data_lists, positions=positions,
        showmeans=False, showmedians=False, showextrema=False, widths=0.75)
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(palette[i]); pc.set_alpha(0.35); pc.set_edgecolor('none')

    bp = ax.boxplot(data_lists, positions=positions, widths=0.35,
        patch_artist=True, showfliers=False, medianprops={'color': 'black', 'lw': 1.2})
    for i, box in enumerate(bp['boxes']):
        box.set_facecolor(palette[i]); box.set_alpha(0.85); box.set_edgecolor('black')
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(['1st', '2nd', '3rd (wobble)'])
    ax.set_xlabel('Codon position (transcript orientation)')
    ax.set_ylabel(r'gDTR settling depth $c_{interp}$ (layer)')
    ax.set_title(f'TP53 CDS: codon-position stratification (n={len(df)})')

    # Annotate KW + post-hoc
    txt = f'Kruskal-Wallis: H={H:.2f}, p={p_kw:.2e}\n'
    for r in posthoc:
        txt += f"{r['pair']}: p_bonf={r['p_bonf']:.2e}\n"
    ax.text(0.02, 0.98, txt.strip(), transform=ax.transAxes,
            ha='left', va='top', fontsize=7,
            bbox=dict(boxstyle='round', facecolor='white',
                      alpha=0.85, edgecolor='gray'))
    plt.tight_layout()
    fig.savefig(FIGURES / 'E2_codon_gdtr.pdf')
    fig.savefig(FIGURES / 'E2_codon_gdtr.png', dpi=200)
    plt.close(fig)
    log.info('wrote E2_codon_gdtr.{pdf,png}')

    # JSON output
    out = {
        'script': 'E2_codon_gdtr.py',
        'transcript': TP53_TRANSCRIPT,
        'strand': '-',
        'n_cds_features': len(cds_list),
        'cds_features': cds_list,
        'lens': 'UR-gDTR (cosine)',
        'gamma_cos_q70_penultimate': gamma_cos,
        'rho': RHO, 'L': int(L),
        'central_width': CENTRAL_WIDTH,
        'edge_warmup': EDGE_WARMUP,
        'window': WINDOW, 'stride': STRIDE,
        'n_total_cds_positions': int(in_cds.sum()),
        'n_with_profile': int(valid.sum()),
        'per_codon_summary': summary_rows,
        'kruskal_wallis': {'H': float(H),
                           'p': float(p_kw),
                           'df': 2,
                           'n_total': int(len(df))},
        'posthoc_pairwise_mwu': posthoc,
    }
    out['runtime_s'] = time.time() - t0
    out['host'] = socket.gethostname()
    out['platform'] = platform.platform()
    out['torch'] = torch.__version__

    with open(RUNS / 'E2_codon_gdtr.json', 'w') as f:
        json.dump(out, f, indent=2, default=str)
    log.info('wrote %s', RUNS / 'E2_codon_gdtr.json')

    log.info('=' * 60)
    log.info('E2 verdict:')
    for s_row in summary_rows:
        log.info('  pos %d (n=%d): mean c=%.3f [CI %.3f, %.3f]',
                 s_row['codon_pos'], s_row['n'],
                 s_row['mean_c_interp'],
                 s_row['ci95_low'], s_row['ci95_high'])
    log.info('  Kruskal-Wallis H=%.3f p=%.3e', H, p_kw)
    log.info('=' * 60)


if __name__ == '__main__':
    main()
