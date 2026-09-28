"""Phase 4 concordance — 4-way Spearman + per-model splice signal."""
from __future__ import annotations
import json, time, warnings
warnings.filterwarnings('ignore')
from pathlib import Path
import numpy as np
import h5py
import _runner_utils as ru
ru.add_repo_paths()
PHASE = 'phase4_concordance'
PHASE_OUT_DIR = ru.GDTR_ROOT / 'results' / 'phase4'
PHASE_OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG = ru.setup_logging(PHASE)

CACHES = {
    'evo2':     ru.GDTR_ROOT / 'results' / 'phase1.6' / 'chr22_cache.h5',
    'hyenadna': ru.GDTR_ROOT / 'results' / 'phase4'   / 'chr22_cache_hyenadna.h5',
    'nt_v2':    ru.GDTR_ROOT / 'results' / 'phase4'   / 'chr22_cache_nt.h5',
    'dnabert2': ru.GDTR_ROOT / 'results' / 'phase4'   / 'chr22_cache_dnabert.h5',
}

CTX = {0:'intergenic',1:'intron',2:'coding_exon',3:'5utr',4:'3utr',5:'splice_donor',6:'splice_acceptor',7:'repeat'}


def settling_depth(D, gamma):
    rmin = np.minimum.accumulate(D, axis=0)
    below = rmin <= gamma
    any_b = below.any(axis=0)
    fi = below.argmax(axis=0)
    L = D.shape[0]
    return np.where(any_b, fi + 1, L).astype(np.int64)

def load_model_summary(name, path):
    if not path.exists():
        LOG.warning('cache missing for %s: %s', name, path)
        return None
    with h5py.File(path, 'r') as h:
        N, L, T = h['D_cos'].shape
        done = h['done_mask'][:]
        n_done = int(done.sum())
        gamma_layer = max(0, L - 2)
        idxs = np.where(done == 1)[0]
        rng = np.random.default_rng(42)
        if len(idxs) > 2000:
            idxs = rng.choice(idxs, size=2000, replace=False)
        T_real_arr = h['T_real'][:] if 'T_real' in h else np.full(N, T, dtype=np.int32)
        D_pen = []
        for j in idxs:
            tr = int(T_real_arr[j]) if T_real_arr[j] > 0 else T
            d = h['D_cos'][j, gamma_layer, :tr].astype(np.float32)
            D_pen.append(d)
        D_pen = np.concatenate(D_pen) if D_pen else np.array([0.0])
        gamma = float(np.quantile(D_pen, 0.70))
        LOG.info('%s: N=%d done=%d L=%d T=%d gamma_q70(L%d)=%.4f', name, N, n_done, L, T, gamma_layer, gamma)
        return {'name': name, 'N': N, 'L': L, 'T': T, 'gamma': gamma, 'gamma_layer': gamma_layer, 'n_done': n_done, 'path': str(path)}

def per_window_mean_c(name, path, gamma):
    """Return [N] mean settling depth per window."""
    with h5py.File(path, 'r') as h:
        N, L, T = h['D_cos'].shape
        done = h['done_mask'][:]
        T_real_arr = h['T_real'][:] if 'T_real' in h else np.full(N, T, dtype=np.int32)
        out = np.full(N, np.nan, dtype=np.float64)
        for i in range(N):
            if not done[i]:
                continue
            tr = int(T_real_arr[i]) if T_real_arr[i] > 0 else T
            D = h['D_cos'][i, :, :tr].astype(np.float32)
            c = settling_depth(D, gamma)
            out[i] = float(c.mean())
        return out

def per_position_splice_signal(name, path, gamma, labels):
    with h5py.File(path, 'r') as h:
        N, L, T = h['D_cos'].shape
        done = h['done_mask'][:]
        starts = h['starts'][:]
        ends = h['ends'][:]
        T_real_arr = h['T_real'][:] if 'T_real' in h else np.full(N, T, dtype=np.int32)
        if T != 6000:
            return None
        bins = {1: [], 2: [], 5: [], 6: []}
        for i in range(N):
            if not done[i]:
                continue
            s_, e_ = int(starts[i]), int(ends[i])
            tr = int(T_real_arr[i]) if T_real_arr[i] > 0 else T
            D = h['D_cos'][i, :, :tr].astype(np.float32)
            c = settling_depth(D, gamma)
            lab_slice = labels[s_:e_][:tr]
            for code in bins:
                sel = lab_slice == code
                if sel.any():
                    bins[code].extend(c[sel].tolist())
    return {CTX[k]: {'n': len(v), 'mean_c': (float(np.mean(v)) if v else None), 'median_c': (float(np.median(v)) if v else None)} for k, v in bins.items()}

def per_window_splice_signal(name, path, gamma, win_meta):
    """Non-per-bp models: partition windows by content."""
    mean_c = per_window_mean_c(name, path, gamma)
    splice_mask = win_meta['n_splice'] > 0
    intron_mask = (win_meta['n_intron'] > 4500) & (win_meta['n_splice'] == 0)
    exon_mask = (win_meta['n_coding_exon'] > 1000) & (win_meta['n_splice'] == 0)
    out = {}
    for nm, mask in [('splice_containing', splice_mask), ('intron_dominant', intron_mask), ('exon_dominant', exon_mask)]:
        v = mean_c[mask]
        v = v[~np.isnan(v)]
        out[nm] = {'n': int(v.size), 'mean_c': (float(v.mean()) if v.size else None), 'median_c': (float(np.median(v)) if v.size else None)}
    return out, mean_c

def main():
    with ru.phase_context(PHASE, PHASE_OUT_DIR, step_name='concordance'):
        from scipy.stats import spearmanr
        labels_path = ru.GDTR_ROOT / 'data' / 'annotation' / 'chr22_position_labels.npy'
        labels = np.load(labels_path)
        wins_path = ru.GDTR_ROOT / 'data' / 'baselines' / 'chr22_windows.tsv'
        with wins_path.open() as f:
            hdr = f.readline().rstrip().split('\t')
            rows = [line.rstrip().split('\t') for line in f]
        win_dict = {h: np.array([r[hdr.index(h)] for r in rows]) for h in hdr}
        for k in ('start','end','n_intron','n_coding_exon','n_splice','n_5utr','n_3utr','n_intergenic'):
            if k in win_dict:
                win_dict[k] = win_dict[k].astype(np.int64)
        summary = {}
        for name, path in CACHES.items():
            r = load_model_summary(name, path)
            if r is not None:
                summary[name] = r
        if not summary:
            raise RuntimeError('no caches available')
        # Stage 2: per-window mean_c
        mean_c = {}
        for name, info in summary.items():
            LOG.info('computing per-window mean_c for %s ...', name)
            mc = per_window_mean_c(name, Path(info['path']), info['gamma'])
            mean_c[name] = mc
            LOG.info('  %s mean_c stats: mean=%.3f median=%.3f n=%d', name, np.nanmean(mc), np.nanmedian(mc), int(np.sum(~np.isnan(mc))))
        # Stage 3: pairwise Spearman
        names = list(mean_c.keys())
        K = len(names)
        rho_mat = np.full((K, K), np.nan)
        p_mat = np.full((K, K), np.nan)
        for i, a in enumerate(names):
            for j, b in enumerate(names):
                if i == j:
                    rho_mat[i, j] = 1.0; p_mat[i, j] = 0.0
                    continue
                if i < j:
                    x = mean_c[a]; y = mean_c[b]
                    m = ~(np.isnan(x) | np.isnan(y))
                    if m.sum() < 50:
                        continue
                    rho, pv = spearmanr(x[m], y[m])
                    rho_mat[i, j] = rho_mat[j, i] = float(rho)
                    p_mat[i, j] = p_mat[j, i] = float(pv)
                    LOG.info('Spearman %s vs %s: rho=%.4f p=%.2e n=%d', a, b, rho, pv, int(m.sum()))
        thresh = {}; top_masks = {}
        for nm in names:
            v = mean_c[nm]
            vv = v[~np.isnan(v)]
            if vv.size == 0:
                continue
            t = float(np.quantile(vv, 0.90))
            thresh[nm] = t
            top_masks[nm] = (v >= t) & (~np.isnan(v))
        if top_masks:
            inter = np.ones(len(next(iter(top_masks.values()))), dtype=bool)
            for m in top_masks.values():
                inter &= m
            n_inter = int(inter.sum())
            sizes = np.array([m.sum() for m in top_masks.values()])
            mean_top = max(1, int(sizes.mean()))
            jaccard = n_inter / mean_top
            LOG.info('Top-10%% inter=%d mean top=%d jaccard~%.3f', n_inter, mean_top, jaccard)
        else:
            n_inter = 0; jaccard = float('nan'); inter = np.zeros(0, dtype=bool)
        # Functional enrichment of intersection: fraction of inter windows
        # with >0 splice / coding_exon / 5utr content.
        inter_enrichment = {}
        if 'n_splice' in win_dict and inter.size:
            for ftr in ['n_splice', 'n_coding_exon', 'n_5utr', 'n_3utr', 'n_intron']:
                if ftr in win_dict:
                    arr = win_dict[ftr]
                    if arr.size != inter.size:
                        continue
                    sel = inter
                    inter_enrichment[ftr] = {
                        'frac_inter_with_feature': float(np.mean(arr[sel] > 0)) if sel.any() else None,
                        'frac_all_with_feature':   float(np.mean(arr > 0)),
                    }
        # Stage 5: per-model splice signal
        splice_signal = {}
        for name, info in summary.items():
            if info['T'] == 6000:
                LOG.info('per-position splice signal for %s ...', name)
                sig = per_position_splice_signal(name, Path(info['path']), info['gamma'], labels)
                splice_signal[name] = {'kind': 'per_position', 'data': sig}
            else:
                LOG.info('per-window splice signal for %s ...', name)
                sig, _ = per_window_splice_signal(name, Path(info['path']), info['gamma'], win_dict)
                splice_signal[name] = {'kind': 'per_window', 'data': sig}
        # Save outputs
        per_model_summary = {}
        for nm, info in summary.items():
            per_model_summary[nm] = {
                'N': info['N'], 'L': info['L'], 'T': info['T'],
                'n_done': info['n_done'], 'gamma_q70': info['gamma'],
                'gamma_layer': info['gamma_layer'],
                'mean_c_mean': float(np.nanmean(mean_c[nm])),
                'mean_c_median': float(np.nanmedian(mean_c[nm])),
                'splice_signal': splice_signal.get(nm),
            }
        (PHASE_OUT_DIR / 'per_model_summary.json').write_text(json.dumps(per_model_summary, indent=2))
        rho_dict = {a: {b: float(rho_mat[i, j]) for j, b in enumerate(names)} for i, a in enumerate(names)}
        p_dict = {a: {b: float(p_mat[i, j]) for j, b in enumerate(names)} for i, a in enumerate(names)}
        concord = {
            'models': names,
            'spearman_rho': rho_dict,
            'spearman_p': p_dict,
            'top_decile_intersection': {
                'n_intersect': n_inter,
                'jaccard_approx': jaccard,
                'inter_enrichment': inter_enrichment,
            },
        }
        (PHASE_OUT_DIR / 'concordance_matrix.json').write_text(json.dumps(concord, indent=2))
        # Figures
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(5.2, 4.5))
            im = ax.imshow(rho_mat, vmin=0, vmax=1, cmap='viridis')
            ax.set_xticks(range(len(names)))
            ax.set_xticklabels(names, rotation=30, ha='right')
            ax.set_yticks(range(len(names)))
            ax.set_yticklabels(names)
            for i in range(len(names)):
                for j in range(len(names)):
                    val = rho_mat[i, j]
                    color = 'white' if val < 0.6 else 'black'
                    ax.text(j, i, f'{val:.2f}', ha='center', va='center', color=color, fontsize=9)
            ax.set_title('Cross-arch Spearman rho (per-window mean c)')
            plt.colorbar(im, ax=ax, fraction=0.046)
            fig.tight_layout()
            for ext in ('pdf', 'png'):
                fig.savefig(PHASE_OUT_DIR / f'F12_cross_arch_heatmap.{ext}', dpi=150)
            plt.close(fig)
            n_models = len(splice_signal)
            ncol = max(1, n_models)
            fig, axes = plt.subplots(1, ncol, figsize=(3.6 * ncol, 4))
            if ncol == 1:
                axes = [axes]
            for ax, (nm, s) in zip(axes, splice_signal.items()):
                if s is None:
                    ax.set_title(f'{nm}: n/a'); continue
                data = s['data']
                keys = list(data.keys())
                vals = [data[k]['mean_c'] if data[k]['mean_c'] is not None else 0 for k in keys]
                ax.bar(range(len(keys)), vals)
                ax.set_xticks(range(len(keys)))
                ax.set_xticklabels(keys, rotation=20, ha='right', fontsize=8)
                ax.set_ylabel('mean c')
                ax.set_title(f"{nm} ({s['kind']})", fontsize=9)
            fig.tight_layout()
            for ext in ('pdf', 'png'):
                fig.savefig(PHASE_OUT_DIR / f'F13_per_model_splice.{ext}', dpi=150)
            plt.close(fig)
        except Exception as e:
            LOG.warning('figures failed: %s', e)

        ru.write_done(PHASE, PHASE_OUT_DIR, {'models': names, 'n_inter_top10': n_inter}, step_name='concordance')
        # Touch overall _done marker
        (PHASE_OUT_DIR / '_done').write_text(json.dumps({'phase': 'phase4', 'models': names}, indent=2))
        LOG.info('Phase 4 concordance done.')


if __name__ == '__main__':
    main()
