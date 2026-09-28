"""Phase 4 — DNABERT-2 117M forward over chr22 windows."""
from __future__ import annotations
import json, time, warnings, sys
warnings.filterwarnings('ignore')
from pathlib import Path
import numpy as np
import torch
import h5py
import _runner_utils as ru
ru.add_repo_paths()

PHASE = 'phase4_dnabert'
PHASE_OUT_DIR = ru.GDTR_ROOT / 'results' / 'phase4'
PHASE_OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG = ru.setup_logging(PHASE)
MODEL_ID = 'zhihan1996/DNABERT-2-117M'
WINDOW_BP = 6000
D2_BP = 3000
SEED = 42


def read_windows(p):
    rows = []
    with p.open() as f:
        h = f.readline().rstrip().split('\t')
        for line in f:
            rows.append(dict(zip(h, line.rstrip().split('\t'))))
    return rows


def disable_triton_flash():
    for name, m in list(sys.modules.items()):
        if 'DNABERT-2-117M' in name and 'bert_layers' in name:
            m.flash_attn_qkvpacked_func = None
            return True
    return False


def main():
    with ru.phase_context(PHASE, PHASE_OUT_DIR, step_name='dnabert_fwd'):
        torch.manual_seed(SEED); np.random.seed(SEED)
        from transformers import AutoTokenizer, AutoModel
        import pysam
        LOG.info('loading %s', MODEL_ID)
        tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
        mdl = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True)
        ok = disable_triton_flash()
        LOG.info('triton flash-attn disabled: %s', ok)
        mdl = mdl.to('cuda').eval()
        N_LAYERS = int(mdl.config.num_hidden_layers)
        LOG.info('D2 n_layers=%d hidden=%d', N_LAYERS, mdl.config.hidden_size)

        # Register hooks to capture per-layer hidden states
        captures = {}
        hooks = []
        for li, layer in enumerate(mdl.encoder.layer):
            def mk(idx):
                def hook(module, inp, out):
                    # out has shape (n_unpadded, hidden) for B=1 fully-attended -> (T, H)
                    captures[idx] = out.detach()
                return hook
            hooks.append(layer.register_forward_hook(mk(li)))

        fa = pysam.FastaFile(str(ru.GDTR_ROOT / 'data' / 'reference' / 'chr22.fa'))
        win = read_windows(ru.GDTR_ROOT / 'data' / 'baselines' / 'chr22_windows.tsv')
        N = len(win)
        T_tok = 1024
        LOG.info('D2_BP=%d -> T_tok upper bound=%d', D2_BP, T_tok)

        h5_path = PHASE_OUT_DIR / 'chr22_cache_dnabert.h5'
        ckpt = PHASE_OUT_DIR / '_dnabert_ckpt.json'
        start_i = 0
        if h5_path.exists() and ckpt.exists():
            start_i = int(json.loads(ckpt.read_text()).get('next', 0))
            LOG.info('resuming i=%d', start_i)
        else:
            with h5py.File(h5_path, 'w') as h:
                h.create_dataset('D_cos', (N, N_LAYERS, T_tok), dtype='float16', compression='gzip', chunks=(1, N_LAYERS, T_tok))
                h.create_dataset('starts', (N,), dtype='int64')
                h.create_dataset('ends', (N,), dtype='int64')
                h.create_dataset('done_mask', (N,), dtype='uint8')
                h.create_dataset('T_real', (N,), dtype='int32')
                h.attrs['T_tok'] = T_tok
                h.attrs['D2_BP'] = D2_BP
                h.attrs['n_layers'] = N_LAYERS

        from src.ur_gdtr_xarch import cosine_lens_xarch
        t0 = time.time()
        save_every = 500
        offset = (WINDOW_BP - D2_BP) // 2
        with h5py.File(h5_path, 'a') as h:
            for i in range(start_i, N):
                w = win[i]
                s = int(w['start']); e = int(w['end'])
                seq = fa.fetch(w['chrom'], s + offset, s + offset + D2_BP).upper()
                if len(seq) != D2_BP:
                    h['done_mask'][i] = 0; continue
                enc = tok(seq, return_tensors='pt').to('cuda')
                captures.clear()
                with torch.no_grad():
                    out = mdl(**enc, output_hidden_states=False)
                layer_hs = []
                for li in range(N_LAYERS):
                    t = captures[li]
                    if t.dim() == 2:
                        t = t.unsqueeze(0)
                    layer_hs.append(t)
                ref = layer_hs[-1]
                D = cosine_lens_xarch(layer_hs, ref, bos_offset=1)
                Dn = D.numpy()
                Tcur = Dn.shape[1]
                if Tcur < T_tok:
                    pad = np.zeros((N_LAYERS, T_tok - Tcur), dtype=Dn.dtype)
                    Dn = np.concatenate([Dn, pad], axis=1)
                Dn = Dn[:, :T_tok]
                h['D_cos'][i] = Dn.astype(np.float16)
                h['starts'][i] = s; h['ends'][i] = e; h['done_mask'][i] = 1; h['T_real'][i] = int(Tcur)
                del out, layer_hs, ref, D, Dn
                if (i + 1) % 50 == 0:
                    el = time.time() - t0
                    rate = (i + 1 - start_i) / max(1.0, el)
                    eta = (N - i - 1) / max(1e-6, rate)
                    LOG.info('i=%d/%d  rate=%.2f/s  ETA=%.1fmin', i + 1, N, rate, eta / 60)
                if (i + 1) % save_every == 0:
                    h.flush()
                    ckpt.write_text(json.dumps({'next': i + 1}))
        ckpt.write_text(json.dumps({'next': N}))
        for h in hooks:
            h.remove()
        ru.write_done(PHASE, PHASE_OUT_DIR, {'n_windows': N, 'h5': str(h5_path), 'n_layers': N_LAYERS, 'D2_BP': D2_BP, 'T_tok': T_tok}, step_name='dnabert_fwd')
        LOG.info('Phase 4 D2 done: %d windows', N)


if __name__ == '__main__':
    main()
