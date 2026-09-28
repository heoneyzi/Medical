"""Phase 4 — HyenaDNA-large 6kb forward over chr22 windows.

Captures per-block residuals (hidden_states[1..n_layer]) plus final-norm as
the cosine-lens reference. Saves per-window D_cos to chr22_cache_hyenadna.h5.
"""
from __future__ import annotations
import json, time, warnings
warnings.filterwarnings('ignore')
from pathlib import Path
import numpy as np
import torch
import h5py
import _runner_utils as ru
ru.add_repo_paths()

PHASE = 'phase4_hyenadna'
PHASE_OUT_DIR = ru.GDTR_ROOT / 'results' / 'phase4'
PHASE_OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG = ru.setup_logging(PHASE)

MODEL_ID = 'LongSafari/hyenadna-large-1m-seqlen-hf'
WINDOW_BP = 6000
SEED = 42


def read_windows(p: Path):
    rows = []
    with p.open() as f:
        h = f.readline().rstrip().split('\t')
        for line in f:
            rows.append(dict(zip(h, line.rstrip().split('\t'))))
    return rows


def main():
    with ru.phase_context(PHASE, PHASE_OUT_DIR, step_name='hyenadna_fwd'):
        torch.manual_seed(SEED); np.random.seed(SEED)
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import pysam

        LOG.info('loading %s', MODEL_ID)
        tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
        mdl = AutoModelForCausalLM.from_pretrained(MODEL_ID, trust_remote_code=True).to('cuda').eval()
        cfg = mdl.config
        N_LAYERS = int(cfg.n_layer)
        LOG.info('Hyena n_layer=%d d_model=%d vocab=%d', N_LAYERS, cfg.d_model, cfg.vocab_size)
        fa = pysam.FastaFile(str(ru.GDTR_ROOT / 'data' / 'reference' / 'chr22.fa'))
        win = read_windows(ru.GDTR_ROOT / 'data' / 'baselines' / 'chr22_windows.tsv')
        N = len(win); T = WINDOW_BP
        LOG.info('windows=%d T=%d', N, T)
        h5_path = PHASE_OUT_DIR / 'chr22_cache_hyenadna.h5'
        ckpt = PHASE_OUT_DIR / '_hyenadna_ckpt.json'
        start_i = 0
        if h5_path.exists() and ckpt.exists():
            start_i = int(json.loads(ckpt.read_text()).get('next', 0))
            LOG.info('resuming i=%d', start_i)
        else:
            with h5py.File(h5_path, 'w') as h:
                h.create_dataset('D_cos', (N, N_LAYERS, T), dtype='float16', compression='gzip', chunks=(1, N_LAYERS, T))
                h.create_dataset('starts', (N,), dtype='int64')
                h.create_dataset('ends', (N,), dtype='int64')
                h.create_dataset('done_mask', (N,), dtype='uint8')
                h.create_dataset('T_real', (N,), dtype='int32')
                h.attrs['T_tok'] = T
                h.attrs['n_layers'] = N_LAYERS
        from src.ur_gdtr_xarch import cosine_lens_xarch
        t0 = time.time()
        save_every = 1000
        with h5py.File(h5_path, 'a') as h:
            for i in range(start_i, N):
                w = win[i]
                s = int(w['start']); e = int(w['end'])
                seq = fa.fetch(w['chrom'], s, e).upper()
                if len(seq) != T:
                    h['done_mask'][i] = 0; continue
                enc = tok(seq, return_tensors='pt').to('cuda')
                with torch.no_grad():
                    out = mdl(**enc, output_hidden_states=True)
                hs = out.hidden_states
                layer_hs = list(hs[1:1 + N_LAYERS])
                D = cosine_lens_xarch(layer_hs, hs[-1], bos_offset=1)
                h['D_cos'][i] = D.numpy().astype(np.float16)
                h['starts'][i] = s; h['ends'][i] = e; h['done_mask'][i] = 1; h['T_real'][i] = T
                del out, hs, layer_hs, D
                if (i + 1) % 100 == 0:
                    el = time.time() - t0
                    rate = (i + 1 - start_i) / max(1.0, el)
                    eta = (N - i - 1) / max(1e-6, rate)
                    LOG.info('i=%d/%d  rate=%.2f/s  ETA=%.1fmin', i + 1, N, rate, eta / 60)
                if (i + 1) % save_every == 0:
                    h.flush()
                    ckpt.write_text(json.dumps({'next': i + 1}))
        ckpt.write_text(json.dumps({'next': N}))
        ru.write_done(PHASE, PHASE_OUT_DIR, {'n_windows': N, 'h5': str(h5_path), 'n_layers': N_LAYERS}, step_name='hyenadna_fwd')
        LOG.info('Phase 4 Hyena done: %d windows', N)


if __name__ == '__main__':
    main()
