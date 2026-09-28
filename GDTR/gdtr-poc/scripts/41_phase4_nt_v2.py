"""Phase 4 — NT-v2 500M 4kb forward over chr22 windows."""
from __future__ import annotations
import json, time, warnings
warnings.filterwarnings('ignore')
from pathlib import Path
import numpy as np
import torch
import h5py
import _runner_utils as ru
ru.add_repo_paths()

PHASE = 'phase4_nt'
PHASE_OUT_DIR = ru.GDTR_ROOT / 'results' / 'phase4'
PHASE_OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG = ru.setup_logging(PHASE)
MODEL_ID = 'InstaDeepAI/nucleotide-transformer-v2-500m-multi-species'
WINDOW_BP = 6000
NT_BP = 4000
SEED = 42


def read_windows(p: Path):
    rows = []
    with p.open() as f:
        h = f.readline().rstrip().split('\t')
        for line in f:
            rows.append(dict(zip(h, line.rstrip().split('\t'))))
    return rows


def main():
    with ru.phase_context(PHASE, PHASE_OUT_DIR, step_name='nt_fwd'):
        torch.manual_seed(SEED); np.random.seed(SEED)
        from transformers import AutoTokenizer, AutoModelForMaskedLM
        import pysam
        LOG.info('loading %s', MODEL_ID)
        tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
        mdl = AutoModelForMaskedLM.from_pretrained(MODEL_ID, trust_remote_code=True).to('cuda').eval()
        N_LAYERS = int(mdl.config.num_hidden_layers)
        LOG.info('NT n_layers=%d hidden=%d vocab=%d', N_LAYERS, mdl.config.hidden_size, mdl.config.vocab_size)
        fa = pysam.FastaFile(str(ru.GDTR_ROOT / 'data' / 'reference' / 'chr22.fa'))
        win = read_windows(ru.GDTR_ROOT / 'data' / 'baselines' / 'chr22_windows.tsv')
        N = len(win)
        # Determine token count for the centered NT_BP slice
        seq0 = fa.fetch(win[0]['chrom'], int(win[0]['start']), int(win[0]['start']) + NT_BP).upper()
        T_tok = tok(seq0, return_tensors='pt')['input_ids'].shape[1]
        LOG.info('NT_BP=%d -> T_tok=%d', NT_BP, T_tok)
        h5_path = PHASE_OUT_DIR / 'chr22_cache_nt.h5'
        ckpt = PHASE_OUT_DIR / '_nt_ckpt.json'
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
                h.attrs['NT_BP'] = NT_BP
                h.attrs['n_layers'] = N_LAYERS
        from src.ur_gdtr_xarch import cosine_lens_xarch
        t0 = time.time()
        save_every = 500
        offset = (WINDOW_BP - NT_BP) // 2
        with h5py.File(h5_path, 'a') as h:
            for i in range(start_i, N):
                w = win[i]
                s = int(w['start']); e = int(w['end'])
                seq = fa.fetch(w['chrom'], s + offset, s + offset + NT_BP).upper()
                if len(seq) != NT_BP:
                    h['done_mask'][i] = 0; continue
                enc = tok(seq, return_tensors='pt').to('cuda')
                with torch.no_grad():
                    out = mdl(**enc, output_hidden_states=True)
                hs = out.hidden_states
                layer_hs = list(hs[1:1 + N_LAYERS])
                ref = hs[-1]
                D = cosine_lens_xarch(layer_hs, ref, bos_offset=1)
                Dn = D.numpy()
                Tcur = Dn.shape[1]
                if Tcur < T_tok:
                    pad = np.zeros((N_LAYERS, T_tok - Tcur), dtype=Dn.dtype)
                    Dn = np.concatenate([Dn, pad], axis=1)
                Dn = Dn[:, :T_tok]
                h['D_cos'][i] = Dn.astype(np.float16)
                h['starts'][i] = s; h['ends'][i] = e; h['done_mask'][i] = 1; h['T_real'][i] = int(Tcur)
                del out, hs, layer_hs, ref, D, Dn
                if (i + 1) % 50 == 0:
                    el = time.time() - t0
                    rate = (i + 1 - start_i) / max(1.0, el)
                    eta = (N - i - 1) / max(1e-6, rate)
                    LOG.info('i=%d/%d  rate=%.2f/s  ETA=%.1fmin', i + 1, N, rate, eta / 60)
                if (i + 1) % save_every == 0:
                    h.flush()
                    ckpt.write_text(json.dumps({'next': i + 1}))
        ckpt.write_text(json.dumps({'next': N}))
        ru.write_done(PHASE, PHASE_OUT_DIR, {'n_windows': N, 'h5': str(h5_path), 'n_layers': N_LAYERS, 'NT_BP': NT_BP, 'T_tok': T_tok}, step_name='nt_fwd')
        LOG.info('Phase 4 NT done: %d windows', N)


if __name__ == '__main__':
    main()
