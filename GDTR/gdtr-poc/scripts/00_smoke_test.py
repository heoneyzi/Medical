"""
Phase 0 — 00_smoke_test.py
Verifies HyenaDNA loads, forward pass works, hidden states are accessible,
hooks work, final layernorm is locatable, tied-weight LM head is checked.
Saves all findings to results/runs/00_smoke_test.json.
"""
from __future__ import annotations
import json, time, logging
from pathlib import Path
import torch

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger('smoke')

PROJECT_ROOT = Path('/path/to/gDTR-PoC')
RESULTS_DIR  = PROJECT_ROOT / 'results' / 'runs'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
OUT_JSON     = RESULTS_DIR / '00_smoke_test.json'

MODEL_ID = 'LongSafari/hyenadna-medium-160k-seqlen-hf'

def main() -> dict:
    findings: dict = {'model_id': MODEL_ID, 'timestamp': time.time()}
    torch.manual_seed(42)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    log.info('Loading tokenizer ...')
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    log.info('Loading model bf16 cuda ...')
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to('cuda').eval()

    findings['transformers_version'] = __import__('transformers').__version__
    findings['torch_version']        = torch.__version__
    findings['cuda_device']          = torch.cuda.get_device_name(0)
    findings['model_class']          = type(model).__name__
    findings['tokenizer_class']      = type(tokenizer).__name__
    log.info(f"loaded {findings['model_class']} on {findings['cuda_device']}")

    findings['n_params_total'] = sum(p.numel() for p in model.parameters())
    findings['n_params_trainable'] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    repr_str = repr(model)
    findings['model_repr_truncated'] = repr_str[:6000]
    findings['model_repr_total_len'] = len(repr_str)

    backbone_attr = None; layers_attr = None; backbone_full = None
    cands = [('hyena.backbone','layers'),('backbone','layers'),('hyena','layers'),('model','layers'),('transformer','h')]
    for outer, inner in cands:
        obj = model; ok = True
        for part in outer.split('.'):
            if hasattr(obj, part): obj = getattr(obj, part)
            else: ok = False; break
        if ok and hasattr(obj, inner):
            backbone_attr, layers_attr = outer, inner
            backbone_full = obj; break
    findings['backbone_attr_path'] = f"{backbone_attr}.{layers_attr}" if backbone_attr else None
    if backbone_attr:
        layers = getattr(backbone_full, layers_attr)
        findings['n_backbone_layers'] = len(layers)
        findings['backbone_layer_class'] = type(layers[0]).__name__
    else:
        findings['n_backbone_layers'] = None

    cfg = model.config
    findings['config_hidden_size']  = getattr(cfg, 'hidden_size', getattr(cfg, 'd_model', None))
    findings['config_vocab_size']   = getattr(cfg, 'vocab_size', None)
    findings['config_n_layer']      = getattr(cfg, 'n_layer', getattr(cfg, 'num_hidden_layers', None))
    findings['config_max_seq_len']  = getattr(cfg, 'max_seq_len', getattr(cfg, 'max_position_embeddings', None))

    if hasattr(model, 'lm_head'):
        lm_head = model.lm_head
        findings['lm_head_class'] = type(lm_head).__name__
        if hasattr(lm_head, 'weight'):
            findings['lm_head_weight_shape'] = list(lm_head.weight.shape)
            findings['lm_head_out_features'] = (lm_head.weight.shape[0]
                                                 if lm_head.weight.ndim==2 else None)

    emb = model.get_input_embeddings()
    findings['input_embedding_class'] = type(emb).__name__
    findings['input_embedding_weight_shape'] = list(emb.weight.shape)

    same_id = False; same_data_ptr = False
    if hasattr(model, 'lm_head') and hasattr(model.lm_head, 'weight'):
        same_id = id(model.lm_head.weight) == id(emb.weight)
        same_data_ptr = model.lm_head.weight.data_ptr() == emb.weight.data_ptr()
    findings['lm_head_tied_id_match']      = bool(same_id)
    findings['lm_head_tied_dataptr_match'] = bool(same_data_ptr)

    ln_candidates = []
    for name, mod in model.named_modules():
        cls = type(mod).__name__.lower()
        if 'layernorm' in cls or 'rmsnorm' in cls or name.endswith('.ln_f') or name.endswith('.norm'):
            ln_candidates.append((name, type(mod).__name__))
    findings['layernorm_modules_all'] = ln_candidates[:30]
    final_ln = None
    for name, cls in ln_candidates:
        if 'ln_f' in name or name.endswith('.norm') or 'final' in name.lower():
            final_ln = name
    findings['final_layernorm_attr_guess'] = final_ln

    seq = 'ACGT' * 1500
    findings['test_seq_len_chars'] = len(seq)
    enc = tokenizer(seq, return_tensors='pt')
    input_ids = enc.input_ids
    findings['input_ids_shape'] = list(input_ids.shape)
    findings['input_ids_first_5']  = input_ids[0, :5].tolist()
    findings['input_ids_last_5']   = input_ids[0, -5:].tolist()
    findings['tokenizer_special_tokens_map'] = {k: v for k, v in tokenizer.special_tokens_map.items()}
    findings['tokenizer_bos_token_id'] = tokenizer.bos_token_id
    findings['tokenizer_eos_token_id'] = tokenizer.eos_token_id
    findings['tokenizer_pad_token_id'] = tokenizer.pad_token_id
    findings['tokenizer_vocab_size']   = tokenizer.vocab_size
    findings['tokenizer_get_vocab_size'] = len(tokenizer.get_vocab())
    findings['tokenizer_offset_input_ids_minus_seq_len'] = int(input_ids.shape[1]) - len(seq)

    captured_via_hook = []; hook_handles = []
    if findings['backbone_attr_path']:
        layers = getattr(backbone_full, layers_attr)
        def make_hook(idx):
            def hook(mod, inp, out):
                t = out[0] if isinstance(out, tuple) else out
                captured_via_hook.append((idx, tuple(t.shape) if hasattr(t,'shape') else None))
            return hook
        for i, blk in enumerate(layers):
            hook_handles.append(blk.register_forward_hook(make_hook(i)))

    input_ids = input_ids.to('cuda')
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    with torch.no_grad():
        try:
            out = model(input_ids, output_hidden_states=True)
            findings['forward_ok'] = True
        except Exception as e:
            findings['forward_ok'] = False
            findings['forward_error'] = repr(e)
            for h in hook_handles: h.remove()
            return findings
    findings['forward_seconds'] = time.time() - t0
    findings['gpu_peak_bytes']  = torch.cuda.max_memory_allocated()
    findings['gpu_peak_GB']     = findings['gpu_peak_bytes'] / 1e9

    findings['out_logits_shape']      = list(out.logits.shape) if hasattr(out, 'logits') else None
    findings['out_has_hidden_states'] = hasattr(out, 'hidden_states') and out.hidden_states is not None
    if findings['out_has_hidden_states']:
        hs = out.hidden_states
        findings['out_hidden_states_count'] = len(hs)
        findings['out_hidden_states_shapes'] = [list(h.shape) for h in hs]
    else:
        findings['out_hidden_states_count'] = 0

    findings['hook_capture_count']  = len(captured_via_hook)
    findings['hook_capture_shapes'] = [list(s) if s else None for _, s in captured_via_hook]
    for h in hook_handles: h.remove()

    if findings['out_has_hidden_states']:
        h0 = out.hidden_states[0].float()
        hL = out.hidden_states[-1].float()
        findings['hidden_state_0_mean']    = float(h0.mean().item())
        findings['hidden_state_0_std']     = float(h0.std().item())
        findings['hidden_state_last_mean'] = float(hL.mean().item())
        findings['hidden_state_last_std']  = float(hL.std().item())

    found_final_norm = None
    for path in ['hyena.backbone.ln_f','backbone.ln_f','transformer.ln_f','hyena.backbone.norm','backbone.norm','model.norm','norm']:
        obj = model; ok = True
        for part in path.split('.'):
            if hasattr(obj, part): obj = getattr(obj, part)
            else: ok = False; break
        if ok and hasattr(obj, 'forward'):
            found_final_norm = (path, type(obj).__name__); break
    findings['final_norm_resolved_path'] = found_final_norm

    if found_final_norm and findings['out_has_hidden_states']:
        path, _ = found_final_norm
        norm = model
        for part in path.split('.'): norm = getattr(norm, part)
        try:
            with torch.no_grad():
                hL_norm = norm(out.hidden_states[-1])
                if hasattr(model, 'lm_head'):
                    logits_recomputed = model.lm_head(hL_norm)
                    diff = (logits_recomputed.float() - out.logits.float()).abs().max().item()
                    findings['final_norm_logit_recompute_max_abs_diff'] = float(diff)
        except Exception as e:
            findings['final_norm_logit_recompute_error'] = repr(e)

    findings['summary_ok'] = findings.get('forward_ok', False)
    return findings


if __name__ == '__main__':
    findings = main()
    OUT_JSON.write_text(json.dumps(findings, indent=2, default=str))
    print('SAVED', OUT_JSON, 'size', OUT_JSON.stat().st_size)
    print('===== SMOKE TEST KEY FINDINGS =====')
    keys = ['transformers_version','torch_version','cuda_device','model_class',
            'n_params_total','n_backbone_layers','backbone_attr_path',
            'config_hidden_size','config_vocab_size','config_n_layer',
            'tokenizer_vocab_size','tokenizer_bos_token_id','tokenizer_eos_token_id',
            'input_ids_shape','tokenizer_offset_input_ids_minus_seq_len',
            'lm_head_tied_id_match','lm_head_tied_dataptr_match',
            'final_norm_resolved_path','final_norm_logit_recompute_max_abs_diff',
            'forward_seconds','gpu_peak_GB','out_logits_shape',
            'out_hidden_states_count','hook_capture_count']
    for k in keys:
        if k in findings: print(f"  {k}: {findings[k]}")
