#!/usr/bin/env python3
"""Expand ONE hit -> ranked candidates, with a visual, inspectable report.

Given a single hit compound, PhenoCompass embeds it (structure tower) and ranks
every other compound in your library by similarity in the model's learned latent
space. The output answers two questions a reviewer/investor will ask:

  1. "What does it actually return for one hit?"  -> a molecule grid of the
     top-K candidates, each labelled with rank, model score, MoA, and Tanimoto.
  2. "Is that meaningful, or noise?"  -> a significance panel: are the hit's own
     mechanism (MoA) compounds enriched at the top (hypergeometric p-value,
     fold-enrichment, ranking AUROC), and are the top hits structurally NOVEL
     (low Tanimoto) rather than trivial look-alikes.

Outputs per hit (in ``--out-dir``):
    hit_<id>_candidates.csv     ranked table
    hit_<id>_molgrid.png        hit + top-K candidate structures (needs RDKit)
    hit_<id>_significance.png   rank vs score, same-MoA highlighted
    hit_<id>_report.html        self-contained one-pager

Usage
-----
    python a549_mvp/expand_hit.py \
        --final-model-dir ./final_model \
        --a549-compounds ./a549_compounds.csv \
        --hit auto:hdac --top-k 20 --out-dir ./a549_mvp_out/expansions
    # --hit can be: a SMILES, a compound id, "auto" (best-populated MoA),
    #   or "auto:<moa_cluster>"
"""

from __future__ import annotations

import argparse
import base64
import html
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import mvp_core as core  # noqa: E402
import moa_mapping as mm  # noqa: E402
import run_mvp  # noqa: E402  (reuse load_compounds / model loaders)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def pick_hit(cdf: pd.DataFrame, hit_arg: str):
    """Resolve --hit into (index, row)."""
    if hit_arg and hit_arg.startswith("auto"):
        want = hit_arg.split(":", 1)[1] if ":" in hit_arg else None
        labelled = cdf[cdf["moa_cluster"].notna()]
        if want:
            labelled = labelled[labelled["moa_cluster"] == want]
        if len(labelled) == 0:
            labelled = cdf[cdf["moa_cluster"].notna()]
        # choose a hit from the best-populated cluster so "expansion" is testable
        best = labelled["moa_cluster"].value_counts().idxmax()
        idx = labelled[labelled["moa_cluster"] == best].index[0]
        return idx, cdf.loc[idx]
    # exact SMILES or id
    m = cdf.index[(cdf["smiles"] == hit_arg) | (cdf["id"].astype(str) == str(hit_arg))]
    if len(m) == 0:
        raise SystemExit(f"--hit '{hit_arg}' not found in the library (SMILES or id).")
    return m[0], cdf.loc[m[0]]


def significance(same_moa: np.ndarray, sim: np.ndarray, top_k: int):
    """Hypergeometric enrichment of same-MoA in top-K + ranking AUROC."""
    from scipy.stats import hypergeom
    N = len(same_moa)
    n_same = int(same_moa.sum())
    order = np.argsort(-sim)
    topk_idx = order[:top_k]
    k_same = int(same_moa[topk_idx].sum())
    expected = top_k * n_same / N if N else np.nan
    fold = (k_same / expected) if expected and expected > 0 else np.nan
    # P(>= k_same same-MoA among top_k draws)
    pval = float(hypergeom.sf(k_same - 1, N, n_same, top_k)) if n_same and N else np.nan
    auroc = np.nan
    if 0 < n_same < N:
        try:
            from sklearn.metrics import roc_auc_score
            auroc = float(roc_auc_score(same_moa, sim))
        except Exception:
            pass
    return {"N": N, "n_same": n_same, "top_k": top_k, "k_same": k_same,
            "expected": expected, "fold_enrichment": fold, "p_value": pval,
            "auroc": auroc}


def draw_molgrid(hit_smiles, hit_label, cand_df, out_png, mock=False):
    """RDKit molecule grid of hit + top candidates.

    Returns (True, None) on success or (False, reason) so the caller can hint
    (e.g. molecule drawing needs Pillow: ``pip install pillow``).
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import Draw
    except Exception as e:  # noqa: BLE001
        return False, f"RDKit not importable ({e})"
    try:
        smis = [hit_smiles] + cand_df["smiles"].tolist()
        mols, legends = [], []
        for i, s in enumerate(smis):
            mol = Chem.MolFromSmiles(s)
            if mol is None:
                continue
            mols.append(mol)
            if i == 0:
                legends.append(f"{'[MOCK] ' if mock else ''}HIT: {hit_label}")
            else:
                r = cand_df.iloc[i - 1]
                same = "*SAME MoA*" if r["same_moa"] else (r.get("moa_cluster") or "?")
                legends.append(f"#{int(r['rank'])} sim={r['score']:.2f} | {same} | T={r['tanimoto']:.2f}")
        if not mols:
            return False, "no valid molecules"
        img = Draw.MolsToGridImage(mols, molsPerRow=4, subImgSize=(260, 200), legends=legends)
        img.save(out_png)
        return True, None
    except Exception as e:  # noqa: BLE001  (usually missing Pillow)
        return False, f"drawing failed ({e}); try: pip install pillow"


def plot_significance(sim, same_moa, sig, hit_moa, out_png, mock=False):
    order = np.argsort(-sim)
    ranks = np.arange(1, len(sim) + 1)
    s_sorted = sim[order]
    same_sorted = same_moa[order].astype(bool)
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    ax.scatter(ranks[~same_sorted], s_sorted[~same_sorted], s=10, c="#b7b7b7",
               alpha=0.5, label="other MoA")
    ax.scatter(ranks[same_sorted], s_sorted[same_sorted], s=26, c="#c1432c",
               edgecolor="black", linewidth=0.3, label=f"same MoA ({hit_moa})")
    ax.axvline(sig["top_k"], ls="--", c="#2c6fbb", lw=1, label=f"top-{sig['top_k']} cut")
    ax.set_xlabel("candidate rank (by model score)")
    ax.set_ylabel("model similarity to hit")
    p = sig["p_value"]
    ax.set_title(f"Top-{sig['top_k']}: {sig['k_same']}/{sig['top_k']} same-MoA "
                 f"({sig['fold_enrichment']:.1f}x, p={p:.1e}) | AUROC={sig['auroc']:.2f}")
    ax.legend(fontsize=8, loc="upper right")
    if mock:
        fig.text(0.5, 0.5, "SYNTHETIC / MOCK", fontsize=30, color="red", alpha=0.18,
                 ha="center", va="center", rotation=25, weight="bold")
    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_tanimoto_heatmap(labels, matrix, out_png, mock=False):
    """Pairwise ECFP Tanimoto heatmap for [hit + top candidates].

    Blue/low off-diagonal = structurally diverse (independent chemotypes).
    """
    import numpy as np
    M = np.asarray(matrix, dtype=float)
    n = len(labels)
    fig, ax = plt.subplots(figsize=(max(5.0, 0.5 * n + 1.5), max(4.2, 0.5 * n + 1.2)))
    im = ax.imshow(M, cmap="magma_r", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    for i in range(n):
        for j in range(n):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if M[i, j] > 0.5 else "black")
    ax.set_title("Pairwise structural similarity (ECFP Tanimoto)\nlow = structurally independent", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Tanimoto")
    if mock:
        fig.text(0.5, 0.5, "MOCK", fontsize=30, color="red", alpha=0.15, ha="center", va="center", rotation=25, weight="bold")
    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _b64(path):
    if path and os.path.isfile(path):
        with open(path, "rb") as f:
            return "data:image/png;base64," + base64.b64encode(f.read()).decode()
    return None


def write_html(hit, hit_moa, sig, molgrid_png, sig_png, tani_png, cand_df, out_html, mock):
    novel = cand_df[(cand_df["same_moa"]) & (cand_df["tanimoto"] < 0.4)]
    banner = ('<div style="background:#c1432c;color:#fff;padding:6px 12px;border-radius:6px;'
              'display:inline-block;font-weight:bold">SYNTHETIC / MOCK — not real model output</div>'
              if mock else "")
    p = sig["p_value"]
    rows = "".join(
        f"<tr><td>{int(r['rank'])}</td><td style='font-family:monospace;font-size:11px'>{html.escape(str(r['id']))}</td>"
        f"<td>{r['score']:.3f}</td><td>{'✓ ' if r['same_moa'] else ''}{html.escape(str(r.get('moa_cluster') or '—'))}</td>"
        f"<td>{r['tanimoto']:.2f}</td></tr>"
        for _, r in cand_df.iterrows()
    )
    grid_img = _b64(molgrid_png)
    grid_html = (f'<img src="{grid_img}" style="max-width:100%">' if grid_img
                 else '<p style="color:#888">[molecule grid needs RDKit — rendered on your run]</p>')
    sig_img = _b64(sig_png)
    doc = f"""<!doctype html><html><head><meta charset="utf-8">
<title>PhenoBridge expansion — {html.escape(str(hit['id']))}</title>
<style>body{{font-family:-apple-system,Arial,sans-serif;max-width:920px;margin:24px auto;color:#222}}
h1{{font-size:20px}} .k{{color:#2c6fbb;font-weight:bold}} table{{border-collapse:collapse;width:100%;font-size:13px}}
td,th{{border:1px solid #ddd;padding:4px 8px;text-align:left}} th{{background:#f4f6f9}}
.card{{border:1px solid #e2e2e2;border-radius:10px;padding:16px;margin:14px 0}}</style></head><body>
{banner}
<h1>Expand one hit → candidate series</h1>
<div class="card"><b>Input hit:</b> <code>{html.escape(str(hit['id']))}</code><br>
<b>Mechanism (MoA):</b> {html.escape(str(hit_moa))}<br>
<b>SMILES:</b> <code style="font-size:11px">{html.escape(str(hit['smiles']))}</code></div>

<div class="card"><b>Is the result meaningful?</b><br>
Among <span class="k">{sig['N']}</span> library compounds, PhenoCompass ranked them by similarity to this hit.
The top <span class="k">{sig['top_k']}</span> contain <span class="k">{sig['k_same']}</span> compounds of the
hit's own mechanism — vs <b>{sig['expected']:.1f}</b> expected by chance →
<span class="k">{sig['fold_enrichment']:.1f}× enrichment</span>, hypergeometric <span class="k">p = {p:.1e}</span>.
Ranking AUROC = <span class="k">{sig['auroc']:.2f}</span>.<br>
Structurally novel same-MoA hits (Tanimoto &lt; 0.4): <span class="k">{len(novel)}</span> — i.e. same mechanism,
independent chemotype.</div>

<div class="card"><b>What it returned (top {len(cand_df)}):</b><br>{grid_html}</div>
<div class="card"><b>Why it's not noise:</b><br>
{'<img src="'+sig_img+'" style="max-width:100%">' if sig_img else ''}</div>

<div class="card"><b>Are the candidates structurally distinct?</b> (pairwise ECFP Tanimoto — low = independent scaffolds)<br>
{'<img src="'+_b64(tani_png)+'" style="max-width:100%">' if _b64(tani_png) else '<p style="color:#888">[Tanimoto heatmap needs RDKit — rendered on your run]</p>'}</div>

<div class="card"><b>Ranked candidates</b>
<table><tr><th>rank</th><th>id</th><th>model score</th><th>MoA</th><th>Tanimoto to hit</th></tr>{rows}</table></div>
</body></html>"""
    with open(out_html, "w") as f:
        f.write(doc)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--final-model-dir")
    ap.add_argument("--a549-compounds")
    ap.add_argument("--hit", default="auto", help="SMILES, id, 'auto', or 'auto:<moa>'")
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--out-dir", default="./a549_mvp_out/expansions")
    ap.add_argument("--anchor-modality", default="joint", choices=["joint", "struct", "morph"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--synthetic-compounds", action="store_true")
    args = ap.parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)

    # compounds
    if args.synthetic_compounds or (args.mock and not args.a549_compounds):
        cdf = run_mvp.make_synthetic_compounds()
    else:
        cdf = run_mvp.load_compounds(args.a549_compounds)
    cdf = cdf.reset_index(drop=True)

    idx, hit = pick_hit(cdf, args.hit)
    hit_moa = hit["moa_cluster"] if hit["moa_cluster"] else "(unmapped)"
    print(f"[hit] {hit['id']} | MoA={hit_moa}")

    # model + embeddings
    if args.mock:
        from mock_model import MockPhenoCompass
        pc = MockPhenoCompass(mm.ANCHOR_MOAS)
        pc.prime_true_moa(cdf["smiles"].tolist(), list(cdf["moa_cluster"]))
    else:
        if not args.final_model_dir:
            raise SystemExit("--final-model-dir required (or --mock).")
        pc = run_mvp.load_real_phenocompass(args.final_model_dir, args.anchor_modality, args.device, False)

    latent = core.ensemble_average(pc.compute_struct_embeddings(cdf["smiles"].tolist()))
    sim = core.rank_by_similarity(latent[idx], latent)
    sim[idx] = -np.inf  # exclude the hit itself

    same_moa = (cdf["moa_cluster"] == hit_moa).to_numpy() & (np.arange(len(cdf)) != idx)
    tani = core.tanimoto_to_one(hit["smiles"], cdf["smiles"].tolist())

    ranked = cdf.copy()
    ranked["score"] = sim
    ranked["same_moa"] = same_moa
    ranked["tanimoto"] = tani
    ranked = ranked[ranked.index != idx].sort_values("score", ascending=False).reset_index(drop=True)
    ranked.insert(0, "rank", np.arange(1, len(ranked) + 1))

    sig = significance(same_moa[np.arange(len(cdf)) != idx],
                       sim[np.arange(len(cdf)) != idx], args.top_k)
    print(f"[significance] top-{args.top_k}: {sig['k_same']}/{sig['top_k']} same-MoA | "
          f"{sig['fold_enrichment']:.1f}x | p={sig['p_value']:.2e} | AUROC={sig['auroc']:.2f}")

    hid = "".join(ch if ch.isalnum() else "_" for ch in str(hit["id"]))[:40]
    top = ranked.head(args.top_k)
    top.to_csv(os.path.join(args.out_dir, f"hit_{hid}_candidates.csv"), index=False)

    molgrid = os.path.join(args.out_dir, f"hit_{hid}_molgrid.png")
    drew, reason = draw_molgrid(hit["smiles"], f"{hit['id']} ({hit_moa})", top, molgrid, mock=args.mock)
    if not drew:
        molgrid = None
        print(f"[molgrid] skipped structure grid: {reason}")

    sig_png = os.path.join(args.out_dir, f"hit_{hid}_significance.png")
    plot_significance(sim[np.arange(len(cdf)) != idx],
                      same_moa[np.arange(len(cdf)) != idx], sig, hit_moa, sig_png, mock=args.mock)

    # pairwise structural-diversity heatmap for hit + top candidates
    tani_png = os.path.join(args.out_dir, f"hit_{hid}_tanimoto_heatmap.png")
    labels = ["HIT"] + [f"#{int(r)}" for r in top["rank"]]
    _pwt = getattr(core, "pairwise_tanimoto_matrix", None)
    if _pwt is None:
        print("[diversity] mvp_core is stale (no pairwise_tanimoto_matrix) — re-sync a549_mvp/mvp_core.py")
    tmat = _pwt([hit["smiles"]] + top["smiles"].tolist()) if _pwt else None
    if tmat is not None:
        plot_tanimoto_heatmap(labels, tmat, tani_png, mock=args.mock)
        offdiag = tmat[np.triu_indices(len(labels), k=1)]
        med = float(np.nanmedian(offdiag))
        print(f"[diversity] median pairwise Tanimoto among hit+top: {med:.2f} "
              f"({'diverse' if med < 0.4 else 'similar'} scaffolds)")
    else:
        tani_png = None
        print("[diversity] RDKit unavailable — skipped Tanimoto heatmap (renders on your server)")

    html_path = os.path.join(args.out_dir, f"hit_{hid}_report.html")
    write_html(hit, hit_moa, sig, molgrid, sig_png, tani_png, top, html_path, args.mock)
    print(f"[report] {html_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
