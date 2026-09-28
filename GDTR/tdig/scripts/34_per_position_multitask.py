"""Exp H4 — Per-position multi-task functional prediction from h_ell.

For each layer, train a 6-way softmax classifier predicting context label from
h_ell. Reports per-class F1 and per-layer aggregate accuracy. Identifies which
layer's representation best encodes "what context is this position".

Outputs: results/multitask_per_position/
  per_layer_metrics.csv         (layer, class_id, class_name, precision, recall, f1, AUROC_ovr)
  per_layer_metrics.png         multi-line plot per class
  best_layer_per_class.json     summary
"""

from __future__ import annotations
import argparse, json
from pathlib import Path
import h5py, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

L_STAR = 29
POS_CODEBOOK = {0: "intergenic", 1: "intron", 2: "coding_exon", 3: "5utr",
                4: "3utr", 5: "splice_donor", 6: "splice_acceptor"}


def build_context_map(meta_df, wids, pos_labels, n_tokens=6000):
    ctx = np.zeros((len(wids), n_tokens), dtype=np.uint8)
    for i, wid in enumerate(wids):
        row = meta_df[meta_df["window_idx"] == int(wid)].iloc[0]
        start = int(row["start"])
        pos = np.clip(start + np.arange(n_tokens), 0, len(pos_labels) - 1)
        ctx[i] = pos_labels[pos]
    return ctx


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tier3", type=Path,
                   default=Path("/path/to/TDiG/data/cache/chr22_v2/tier3_raw_v2.h5"))
    p.add_argument("--meta", type=Path,
                   default=Path("/path/to/TDiG/data/cache/chr22_v2/window_metadata.parquet"))
    p.add_argument("--pos-labels", type=Path,
                   default=Path("/path/to/gDTR/data/annotation/chr22_position_labels.npy"))
    p.add_argument("--out-dir", type=Path,
                   default=Path("/path/to/TDiG/data/cache/_v2_analysis/multitask_per_position"))
    p.add_argument("--max-per-class", type=int, default=3000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(context="paper", style="whitegrid", font_scale=0.95)

    print("[load] tier3 + ctx ...")
    meta = pd.read_parquet(args.meta)
    pos_labels = np.load(args.pos_labels)
    with h5py.File(args.tier3, "r") as h5:
        raw_rms = h5["raw_h_ell_rmsnormed"][:].astype(np.float32)
        wids = h5["window_idx"][:]
    print(f"  raw_rms {raw_rms.shape}")
    ctx_full = build_context_map(meta, wids, pos_labels)
    ctx = ctx_full[:, ::10][:, :raw_rms.shape[2]]
    n_w, n_l, n_t, H = raw_rms.shape
    flat_ctx = ctx.flatten()
    rng = np.random.default_rng(args.seed)

    # Balanced subsample per class
    print("[sample] balanced per class ...")
    take_idx = []
    take_y = []
    for ctx_id, name in POS_CODEBOOK.items():
        mask = flat_ctx == ctx_id
        if mask.sum() < 30:
            continue
        idx = np.where(mask)[0]
        rng.shuffle(idx)
        n = min(args.max_per_class, len(idx))
        take_idx.extend(idx[:n].tolist())
        take_y.extend([ctx_id] * n)
        print(f"  {name}: {n}")
    take_idx = np.array(take_idx); take_y = np.array(take_y)
    wi = take_idx // n_t; ti = take_idx % n_t

    records = []
    print(f"\n[fit] per-layer 6-way classifier ...")
    for ell in range(n_l):
        X = raw_rms[wi, ell, ti, :]
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, take_y, test_size=0.3, stratify=take_y, random_state=args.seed)
        clf = LogisticRegression(max_iter=300, C=1.0, n_jobs=-1, solver="lbfgs",
                                  multi_class="multinomial")
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_te)
        y_prob = clf.predict_proba(X_te)
        for ctx_id, name in POS_CODEBOOK.items():
            if ctx_id not in clf.classes_:
                continue
            cls_idx = list(clf.classes_).index(ctx_id)
            y_te_bin = (y_te == ctx_id).astype(int)
            f1 = f1_score(y_te_bin, (y_pred == ctx_id).astype(int))
            try:
                auroc = roc_auc_score(y_te_bin, y_prob[:, cls_idx])
            except Exception:
                auroc = np.nan
            records.append({"layer": ell, "class_id": ctx_id, "class_name": name,
                              "f1": float(f1), "AUROC_ovr": float(auroc)})
        acc = float((y_pred == y_te).mean())
        print(f"  L={ell:2d}  acc={acc:.3f}")

    df = pd.DataFrame(records)
    df.to_csv(args.out_dir / "per_layer_metrics.csv", index=False)

    # Best layer per class
    best = {}
    for ctx_id, name in POS_CODEBOOK.items():
        sub = df[df.class_id == ctx_id]
        if sub.empty:
            continue
        b = sub.loc[sub.AUROC_ovr.idxmax()]
        best[name] = {"best_layer": int(b.layer), "AUROC_ovr": float(b.AUROC_ovr),
                       "f1_at_best": float(b.f1)}
    (args.out_dir / "best_layer_per_class.json").write_text(json.dumps(best, indent=2))
    print(json.dumps(best, indent=2))

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    for ctx_id, name in POS_CODEBOOK.items():
        sub = df[df.class_id == ctx_id]
        if sub.empty:
            continue
        axes[0].plot(sub.layer, sub.AUROC_ovr, marker="o", ms=3, lw=1.5, label=name)
        axes[1].plot(sub.layer, sub.f1, marker="o", ms=3, lw=1.5, label=name)
    for ax, title, ylab in zip(axes, ("AUROC (OVR)", "F1 score"),
                                 ("AUROC", "F1")):
        ax.axvline(L_STAR, color="k", ls="--", lw=0.6, alpha=0.4)
        ax.set_xlabel("Layer ℓ"); ax.set_ylabel(ylab); ax.set_title(f"(H4) Per-layer per-class {title}")
        ax.legend(loc="best", fontsize=8, ncol=2)
    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(args.out_dir / f"per_layer_metrics.{ext}", dpi=150)
    plt.close(fig)
    print("[done]")


if __name__ == "__main__":
    main()
