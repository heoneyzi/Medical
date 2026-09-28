from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm


ROOTS = {"GO:0003674", "GO:0008150", "GO:0005575"}


def iter_fasta_ids(path: Path) -> set[str]:
    ids: set[str] = set()
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith(">"):
                entry = line[1:].strip().split()[0]
                ids.add(entry)
    return ids


def parse_obo_parents(obo_path: Path) -> dict[str, set[str]]:
    term_parents: dict[str, set[str]] = defaultdict(set)
    with obo_path.open("r", encoding="utf-8", errors="replace") as f:
        cur_id: str | None = None
        for raw in f:
            line = raw.strip()
            if line.startswith("id: "):
                cur_id = line.split("id: ", 1)[1].strip()
            elif line.startswith("is_a: ") and cur_id:
                term_parents[cur_id].add(line.split()[1].strip())
            elif line.startswith("relationship: part_of ") and cur_id:
                term_parents[cur_id].add(line.split()[2].strip())
    return term_parents


def compute_ancestor_closure(term_parents: dict[str, set[str]]) -> dict[str, set[str]]:
    ancestors_map: dict[str, set[str]] = {}

    def get_ancestors(term: str) -> set[str]:
        cached = ancestors_map.get(term)
        if cached is not None:
            return cached
        parents = term_parents.get(term, set())
        all_anc = set(parents)
        for p in parents:
            all_anc |= get_ancestors(p)
        ancestors_map[term] = all_anc
        return all_anc

    for term in tqdm(list(term_parents.keys()), desc="Computing ancestor closure"):
        get_ancestors(term)

    return ancestors_map


def load_predictions(filepath: Path, allowed_ids: set[str]) -> dict[str, dict[str, float]]:
    data: dict[str, dict[str, float]] = defaultdict(dict)
    with filepath.open("r", encoding="utf-8", errors="replace") as f:
        for line in tqdm(f, desc=f"Loading {filepath.name}"):
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            pid, go, score = parts[0], parts[1], float(parts[2])
            if pid not in allowed_ids:
                continue
            prev = data[pid].get(go)
            if prev is None or score > prev:
                data[pid][go] = score
    return data


def ensemble_predictions(
    goa_preds: dict[str, dict[str, float]],
    prott5_preds: dict[str, dict[str, float]],
    weight_goa: float,
    weight_prott5: float,
) -> dict[str, dict[str, float]]:
    all_proteins = set(goa_preds.keys()) | set(prott5_preds.keys())
    ensemble: dict[str, dict[str, float]] = defaultdict(dict)

    for pid in tqdm(all_proteins, desc="Creating ensemble"):
        goa = goa_preds.get(pid, {})
        prott5 = prott5_preds.get(pid, {})
        all_terms = set(goa.keys()) | set(prott5.keys())

        for term in all_terms:
            s_goa = goa.get(term, 0.0)
            s_prott5 = prott5.get(term, 0.0)
            if s_goa > 0 and s_prott5 > 0:
                ensemble[pid][term] = weight_goa * s_goa + weight_prott5 * s_prott5
            elif s_goa > 0:
                ensemble[pid][term] = s_goa
            else:
                ensemble[pid][term] = s_prott5

    return ensemble


def process_protein(
    scores_dict: dict[str, float],
    ancestors_map: dict[str, set[str]],
    neg_prop_alpha: float,
    scaling_power: float,
    max_score: float,
) -> dict[str, float]:
    updated = dict(scores_dict)

    def get_ancestors(term: str) -> set[str]:
        return ancestors_map.get(term, set())

    # Positive propagation: ensure parent >= child
    for term, score in scores_dict.items():
        for anc in get_ancestors(term):
            prev = updated.get(anc)
            if prev is None or score > prev:
                updated[anc] = score

    # Negative propagation: constrain child by parent
    for term in list(updated.keys()):
        if term in ROOTS:
            continue
        ancs = get_ancestors(term)
        if not ancs:
            continue
        anc_scores = [updated[a] for a in ancs if a in updated]
        if not anc_scores:
            continue
        min_anc = min(anc_scores)
        if min_anc < updated[term]:
            updated[term] = neg_prop_alpha * min_anc + (1 - neg_prop_alpha) * updated[term]

    # Power scaling
    non_root = [s for t, s in updated.items() if t not in ROOTS]
    if non_root:
        current_max = max(non_root)
        if 0 < current_max < max_score:
            for t in list(updated.keys()):
                if t in ROOTS:
                    continue
                updated[t] = min(1.0, float(np.power(updated[t] / current_max, scaling_power) * max_score))

    for r in ROOTS:
        updated[r] = 1.0

    return updated


def main() -> None:
    parser = argparse.ArgumentParser(description="Create CAFA-6 submission by ensembling GOA + ProtT5 predictions with GO propagation.")
    parser.add_argument("--competition-data", type=Path, default=Path("data"), help="Path containing Train/ and Test/ folders.")
    parser.add_argument("--prediction-data", type=Path, default=Path("data/kaggle"), help="Path containing goa_submission.tsv and prott5_interpro_predictions.tsv.")
    parser.add_argument("--out", type=Path, default=Path("submissions/submission.tsv"), help="Output submission TSV path.")
    parser.add_argument("--weight-goa", type=float, default=0.55)
    parser.add_argument("--weight-prott5", type=float, default=0.45)
    parser.add_argument("--top-k", type=int, default=200)
    parser.add_argument("--min-score", type=float, default=0.001)
    parser.add_argument("--neg-prop-alpha", type=float, default=0.7)
    parser.add_argument("--scaling-power", type=float, default=0.8)
    parser.add_argument("--max-score", type=float, default=0.95)
    args = parser.parse_args()

    obo_path = args.competition_data / "Train" / "go-basic.obo"
    test_fasta = args.competition_data / "Test" / "testsuperset.fasta"
    goa_path = args.prediction_data / "goa_submission.tsv"
    prott5_path = args.prediction_data / "prott5_interpro_predictions.tsv"

    for p in [obo_path, test_fasta, goa_path, prott5_path]:
        if not p.exists():
            raise FileNotFoundError(str(p))

    test_ids = iter_fasta_ids(test_fasta)
    term_parents = parse_obo_parents(obo_path)
    ancestors_map = compute_ancestor_closure(term_parents)

    goa_preds = load_predictions(goa_path, test_ids)
    prott5_preds = load_predictions(prott5_path, test_ids)
    ens = ensemble_predictions(goa_preds, prott5_preds, args.weight_goa, args.weight_prott5)

    final_rows: list[str] = []
    for pid, scores in tqdm(ens.items(), desc="Applying propagation"):
        updated = process_protein(
            scores,
            ancestors_map=ancestors_map,
            neg_prop_alpha=args.neg_prop_alpha,
            scaling_power=args.scaling_power,
            max_score=args.max_score,
        )
        for go, s in sorted(updated.items(), key=lambda x: -x[1])[: args.top_k]:
            if s >= args.min_score:
                final_rows.append(f"{pid}\t{go}\t{s:.6f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(final_rows) + "\n", encoding="utf-8")

    size_mb = args.out.stat().st_size / (1024 * 1024)
    print(f"Wrote: {args.out} ({size_mb:.1f} MB) rows={len(final_rows):,} proteins={len(ens):,}")


if __name__ == "__main__":
    main()

