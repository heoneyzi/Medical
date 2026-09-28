from __future__ import annotations

import argparse
from pathlib import Path

import torch

from ..common.fasta import read_fasta
from ..common.go_obo import (
    cap_top_terms,
    format_score_3sig,
    merge_score_dicts,
    parse_go_parents,
    propagate_scores_to_parents,
)
from ..common.protein import prott5_tokenize_sequence
from ..train.prott5_go_train import load_trained_model


@torch.inference_mode()
def predict(
    *,
    model_dir: str | Path,
    test_fasta: str | Path,
    out_tsv: str | Path,
    batch_size: int = 1,
    max_length: int | None = None,
    top_k: int = 1500,
    min_score: float = 0.001,
    max_terms_per_protein: int = 1500,
    obo_path: str | Path | None = None,
    propagate: bool = False,
) -> Path:
    model, tokenizer, labels, model_cfg = load_trained_model(model_dir)
    device = next(model.parameters()).device

    if max_length is None:
        max_length = int(model_cfg.get("max_length", 1024))

    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id

    out_tsv = Path(out_tsv)
    out_tsv.parent.mkdir(parents=True, exist_ok=True)

    term_parents = None
    if propagate:
        if obo_path is None:
            raise ValueError("propagate=True requires obo_path (e.g., data/Train/go-basic.obo).")
        term_parents = parse_go_parents(obo_path)

    batch_ids: list[str] = []
    batch_seqs: list[str] = []

    def flush() -> list[tuple[str, dict[str, float]]]:
        if not batch_ids:
            return []

        tokenized = tokenizer(
            [prott5_tokenize_sequence(s) for s in batch_seqs],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        input_ids = tokenized["input_ids"].to(device)
        attention_mask = tokenized["attention_mask"].to(device)
        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pad_token_id=pad_id,
            eos_token_id=eos_id,
        )
        probs = torch.sigmoid(logits).detach().cpu()

        results: list[tuple[str, dict[str, float]]] = []
        for pid, row in zip(batch_ids, probs):
            k = min(top_k, row.numel())
            values, indices = torch.topk(row, k=k, largest=True, sorted=True)
            picked: dict[str, float] = {}
            for i, v in zip(indices.tolist(), values.tolist()):
                score = float(v)
                if score >= min_score and score > 0:
                    picked[labels[int(i)]] = score
            results.append((pid, picked))

        batch_ids.clear()
        batch_seqs.clear()
        return results

    with out_tsv.open("w", encoding="utf-8") as out:
        for pid, seq in read_fasta(test_fasta):
            batch_ids.append(pid)
            batch_seqs.append(seq)
            if len(batch_ids) >= batch_size:
                for protein_id, scores in flush():
                    if term_parents is not None:
                        scores = propagate_scores_to_parents(scores, term_parents=term_parents)
                    for go_term, score in cap_top_terms(
                        scores, max_terms=max_terms_per_protein, min_score=min_score
                    ):
                        score_str = format_score_3sig(score)
                        if float(score_str) <= 0.0:
                            continue
                        out.write(f"{protein_id}\t{go_term}\t{score_str}\n")

        for protein_id, scores in flush():
            if term_parents is not None:
                scores = propagate_scores_to_parents(scores, term_parents=term_parents)
            for go_term, score in cap_top_terms(
                scores, max_terms=max_terms_per_protein, min_score=min_score
            ):
                score_str = format_score_3sig(score)
                if float(score_str) <= 0.0:
                    continue
                out.write(f"{protein_id}\t{go_term}\t{score_str}\n")

    return out_tsv


@torch.inference_mode()
def predict_multi(
    *,
    model_dirs: list[str | Path],
    test_fasta: str | Path,
    out_tsv: str | Path,
    batch_size: int = 1,
    max_length: int | None = None,
    top_k: int = 1500,
    min_score: float = 0.001,
    max_terms_per_protein: int = 1500,
    obo_path: str | Path | None = None,
    propagate: bool = False,
) -> Path:
    if not model_dirs:
        raise ValueError("model_dirs must be non-empty.")

    term_parents = None
    if propagate:
        if obo_path is None:
            raise ValueError("propagate=True requires obo_path (e.g., data/Train/go-basic.obo).")
        term_parents = parse_go_parents(obo_path)

    out_tsv = Path(out_tsv)
    out_tsv.parent.mkdir(parents=True, exist_ok=True)

    models = [load_trained_model(d) for d in model_dirs]

    if max_length is None:
        max_length = int(models[0][3].get("max_length", 1024))

    batch_ids: list[str] = []
    batch_seqs: list[str] = []

    def flush_batch() -> list[tuple[str, dict[str, float]]]:
        if not batch_ids:
            return []

        merged_per_protein: list[dict[str, float]] = [dict() for _ in batch_ids]

        for model, tokenizer, labels, model_cfg in models:
            device = next(model.parameters()).device
            tokenized = tokenizer(
                [prott5_tokenize_sequence(s) for s in batch_seqs],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            input_ids = tokenized["input_ids"].to(device)
            attention_mask = tokenized["attention_mask"].to(device)
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            probs = torch.sigmoid(logits).detach().cpu()

            for row_idx, row in enumerate(probs):
                k = min(top_k, row.numel())
                values, indices = torch.topk(row, k=k, largest=True, sorted=True)
                merged = merged_per_protein[row_idx]
                for i, v in zip(indices.tolist(), values.tolist()):
                    score = float(v)
                    if score < min_score or score <= 0:
                        continue
                    term = labels[int(i)]
                    prev = merged.get(term)
                    if prev is None or score > prev:
                        merged[term] = score

        out_rows = list(zip(batch_ids, merged_per_protein))
        batch_ids.clear()
        batch_seqs.clear()
        return out_rows

    with out_tsv.open("w", encoding="utf-8") as out:
        for protein_id, seq in read_fasta(test_fasta):
            batch_ids.append(protein_id)
            batch_seqs.append(seq)
            if len(batch_ids) >= batch_size:
                for pid, merged in flush_batch():
                    if term_parents is not None:
                        merged = propagate_scores_to_parents(merged, term_parents=term_parents)
                    for go_term, score in cap_top_terms(
                        merged, max_terms=max_terms_per_protein, min_score=min_score
                    ):
                        score_str = format_score_3sig(score)
                        if float(score_str) <= 0.0:
                            continue
                        out.write(f"{pid}\t{go_term}\t{score_str}\n")

        for pid, merged in flush_batch():
            if term_parents is not None:
                merged = propagate_scores_to_parents(merged, term_parents=term_parents)
            for go_term, score in cap_top_terms(
                merged, max_terms=max_terms_per_protein, min_score=min_score
            ):
                score_str = format_score_3sig(score)
                if float(score_str) <= 0.0:
                    continue
                out.write(f"{pid}\t{go_term}\t{score_str}\n")

    return out_tsv


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Predict GO terms with a trained ProtT5 classifier.")
    p.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        action="append",
        help="One or more model directories. Repeat flag to combine multiple models.",
    )
    p.add_argument("--test-fasta", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--max-length", type=int, default=None)
    p.add_argument("--top-k", type=int, default=1500, help="Pre-cap top-k per model before propagation.")
    p.add_argument("--min-score", type=float, default=0.001)
    p.add_argument("--max-terms-per-protein", type=int, default=1500)
    p.add_argument("--obo", type=Path, default=None, help="Path to go-basic.obo for ontology propagation.")
    p.add_argument(
        "--propagate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Propagate scores to GO parents using max(child) semantics.",
    )
    args = p.parse_args(argv)

    if len(args.model_dir) == 1:
        out = predict(
            model_dir=args.model_dir[0],
            test_fasta=args.test_fasta,
            out_tsv=args.out,
            batch_size=args.batch_size,
            max_length=args.max_length,
            top_k=args.top_k,
            min_score=args.min_score,
            max_terms_per_protein=args.max_terms_per_protein,
            obo_path=args.obo,
            propagate=args.propagate,
        )
    else:
        out = predict_multi(
            model_dirs=args.model_dir,
            test_fasta=args.test_fasta,
            out_tsv=args.out,
            batch_size=args.batch_size,
            max_length=args.max_length,
            top_k=args.top_k,
            min_score=args.min_score,
            max_terms_per_protein=args.max_terms_per_protein,
            obo_path=args.obo,
            propagate=args.propagate,
        )
    print(f"wrote={out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
