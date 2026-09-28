"""
Prediction script for JEPA encoder-based GO term prediction.

Generates predictions in exact Kaggle submission format:
- No header
- One line per (protein, GO_term, score) triplet
- Score with 3 decimal places
- ALL GO terms for ALL proteins (no threshold filtering)

Usage:
    python -m src.jepa_go.predict_jepa_encoder \
        --model_dir models/jepa-go \
        --test_fasta data/Test/testsuperset.fasta \
        --output submission.tsv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.common.fasta import read_fasta
from src.jepa_go.dataset import InferenceDataset
from src.jepa_go.models import load_jepa_go_model


@torch.inference_mode()
def predict(
    model_dir: Path,
    test_fasta: Path,
    output_path: Path,
    batch_size: int = 16,
    dtype: str = "bfloat16",
    device: str = None,
    min_score: float = 0.001,
) -> None:
    """
    Generate predictions for test sequences.
    
    Outputs in exact Kaggle submission format:
    - No header line
    - Format: protein_id<TAB>GO:XXXXXXX<TAB>0.XXX
    - Score with 3 decimal places
    - All predictions with score >= min_score (to avoid outputting zeros)
    
    Args:
        model_dir: Directory containing trained model
        test_fasta: Path to test FASTA file
        output_path: Path to output TSV file
        batch_size: Batch size for inference
        dtype: Model dtype
        device: Device to use
        min_score: Minimum score to output (scores of 0 are not allowed per rules)
    """
    # Device setup
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    
    # Dtype
    if dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    elif dtype == "float16":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32
    
    print(f"Using device: {device}, dtype: {dtype}")
    
    # Load model
    model_dir = Path(model_dir)
    print(f"Loading model from {model_dir}...")
    model, tokenizer, config = load_jepa_go_model(model_dir, device, torch_dtype)
    model.eval()
    
    # Load label mapping
    with open(model_dir / "labels.json") as f:
        labels_data = json.load(f)
    label_list = labels_data["label_list"]
    num_labels = len(label_list)
    print(f"Model has {num_labels} labels")
    
    # Load test sequences
    print(f"Loading test sequences from {test_fasta}...")
    test_sequences = dict(read_fasta(test_fasta))
    num_proteins = len(test_sequences)
    print(f"Loaded {num_proteins} test sequences")
    
    # Create dataset
    dataset = InferenceDataset(
        sequences=test_sequences,
        tokenizer=tokenizer,
        max_length=config.get("max_length", 1024),
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    
    # Prepare output file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    use_amp = torch_dtype != torch.float32
    total_predictions = 0
    
    print(f"Writing predictions to {output_path}...")
    print(f"Output format: protein_id<TAB>GO:term<TAB>score (3 decimal places)")
    print(f"Minimum score threshold: {min_score} (to exclude zeros)")
    
    # Write predictions directly to file (no header per Kaggle format)
    with open(output_path, "w") as f:
        for batch in tqdm(dataloader, desc="Predicting"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            protein_ids = batch["protein_id"]
            current_batch_size = len(protein_ids)
            
            with torch.autocast(device_type="cuda", enabled=use_amp, dtype=torch_dtype):
                logits = model(input_ids, attention_mask)
            
            # Batched sigmoid and move to CPU
            probs = torch.sigmoid(logits).cpu()
            
            # Process each sample in batch
            for i in range(current_batch_size):
                pid = protein_ids[i]
                protein_probs = probs[i]
                
                # Get all predictions above minimum score
                # Scores of exactly 0 are not allowed per Kaggle rules
                mask = protein_probs >= min_score
                indices = mask.nonzero(as_tuple=True)[0]
                
                if len(indices) == 0:
                    # If all scores are below min_score, output top 1 prediction
                    # (every protein must have at least some prediction)
                    top_idx = protein_probs.argmax().item()
                    score = max(float(protein_probs[top_idx]), min_score)
                    f.write(f"{pid}\t{label_list[top_idx]}\t{score:.3f}\n")
                    total_predictions += 1
                else:
                    # Output all predictions above min_score
                    selected_probs = protein_probs[indices]
                    
                    # Sort by probability descending
                    sorted_order = selected_probs.argsort(descending=True)
                    sorted_indices = indices[sorted_order]
                    sorted_probs = selected_probs[sorted_order]
                    
                    # Write each prediction
                    for idx, prob in zip(sorted_indices.tolist(), sorted_probs.tolist()):
                        f.write(f"{pid}\t{label_list[idx]}\t{prob:.3f}\n")
                        total_predictions += 1
    
    print(f"\nPrediction complete!")
    print(f"Total predictions written: {total_predictions:,}")
    print(f"Average predictions per protein: {total_predictions / num_proteins:.1f}")
    print(f"Saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate GO term predictions in Kaggle submission format"
    )
    
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Directory containing trained model")
    parser.add_argument("--test_fasta", type=str, default="data/Test/testsuperset.fasta",
                        help="Path to test FASTA file")
    parser.add_argument("--output", type=str, default="submission.tsv",
                        help="Output submission file")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for inference")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (default: auto)")
    parser.add_argument("--min_score", type=float, default=0.001,
                        help="Minimum score to output (scores of 0 not allowed)")
    
    args = parser.parse_args()
    
    predict(
        model_dir=Path(args.model_dir),
        test_fasta=Path(args.test_fasta),
        output_path=Path(args.output),
        batch_size=args.batch_size,
        dtype=args.dtype,
        device=args.device,
        min_score=args.min_score,
    )


if __name__ == "__main__":
    main()
