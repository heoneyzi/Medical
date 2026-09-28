from __future__ import annotations

import argparse
import copy
import random
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from geoflowagent.constants import CHECKPOINT_VERSION, FEATURE_SPEC_VERSION
from geoflowagent.models.functional import (
    FunctionalGeometryModel,
    regret_ranking_loss,
    set_valued_nll,
)
from geoflowagent.training.features import FeatureStore
from geoflowagent.utils.io import canonical_json, read_yaml, sha256_file, sha256_text, write_json
from geoflowagent.utils.reproducibility import (
    choose_device,
    collect_run_provenance,
    seed_everything,
)


@torch.inference_mode()
def evaluate_metric_model(
    model: FunctionalGeometryModel,
    store: FeatureStore,
    split: str,
    device: torch.device,
    *,
    apply_contract_mask: bool,
) -> dict[str, float | int | None]:
    indices = store.indices(split, include_terminal=False)
    if not indices:
        return {
            "count": 0,
            "valid_set_accuracy": 0.0,
            "single_trace_accuracy": 0.0,
            "regret_at_1": None,
            "regret_label_coverage": 0.0,
        }
    batch = store.batch(indices, device)
    result = model(
        batch["state_views"],
        batch["tool_views"],
        batch["structured_state"],
        batch["structured_tools"],
        aux_views=batch["aux_views"],
        candidate_mask=batch["candidate_mask"],
        contract_mask=batch["contract_mask"],
        apply_contract_mask=apply_contract_mask,
    )
    prediction = result["energy"].argmin(dim=1)
    valid_hit = batch["valid_mask"].gather(1, prediction[:, None]).squeeze(1)
    exact = prediction == batch["gold"]
    known = batch["regret_mask"].gather(1, prediction[:, None]).squeeze(1)
    selected_regret = batch["regret"].gather(1, prediction[:, None]).squeeze(1)
    return {
        "count": len(indices),
        "valid_set_accuracy": float(valid_hit.float().mean().item()),
        "single_trace_accuracy": float(exact.float().mean().item()),
        "regret_at_1": float(selected_regret[known].mean().item()) if known.any() else None,
        "regret_label_coverage": float(known.float().mean().item()),
        "mean_gate_entropy": float(
            (-(result["view_weights"] * result["view_weights"].clamp_min(1e-8).log()).sum(1))
            .mean()
            .item()
        ),
    }


def train_metric_model(
    processed_dir: str | Path,
    cache_dir: str | Path,
    output_dir: str | Path,
    config_path: str | Path,
    *,
    distance_override: str | None = None,
    include_test: bool = False,
) -> dict[str, Any]:
    config = read_yaml(config_path)
    train_config = dict(config.get("metric_training", {}))
    seed = int(config.get("seed", 17))
    experiment_config_sha256 = sha256_file(config_path)
    distance = distance_override or str(train_config.get("distance", "cosine"))
    effective_train_config = {**train_config, "distance": distance}
    training_config_sha256 = sha256_text(canonical_json(effective_train_config))
    seed_everything(seed)
    device = choose_device(str(config.get("device", "auto")))
    run_provenance = collect_run_provenance(config_path, device=device)
    structured_dim = int(train_config.get("structured_dim", 64))
    store = FeatureStore(processed_dir, cache_dir, structured_dim=structured_dim)
    train_indices = store.indices("train", include_terminal=False)
    if not train_indices:
        raise ValueError("No nonterminal training rows")
    if not store.indices("dev", include_terminal=False):
        raise ValueError("No nonterminal dev rows; cannot select a metric checkpoint safely")
    if include_test and not store.indices("test", include_terminal=False):
        raise ValueError(
            "Test was explicitly unsealed, but there are no nonterminal test rows "
            "for metric evaluation"
        )
    model = FunctionalGeometryModel(
        store.view_dims,
        aux_dims=store.aux_dims,
        structured_dim=structured_dim,
        shared_dim=int(train_config.get("shared_dim", 64)),
        hidden_dim=int(train_config.get("hidden_dim", 128)),
        distance=distance,
        distance_rank=train_config.get("distance_rank"),
        temperature=float(train_config.get("temperature", 0.1)),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_config.get("learning_rate", 3e-4)),
        weight_decay=float(train_config.get("weight_decay", 1e-4)),
    )
    epochs = int(train_config.get("epochs", 100))
    batch_size = int(train_config.get("batch_size", 32))
    patience = int(train_config.get("patience", 20))
    rank_weight = float(train_config.get("regret_weight", 0.5))
    transition_weight = float(train_config.get("transition_weight", 0.2))
    margin = float(train_config.get("rank_margin", 0.1))
    apply_contract_mask = bool(train_config.get("apply_contract_mask", False))
    best_score = -float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = -1
    history = []
    stale = 0
    rng = random.Random(seed)
    for epoch in range(1, epochs + 1):
        rng.shuffle(train_indices)
        model.train()
        epoch_loss = 0.0
        examples_seen = 0
        for start in range(0, len(train_indices), batch_size):
            batch_indices = train_indices[start : start + batch_size]
            batch = store.batch(batch_indices, device)
            result = model(
                batch["state_views"],
                batch["tool_views"],
                batch["structured_state"],
                batch["structured_tools"],
                aux_views=batch["aux_views"],
                candidate_mask=batch["candidate_mask"],
                contract_mask=batch["contract_mask"],
                apply_contract_mask=apply_contract_mask,
            )
            scoring_mask = (
                batch["candidate_mask"] & batch["contract_mask"]
                if apply_contract_mask
                else batch["candidate_mask"]
            )
            selection_loss = set_valued_nll(result["energy"], batch["valid_mask"], scoring_mask)
            rank_loss = regret_ranking_loss(
                result["energy"],
                batch["regret"],
                batch["regret_mask"],
                scoring_mask,
                margin=margin,
            )
            prototypes = model.tool_prototypes(batch["tool_views"])
            selected = prototypes[batch["gold"]]
            predicted_next = model.predict_next_snapshot(result["context"], selected)
            target_next = model.encode_snapshot(
                batch["next_state_views"],
                batch["structured_next_state"],
                batch["aux_next_state_views"],
            ).detach()
            transition_loss = 1.0 - F.cosine_similarity(predicted_next, target_next, dim=-1).mean()
            loss = selection_loss + rank_weight * rank_loss + transition_weight * transition_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += float(loss.item()) * len(batch_indices)
            examples_seen += len(batch_indices)
        model.eval()
        dev = evaluate_metric_model(
            model, store, "dev", device, apply_contract_mask=apply_contract_mask
        )
        score = float(dev["valid_set_accuracy"])
        history.append(
            {
                "epoch": epoch,
                "train_loss": epoch_loss / max(1, examples_seen),
                "dev_valid_set_accuracy": score,
                "dev_regret_at_1": dev.get("regret_at_1", 0.0),
            }
        )
        if score > best_score + 1e-8:
            best_score = score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        raise RuntimeError("Metric training did not produce a checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    report_splits = ("train", "dev", "test") if include_test else ("train", "dev")
    metrics = {
        split: evaluate_metric_model(
            model, store, split, device, apply_contract_mask=apply_contract_mask
        )
        for split in report_splits
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "kind": "functional_geometry",
        "seed": seed,
        "experiment_config_sha256": experiment_config_sha256,
        "training_config_sha256": training_config_sha256,
        "training_config": effective_train_config,
        "run_provenance": run_provenance,
        "model_config": model.export_config(),
        "model_state": model.state_dict(),
        "tool_ids": store.tool_ids,
        "views": store.views,
        "feature_spec_version": store.cache.manifest["feature_spec_version"],
        "cache_config_sha256": store.cache.manifest["config_sha256"],
        "cache_content_sha256": store.cache.content_sha256,
        "source_manifest_sha256": store.cache.manifest["source_manifest_sha256"],
        "apply_contract_mask": apply_contract_mask,
        "best_epoch": best_epoch,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }
    checkpoint_path = output_dir / "metric_model.pt"
    torch.save(checkpoint, checkpoint_path)
    checkpoint_sha256 = sha256_file(checkpoint_path)
    result = {
        "distance": distance,
        "seed": seed,
        "experiment_config_sha256": experiment_config_sha256,
        "training_config_sha256": training_config_sha256,
        "training_config": effective_train_config,
        "run_provenance": run_provenance,
        "cache_content_sha256": store.cache.content_sha256,
        "metric_checkpoint_sha256": checkpoint_sha256,
        "best_epoch": best_epoch,
        "best_dev_valid_set_accuracy": best_score,
        "parameter_count": checkpoint["parameter_count"],
        "metrics": metrics,
        "history": history,
        "test_reported": include_test,
    }
    write_json(output_dir / "metric_metrics.json", result)
    return result


def load_metric_checkpoint(
    path: str | Path, device: torch.device
) -> tuple[FunctionalGeometryModel, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("checkpoint_version") != CHECKPOINT_VERSION:
        raise ValueError(
            f"Unsupported metric checkpoint version: {checkpoint.get('checkpoint_version')!r}"
        )
    if checkpoint.get("kind") != "functional_geometry":
        raise ValueError(f"Not a functional geometry checkpoint: {path}")
    if checkpoint.get("feature_spec_version") != FEATURE_SPEC_VERSION:
        raise ValueError(
            "Metric checkpoint feature semantics differ from this runtime: "
            f"{checkpoint.get('feature_spec_version')!r} != {FEATURE_SPEC_VERSION!r}"
        )
    model = FunctionalGeometryModel(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a small functional geometry module")
    parser.add_argument("--processed-dir", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--distance",
        choices=[
            "euclidean",
            "cosine",
            "diagonal_mahalanobis",
            "lowrank_mahalanobis",
            "bilinear",
            "poincare",
        ],
    )
    parser.add_argument(
        "--include-test",
        action="store_true",
        help="Explicitly unseal and report the test split after dev-based selection.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = train_metric_model(
        args.processed_dir,
        args.cache_dir,
        args.output_dir,
        args.config,
        distance_override=args.distance,
        include_test=args.include_test,
    )
    print(result["metrics"])


if __name__ == "__main__":
    main()
