"""RQ1: is the execution structure already in the frozen representation?

Three questions that are usually run together and should not be.

**Is the information there?**  A *linear* probe on frozen vectors predicts the
oracle's remaining cost and its optimal action.  Linear on purpose: a deep head
would measure the head.

**Is the geometry aligned?**  Information being decodable is not the same as
distance meaning something.  Alignment is measured directly, with the depth
confound removed -- states late in a trajectory are both closer to the goal and
carry more history text, so an uncontrolled correlation can be produced by
counting characters.

**Is it biological?**  Minimal pairs and a contamination holdout separate
"carries variant meaning" from "carries notation" and from "read the answer in
pretraining".

Every statistic is returned as a :class:`Finding` with a gene-clustered interval
or a position in a permutation null.  Nothing here compares a number to a
threshold.
"""

from __future__ import annotations

import collections
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from geoflowagent.geoacmg.claims import Finding, Role
from geoflowagent.geoacmg.inference import null_referenced, paired_contrast

# --------------------------------------------------------------------- probes


def _design(features: np.ndarray) -> np.ndarray:
    return np.hstack([features, np.ones((features.shape[0], 1), dtype=features.dtype)])


def _gram(design: np.ndarray, penalty: float) -> np.ndarray:
    gram = design.T @ design
    gram[np.diag_indices_from(gram)] += penalty
    return gram


def _ridge_fit(features: np.ndarray, targets: np.ndarray, penalty: float) -> np.ndarray:
    design = _design(features)
    return np.linalg.solve(_gram(design, penalty), design.T @ targets)


def _apply(weights: np.ndarray, features: np.ndarray) -> np.ndarray:
    return _design(features) @ weights


# The permutation null re-fits the same design thousands of times.
#
# Only the labels move between draws: the design matrix and its regularised Gram
# depend on the features and the fold assignment alone.  Rebuilding them per
# class per fold per draw made the label-permutation null for one view a
# 16-hour job -- 10,050 Gram matrices of 448,704 x 769 where five suffice.
#
# The cache holds one entry, keyed by the identity of the feature array it was
# built from, and keeps a reference to that array so the id cannot be reused
# while the entry is live.  Every arithmetic operation downstream sees the same
# arrays it would have built itself, so the results are bit-identical rather
# than merely close; a run of both paths over the real probe table is recorded
# in RUN_LOG.md.
_FOLD_CACHE: dict[tuple[Any, ...], tuple[np.ndarray, list[Any]]] = {}


def _fold_plan(groups: Sequence[Any], folds: int) -> tuple[np.ndarray, int]:
    keys = sorted(set(groups))
    if len(keys) < folds:
        folds = max(2, len(keys))
    assignment = {key: index % folds for index, key in enumerate(keys)}
    return np.asarray([assignment[key] for key in groups]), folds


def _folds_for(
    features: np.ndarray, fold_of: np.ndarray, folds: int, penalty: float
) -> list[Any]:
    """Per-fold ``(train, test, design_train, design_test, gram)``, built once."""

    key = (
        id(features),
        features.shape,
        features.dtype.str,
        folds,
        penalty,
        hash(fold_of.tobytes()),
    )
    cached = _FOLD_CACHE.get(key)
    if cached is not None:
        return cached[1]
    plan: list[Any] = []
    for fold in range(folds):
        test = fold_of == fold
        train = ~test
        if not train.any() or not test.any():
            plan.append(None)
            continue
        design_train = _design(features[train])
        plan.append(
            (train, test, design_train, _design(features[test]), _gram(design_train, penalty))
        )
    # These entries are gigabytes; keep exactly one.
    _FOLD_CACHE.clear()
    _FOLD_CACHE[key] = (features, plan)
    return plan


def grouped_cv_r2(
    features: np.ndarray,
    targets: np.ndarray,
    groups: Sequence[Any],
    *,
    folds: int = 5,
    penalty: float = 1.0,
) -> float:
    """Cross-validated R^2 with whole groups held out.

    Groups are genes.  Splitting rows would let a variant's own gene sit in the
    training fold, and the probe would be scored on how well it memorised that
    gene rather than on what the representation encodes.
    """

    fold_of, folds = _fold_plan(groups, folds)
    predictions = np.zeros_like(targets, dtype=np.float64)
    for entry in _folds_for(features, fold_of, folds, penalty):
        if entry is None:
            continue
        train, test, design_train, design_test, gram = entry
        weights = np.linalg.solve(gram, design_train.T @ targets[train])
        predictions[test] = design_test @ weights
    residual = float(np.sum((targets - predictions) ** 2))
    total = float(np.sum((targets - targets.mean()) ** 2))
    return 1.0 - residual / total if total > 0 else 0.0


def grouped_cv_accuracy(
    features: np.ndarray,
    labels: Sequence[Any],
    groups: Sequence[Any],
    *,
    folds: int = 5,
    penalty: float = 1.0,
) -> float:
    """One-vs-rest ridge classification accuracy, groups held out."""

    classes = sorted({str(label) for label in labels})
    if len(classes) < 2:
        return 0.0
    label_array = np.asarray([str(label) for label in labels])
    fold_of, folds = _fold_plan(groups, folds)
    predicted = np.empty(len(label_array), dtype=object)
    for entry in _folds_for(features, fold_of, folds, penalty):
        if entry is None:
            continue
        train, test, design_train, design_test, gram = entry
        # Sliced once per fold rather than once per class: the slice is a copy of
        # a few hundred thousand strings, and taking it ten times over was most
        # of the wall time of a permutation draw. The values are identical.
        train_labels = label_array[train]
        scores = np.zeros((int(test.sum()), len(classes)))
        for index, name in enumerate(classes):
            target = (train_labels == name).astype(np.float64)
            # Left as a per-column solve on purpose. Batching the right-hand
            # sides, or caching the factorisation through scipy, moves the
            # weights by about 1e-17 -- harmless in size, but the scores feed an
            # argmax, and the run that produces the study's numbers should not
            # depend on a tie broken at the last bit.
            weights = np.linalg.solve(gram, design_train.T @ target)
            scores[:, index] = design_test @ weights
        predicted[test] = [classes[i] for i in scores.argmax(axis=1)]
    return float(np.mean(predicted == label_array))


# ----------------------------------------------------------------- E1 controls


def random_projection(features: np.ndarray, *, seed: int = 17) -> np.ndarray:
    """An isotropic control of identical dimension.

    Rules out "the probe works because the vector is wide".
    """

    rng = np.random.default_rng(seed)
    basis = rng.normal(size=(features.shape[1], features.shape[1]))
    basis /= np.linalg.norm(basis, axis=0, keepdims=True)
    projected = rng.normal(size=features.shape) @ basis
    return projected.astype(features.dtype)


def depth_matched_pairs(
    values: Sequence[float],
    targets: Sequence[float],
    depths: Sequence[int],
    tasks: Sequence[Any],
) -> tuple[list[float], list[float], list[Any]]:
    """Ordering accuracy restricted to states at the same depth in the same task.

    Removes the confound phase 1 left open: within a task, a later state is both
    nearer the goal and described by more text, so a correlation between distance
    and remaining cost can be produced by trajectory length alone.  Holding depth
    fixed removes that channel entirely.
    """

    buckets: dict[tuple[Any, int], list[int]] = collections.defaultdict(list)
    for index, (task, depth) in enumerate(zip(tasks, depths, strict=True)):
        buckets[(task, depth)].append(index)
    correct: list[float] = []
    chance: list[float] = []
    clusters: list[Any] = []
    for (task, _), members in buckets.items():
        for left in range(len(members)):
            for right in range(left + 1, len(members)):
                a, b = members[left], members[right]
                if targets[a] == targets[b]:
                    continue
                predicted_order = values[a] < values[b]
                true_order = targets[a] < targets[b]
                correct.append(1.0 if predicted_order == true_order else 0.0)
                chance.append(0.5)
                clusters.append(task)
    return correct, chance, clusters


# ------------------------------------------------------------------ E1 and E2


def information_findings(
    features: np.ndarray,
    *,
    remaining_cost: Sequence[float],
    optimal_action: Sequence[str],
    genes: Sequence[str],
    view: str,
    surface_features: np.ndarray | None = None,
    claim_id: str = "C1",
    role: Role = Role.EXPLORATORY,
    seed: int = 17,
    permutations: int = 200,
) -> list[Finding]:
    """E1: how much of the oracle's structure a linear read-out recovers.

    Four controls, each killing a different alternative explanation:
    a label permutation null (chance is measured, not assumed), a random
    projection of the same width (dimension), a surface-only encoding if one is
    supplied (biology versus template), and gene-held-out folds throughout
    (memorisation).
    """

    targets = np.asarray(remaining_cost, dtype=np.float64)
    findings: list[Finding] = []

    observed_r2 = grouped_cv_r2(features, targets, genes)
    control_r2 = grouped_cv_r2(random_projection(features, seed=seed), targets, genes)
    findings.append(
        paired_contrast(
            claim_id=claim_id,
            name=f"E1_remaining_cost_r2_{view}_minus_random_projection",
            left=[observed_r2], right=[control_r2], clusters=["corpus"],
            unit="corpus", role=role,
            detail={"view": view, "observed_r2": observed_r2, "random_projection_r2": control_r2,
                    "note": "single-corpus contrast; the interval that matters is the "
                            "permutation null below"},
        )
    )

    def _accuracy(pairs: Sequence[tuple[int, str]]) -> float:
        labels = [label for _, label in pairs]
        return grouped_cv_accuracy(features, labels, genes)

    findings.append(
        null_referenced(
            claim_id=claim_id,
            name=f"E1_optimal_action_accuracy_{view}",
            statistic=_accuracy,
            values=list(range(len(optimal_action))),
            labels=list(optimal_action),
            unit="gene", n_units=len(set(genes)), role=role,
            draws=permutations, seed=seed,
            detail={"view": view, "classes": sorted(set(optimal_action)),
                    "majority_rate": max(collections.Counter(optimal_action).values())
                    / len(optimal_action)},
        )
    )

    if surface_features is not None:
        surface_r2 = grouped_cv_r2(surface_features, targets, genes)
        findings.append(
            paired_contrast(
                claim_id="C2",
                name=f"E1_biological_information_{view}",
                left=[observed_r2], right=[surface_r2], clusters=["corpus"],
                unit="corpus", role=role,
                detail={"view": view, "full_r2": observed_r2, "surface_only_r2": surface_r2,
                        "note": "full minus surface-only is the share of the decodable "
                                "structure that is not template form"},
            )
        )
    return findings


def alignment_findings(
    distances: Sequence[float],
    *,
    remaining_cost: Sequence[float],
    depths: Sequence[int],
    tasks: Sequence[Any],
    genes: Sequence[str],
    view: str,
    claim_id: str = "C1",
    role: Role = Role.EXPLORATORY,
) -> list[Finding]:
    """E2: does distance in the space mean remaining work?"""

    correct, chance, clusters = depth_matched_pairs(distances, remaining_cost, depths, tasks)
    findings: list[Finding] = []
    if correct:
        gene_of = dict(zip(tasks, genes, strict=False))
        findings.append(
            paired_contrast(
                claim_id=claim_id,
                name=f"E2_depth_matched_ordering_{view}",
                left=correct, right=chance,
                clusters=[gene_of.get(task, task) for task in clusters],
                unit="gene", role=role,
                detail={"view": view, "pairs": len(correct),
                        "note": "states compared only within one task at one depth, so "
                                "trajectory length cannot produce the effect"},
            )
        )
    return findings


# ------------------------------------------------------------------ diagnostics


def hubness(vectors: np.ndarray, *, k: int = 10) -> dict[str, Any]:
    """Concentration diagnostics for a retrieval space.

    Reported because a space can carry information and still be unusable for
    nearest-neighbour retrieval.  Recent work finds hub mass, not anisotropy,
    explains retrieval asymmetry, so hub share is reported alongside the usual
    anisotropy and effective-rank numbers rather than instead of them.
    """

    normed = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12)
    similarity = normed @ normed.T
    np.fill_diagonal(similarity, -np.inf)
    neighbours = np.argsort(-similarity, axis=1)[:, :k]
    counts = collections.Counter(neighbours.flatten().tolist())
    occurrence = np.asarray([counts.get(i, 0) for i in range(len(normed))], dtype=np.float64)
    order = np.sort(occurrence)
    cumulative = np.cumsum(order)
    gini = (
        float((len(order) + 1 - 2 * np.sum(cumulative) / cumulative[-1]) / len(order))
        if cumulative[-1] > 0
        else 0.0
    )
    centroid = normed.mean(axis=0)
    spectrum = np.linalg.svd(normed - centroid, compute_uv=False)
    power = spectrum**2
    share = power / power.sum()
    entropy = float(-np.sum(share * np.log(share + 1e-12)))
    upper = np.triu_indices(len(normed), k=1)
    return {
        "k": k,
        "hub_occurrence_gini": gini,
        "top1_hub_share": float(np.max(occurrence) / max(occurrence.sum(), 1.0)),
        "mean_cosine_similarity": float(similarity[upper][np.isfinite(similarity[upper])].mean()),
        "effective_rank": float(math.exp(entropy)),
        "dimension": int(normed.shape[1]),
        "note": (
            "effective rank far below the dimension, or a large hub share, means the "
            "space is not usable as a global retrieval index even if a probe reads it"
        ),
    }


def pair_separation_findings(
    pairs: Sequence[Mapping[str, Any]],
    distances: Sequence[float],
    *,
    view: str,
    claim_id: str = "C2",
    role: Role = Role.EXPLORATORY,
) -> list[Finding]:
    """E3: do minimal pairs separate the way biology says they should?

    Functional pairs (same gene, opposite classification) are read against the
    same-classification control from the same genes.  That control is the null:
    two different variants are always somewhat apart, and the question is whether
    disagreeing ones are further.
    """

    grouped: dict[str, list[tuple[float, str]]] = collections.defaultdict(list)
    for row, distance in zip(pairs, distances, strict=True):
        kind = str(row.get("provenance", {}).get("kind", row["relation"]))
        grouped[kind].append((float(distance), str(row.get("provenance", {}).get("gene", ""))))

    findings: list[Finding] = []
    functional = grouped.get("opposite_classification", [])
    control = grouped.get("same_classification_control", [])
    if functional and control:
        size = min(len(functional), len(control))
        findings.append(
            paired_contrast(
                claim_id=claim_id,
                name=f"E3_functional_pair_separation_{view}",
                left=[d for d, _ in functional[:size]],
                right=[d for d, _ in control[:size]],
                clusters=[gene or f"g{i}" for i, (_, gene) in enumerate(functional[:size])],
                unit="gene", role=role,
                detail={"view": view, "functional_pairs": len(functional),
                        "control_pairs": len(control)},
            )
        )
    spelling = grouped.get("hgvs_spelling", [])
    if spelling and control:
        size = min(len(spelling), len(control))
        findings.append(
            paired_contrast(
                claim_id=claim_id,
                name=f"E3_notation_invariance_{view}",
                left=[d for d, _ in control[:size]],
                right=[d for d, _ in spelling[:size]],
                clusters=[gene or f"g{i}" for i, (_, gene) in enumerate(spelling[:size])],
                unit="gene", role=role,
                detail={"view": view,
                        "note": "positive means two spellings of one variant sit closer "
                                "than two genuinely different variants, which is the "
                                "invariance the encoder should have"},
            )
        )
    return findings


def contamination_finding(
    before_scores: Sequence[float],
    after_scores: Sequence[float],
    before_genes: Sequence[str],
    after_genes: Sequence[str],
    *,
    view: str,
    cutoff: str,
    claim_id: str = "C2",
    role: Role = Role.EXPLORATORY,
) -> Finding:
    """E3: how much of the performance survives the encoder cut-off.

    A large positive gap means the representation was partly recalling published
    answers rather than encoding structure.  The gap is reported with an interval;
    it is never converted into a pass or fail.
    """

    size = min(len(before_scores), len(after_scores))
    return paired_contrast(
        claim_id=claim_id,
        name=f"E3_contamination_gap_{view}",
        left=list(before_scores)[:size],
        right=list(after_scores)[:size],
        clusters=list(before_genes)[:size],
        unit="gene", role=role,
        detail={"view": view, "cutoff": cutoff,
                "limitation": "controls database contamination only; a variant absent "
                              "from a database may still be discussed in the literature "
                              "the encoder read"},
    )
