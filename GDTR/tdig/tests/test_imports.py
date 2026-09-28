"""Smoke test — every module imports cleanly."""

from __future__ import annotations


def test_references_imports():
    from tdig.references import get_reference_a, get_reference_b, get_reference_c, apply_rmsnorm  # noqa: F401


def test_metrics_imports():
    from tdig.metrics import m1_direction, m2_magnitude, m3_geo, m4_mahalanobis, m5_tortuosity, m6_l2_residual  # noqa: F401


def test_gates_imports():
    from tdig.gates import splice_sanity, entropy_gate, motif_gate  # noqa: F401


def test_analyses_imports():
    from tdig.analyses import concordance, context_stratify, joint_lr, trajectory_decomposition  # noqa: F401


def test_viz_imports():
    from tdig.viz import (
        v1_trajectory_pca, v2_velocity_heatmap, v3_curvature_heatmap,
        v4_cumulative_path, v5_2d_signature, v6_cka_matrix,
        v7_context_heatmap, v8_dissociation_map, v9_tortuosity_profile,
    )  # noqa: F401


def test_tasks_imports():
    from tdig.tasks import splice_pred, variant_clf  # noqa: F401


def test_controls_imports():
    from tdig.controls import cka_control  # noqa: F401
