"""Variant Delta-metric tests."""
import torch

from src.variant_delta import compute_delta_metrics


def test_idempotence_ref_equals_alt():
    """If D_ref == D_alt, every Delta-metric must be zero."""
    torch.manual_seed(0)
    L, T = 8, 100
    D = torch.rand(L, T) * 0.4   # in [0, 0.4]
    out = compute_delta_metrics(D, D)
    assert out["delta_c_discrete"] == 0
    assert out["delta_c_interp"] == 0.0
    assert out["max_abs_delta_D"] == 0.0
    assert out["signed_argmax_delta_D"] == 0.0
    assert torch.equal(out["delta_D"], torch.zeros(L))


def test_synthetic_signal_sign_known():
    """Construct ref/alt with known sign of layer-3 difference."""
    L, T = 8, 21
    pos = T // 2
    D_ref = torch.full((L, T), 0.3)
    D_alt = D_ref.clone()
    # alt has higher JSD at layer 3 (i.e. *less* converged at that layer)
    D_alt[2, pos] = 0.8
    out = compute_delta_metrics(D_ref, D_alt, gamma=0.5)
    # delta_D[2] = +0.5, max_abs = 0.5
    assert abs(out["max_abs_delta_D"] - 0.5) < 1e-6
    assert out["argmax_layer_1based"] == 3
    assert out["signed_argmax_delta_D"] > 0


def test_variant_position_default_is_central():
    L, T = 8, 5
    D_ref = torch.full((L, T), 0.4)
    D_alt = torch.full((L, T), 0.4)
    out = compute_delta_metrics(D_ref, D_alt)
    assert out["variant_position"] == 2  # T // 2


def test_variant_position_custom():
    L, T = 8, 5
    D_ref = torch.full((L, T), 0.4)
    D_alt = D_ref.clone()
    D_alt[5, 0] = 0.99
    out = compute_delta_metrics(D_ref, D_alt, variant_position=0)
    assert out["variant_position"] == 0
    assert out["argmax_layer_1based"] == 6
