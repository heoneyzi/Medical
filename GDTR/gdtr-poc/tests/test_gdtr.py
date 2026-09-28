"""Synthetic-D tests for src.gdtr.

Constructs a [L, T] matrix with hand-engineered trajectories per column to
verify that running_min, settling_depth_discrete, and the deep-thinking
ratio behave as documented.
"""
import math
import numpy as np
import torch

from src.gdtr import (
    deep_thinking_mask,
    gdtr,
    running_min,
    settling_depth_discrete,
)


def _build_D() -> torch.Tensor:
    # L=8, T=4 columns:
    # col 0: D = [0.9, 0.8, 0.7, 0.6, 0.4, 0.3, 0.2, 0.0]   c = 5 (first <=0.5)
    # col 1: D = [0.4, 0.5, 0.6, 0.6, 0.6, 0.5, 0.4, 0.0]   c = 1
    # col 2: D = [0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.0]   c = 8 (only crosses at L)
    # col 3: D = [0.95,0.95,0.95,0.95,0.95,0.95,0.95,0.6]   c = 8 (saturated, never <=0.5)
    cols = [
        [0.9, 0.8, 0.7, 0.6, 0.4, 0.3, 0.2, 0.0],
        [0.4, 0.5, 0.6, 0.6, 0.6, 0.5, 0.4, 0.0],
        [0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.0],
        [0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.6],
    ]
    return torch.tensor(cols, dtype=torch.float32).T  # [L=8, T=4]


def test_running_min_monotone_nonincreasing():
    D = _build_D()
    rmin = running_min(D)
    diffs = rmin[1:] - rmin[:-1]
    assert (diffs <= 0).all(), "running_min must be non-increasing along layers"


def test_settling_depth_discrete_known():
    D = _build_D()
    c = settling_depth_discrete(D, gamma=0.5)
    # col 0: first <=0.5 is layer 5 (index 4, value 0.4)
    assert int(c[0]) == 5
    # col 1: layer 1 (value 0.4 <= 0.5)
    assert int(c[1]) == 1
    # col 2: only crosses at layer 8 (final 0.0)
    assert int(c[2]) == 8
    # col 3: never <= 0.5 -> saturated to L=8
    assert int(c[3]) == 8


def test_gdtr_threshold():
    # rho * L = 0.85 * 8 = 6.8 -> "deep" iff c > 6.8 -> c in {7, 8}
    c = torch.tensor([5, 1, 8, 8])
    val = gdtr(c, rho=0.85, L=8)
    # 2/4 deep
    assert math.isclose(val, 0.5, abs_tol=1e-6)


def test_deep_thinking_mask():
    c = torch.tensor([5, 1, 8, 7])
    m = deep_thinking_mask(c, rho=0.85, L=8)
    assert m.tolist() == [False, False, True, True]


def test_gdtr_empty_returns_nan():
    val = gdtr(torch.tensor([], dtype=torch.float32))
    assert math.isnan(val)


def test_running_min_validate_nan():
    bad = torch.tensor([[float("nan")]])
    try:
        running_min(bad)
    except ValueError:
        return
    assert False, "running_min should reject NaN"
