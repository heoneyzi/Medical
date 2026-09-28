"""Edge-case tests for settling_depth_interp."""
import math
import torch

from src.gdtr import settling_depth_interp


def test_immediate_below_returns_one():
    # D[0, 0] = 0.4 <= gamma=0.5
    D = torch.tensor([[0.4, 0.6, 0.7, 0.0]]).T  # shape [4,1]
    c, sat = settling_depth_interp(D, gamma=0.5)
    assert math.isclose(float(c[0]), 1.0, abs_tol=1e-6)
    assert not bool(sat[0])


def test_never_crosses_marks_saturated():
    D = torch.tensor([[0.95, 0.9, 0.85, 0.8]]).T  # never <= 0.5
    c, sat = settling_depth_interp(D, gamma=0.5)
    # L = 4
    assert math.isclose(float(c[0]), 4.0, abs_tol=1e-6)
    assert bool(sat[0])


def test_exact_equality_returns_layer():
    # D[1, 0] = 0.5 exactly -> first_idx=1 (0-based), interpolate uses
    # rmin_above=0.7, rmin_below=0.5 -> frac=(0.7-0.5)/0.2=1.0 -> c=1+1=2
    D = torch.tensor([[0.7, 0.5, 0.4, 0.0]]).T
    c, _ = settling_depth_interp(D, gamma=0.5)
    assert math.isclose(float(c[0]), 2.0, abs_tol=1e-6)


def test_crosses_between_layers_linear_interp():
    # D[0]=0.7, D[1]=0.3 (crosses 0.5 between layer 1 and 2)
    # rmin_above=0.7 at layer 1 (idx=0); rmin_below=0.3 at layer 2 (idx=1)
    # frac = (0.7 - 0.5) / (0.7 - 0.3) = 0.5 -> c = 1 + 0.5 = 1.5
    D = torch.tensor([[0.7, 0.3, 0.2, 0.1]]).T
    c, _ = settling_depth_interp(D, gamma=0.5)
    assert math.isclose(float(c[0]), 1.5, abs_tol=1e-6)


def test_uses_running_min_not_raw_jsd():
    # If raw JSD goes 0.7, 0.4, 0.6, 0.0:
    #   running_min = 0.7, 0.4, 0.4, 0.0
    # gamma=0.5 crosses between layer 1 (0.7) and layer 2 (0.4)
    # frac = (0.7 - 0.5) / (0.7 - 0.4) = 2/3  -> c = 1 + 2/3 = 5/3
    D = torch.tensor([[0.7, 0.4, 0.6, 0.0]]).T
    c, _ = settling_depth_interp(D, gamma=0.5)
    assert math.isclose(float(c[0]), 1.0 + 2.0 / 3.0, abs_tol=1e-6)
