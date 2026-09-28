"""Synthetic tests for converging-tail settling depths."""
import torch

from src.gdtr import settling_depth_tail_soft, settling_depth_tail_strict


def test_strict_tail_ignores_transient_crossing():
    D = torch.tensor([[0.9, 0.4, 0.8, 0.3, 0.2]], dtype=torch.float32).T
    c, unresolved = settling_depth_tail_strict(D, gamma=0.5)
    assert int(c[0]) == 4
    assert not bool(unresolved[0])


def test_strict_tail_stable_convergence():
    D = torch.tensor([[0.9, 0.8, 0.4, 0.3, 0.2]], dtype=torch.float32).T
    c, unresolved = settling_depth_tail_strict(D, gamma=0.5)
    assert int(c[0]) == 3
    assert not bool(unresolved[0])


def test_strict_tail_unresolved_uses_l_plus_one():
    D = torch.tensor([[0.9, 0.8, 0.7, 0.6]], dtype=torch.float32).T
    c, unresolved = settling_depth_tail_strict(D, gamma=0.5)
    assert int(c[0]) == 5
    assert bool(unresolved[0])


def test_soft_tail_allows_one_late_violation():
    D = torch.tensor([[0.9, 0.4, 0.8, 0.3, 0.2]], dtype=torch.float32).T
    c, unresolved = settling_depth_tail_soft(D, gamma=0.5, rho=0.75)
    assert int(c[0]) == 2
    assert not bool(unresolved[0])


def test_soft_tail_can_be_unresolved_without_final_anchor():
    D = torch.tensor([[0.9, 0.8, 0.7, 0.6]], dtype=torch.float32).T
    c, unresolved = settling_depth_tail_soft(D, gamma=0.5, rho=0.8)
    assert int(c[0]) == 5
    assert bool(unresolved[0])


def test_soft_tail_validates_rho():
    D = torch.zeros((2, 1), dtype=torch.float32)
    try:
        settling_depth_tail_soft(D, gamma=0.5, rho=1.1)
    except ValueError:
        return
    assert False, "settling_depth_tail_soft should reject rho > 1"
