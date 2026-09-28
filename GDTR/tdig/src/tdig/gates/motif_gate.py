"""Motif / flank bidirectional gate (S2.3) — E9 protocol carryover.

1,000 chr22 canonical GT-AG donors, +/- 100bp:
  - Real flanks:                  baseline c
  - Motif edit (GT -> AA), flank preserved:    c should change minimally
  - Flank shuffle (GT preserved):              c should shift markedly

Bidirectional pattern means the metric reads grammar integration rather than
motif strength alone. Reference: gDTR-PoC scripts/exp2_shuffled_motif_control.py.
"""

from __future__ import annotations


def motif_edit_response(c_real, c_motif_edit):
    """Paired Wilcoxon test: c_real vs c_motif_edit.

    Args:
        c_real: [n_donors] baseline settling depths.
        c_motif_edit: [n_donors] settling depths after GT -> AA.

    Returns:
        dict with 'd', 'p_wilcoxon', 'mean_shift'.
    """
    raise NotImplementedError("S2 implementation")


def flank_shuffle_response(c_real, c_flank_shuffled):
    """Paired Wilcoxon: c_real vs c_flank_shuffled (5 shuffles each)."""
    raise NotImplementedError("S2 implementation")


def bidirectional_check(c_real, c_motif_edit, c_flank_shuffled):
    """Both perturbations produce non-trivial responses, in opposite directions.

    Returns:
        dict with motif_edit + flank_shuffle results, plus 'bidirectional_pass'.
    """
    raise NotImplementedError("S2 implementation")
