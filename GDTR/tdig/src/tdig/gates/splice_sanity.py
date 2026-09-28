"""Splice signal sanity gate (S2.1).

Cohen's d between splice donor and intron settling depths on chr22.
Pass criterion: |d| >= 0.20.

Carried over from gDTR-PoC Phase 1 (chr22 12,978 windows x 6kb, 77M positions).
"""

from __future__ import annotations


def splice_donor_vs_intron_d(c_per_token, context_labels):
    """Cohen's d between splice_donor and intron positions.

    Args:
        c_per_token: [n_tokens] settling depths from a metric x ref cell.
        context_labels: [n_tokens] str array with values in
            {'intergenic', 'intron', 'coding_exon', '5utr', '3utr',
             'splice_donor', 'splice_acceptor'}.

    Returns:
        dict with keys 'd', 'donor_mean', 'intron_mean', 'donor_n', 'intron_n', 'pass'.
    """
    raise NotImplementedError("S2 implementation")
