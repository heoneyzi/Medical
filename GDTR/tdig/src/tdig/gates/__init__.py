"""Sanity gates — S2 of the experimental sequence.

Three gates applied uniformly to every settling cell:
- splice_sanity: chr22 splice donor vs intron Cohen's d
- entropy_gate:  E8 protocol (entropy decoupling)
- motif_gate:    E9 protocol (motif edit + flank shuffle bidirectional)

A cell failing a gate is NOT dropped automatically — failure mode itself is
information about what the metric x reference cell captures.
"""
