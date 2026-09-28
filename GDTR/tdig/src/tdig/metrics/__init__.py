"""TDiG metric family — five main + one consistency-check metric.

M1 c_dir       — direction settling (cosine)
M2 c_mag       — magnitude settling (|r - 1|)
M3 c_geo       — trajectory dynamics (velocity + curvature, reference-free)
M4 c_M         — Mahalanobis residual (distribution anisotropy)
M5 c_tau       — path tortuosity (cumulative path efficiency)
M6 D_L2        — L2 residual (function of M1+M2; appendix consistency check)

See docs/metric_definitions.md for full math; PLAN.md for the cell matrix.
"""
