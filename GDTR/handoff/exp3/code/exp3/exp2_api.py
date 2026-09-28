"""The complete allow-list of EXP2 APIs that EXP3 may import.

Only generic runtime, tap, endpoint and statistics contracts are re-exported.
Importing ``exp2.step*`` from anywhere in EXP3 is a boundary violation and is
checked by the test suite.
"""
from exp2 import naming as N
from exp2.endpoints import branch_geometry, solve_beta_shape
from exp2.manifest import ArchitectureManifest, BlockType, Manifest, sha256_path
from exp2.stats import Row, cluster_bootstrap, paired_contrast
from exp2.taps import Edit, Evo2Runner, replace_with

__all__ = [
    "N", "ArchitectureManifest", "BlockType", "Edit", "Evo2Runner", "Manifest", "Row",
    "branch_geometry", "cluster_bootstrap", "paired_contrast", "replace_with",
    "sha256_path", "solve_beta_shape",
]
