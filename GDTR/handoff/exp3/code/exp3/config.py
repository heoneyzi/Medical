"""Pre-registered EXP3 choices, separate from EXP2 configuration."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Exp3Margins:
    """Pre-registered thresholds used by EXP3 verdict functions."""

    m28_damage: float = 0.10
    rescue: float = 0.10
    equivalence: float = 0.05
    m28_product_change: float = 0.10
    b29_effect: float = 0.10
    b29_perpendicular_fraction: float = 0.05
    b29_independent_effect: float = 0.10
    b29_plateau_increment: float = 0.05
    b29_log_q_boundary: float = 0.50
    b29_plateau_remaining_fraction: float = 0.10
    b29_plateau_boundary_spread: float = 0.50
    scale_plateau_increment: float = 0.01
    scale_transition_effect: float = 0.10
    scale_log_q_boundary: float = 0.50
    scale_plateau_remaining_fraction: float = 0.10
    scale_plateau_boundary_spread: float = 0.50
    angular_specificity: float = 0.10
    curve_collapse_rmse: float = 0.05
    cube_scale_effect: float = 0.10
    cube_content_effect: float = 0.10
    cube_carrier_effect: float = 0.10
    cube_axis_equivalence: float = 0.05
    b30_mediation_effect: float = 0.10
    b30_mediation_rescue: float = 0.10
    b30_mediation_equivalence: float = 0.05
    direction_transfer_effect: float = 0.10
    direction_specificity: float = 0.10
    direction_transfer_equivalence: float = 0.05
    direction_transfer_rescue: float = 0.10
    min_dependency_clusters: int = 5
    bootstrap_replicates: int = 10_000
    alpha: float = 0.05

    def __post_init__(self) -> None:
        numeric = (
            self.m28_damage, self.rescue, self.equivalence,
            self.m28_product_change, self.b29_effect,
            self.b29_perpendicular_fraction, self.b29_independent_effect,
            self.b29_plateau_increment, self.b29_log_q_boundary,
            self.b29_plateau_remaining_fraction,
            self.b29_plateau_boundary_spread,
            self.scale_plateau_increment,
            self.scale_transition_effect, self.scale_log_q_boundary,
            self.scale_plateau_remaining_fraction,
            self.scale_plateau_boundary_spread,
            self.angular_specificity, self.curve_collapse_rmse,
            self.cube_scale_effect, self.cube_content_effect,
            self.cube_carrier_effect, self.cube_axis_equivalence,
            self.b30_mediation_effect, self.b30_mediation_rescue,
            self.b30_mediation_equivalence, self.direction_transfer_effect,
            self.direction_specificity, self.direction_transfer_equivalence,
            self.direction_transfer_rescue, self.alpha,
        )
        if any(not np.isfinite(value) or value <= 0 for value in numeric):
            raise ValueError("all EXP3 scientific margins must be finite and positive")
        if (self.b29_perpendicular_fraction > 1
                or self.b29_plateau_remaining_fraction >= 1
                or self.scale_plateau_remaining_fraction >= 1
                or self.alpha >= 1):
            raise ValueError("fraction margins and alpha must lie in (0,1)")
        if (type(self.min_dependency_clusters) is not int
                or self.min_dependency_clusters < 2):
            raise ValueError("min_dependency_clusters must be an integer >=2")
        if (type(self.bootstrap_replicates) is not int
                or self.bootstrap_replicates < 100):
            raise ValueError("bootstrap_replicates must be an integer >=100")


@dataclass(frozen=True)
class Exp3Config:
    alpha_doses: tuple[float, ...] = tuple(float(x) for x in np.logspace(-5, 0, 21))
    context_windows_bp: tuple[int, ...] = (5, 9, 17, 33, 65, 129)
    direction_bank_ranks: tuple[int, ...] = (1, 2, 4, 8, 16)
    transport_ranks: tuple[int, ...] = (1, 2, 4, 8, 16, 32)
    ridge_grid: tuple[float, ...] = (1e-6, 1e-4, 1e-2, 1.0)
    label_budgets: tuple[int | str, ...] = (0, 1, 5, 10, 25, 50, 100, 250, "full")
    discovery_chromosome: str = "chr22"
    development_chromosome: str = "chr21"
    locked_chromosome: str = "chr17"
    comparison_models: tuple[str, ...] = ("evo2_7b", "evo2_1b_base")
    required_scale_sites: tuple[tuple[str, tuple[str, ...]], ...] = (
        ("evo2_7b", ("b28_g", "b29_g", "b30_m")),
        ("evo2_1b_base", ("late_g",)),
    )
    required_candidate_methods: int = 2
    required_specificity_families: int = 3
    margins: Exp3Margins = field(default_factory=Exp3Margins)
    continue_after_scientific_non_support: bool = True
    seed: int = 42
    notes: str = ""
    config_sha256: str = field(default="", compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.margins, Exp3Margins):
            object.__setattr__(self, "margins", Exp3Margins(**dict(self.margins)))
        a = np.asarray(self.alpha_doses, dtype=float)
        preregistered = np.logspace(-5, 0, 21)
        if (a.shape != preregistered.shape or np.any(~np.isfinite(a))
                or not np.allclose(a, preregistered, rtol=1e-12, atol=1e-15)):
            raise ValueError(
                "alpha_doses must equal the preregistered 21-point logspace "
                "grid from 1e-5 through 1")
        if len({self.discovery_chromosome, self.development_chromosome,
                self.locked_chromosome}) != 3:
            raise ValueError("discovery, development and locked chromosomes must differ")
        if any(not isinstance(value, str) or not value.strip() for value in (
                self.discovery_chromosome, self.development_chromosome,
                self.locked_chromosome)):
            raise ValueError("chromosome identifiers must be nonblank strings")
        if (len(self.comparison_models) < 2
                or len(set(self.comparison_models)) != len(self.comparison_models)
                or any(not isinstance(v, str) or not v.strip()
                       for v in self.comparison_models)):
            raise ValueError("comparison_models needs at least two distinct names")
        scale_sites = dict(self.required_scale_sites)
        if (len(scale_sites) != len(self.required_scale_sites)
                or set(scale_sites) != set(self.comparison_models)
                or any(not sites or len(set(sites)) != len(sites)
                       or any(not isinstance(site, str) or not site.strip()
                              for site in sites)
                       for sites in scale_sites.values())):
            raise ValueError(
                "required_scale_sites must give non-empty distinct sites for "
                "every comparison model")
        if "evo2_7b" in scale_sites and not {
                "b28_g", "b29_g", "b30_m"}.issubset(scale_sites["evo2_7b"]):
            raise ValueError("the 7B scale panel must include b28_g, b29_g, and b30_m")

        def positive_int_grid(name: str, values: tuple[int, ...]) -> None:
            if (not values or any(type(v) is not int or v <= 0 for v in values)
                    or tuple(sorted(set(values))) != tuple(values)):
                raise ValueError(f"{name} must be a non-empty strictly increasing integer grid")

        positive_int_grid("context_windows_bp", self.context_windows_bp)
        positive_int_grid("direction_bank_ranks", self.direction_bank_ranks)
        positive_int_grid("transport_ranks", self.transport_ranks)
        ridge = np.asarray(self.ridge_grid, dtype=float)
        if (ridge.ndim != 1 or ridge.size == 0 or np.any(~np.isfinite(ridge))
                or np.any(ridge <= 0) or np.any(np.diff(ridge) <= 0)):
            raise ValueError("ridge_grid must be finite, positive, and increasing")
        if (not self.label_budgets or self.label_budgets[-1] != "full"
                or any(type(v) is not int or v < 0
                       for v in self.label_budgets[:-1])
                or tuple(sorted(set(self.label_budgets[:-1])))
                   != tuple(self.label_budgets[:-1])):
            raise ValueError(
                "label_budgets must be increasing non-negative integers followed by 'full'")
        if (type(self.required_candidate_methods) is not int
                or type(self.required_specificity_families) is not int
                or self.required_candidate_methods < 2
                or self.required_specificity_families < 3):
            raise ValueError("EXP3 needs >=2 candidate methods and >=3 control families")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if not isinstance(self.notes, str):
            raise TypeError("notes must be a string")
        if not self.continue_after_scientific_non_support:
            raise ValueError(
                "EXP3 is adaptive: scientific non-support must route, not stop the program")
        observed = self.compute_hash()
        if self.config_sha256 and self.config_sha256 != observed:
            raise RuntimeError("EXP3 config changed after sealing")
        object.__setattr__(self, "config_sha256", observed)

    def compute_hash(self) -> str:
        value = asdict(self)
        value["config_sha256"] = ""
        raw = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(raw.encode()).hexdigest()

    def save(self, path: str | Path) -> None:
        target = Path(path).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_text(json.dumps(asdict(self), indent=2, sort_keys=True))
        temporary.replace(target)

    @staticmethod
    def load(path: str | Path) -> "Exp3Config":
        raw: dict[str, Any] = json.loads(Path(path).expanduser().read_text())
        for key in ("alpha_doses", "context_windows_bp", "direction_bank_ranks",
                    "transport_ranks", "ridge_grid", "label_budgets",
                    "comparison_models"):
            raw[key] = tuple(raw[key])
        if "required_scale_sites" in raw:
            raw["required_scale_sites"] = tuple(
                (str(model), tuple(sites))
                for model, sites in raw["required_scale_sites"])
        if "margins" in raw:
            raw["margins"] = Exp3Margins(**raw["margins"])
        return Exp3Config(**raw)


__all__ = ["Exp3Config", "Exp3Margins"]
