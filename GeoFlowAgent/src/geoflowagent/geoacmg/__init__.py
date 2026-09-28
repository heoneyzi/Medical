"""GeoACMG: a benchmark and analysis frame for frozen-representation tool agents.

The package is organised so that the parts which can be wrong independently are
testable independently:

``cited``        constants that carry their provenance; a threshold cannot be
                 introduced without a source
``evidence``     ACMG criteria, strength modifiers and the Tavtigian point scale
``clingen``      the Evidence Repository corpus, parsed fail-loud
``toolmap``      which query decides which criterion, and which criteria no API
                 can decide
``tasks``        curated variants to tasks, with the sufficiency rule that makes
                 "commit without looking" impossible
``diagnostics``  whether the benchmark contains a planning problem at all
``estimators``   intervals and tests, resampling genes rather than rows
``inference``    estimators to findings
``claims``       claims, findings, verdicts, preregistration

The dependency direction is strictly downward in that list. Nothing in
``evidence`` knows about a claim; nothing in ``claims`` knows about genomics.
"""

from geoflowagent.geoacmg.cited import Cited, guideline, measured
from geoflowagent.geoacmg.claims import (
    Claim,
    Evidence,
    Finding,
    Outcome,
    Preregistration,
    Role,
    Verdict,
    adjudicate,
)

__all__ = [
    "Claim",
    "Cited",
    "Evidence",
    "Finding",
    "Outcome",
    "Preregistration",
    "Role",
    "Verdict",
    "adjudicate",
    "guideline",
    "measured",
]
