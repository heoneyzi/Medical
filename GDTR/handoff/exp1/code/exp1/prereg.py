"""Step 6: freeze the analysis plan.

The freeze is mechanical on purpose.  The file is written once, hashed, and the
hash is recorded; afterwards any attempt to overwrite it fails unless
``--force`` is given, and the force is itself logged.  What this buys in review
is the ability to say that the predictions were fixed before the contrasts were
estimated -- the single cheapest defence against the researcher-degrees-of-
freedom objection that interpretability papers attract.

What may be inspected before freezing: sample sizes, variances, pilot norm and
cosine profiles.  What may not: any context contrast, any channel effect size.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from .io import run_context, sha256_file

DEFAULT_PLAN: Dict[str, object] = {
    "experiment": "Evo2 handoff — Experiment 1",
    "model_scope": ["evo2_7b"],
    "frozen_before": "any context contrast or channel effect size is computed",
    "primary_endpoints": {
        "F1": "sign and magnitude of mean |p| and mean n across the onset block",
        "F2": "channel-wise Pearson r with next-base entropy, and channel-wise "
              "within-window paired d against intron",
        "stage2": "per-block KL to the final distribution and top-1 match rate",
        "bridge": "within-window partial effect of the pre-rotation content readout "
                  "on the post-onset decision ratio, controlling for entropy and composition",
    },
    "predictions": {
        "P0_operator": "onset block is hyena-short with index = 0 (mod 7); rotation is "
                       "long-hyena at onset + 2",
        "P1_carrier_entropy": "the +0.57 correlation with next-base entropy is carried by "
                              "the carrier channel a_u, not by the content channel a_v",
        "P2_content_biology": "the annotation contrast that survives full adjustment is "
                              "present in the content channel a_v",
        "P3_donor": "splice donor separates from intron in the content channel even though "
                    "it does not in the pooled readout",
        "P4_peak": "the mid-stack alignment peak is dominated by the carrier channel",
        "F1_hypotheses": {
            "A_dilution": "|p| preserved across the onset while n explodes",
            "B_orthogonal_overwrite": "onset branch update is large and nearly orthogonal "
                                      "to the output frame (|dp| / ||update|| < 0.02)",
            "C_active_cancellation": "onset branch update has a negative projection larger "
                                     "in magnitude than the pre-onset aligned component",
            "D_rotation": "the rotation block turns the existing aligned component",
            "E_new_write": "the rotation block writes >90% of the final aligned component",
        },
    },
    "quadrants": {
        "definition": "within-window medians of the pre-rotation content readout and the "
                      "post-onset decision ratio",
        "pre_hi/post_hi": "locally determined positions (repeat, poly-A, strong motifs) — many expected",
        "pre_hi/post_lo": "local grammar clear but output context dependent — splice donor predicted here",
        "pre_lo/post_hi": "long integration but simple answer — few expected",
        "pre_lo/post_lo": "hard positions — many expected",
    },
    "equivalence_margins": {
        "next_base_nll_relative_increase": 0.01,
        "auroc_decrease": 0.01,
        "teacher_kl_increase_nats": 0.02,
    },
    "statistics": {
        "resampling_unit": "window",
        "estimand": "within-window paired standardised difference vs intron",
        "bootstrap_B": 2000,
        "bootstrap_seed": 42,
        "min_positions_per_class_per_window": 30,
        "position_level_p_values": "none reported",
        "multiplicity": "hierarchical FDR over context x block x channel",
    },
    "controls": [
        "weight-shuffled model (within-tensor permutation, seed recorded)",
        "within-window label permutation",
        "repeat vs non-repeat inside intron as the size yardstick",
        "three references: raw h29, RMSNorm-both, h_norm",
        "input variants: real / random / polyA / dinucleotide shuffle",
    ],
    "holdout": {
        "non_human": "reserved, not opened in Experiment 1",
    },
    "stopping_rules": {
        "reconstruction_fidelity": "abort if x_in + mixer + mlp does not reconstruct the block "
                                   "output to better than 1e-2 relative error",
        "channel_identity": "abort if a != alpha*a_u + beta*a_v to better than 1e-6",
    },
}


def plan_path(root: Path) -> Path:
    return Path(root) / "prereg" / "experiment1_plan.yaml"


def freeze(root: Path, plan: Optional[Dict] = None, force: bool = False) -> Dict[str, object]:
    p = plan_path(root)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists() and not force:
        raise FileExistsError(
            f"{p} already frozen; pass --force only if you intend to record an amendment"
        )
    payload = dict(plan or DEFAULT_PLAN)
    payload["_meta"] = run_context()
    if p.exists() and force:
        payload["_meta"]["amends_sha256"] = sha256_file(p)
    p.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True))
    digest = sha256_file(p)
    (p.parent / "experiment1_plan.sha256").write_text(digest + "\n")
    return dict(path=str(p), sha256=digest, amended=bool(p.exists() and force))


def verify(root: Path) -> Dict[str, object]:
    p = plan_path(root)
    if not p.exists():
        return dict(ok=False, reason="no frozen plan")
    recorded = (p.parent / "experiment1_plan.sha256")
    digest = sha256_file(p)
    return dict(
        ok=recorded.exists() and recorded.read_text().strip() == digest,
        sha256=digest,
        recorded=recorded.read_text().strip() if recorded.exists() else None,
    )
