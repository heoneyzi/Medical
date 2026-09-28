#!/usr/bin/env python3
"""Write machine-readable eligibility decisions for the strict zero-shot run."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--state", default="models/SE-100M/model.safetensors")
    p.add_argument("--stack", default="models/Stack-Large/bc_large.ckpt")
    args = p.parse_args()
    rows = [
        {
            "method": "no_effect", "checkpoint_present": True,
            "target_specific_transition_head": False, "supports_HAR_targets": False,
            "GSE270828_training_exclusion_proven": True, "strict_score_eligible": True,
            "status": "eligible_negative_control",
            "reason": "Uses only evaluation-context NTC cells; has no learned parameters.",
        },
        {
            "method": "STATE-SE-100M", "checkpoint_present": Path(args.state).is_file(),
            "target_specific_transition_head": False, "supports_HAR_targets": False,
            "GSE270828_training_exclusion_proven": False, "strict_score_eligible": False,
            "status": "excluded_encoder_only_and_training_overlap_unknown",
            "reason": "The local artifact is a cell-state embedding checkpoint, not a perturbation transition predictor; its public training manifest does not certify GSE270828 exclusion.",
        },
        {
            "method": "STACK-Large", "checkpoint_present": Path(args.stack).is_file(),
            "target_specific_transition_head": True, "supports_HAR_targets": False,
            "GSE270828_training_exclusion_proven": False, "strict_score_eligible": False,
            "status": "excluded_requires_response_prompt_and_training_overlap_unknown",
            "reason": "In-context generation requires perturbed example data and has no direct HAR-ID intervention interface; scBaseCount training membership cannot be audited from the published manifest.",
        },
        {
            "method": "Jiang24_LOCO_GWPS", "checkpoint_present": True,
            "target_specific_transition_head": True, "supports_HAR_targets": False,
            "GSE270828_training_exclusion_proven": True, "strict_score_eligible": False,
            "status": "excluded_same_dataset_response_reference",
            "reason": "Uses perturbation responses from other Jiang24 cell lines and therefore violates the strengthened complete-zero-shot contract.",
        },
    ]
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
