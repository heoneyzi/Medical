from __future__ import annotations

import re

_AMBIGUOUS_AAS = re.compile(r"[UZOB]")


def sanitize_sequence(sequence: str) -> str:
    sequence = sequence.strip().upper()
    sequence = re.sub(r"[^A-Z]", "", sequence)
    return _AMBIGUOUS_AAS.sub("X", sequence)


def prott5_tokenize_sequence(sequence: str) -> str:
    sequence = sanitize_sequence(sequence)
    return " ".join(sequence)

