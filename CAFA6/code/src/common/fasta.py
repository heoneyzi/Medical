from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path


def extract_protein_id(header: str) -> str:
    header = header.strip()
    if header.startswith(">"):
        header = header[1:]

    first_token = header.split(maxsplit=1)[0]
    if "|" in first_token:
        parts = first_token.split("|")
        if len(parts) >= 2 and parts[1]:
            return parts[1]
    return first_token


def read_fasta(path: str | Path) -> Iterator[tuple[str, str]]:
    path = Path(path)
    protein_id: str | None = None
    chunks: list[str] = []

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue

            if line.startswith(">"):
                if protein_id is not None:
                    yield protein_id, "".join(chunks)
                protein_id = extract_protein_id(line)
                chunks = []
                continue

            chunks.append(line)

    if protein_id is not None:
        yield protein_id, "".join(chunks)

