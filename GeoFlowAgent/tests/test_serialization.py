from __future__ import annotations

import numpy as np
import pytest

from geoflowagent.data.serialize import serialize_entities, serialize_sequence
from geoflowagent.embeddings.base import FrozenEncoder, encode_nonempty
from geoflowagent.embeddings.hashing import HashEncoder


def test_entity_serializer_selects_biomedical_entities_not_build_metadata() -> None:
    text = serialize_entities(
        {
            "assembly": "GRCh38",
            "raw_variant": "NM_SYNTH.1:c.1A>G",
            "hpo_terms": ["HP:SYNTH001"],
            "gene_symbol": "SYNTH_GENE",
            "coordinate_system": "one_based_closed",
            "sequence_context": "ACGT",
        }
    )

    assert "NM_SYNTH.1:c.1A>G" in text
    assert "HP:SYNTH001" in text
    assert "SYNTH_GENE" in text
    assert "GRCh38" not in text
    assert "one_based_closed" not in text
    assert "ACGT" not in text


def test_sequence_serializer_normalizes_iupac_and_rejects_metadata() -> None:
    assert serialize_sequence({"sequence_context": "acgt n\n"}) == "ACGTN"
    with pytest.raises(ValueError, match="IUPAC DNA"):
        serialize_sequence({"sequence_context": "GRCh38"})


def test_missing_modality_is_an_exact_zero_vector() -> None:
    vectors = encode_nonempty(HashEncoder(dim=16), ["", "SYNTH_GENE"], batch_size=2)

    np.testing.assert_array_equal(vectors[0], np.zeros(16, dtype=np.float32))
    assert np.linalg.norm(vectors[1]) > 0


class _CountingEncoder(FrozenEncoder):
    def __init__(self) -> None:
        self.seen: list[str] = []

    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        del batch_size
        self.seen.extend(texts)
        return np.asarray([[len(text), text.count("a")] for text in texts], dtype=np.float32)

    @property
    def metadata(self) -> dict[str, object]:
        return {"dim": 2}


def test_nonempty_encoding_deduplicates_exact_repeated_text() -> None:
    encoder = _CountingEncoder()
    vectors = encode_nonempty(encoder, ["alpha", "", "beta", "alpha"], batch_size=2)

    assert encoder.seen == ["alpha", "beta"]
    np.testing.assert_array_equal(vectors[0], vectors[3])
    np.testing.assert_array_equal(vectors[1], np.zeros(2, dtype=np.float32))
