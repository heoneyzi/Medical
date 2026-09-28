from types import SimpleNamespace

import pytest
import torch

from geoflowagent.embeddings.huggingface import _last_hidden_state


def test_last_hidden_state_accepts_modern_model_output() -> None:
    expected = torch.randn(2, 3, 4)
    assert _last_hidden_state(SimpleNamespace(last_hidden_state=expected)) is expected


def test_last_hidden_state_accepts_legacy_tuple() -> None:
    expected = torch.randn(2, 3, 4)
    assert _last_hidden_state((expected, None)) is expected


def test_last_hidden_state_rejects_unknown_output() -> None:
    with pytest.raises(TypeError, match="last hidden-state tensor"):
        _last_hidden_state({"hidden_states": []})
