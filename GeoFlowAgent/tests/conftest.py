from __future__ import annotations

from pathlib import Path

import pytest

from geoflowagent.data.preprocess import prepare_dataset
from geoflowagent.embeddings.cache import build_embedding_cache

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SMOKE_RAW_DIR = REPOSITORY_ROOT / "data" / "raw" / "smoke"
SMOKE_CONFIG = REPOSITORY_ROOT / "configs" / "smoke.yaml"


@pytest.fixture(scope="session")
def repository_root() -> Path:
    return REPOSITORY_ROOT


@pytest.fixture(scope="session")
def smoke_raw_dir() -> Path:
    return SMOKE_RAW_DIR


@pytest.fixture(scope="session")
def processed_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    output = tmp_path_factory.mktemp("processed")
    prepare_dataset(SMOKE_RAW_DIR, output, seed=17, max_search_depth=12)
    return output


@pytest.fixture(scope="session")
def cache_dir(
    tmp_path_factory: pytest.TempPathFactory,
    processed_dir: Path,
) -> Path:
    output = tmp_path_factory.mktemp("cache")
    build_embedding_cache(processed_dir, output, SMOKE_CONFIG)
    return output
