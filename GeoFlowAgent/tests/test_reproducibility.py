from __future__ import annotations

import re

from geoflowagent import __version__
from geoflowagent.utils.reproducibility import collect_run_provenance


def test_run_provenance_binds_source_lock_and_runtime(repository_root) -> None:
    provenance = collect_run_provenance(
        repository_root / "configs" / "smoke.yaml",
        device="cpu",
    )

    assert provenance["package_version"] == __version__
    assert re.fullmatch(r"[0-9a-f]{64}", provenance["source_tree_sha256"])
    assert re.fullmatch(r"[0-9a-f]{64}", provenance["dependency_lock_sha256"])
    assert re.fullmatch(r"[0-9a-f]{64}", provenance["pyproject_sha256"])
    assert provenance["python_version"]
    assert provenance["torch_version"]
    assert provenance["numpy_version"]
    assert provenance["device"] == "cpu"
