from __future__ import annotations

import hashlib
import os
import platform
import random
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch

from geoflowagent import __version__
from geoflowagent.utils.io import read_yaml, resolve_path, sha256_file


def seed_everything(seed: int, deterministic: bool = True) -> None:
    # cuBLAS reads this before creating its first workspace/handle, so set it
    # before any CUDA seeding or tensor operation in this process.
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        # TransformerEncoder otherwise selects a memory-efficient CUDA SDP
        # backward kernel that PyTorch explicitly marks non-deterministic.
        if torch.cuda.is_available() and hasattr(torch.backends, "cuda"):
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_math_sdp(True)


def choose_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: to_device(item, device) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        converted = [to_device(item, device) for item in value]
        return type(value)(converted)
    return value


def _source_tree_sha256(source_root: Path) -> str | None:
    if not source_root.is_dir():
        return None
    digest = hashlib.sha256()
    paths = sorted(path for path in source_root.rglob("*.py") if path.is_file())
    if not paths:
        return None
    for path in paths:
        digest.update(path.relative_to(source_root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _git_state(project_root: Path) -> tuple[str | None, bool | None]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=False,
            capture_output=True,
            text=True,
            timeout=3,
        )
        if commit.returncode != 0:
            return None, None
        status = subprocess.run(
            ["git", "status", "--porcelain", "--", "."],
            cwd=project_root,
            check=False,
            capture_output=True,
            text=True,
            timeout=3,
        )
        dirty = bool(status.stdout.strip()) if status.returncode == 0 else None
        return commit.stdout.strip(), dirty
    except (OSError, subprocess.SubprocessError):
        return None, None


def collect_run_provenance(
    config_path: str | Path,
    *,
    device: torch.device | str,
) -> dict[str, Any]:
    """Capture code, dependency, interpreter, and device identity without host paths."""

    config_path = Path(config_path).resolve()
    config = read_yaml(config_path)
    project_root = resolve_path(config_path.parent, config.get("project_root", ".")).resolve()
    source_root = project_root / "src" / "geoflowagent"
    if not source_root.is_dir():
        source_root = Path(__file__).resolve().parents[1]
    lockfile = project_root / "uv.lock"
    pyproject = project_root / "pyproject.toml"
    git_commit, git_dirty = _git_state(project_root)
    return {
        "package_version": __version__,
        "source_tree_sha256": _source_tree_sha256(source_root),
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "dependency_lock_sha256": sha256_file(lockfile) if lockfile.is_file() else None,
        "pyproject_sha256": sha256_file(pyproject) if pyproject.is_file() else None,
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "numpy_version": str(np.__version__),
        "platform": platform.system(),
        "machine": platform.machine(),
        "device": str(device),
    }
