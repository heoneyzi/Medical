"""저장 규율.

2026-09-19 의 15시간 실행은 산출물을 전부 ``/tmp`` (컨테이너 overlay) 에만 써서
컨테이너 재시작과 함께 16GB 를 잃었다.  1차 실험에는 이미 45초 주기 미러 데몬이
있었는데 2차가 그것을 ``artifacts/acmg`` 로 가리키지 않았다.  같은 사고를 사람의
기억에 맡기지 않기 위해, 이 모듈을 통과하지 않고는 결과 파일을 쓸 수 없게 한다.

규칙 셋:

* 휘발성 파일시스템(overlay/tmpfs)에 결과를 쓰려 하면 거부한다.
* 긴 작업 시작 전에 영구 경로의 여유 공간을 확인하고, 모자라면 시작을 거부한다.
* finding 은 생성 즉시 영구 경로로 write-through 한다. 마지막에 모아 백업하지 않는다.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

VOLATILE_FILESYSTEMS = frozenset({"overlay", "tmpfs", "ramfs", "aufs"})


class VolatileStorageError(RuntimeError):
    """결과를 휘발성 경로에 쓰려 했다."""


class InsufficientDurableSpace(RuntimeError):
    """영구 경로의 여유 공간이 선언한 필요량보다 적다."""


def filesystem_type(path: str | Path) -> str:
    """``path`` 를 담고 있는 파일시스템 종류. 알 수 없으면 빈 문자열."""

    target = Path(path)
    while not target.exists() and target != target.parent:
        target = target.parent
    try:
        out = subprocess.run(
            ["df", "-PT", str(target)], capture_output=True, text=True, check=True
        ).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError):
        return ""
    if len(out) < 2:
        return ""
    fields = out[1].split()
    return fields[1] if len(fields) > 1 else ""


def free_megabytes(path: str | Path) -> int:
    target = Path(path)
    while not target.exists() and target != target.parent:
        target = target.parent
    usage = shutil.disk_usage(target)
    return int(usage.free // (1024 * 1024))


@dataclass(frozen=True)
class DurableRoot:
    """결과가 살아남을 경로.

    ``require`` 를 통과하지 못하면 어떤 스테이지도 시작하지 않는다.  실패를
    조용히 넘기지 않는 것이 이 클래스의 유일한 목적이다.
    """

    path: Path
    min_free_mb: int = 2048
    allow_volatile: bool = False

    @classmethod
    def from_env(cls, *, min_free_mb: int = 2048) -> DurableRoot:
        raw = os.environ.get("GEOACMG_DURABLE")
        if not raw:
            raise RuntimeError(
                "GEOACMG_DURABLE 이 설정되지 않았다. 영구 볼륨 경로를 지정하고 "
                "sync_geoacmg_resume.sh 를 먼저 띄울 것."
            )
        allow = os.environ.get("GEOACMG_ALLOW_VOLATILE") == "1"
        return cls(Path(raw), min_free_mb=min_free_mb, allow_volatile=allow)

    def require(self) -> DurableRoot:
        self.path.mkdir(parents=True, exist_ok=True)
        fs = filesystem_type(self.path)
        if fs in VOLATILE_FILESYSTEMS and not self.allow_volatile:
            raise VolatileStorageError(
                f"{self.path} 는 {fs} 파일시스템이다. 컨테이너가 재시작되면 사라진다.\n"
                "영구 볼륨을 마운트해 GEOACMG_DURABLE 을 그쪽으로 지정할 것.\n"
                "정말로 휘발성 경로를 쓰겠다면 GEOACMG_ALLOW_VOLATILE=1 (권장하지 않음)."
            )
        free = free_megabytes(self.path)
        if free < self.min_free_mb:
            raise InsufficientDurableSpace(
                f"{self.path} 여유 {free}MB < 필요 {self.min_free_mb}MB. "
                "공간을 확보하거나 다른 영구 경로를 지정할 것."
            )
        for sub in ("findings", "manifests", "pairs", "logs", "reports"):
            (self.path / sub).mkdir(parents=True, exist_ok=True)
        return self

    def mirror_running(self) -> bool:
        """미러 데몬이 락을 잡고 있는가."""

        lock = self.path / ".sync.lock"
        if not lock.exists():
            return False
        try:
            out = subprocess.run(
                ["fuser", str(lock)], capture_output=True, text=True, check=False
            )
            return bool(out.stdout.strip())
        except OSError:
            return False

    # ------------------------------------------------------------ write-through

    def write_json(self, relative: str, payload: Any) -> Path:
        target = self.path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + ".part")
        tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), "utf-8")
        tmp.replace(target)
        return target

    def write_pairs(self, relative: str, **arrays: Any) -> Path:
        """per-pair 원시 벡터를 남긴다.

        저널 21절의 교훈: per-pair 를 남겨두면 집계 단위나 가중치를 바꿔도
        재학습이 필요 없다.  모든 finding 은 자기 per-pair 파일을 함께 쓴다.
        """

        import numpy as np

        target = (self.path / "pairs" / relative).with_suffix(".npz")
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(target.name + ".part")
        # 파일 핸들로 넘긴다: 경로로 넘기면 numpy 가 .npz 를 덧붙여
        # x.npz.part.npz 가 만들어지고 원자적 교체가 깨진다.
        with tmp.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        tmp.replace(target)
        return target

    def append_log(self, record: Mapping[str, Any]) -> None:
        line = json.dumps(record, ensure_ascii=False, default=str)
        with (self.path / "logs" / "finish.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
