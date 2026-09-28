"""CPU 병목 타파.

저널 §21 의 진단: **병목 셋이 전부 "GPU 가 노는데 CPU 한 코어가 병목"이었다.**

* 조합마다 프로세스를 띄워 store 를 15번 적재  → 시드별 배치로 3회
* flow 평가의 전수 탐색                          → root-only
* 이미 코드에 있는 축소 스위치를 못 봄           → 실행 전 스위치 탐색

그리고 가장 아픈 것: flow 평가가 상태마다 "계획 4샘플 × nfe 12 + 재계획 롤아웃
3종"을 파이썬 루프로 전개해 **128코어 장비에서 CPU 1.2코어에 묶였다.**
전수 평가는 8.1시간 뒤에도 미완, root-only 로도 3시간 초과.

이 모듈이 주는 것:

1. ``worker_budget`` — 동시 실행 수를 ``nproc`` 가 아니라 **cgroup 메모리 한도**로
   정한다.  ``free`` 는 호스트를 보여주므로 컨테이너 안에서는 거짓말이다.
   여유 판단은 페이지 캐시를 뺀 **anon(rss)** 기준으로 한다.
2. ``batched`` — 상태 단위 파이썬 루프를 배치 텐서 연산으로 바꾸는 헬퍼.
   nfe 12 스텝을 상태마다 도는 대신 B개 상태를 쌓아 한 번에 12스텝 돈다.
   이것이 1.2코어를 GPU 로 옮기는 실제 수단이다.
3. ``parallel_map`` — 남은 진짜 파이썬 루프를 프로세스로 펼친다.
4. ``Progress`` — 3.5시간 침묵을 금지한다. 모든 긴 루프는 ETA 를 찍는다.
"""

from __future__ import annotations

import os
import sys
import time
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

T = TypeVar("T")
S = TypeVar("S")


# ----------------------------------------------------------------- cgroup 인식


def _read_int(path: str) -> int | None:
    try:
        raw = Path(path).read_text().strip()
    except OSError:
        return None
    if raw in {"max", ""}:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def cgroup_memory_limit_bytes() -> int | None:
    """컨테이너의 메모리 한도. v2 우선, 없으면 v1."""

    for path in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        value = _read_int(path)
        # v1 은 한도 없음을 거대한 수로 표현한다
        if value is not None and value < (1 << 62):
            return value
    return None


def cgroup_anon_bytes() -> int | None:
    """현재 anon(rss) 사용량.

    페이지 캐시는 회수 가능하므로 여유 계산에서 빼야 한다.  ``memory.current``
    를 그대로 쓰면 캐시까지 '사용 중'으로 보여 동시성을 과소평가한다.
    """

    for path in ("/sys/fs/cgroup/memory.stat", "/sys/fs/cgroup/memory/memory.stat"):
        try:
            for line in Path(path).read_text().splitlines():
                key, _, value = line.partition(" ")
                if key in {"anon", "total_rss"}:
                    return int(value)
        except (OSError, ValueError):
            continue
    return None


@dataclass(frozen=True)
class WorkerBudget:
    workers: int
    limit_bytes: int | None
    anon_bytes: int | None
    per_worker_bytes: int
    reason: str

    def to_payload(self) -> dict[str, Any]:
        gib = 1024**3
        return {
            "workers": self.workers,
            "cgroup_limit_gib": round(self.limit_bytes / gib, 1) if self.limit_bytes else None,
            "anon_in_use_gib": round(self.anon_bytes / gib, 1) if self.anon_bytes else None,
            "per_worker_gib": round(self.per_worker_bytes / gib, 2),
            "reason": self.reason,
            "note": "nproc 이 아니라 cgroup 메모리 한도가 동시성을 정한다 (저널 §21)",
        }


def worker_budget(
    *, per_worker_bytes: int, ceiling: int | None = None, reserve_bytes: int = 4 * 1024**3
) -> WorkerBudget:
    """동시 실행 수를 메모리에서 유도한다.

    ``per_worker_bytes`` 는 실제 측정값을 넣는다.  추정이 필요하면 워커 하나를
    먼저 돌려 RSS 를 재고 그 값을 넣는 것이 맞다 — 여기서 상수를 지어내지 않는다.
    """

    cpu = os.cpu_count() or 1
    hard_ceiling = ceiling if ceiling is not None else cpu
    limit = cgroup_memory_limit_bytes()
    anon = cgroup_anon_bytes()

    if limit is None:
        workers = max(1, min(hard_ceiling, cpu))
        return WorkerBudget(workers, None, anon, per_worker_bytes, "cgroup 한도 없음 — CPU 수로 제한")

    available = limit - (anon or 0) - reserve_bytes
    by_memory = max(1, int(available // max(per_worker_bytes, 1)))
    workers = max(1, min(hard_ceiling, by_memory, cpu))
    return WorkerBudget(
        workers,
        limit,
        anon,
        per_worker_bytes,
        f"메모리 기준 {by_memory}, CPU {cpu}, 상한 {hard_ceiling} → {workers}",
    )


def measure_rss_bytes() -> int:
    """현재 프로세스의 RSS. 워커 하나를 돌려 per_worker_bytes 를 재는 데 쓴다."""

    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return 0


# --------------------------------------------------------------------- progress


@dataclass
class Progress:
    """3.5시간 침묵 금지.

    저널 §21: "스크립트는 진행 상황을 출력한다. flow 가 3.5시간 침묵해 진행률을
    알 수 없었다."  긴 루프는 전부 이것을 통과시킨다.
    """

    total: int
    label: str
    every_seconds: float = 30.0
    stream: Any = sys.stderr
    started: float = field(default_factory=time.time)
    done: int = 0
    _last: float = field(default=0.0, init=False)

    def tick(self, count: int = 1) -> None:
        self.done += count
        now = time.time()
        if now - self._last < self.every_seconds and self.done < self.total:
            return
        self._last = now
        elapsed = now - self.started
        rate = self.done / elapsed if elapsed > 0 else 0.0
        remaining = (self.total - self.done) / rate if rate > 0 else float("inf")
        print(
            f"[{time.strftime('%H:%M:%S')}] {self.label}: {self.done}/{self.total} "
            f"({100.0 * self.done / max(self.total, 1):.1f}%) "
            f"elapsed {elapsed / 60:.1f}m  eta {remaining / 60:.1f}m",
            file=self.stream,
            flush=True,
        )

    def eta_seconds(self) -> float:
        elapsed = time.time() - self.started
        if self.done <= 0:
            return float("inf")
        return (self.total - self.done) * (elapsed / self.done)


# ---------------------------------------------------------------------- batching


def batched(items: Sequence[T], size: int) -> Iterator[Sequence[T]]:
    """상태 단위 루프를 배치로 바꾼다.

    flow 평가가 1.2코어에 묶인 이유가 이것이다: 상태마다 4샘플 × nfe 12 를
    파이썬으로 돌았다.  B개 상태를 쌓아 12 스텝을 **한 번에** 돌면 같은 연산이
    GPU 커널 12번이 된다.
    """

    if size <= 0:
        raise ValueError("배치 크기는 1 이상이어야 한다")
    for start in range(0, len(items), size):
        yield items[start : start + size]


def autotune_batch(
    probe: Callable[[int], float],
    *,
    candidates: Sequence[int] = (32, 64, 128, 256, 512, 1024),
    label: str = "batch",
) -> int:
    """배치 크기를 상수로 고르지 않고 측정해서 정한다.

    ``probe(size)`` 는 그 크기로 한 배치를 돌리는 데 걸린 초를 돌려준다.
    항목당 시간이 더 이상 줄지 않는 지점에서 멈춘다.  OOM 은 호출자가 잡아
    ``float('inf')`` 를 돌려주면 된다.
    """

    best_size = candidates[0]
    best_per_item = float("inf")
    for size in candidates:
        try:
            seconds = probe(size)
        except Exception:  # noqa: BLE001 - OOM 등은 그 크기를 포기하는 신호
            break
        per_item = seconds / size
        print(f"  {label} autotune: size={size} {per_item * 1e3:.3f} ms/item", file=sys.stderr)
        if per_item < best_per_item * 0.95:
            best_per_item, best_size = per_item, size
        else:
            break
    return best_size


# --------------------------------------------------------------------- parallel


def parallel_map(
    function: Callable[[T], S],
    items: Sequence[T],
    *,
    budget: WorkerBudget,
    label: str = "map",
    every_seconds: float = 30.0,
) -> list[S]:
    """남은 진짜 파이썬 루프를 프로세스로 펼친다.

    워커가 1이면 프로세스를 띄우지 않는다 — 작은 작업에서 fork 비용이 이득을
    넘기는 경우가 흔하다.
    """

    if not items:
        return []
    progress = Progress(total=len(items), label=label, every_seconds=every_seconds)
    if budget.workers <= 1:
        out: list[S] = []
        for item in items:
            out.append(function(item))
            progress.tick()
        return out

    results: list[Any] = [None] * len(items)
    with ProcessPoolExecutor(max_workers=budget.workers) as pool:
        futures = {pool.submit(function, item): index for index, item in enumerate(items)}
        for future in as_completed(futures):
            results[futures[future]] = future.result()
            progress.tick()
    return results  # type: ignore[return-value]


def find_reduction_switches(source_root: str | Path) -> list[dict[str, Any]]:
    """코드에 이미 있는 축소 스위치를 실행 전에 찾아 보여준다.

    저널 §21: ``GEOFLOW_STATE_FLOW_REPORT_TRAIN=0`` 이 이미 있었는데 못 봐서
    표본 내 train 롤아웃 1,753 루트를 헛돌았다.  긴 작업 전에 자원 프로파일만
    보지 말고 **코드가 제공하는 축소 스위치를 먼저 찾는다.**
    """

    import re

    pattern = re.compile(r"""environ(?:\.get)?\(?\s*["']([A-Z0-9_]+)["']""")
    hits: list[dict[str, Any]] = []
    for path in Path(source_root).rglob("*.py"):
        try:
            text = path.read_text("utf-8", errors="replace")
        except OSError:
            continue
        for number, line in enumerate(text.splitlines(), start=1):
            for name in pattern.findall(line):
                if any(token in name for token in ("REPORT", "EVAL", "LIMIT", "MAX", "SKIP", "ONLY", "FAST", "SAMPLE")):
                    hits.append(
                        {
                            "env": name,
                            "file": str(path),
                            "line": number,
                            "source": line.strip()[:160],
                            "current": os.environ.get(name),
                        }
                    )
    return hits
