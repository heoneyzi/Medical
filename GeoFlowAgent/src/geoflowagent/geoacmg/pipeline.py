"""One command that runs the whole study, in order, resumably.

Each stage declares what it reads, what it writes, and whether it needs a GPU.
The runner skips a stage whose outputs already exist unless ``--force`` is given,
so a long run on a remote box survives a dropped connection without starting
over.

Stages that would open the sealed test split refuse to run unless the study says
so explicitly.  That is the one place the pipeline is deliberately inconvenient.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Stage:
    key: str
    title: str
    outputs: tuple[str, ...]
    run: Callable[[Context], dict[str, Any]]
    needs_gpu: bool = False
    optional: bool = False
    note: str = ""


@dataclass
class Context:
    root: Path
    corpus: Path
    config: Path
    python: str = sys.executable
    max_tasks_per_gene: int | None = 100
    split_by: str = "gene"
    include_test: bool = False
    force: bool = False
    log: list[dict[str, Any]] = field(default_factory=list)

    def path(self, *parts: str) -> Path:
        return self.root.joinpath(*parts)

    def shell(self, args: Sequence[str], *, stage: str) -> dict[str, Any]:
        started = time.time()
        completed = subprocess.run(
            list(args), capture_output=True, text=True, check=False
        )
        record = {
            "stage": stage,
            "command": " ".join(args),
            "returncode": completed.returncode,
            "seconds": round(time.time() - started, 1),
            "stdout_tail": completed.stdout[-2000:],
            "stderr_tail": completed.stderr[-2000:],
        }
        if completed.returncode != 0:
            raise RuntimeError(
                f"stage {stage} failed ({completed.returncode})\n"
                f"{completed.stderr[-2000:]}"
            )
        return record


# ------------------------------------------------------------------- stages


def _stage_prereg(ctx: Context) -> dict[str, Any]:
    return ctx.shell(
        [ctx.python, "-m", "geoflowagent.geoacmg", "prereg",
         "--output", str(ctx.path("preregistration.json"))],
        stage="prereg",
    )


def _stage_corpus(ctx: Context) -> dict[str, Any]:
    args = [ctx.python, "-m", "geoflowagent.geoacmg", "build",
            "--corpus", str(ctx.corpus),
            "--output-dir", str(ctx.path("corpus")),
            "--split-by", ctx.split_by]
    if ctx.max_tasks_per_gene:
        args += ["--max-tasks-per-gene", str(ctx.max_tasks_per_gene)]
    return ctx.shell(args, stage="corpus")


def _stage_diagnose(ctx: Context) -> dict[str, Any]:
    args = [ctx.python, "-m", "geoflowagent.geoacmg", "diagnose",
            "--corpus", str(ctx.corpus),
            "--output", str(ctx.path("diagnostics.json")),
            "--findings-out", str(ctx.path("findings", "b0.jsonl"))]
    if ctx.max_tasks_per_gene:
        args += ["--max-tasks-per-gene", str(ctx.max_tasks_per_gene)]
    return ctx.shell(args, stage="diagnose")


def _stage_oracle(ctx: Context) -> dict[str, Any]:
    """Expand the exact search graph and write V*, Q*, regret and optimal sets.

    This is the repository's own ``prepare-search``; the corpus was written in the
    format it already consumes, so no new oracle was implemented.  Budget roughly
    one second per task, single threaded, and several gigabytes of progress files.
    """

    return ctx.shell(
        [ctx.python, "-m", "geoflowagent.cli", "prepare-search", "--config", str(ctx.config)]
        + (["--include-test", "--test-access-reason", "final evaluation"] if ctx.include_test else []),
        stage="oracle",
    )


def _stage_embed(ctx: Context) -> dict[str, Any]:
    """Encode every text once with the frozen encoders. The only real GPU stage."""

    return ctx.shell(
        [ctx.python, "-m", "geoflowagent.cli", "embed", "--config", str(ctx.config)],
        stage="embed",
    )


def _stage_probe(ctx: Context) -> dict[str, Any]:
    return ctx.shell(
        [ctx.python, "-m", "geoflowagent.geoacmg", "probe",
         "--package", str(ctx.path("corpus")),
         "--processed", str(ctx.path("processed")),
         "--cache", str(ctx.path("cache")),
         "--output", str(ctx.path("findings", "rq1.jsonl"))],
        stage="probe",
    )


def _stage_geometry(ctx: Context) -> dict[str, Any]:
    return ctx.shell(
        [ctx.python, "-m", "geoflowagent.cli", "compare-value-geometries",
         "--config", str(ctx.config)],
        stage="geometry",
    )


def _stage_train_value(ctx: Context) -> dict[str, Any]:
    return ctx.shell(
        [ctx.python, "-m", "geoflowagent.cli", "train-value", "--config", str(ctx.config)],
        stage="train_value",
    )


def _stage_train_flow(ctx: Context) -> dict[str, Any]:
    return ctx.shell(
        [ctx.python, "-m", "geoflowagent.cli", "train-state-flow", "--config", str(ctx.config)],
        stage="train_flow",
    )


def _stage_ladder(ctx: Context) -> dict[str, Any]:
    return ctx.shell(
        [ctx.python, "-m", "geoflowagent.geoacmg", "ladder",
         "--package", str(ctx.path("corpus")),
         "--processed", str(ctx.path("processed")),
         "--output", str(ctx.path("findings", "rq2.jsonl"))],
        stage="ladder",
    )


def _stage_report(ctx: Context) -> dict[str, Any]:
    return ctx.shell(
        [ctx.python, "-m", "geoflowagent.geoacmg", "report",
         "--prereg", str(ctx.path("preregistration.json")),
         "--findings-dir", str(ctx.path("findings")),
         "--card", str(ctx.path("corpus", "manifest.json")),
         "--output", str(ctx.path("REPORT.md"))],
        stage="report",
    )


STAGES: tuple[Stage, ...] = (
    Stage("prereg", "분석 계획 동결", ("preregistration.json",), _stage_prereg),
    Stage("corpus", "ClinGen -> task corpus", ("corpus/manifest.json",), _stage_corpus),
    Stage("diagnose", "벤치마크 진단", ("diagnostics.json", "findings/b0.jsonl"), _stage_diagnose,
          note="여기서 단일 도구군 천장이 전체와 같으면 멈춘다"),
    Stage("oracle", "정확 탐색 그래프와 V*/Q*", ("processed/manifest.json",), _stage_oracle,
          note="CPU. task당 약 1초"),
    Stage("embed", "동결 인코더 캐시", ("cache/manifest.json",), _stage_embed,
          needs_gpu=True, note="추론 1회. 이후 모든 단계가 재사용"),
    Stage("probe", "RQ1 — 표현에 실행 구조가 있는가", ("findings/rq1.jsonl",), _stage_probe),
    Stage("geometry", "거리 함수 비교", ("checkpoints/geometry_comparison",), _stage_geometry,
          needs_gpu=True, optional=True),
    Stage("train_value", "거리 헤드 학습", ("checkpoints/value",), _stage_train_value,
          needs_gpu=True),
    Stage("train_flow", "flow 계획기 학습", ("checkpoints/state_flow",), _stage_train_flow,
          needs_gpu=True),
    Stage("ladder", "RQ2 — 계획 생성 대 탐욕", ("findings/rq2.jsonl",), _stage_ladder),
    Stage("report", "주장 -> 판정", ("REPORT.md",), _stage_report),
)


def run(ctx: Context, *, only: Sequence[str] | None = None, skip: Sequence[str] = ()) -> dict[str, Any]:
    """Run the stages in order, skipping ones whose outputs are already there."""

    selected = [
        stage
        for stage in STAGES
        if (only is None or stage.key in only) and stage.key not in skip
    ]
    results: list[dict[str, Any]] = []

    def persist() -> None:
        ctx.path("pipeline_log.json").parent.mkdir(parents=True, exist_ok=True)
        ctx.path("pipeline_log.json").write_text(
            json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    for stage in selected:
        done = all(ctx.path(output).exists() for output in stage.outputs)
        if done and not ctx.force:
            results.append({"stage": stage.key, "status": "skipped", "reason": "outputs exist"})
            persist()
            continue
        try:
            record = stage.run(ctx)
        except Exception as error:
            # A failure is a result, so it is written down before it is re-raised.
            # An optional stage does not take the rest of the run with it; a
            # required one still stops everything.
            results.append(
                {"stage": stage.key, "status": "failed", "optional": stage.optional,
                 "error": str(error)}
            )
            persist()
            if stage.optional:
                continue
            raise
        record["status"] = "ok"
        results.append(record)
        ctx.log.append(record)
        persist()
    return {"stages": results}


def plan_table() -> str:
    rows = ["| 단계 | 하는 일 | GPU | 산출 |", "|---|---|---|---|"]
    for stage in STAGES:
        rows.append(
            f"| `{stage.key}` | {stage.title} | {'예' if stage.needs_gpu else '아니오'} | "
            f"`{', '.join(stage.outputs)}` |"
        )
    return "\n".join(rows)


def check_environment(ctx: Context) -> dict[str, Any]:
    """What is present before anything is spent."""

    torch_available = False
    try:  # pragma: no cover - environment dependent
        import torch

        torch_available = True
        gpu = torch.cuda.is_available()
        devices = torch.cuda.device_count() if gpu else 0
    except Exception:
        gpu, devices = False, 0
    return {
        "python": ctx.python,
        "corpus_present": ctx.corpus.exists(),
        "config_present": ctx.config.exists(),
        "torch": torch_available,
        "cuda": gpu,
        "cuda_devices": devices,
        "disk_free_gb": round(shutil.disk_usage(ctx.root.parent).free / 1e9, 1),
        "note": (
            "oracle 단계는 CPU만 쓰고 task당 약 1초, 진행 파일이 task당 수 MB 쌓인다. "
            "embed 단계만 GPU가 실제로 필요하다"
        ),
    }
