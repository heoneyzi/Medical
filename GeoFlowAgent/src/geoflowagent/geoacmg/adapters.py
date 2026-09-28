"""실제 산출물 레이아웃에 번들 분석 모듈을 잇는 어댑터.

번들(``geoacmg_finish``)의 분석 모듈은 산출물 형식에 결합되지 않도록 호출자에게
로더를 주입받는다.  이 모듈이 그 로더를 `artifacts/acmg` 의 실제 레이아웃에 맞춰
구현한다.  번들 모듈 자체는 고치지 않는다 — 레이아웃이 바뀌면 여기만 고친다.

확인된 레이아웃 (2026-09-20):

``checkpoints/geometry_comparison/<family>/seed-<n>/``
    ``value_geometry.pt``    dict — ``model_config`` · ``model_state`` · ``seed`` ·
                             ``energy`` · ``training_config`` · 캐시 지문들
    ``value_metrics.json``   평가 지표 (per-task 포함)
    ``training_resume.pt``   재개용 전체 상태
    ``training_progress.json``
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from geoflowagent.models.value_geometry import GoalConditionedValueGeometry
from geoflowagent.utils.reproducibility import seed_everything

#: 에너지 헤드를 제외한 나머지 — 두 arm 이 공유해야 하는 부분.
ENERGY_PREFIX = "energy_head."


class TrunkView:
    """``state_dict()`` 에서 에너지 헤드를 뺀 뷰.

    ``paired.state_dict_fingerprint`` 는 ``state_dict()`` 만 요구하므로 nn.Module 일
    필요가 없다.  에너지 헤드를 빼는 이유는 그것이 두 arm 의 **차이 그 자체**이기
    때문이다 — 공유돼야 하는 것은 trunk 다.
    """

    def __init__(self, model: Any) -> None:
        self._model = model

    def state_dict(self) -> dict[str, Any]:
        return {
            key: value
            for key, value in self._model.state_dict().items()
            if not key.startswith(ENERGY_PREFIX)
        }


def read_geometry_checkpoint(seed_dir: str | Path) -> dict[str, Any]:
    """``value_geometry.pt`` 를 읽는다. 없으면 fail-loud."""

    path = Path(seed_dir) / "value_geometry.pt"
    if not path.is_file():
        raise FileNotFoundError(f"체크포인트가 없다: {path}")
    return torch.load(path, map_location="cpu", weights_only=False)


def build_model_at_init(seed_dir: str | Path, *, energy: str | None = None) -> Any:
    """체크포인트의 ``model_config`` 로 모델을 **초기 상태에서** 재구성한다.

    학습된 가중치를 싣지 않는다.  ``paired.verify_matched_start`` 가 비교하는 것은
    *trunk 초기값*이고, 학습된 가중치는 에너지가 다르면 당연히 갈라지기 때문이다.
    검정하려는 명제는 "같은 시드면 두 arm 이 같은 출발점이었다" 이다.

    ``GoalConditionedValueGeometry.__init__`` 은 ``energy_head`` 를 value/reachability/
    completion 헤드보다 **먼저** 만든다.  따라서 파라미터가 있는 에너지는 뒤따르는
    헤드들의 난수를 밀어내고, 파라미터 0개 에너지는 밀어내지 않는다.  이 성질이
    쌍 설계의 전제이며 가정하지 않고 잰다.
    """

    checkpoint = read_geometry_checkpoint(seed_dir)
    config = dict(checkpoint["model_config"])
    if energy is not None:
        config["energy"] = energy
    seed = int(checkpoint["seed"])
    seed_everything(seed)
    return GoalConditionedValueGeometry(**config)


def load_trunk(seed_dir: str | Path) -> TrunkView:
    """``paired.audit_existing_runs`` 에 넘기는 로더."""

    return TrunkView(build_model_at_init(seed_dir))


def read_value_metrics(seed_dir: str | Path) -> Mapping[str, Any]:
    path = Path(seed_dir) / "value_metrics.json"
    if not path.is_file():
        raise FileNotFoundError(f"지표 파일이 없다: {path}")
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)
