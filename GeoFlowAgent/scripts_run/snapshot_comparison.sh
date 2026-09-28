#!/usr/bin/env bash
# 집계가 끝나는 즉시 comparison.json 을 스냅샷한다.
# 이 파일은 --output-dir 를 공유한 다른 실행이 덮어쓸 수 있으므로 즉시 사본을 뜬다.
set -uo pipefail
cd $GEOACMG_WORK
SRC=artifacts/acmg/checkpoints/geometry_comparison/comparison.json
DST=artifacts/acmg/checkpoints/geometry_comparison_FINAL.json

while pgrep -f 'geoflowagent\.cli compare-value-geometries' >/dev/null 2>&1; do sleep 20; done
sleep 5
if [ ! -f "$SRC" ]; then echo "실패: 집계 종료 후에도 $SRC 가 없다"; exit 1; fi
cp "$SRC" "$DST"
python3 - <<'PY'
import json
d=json.load(open("artifacts/acmg/checkpoints/geometry_comparison_FINAL.json"))
ks=sorted(d)
print("  키:", ks)
if "paired_task_macro" not in d:
    print("  !! paired_task_macro 없음 — 짝지은 구간이 저장되지 않았다"); raise SystemExit(1)
print(f"  energies={d.get('energies')} seeds={d.get('seeds')} runs={len(d.get('runs',[]))}")
print(f"  selected_energy={d.get('selected_energy')} test_sealed={d.get('test_sealed')}")
PY
