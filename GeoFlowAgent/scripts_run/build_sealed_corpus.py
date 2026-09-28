#!/usr/bin/env python3
"""test 를 제외한 corpus 사본을 만든다 — 사다리를 봉인 위반 없이 돌리기 위해.

문제. `runners.ladder_findings` 는 package_dir 의 tasks.jsonl 을 **split 필터 없이** 읽고
`tasks[:max_tasks]` 로 자를 뿐이다. corpus/tasks.jsonl 의 앞부분이 test 라
`--max-tasks 50` 은 100% test 가 된다. 금지사항 4("test split 을 열지 마라") 위반이다.

해결. src/ 를 고치지 않고 test 행을 뺀 사본을 만들어 --package 로 넘긴다.
ladder_findings 는 매니페스트나 해시를 검증하지 않으므로 안전하다.
제외 건수를 전부 보고한다(fail-loud).
"""
import json, shutil
from pathlib import Path
from collections import Counter

SRC=Path("artifacts/acmg/corpus")
DST=Path("artifacts/acmg/corpus_sealed")
DST.mkdir(parents=True, exist_ok=True)

def rd(p): return [json.loads(l) for l in open(p)]
def wr(p, rows):
    with open(p,"w") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False, sort_keys=True)+"\n")

tasks=rd(SRC/"tasks.jsonl")
by_split=Counter(t.get("split") for t in tasks)
keep=[t for t in tasks if t.get("split")!="test"]
kept_ids={t["task_id"] for t in keep}
print(f"  tasks {len(tasks):,d} -> {len(keep):,d}")
for s,n in sorted(by_split.items()): print(f"    split={s}: {n:,d} {'(제외)' if s=='test' else '(유지)'}")

snaps=rd(SRC/"snapshots.jsonl")
ks=[r for r in snaps if r.get("provenance",{}).get("task_id") in kept_ids]
print(f"  snapshots {len(snaps):,d} -> {len(ks):,d}  (제외 {len(snaps)-len(ks):,d})")

vers=rd(SRC/"verifiers.private.jsonl")
kv=[r for r in vers if r.get("task_id") in kept_ids]
print(f"  verifiers {len(vers):,d} -> {len(kv):,d}  (제외 {len(vers)-len(kv):,d})")

wr(DST/"tasks.jsonl", keep)
wr(DST/"snapshots.jsonl", ks)
wr(DST/"verifiers.private.jsonl", kv)
shutil.copy(SRC/"tools.jsonl", DST/"tools.jsonl")
# 매니페스트는 출처와 제외 내역을 적어 새로 쓴다 (원본을 복사하면 개수가 거짓이 된다)
json.dump({"derived_from":str(SRC),"purpose":"test split 을 제외한 사다리용 사본",
           "excluded_split":"test",
           "counts":{"tasks":len(keep),"tasks_excluded":len(tasks)-len(keep),
                     "snapshots":len(ks),"snapshots_excluded":len(snaps)-len(ks),
                     "verifiers":len(kv),"verifiers_excluded":len(vers)-len(kv)},
           "split_counts_original":dict(by_split)},
          open(DST/"manifest.json","w"), ensure_ascii=False, indent=2, sort_keys=True)
print(f"\n  생성: {DST}")
for p in sorted(DST.iterdir()): print(f"    {p.name:<28s} {p.stat().st_size:>12,d} bytes")
print("\n  확인: 남은 tasks 에 test 가 있는가 ->", any(t.get('split')=='test' for t in keep))
