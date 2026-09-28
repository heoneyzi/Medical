<sub>[🏠 Portfolio](https://github.com/heoneyzi) › [🩺 Medical](../../../../README.md) › [GDTR](../../../README.md) › [Line A](../../README.md) › **E9 · Motif vs flank**</sub>

# 🔬 E9 · v11 control — Motif edit vs flank shuffle at splice donors

> **Question —** Is the early settling at splice donors driven by the 2-nt GT motif itself, or by the surrounding sequence context ("grammar")?

<details>
<summary><b>🇰🇷 한국어 요약</b></summary>

스플라이스 부위의 핵심 두 글자(GT)와 그 주변 서열 중 무엇이 빠른 정착을 만드는지 직접 서열을 바꿔(Perturbation) 확인했습니다.
GT를 AA로 바꾸면 정착이 조금 늦어졌고(0.46층), 반대로 GT는 두고 주변 ±100bp를 섞으면 정착이 3.18층이나 빨라졌습니다.
문장에서 핵심 단어는 그대로 두고 앞뒤 문맥만 뒤섞으면 오히려 "쉽게" 읽혀 버리는 것과 비슷합니다 — 실제 스플라이스 부위는 주변 문맥까지 통합하느라 더 깊이 처리된다는 해석입니다.
두 조작이 깊이를 반대 방향으로 움직인다는 점이 논문의 "양방향(bidirectional)" 주장의 근거입니다.

</details>

| | |
|---|---|
| **Status** | ✅ done (2026-05-04) — added during the v11 revision |
| **Model / data** | Evo 2 7B, γ_cos = 0.39663; 1,000 canonical GT-AG donors on chr22, ±3 kb context, 5 shuffles each; settling depth averaged over ±10 bp |
| **Compute** | one H200, ~32 min |
| **Headline** | flank shuffle (GT kept) lifts c̄ 26.77 → 23.59 (*d* = +0.515, *p* = 4.1 × 10⁻⁵⁹); GT→AA (flank kept) → 27.24 (*d* = −0.086, *p* = 2.3 × 10⁻³²) |

## Setup

Three paired conditions per donor ([`scripts/exp2_shuffled_motif_control.py`](../../scripts/exp2_shuffled_motif_control.py)): (1) the real sequence; (2) the ±100 bp flank dinucleotide-shuffled with the central GT preserved; (3) GT replaced by AA with the flank preserved. Paired Wilcoxon tests against the real donor.

## Results

| Condition | mean c̄ | median | Cohen's *d* vs real | paired Wilcoxon *p* |
|---|---|---|---|---|
| real GT-AG donor | 26.77 | 28.90 | — | — |
| flank shuffled, GT kept | **23.59** | 23.38 | **+0.515** | 4.1 × 10⁻⁵⁹ (two-sided) |
| GT → AA, flank kept | 27.24 | 29.86 | −0.086 | 2.3 × 10⁻³² (one-sided) |

Source: [`results/exp2_shuffled_meta.json`](../../results/exp2_shuffled_meta.json); the paper's appendix table of motif and flank controls reports the same values.

## Takeaway

- The two edits move depth in **opposite** directions: breaking the motif delays settling slightly, removing the real context makes the lone motif settle much earlier.
- Read as: real donors need deeper context integration than the motif alone — but depth alone cannot rule out that on shuffled flanks the model simply "gives up" and commits early (the paper states this caveat).
- The interventions cover one locus class (canonical donors), so they do not establish causal circuits genome-wide.

## Files

| File | What it is |
|---|---|
| [`scripts/exp2_shuffled_motif_control.py`](../../scripts/exp2_shuffled_motif_control.py) | the perturbation experiment |
| [`results/exp2_shuffled_meta.json`](../../results/exp2_shuffled_meta.json) | all numbers above |

---
<sub>[← E8](../E8_entropy_control/README.md) · [🏠 Portfolio](https://github.com/heoneyzi) · [E10 →](../E10_v8_revision/README.md)</sub>
