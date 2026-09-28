<sub>[🏠 Portfolio](https://github.com/heoneyzi) › [🩺 Medical](../../../../README.md) › [GDTR](../../../README.md) › [Line A](../../README.md) › **E8 · Entropy control**</sub>

# 🔬 E8 · v11 control — Is settling depth just next-token entropy?

> **Question —** A token might "settle" early simply because the model is confident about the next base. Does the splice signal survive once that uncertainty is removed?

<details>
<summary><b>🇰🇷 한국어 요약</b></summary>

정착 깊이가 단지 "다음 염기를 얼마나 확신하는가(엔트로피)"를 다르게 표현한 것일 수 있다는 반론을 검증했습니다.
22번 염색체 120개 창(72만 위치)에서 두 값의 상관은 −0.079로 작았고, 엔트로피 영향을 빼고 나서도 스플라이스 효과는 오히려 커졌습니다(d −0.452 → −0.583).
시험 점수가 단지 "자신감"을 잰 것인지 확인하려고 자신감 효과를 빼고 다시 채점해 본 것과 같습니다.
다만 5′UTR만은 엔트로피와 강하게 묶여 있어(ρ = +0.41) 논문에서 따로 보고했습니다.

</details>

| | |
|---|---|
| **Status** | ✅ done (2026-05-04) — added during the v11 revision |
| **Model / data** | Evo 2 7B, γ_cos = 0.39663; 120 chr22 windows × 6 kb = 720,000 positions in the result file (the paper reports 719,000 analysed positions) |
| **Compute** | one H200, ~40 s on cached taps |
| **Headline** | overall ρ(c, H) = −0.079; splice-donor vs intron *d* = −0.452 raw → **−0.583** after regressing out entropy |

## Setup

Per-position next-token Shannon entropy *H* from the post-norm logits; Spearman ρ between settling depth *c* and *H*, overall and per context; then *c* is regressed on *H* and the donor-vs-intron effect is re-measured on the residual ([`scripts/exp1_entropy_correlation.py`](../../scripts/exp1_entropy_correlation.py)).

## Results

| Context | *n* | mean *c* | mean *H* | ρ(c, H) |
|---|---|---|---|---|
| intergenic | 243,956 | 28.94 | 0.795 | −0.024 |
| intron | 425,131 | 28.03 | 1.030 | −0.108 |
| coding exon | 36,427 | 28.61 | 0.744 | −0.080 |
| 3′ UTR | 8,238 | 27.61 | 1.073 | −0.028 |
| splice donor | 1,981 | 25.04 | 0.887 | −0.083 |
| splice acceptor | 1,920 | 27.33 | 0.764 | −0.152 |
| **5′ UTR** | 2,347 | 31.43 | 1.121 | **+0.414** |

Source: [`results/exp1_entropy_meta.json`](../../results/exp1_entropy_meta.json) (no figure was produced for this control; the table above is the full result).

## Takeaway

- The splice ordering is not an entropy artefact — removing entropy makes it stronger.
- 5′ UTR is the exception: there depth tracks prediction confidence, so the paper reports it separately and does not compare it on the same axis.
- This controls next-token uncertainty only; sequence composition (GC, k-mer rarity, repeats) is a separate confounder that this test does not remove.

## Files

| File | What it is |
|---|---|
| [`scripts/exp1_entropy_correlation.py`](../../scripts/exp1_entropy_correlation.py) | the control |
| [`results/exp1_entropy_meta.json`](../../results/exp1_entropy_meta.json) | all numbers above |

---
<sub>[← E7](../E7_tier2_robustness/README.md) · [🏠 Portfolio](https://github.com/heoneyzi) · [E9 →](../E9_motif_flank_control/README.md)</sub>
