<sub>[🏠 Portfolio](https://github.com/heoneyzi) › [🩺 Medical](../../../README.md) › [GDTR](../../README.md) › [Line C · Handoff](../README.md) › **EXP2**</sub>

# 🔬 C2 · EXP2 — What each late block does, causally

> **Question —** Does the change really start at block 28, what distinguishes block 28 from block 30, why do these particular branches explode, and does EXP1 hold up under intervention?

<details>
<summary><b>🇰🇷 한국어 요약</b></summary>

EXP2는 블록의 각 가지(mixer, MLP)를 하나씩 끄거나 크기를 바꾸는 개입 실험으로 늦은 층의 역할을 나눴습니다.
28번 블록의 MLP는 새 방향을 **쓰는(writer)** 단계이고, 30번 블록은 MLP 없이 mixer만으로 그것을 출력 형식으로 **다시 부호화(re-encoder)** 하며, 마지막 31번 블록은 아무 기여도 하지 않았습니다.
눈에 띄는 수십만 배의 크기 증가는 정작 출력에 영향이 없었고(크기를 반으로 줄여도 변화 0), 중요한 것은 쓰인 **방향**이었습니다 — 목소리 크기가 아니라 말의 내용이 결과를 바꾸는 것과 같습니다.
또 가중치만 보고도 어느 블록이 writer/re-encoder인지 4개 체크포인트 모두에서 맞혔고, 40B 모델에서는 발표된 onset 위치를 독립적으로 재현했습니다.

</details>

| | |
|---|---|
| **Status** | ✅ done (2026-09-20) — 30 experiments, 39 recorded findings |
| **Model / data** | `evo2_7b` (bf16) on EXP1's frozen chr22 panel with genome-disjoint splits (discovery / development / locked_A / locked_B / locked_chr17); `evo2_7b_base` and `evo2_1b_base` (causal), `evo2_40b_base` (weights only) |
| **Compute** | single A100 (a 40B causal run does not fit: 75.1 GiB checkpoint on 79.2 GiB) |
| **Headline** | writer **b28** (removing its MLP: +2.153 nats, 377× its phase twin b21) → re-encoder **b30** (its MLP contributes exactly 0) → idle **b31**; weights alone name writer and re-encoder in **4/4** checkpoints |

## Setup

Exact taps on every block's branches (`x → m = Mixer(RMSNorm(x)) → r = x + m → g = MLP(RMSNorm(r)) → x_next`), verified by residual identities and a deliberately wrong hook. Interventions: zero a branch (causal atlas), scale it by α (dose), change only its radius (radial scan), clamp one coordinate (carrier), and rescue with self/donor/shuffled/constant values. Endpoints: ΔNLL, the change in output-distribution shape and a temperature parameter log β\*. Inference unit = window; homologous controls use the attention period of 7 (b21 is b28's phase twin).

## Results

| Question | Result |
|---|---|
| Is the change at b28? | branch-max ΔNLL b17–b27 0.003–0.057 → **b28 2.153**, b29 2.076, b30 5.062, b31 0.000 (a ×165 step); b28 MLP − b21 MLP = +2.147 [2.069, 2.224] (377×), while the HCL twin pair b27 − b20 = +0.0008 [−0.0002, 0.0018] (null) |
| b28 vs b30 | removing b28's MLP collapses cos(x29) to 0.097; removing b30's mixer leaves x29/x30 exactly unchanged (cos 1.000000) but turns x31 orthogonal (−0.0075) — b28 **writes**, b30 **re-encodes** |
| Does size matter? | no: halving b30's mixer changes the output by 5.1 × 10⁻¹⁵ and ΔNLL is exactly 0 for α ∈ {0.25, 0.5, 2, 4}; radial sensitivity falls ~100× from x28 to r28 — the mechanism is the direction written |
| Why these blocks? | late MLPs are exact bilinear forms; gain bound G at b28 124× and b29 6,295× the median vs 0.76× at twin b21; W1/W2 stable rank 27.2/28.6 at b28 vs ~400–530 in ordinary blocks |
| Carrier coordinate #3756 | clamping it at x31 costs **+4.3134 nats** (76,500× a size-matched control) but is inert at x28–x30; a single constant restores 99.86 %; 99.4 : 1 selective for temperature → a confidence channel (7B only — absent in 1B) |
| Scale and recipe | `evo2_7b_base`: same stages (b28 MLP 2.754 nats, 620× twin); 1B: writer b22 (31.79 nats), re-encoder b23 (MLP 0.0000), terminal attention b24 idle |
| Weights-only prediction | writer = first bilinear block with G > 10× median, re-encoder = first HCL block with a dead MLP → 28/30, 28/30, 22/23, **21/23** (40B, out-of-sample; matches the onset/rotation found from activations in arch_compare); 40B MLPs b23–b49 are untrained (effective depth ~23/50) |
| Re-check of EXP1 | 5 of 7 recoverability values reproduce within 0.0092; the two crossing b30 do not (h30→h27 ≈ 0.35 vs 0.270) — the bottleneck is real but ~30 % less severe |

Gates: 113/113 model-free tests, residual identity worst 3.62 × 10⁻³, wrong-hook negative test separated 46×. All numbers: [`docs/EXP2_summary_ko.md`](docs/EXP2_summary_ko.md) (§3–§10).

<p align="center"><img src="../figures/figC_magnitude.png" width="760" alt="Dose-response of the writer, re-encoder and twin-block branches"></p>
<p align="center"><sub>(a) arch_compare α sweep; (b) EXP2 dose curves — the re-encoder's output change stays at 10⁻¹⁵–10⁻⁷ for every non-zero scale; (c) 40B — <code>../figures/figC_magnitude.png</code>.</sub></p>

## Takeaway

- The late stack is three stages — **writer → re-encoder → idle readout** — in 7B and 1B, with the same writer/re-encoder pair read off the 40B weights; which block plays which role follows from the weights.
- The famous norm explosion is causally inert: readouts after r28 cannot see the radius, so "the norm locates, the direction matters".
- Limits: 40B is weights-only here; the confidence channel is 7B-specific; statements are about the model, not biology.

## Files

| File | What it is |
|---|---|
| [`code/exp2/`](code/exp2/) | the EXP2 step library: gate, calibration, radial, carrier, content, α, bilinear, HCL, transport, EXP1 extension (`step1_gate.py` … `step10_exp1_extension.py`), plus `taps.py`, `endpoints.py`, `stats.py`, `manifest.py`, `adapters/vortex.py` |
| [`docs/EXP2_summary_ko.md`](docs/EXP2_summary_ko.md) | full results summary (Korean); its run scripts and 81 result files stayed on the GPU volume |

---
<sub>[← EXP1](../exp1/README.md) · [🏠 Portfolio](https://github.com/heoneyzi) · [EXP3 →](../exp3/README.md)</sub>
