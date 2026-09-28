# 02 · 아이디어 (초기 모델링 아이디어)

<sub>[🏠 Portfolio](https://github.com/heoneyzi) › [🩺 Medical](../../README.md) › [CAFA6](../README.md) › [Notes](README.md) › **아이디어**</sub>

> [!NOTE]
> Brainstorming notes (Korean) from the first weeks of the challenge. They start from teammate Yumin Jung's idea (GOA-based data expansion with distillation, GO-term tokenization) and record Jiheon's thoughts on how to realise it, plus two ideas of his own: matching against GO term text from `go-basic.obo`, and using the GO DAG as classifier structure or a $y_{parent} \ge y_{child}$ loss constraint. Discussed, not all implemented; a related parent-≥-child rule exists in the team code as ontology propagation — see [label-space JEPA](../pipelines/05_label_space_jepa/README.md).

유민이 아이디어

→ GOA를 이용해서 데이터를 늘려서 Knowledge Distillation으로, ESM-2+classifier를 사용하는 Ensemble 방안

→ Classification을 넘어서 GO Term을 Tokenizing해서 학습에 추가하자. GO의 DAG 구조에 대한 특수한 상황도 잘 잡아낼 수 있지 않을까? ESM-3가 그런 방식

그래서 어떻게 해볼 수 있을까 고민중

Prot2Token을 사용해서 Tokenizing 자체는 가능함.

Pro2Token + ESM2로 인코더를 잡고 학습해야하는데… 어려울 것 같음

ESM-3의 마스킹 학습 자체도 따라하기 쉽지 않을 것 같음…

- 택스트 대조 방식? → `go-basic.obo` 에 있던 GO의 text 형식을 이용하자.
- DAG 구조에서 이점을 챙길 수 있을까? → Go Term의 인접 행렬 혹은 그래프로 만들고 이를 분류기의 가중치로 사용한다던지… $`y_{parent}>y_{child}`$이도록 손실함수를 강제한다던지…
