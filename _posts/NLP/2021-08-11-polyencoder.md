---
layout: post
title: REVIEW / Poly-encoders - Transformer Architectures and Pre-training Strategies for Fast and Accurate Multi-sentence Scoring
gh-repo: ydy8989/ydy8989.github.io
gh-badge: [follow]
categories: [NLP]
tags: [chatbot, nlp]
comments: true
---

ICLR 2020에서  Facebook AI Research 팀은 발표한 `Poly-encoders` 구조는 두 문장을 비교하는 task에서 일반적으로 사용되던 `Cross-encoder`와 `Bi-encoder` 방식의 장점을 각각 취한 형태의 encoding 방식이다. 

최근 몇 년간의 연구에서 Transformers 구조는 Sequence간의 embedding 방식을 획기적으로 바꾸었으며, 내부적으로 '알아서' 임베딩 되도록 학습된다. 이로 인해 두 Sequence들 간의 비교 task는 매우 쉬워졌으며 다양한 방식이 제안되고 있다. 저자들은 그 중 일반적으로 사용되는 Cross-encoder 방식과 Bi-encoder 방식을 섞은 Poly-encoder를 제안하였다. 



## Overview

- Pretrained Transformer를 이용하여 시퀀스 간의 Pairwise 연산을 할 때 사용하는 방법론은 크게 2가지였다.
	- Cross-encoder
		- 두 sequence를 하나의 encoder에 동시 입력하여 시퀀스 간의  full self-attention을 수행
		- 성능을 좋지만, 실사용하기엔 너무 느리다는 단점
	- Bi-encoder
		- 두 시퀀스를 별도로 인코딩하고 두 Representation 사이의 스코어를 계산하는 방법
		- 일반적으로 성능이 더 낮지만, 실사용에 유리하다
- 본 논문에서는 Cross-encoder보다 실사용이 유리하고, Bi-encoder보다 성능이 좋은 Poly-encoder 방식을 제안하였다.



## Bi-Encoder


$$
y_{ctxt} = red(T_1 (ctxt)), \quad y_{cand} = red(T_2 (cand))
$$




- Context와 Reply Candidate를 별도의 BERT로 인코딩

	- 동일한 가중치 시작
	- 학습하는 동안 두 BERT는 서로 다르게 업데이트 된다.

- Reduction : BERT의 sequence 아웃풋을 Reduction하는 방법

	1. 첫 토큰([S])를 이용
	2. 토큰별 아웃풋을 평균
	3. 첫 토큰부터 m개까지의 토큰을 평균

	- 실험 결과 첫 토큰만 이용하는게 제일 성능이 좋았다.

- Score: 두 인코더의 아웃풋을 dot-product한 값을 스코어로 사용

- 학습시에는 동일한 배치 내의 다른 reply candidates를 **negatives**로 이용한다.
