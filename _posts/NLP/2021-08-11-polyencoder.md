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

<br/>

## Overview

- Pretrained Transformer를 이용하여 시퀀스 간의 Pairwise 연산을 할 때 사용하는 방법론은 크게 2가지였다.
	- `Cross-encoder`
		- 두 sequence를 하나의 encoder에 동시 입력하여 시퀀스 간의  full self-attention을 수행
		- 성능을 좋지만, 실사용하기엔 너무 느리다는 단점
	- `Bi-encoder`
		- 두 시퀀스를 별도로 인코딩하고 두 Representation 사이의 스코어를 계산하는 방법
		- 일반적으로 성능이 더 낮지만, 실사용에 유리하다
- 본 논문에서는 Cross-encoder보다 실사용이 유리하고, Bi-encoder보다 성능이 좋은 Poly-encoder 방식을 제안하였다.

<br/>

## Methods

### Bi-Encoder

![image](https://user-images.githubusercontent.com/38639633/129450358-3d01c035-68d5-4ce8-9649-e86963a889fe.png){:.width="80%"}{:.center}

위 그림처럼 Context encoder와 Candidates encoder가 각각 context 문장과 candidates 문장을 인코딩하는 구조이며, 그 식은 아래와 같다. 


$$
y_{ctxt} = red(T_1 (ctxt)), \quad y_{cand} = red(T_2 (cand))
$$

- 여기서 $T(x) = h_1, h_2, \dots, h_N$​​​은 각 문장을 encoding하는 Transformer를 의미하며(사실상 BERT 구조라고 논문에서는 말하고 있다.), 논문에서 chapter 4.1에 나와있다. 동일한 가중치로 시작하여, 학습하는 동안 두 BERT는 서로 다르게 업데이트 된다. 
- 저자는 $red(\cdot)$​​는 아래 세 가지 과정을 고려했으나 첫 번째 방식을 사용하였다고 한다. 
	- 첫 토큰을 ([S])로 교환
	- 토큰별 아웃풋을 평균 냄
	- 첫 토큰부터 m개까지의 토큰을 평균

<br/>

#### Scoring

다음으로 Score를 계산하는데, 두 인코더의 아웃풋을 dot-product한 값을 스코어로 사용하며 식은 아래와 같다. 


$$
s(ctxt, cand_i) = y_{ctxt} \cdot y_{cand_{i}}
$$


이렇게 구한 $n$​​​​​​​개의 후보들에 대하여 점수를 구한 뒤 가장 높은 점수의 후보 문장을 주어진 context 다음에 올 문장이라고 간주한다. 학습시에는 동일한 batch 내의 다른 candidates 샘플을 negative sampling하여 cross entropy를 최소화한다. 즉, 요약하자면

- $s(ctxt, cand_i) = y_{ctxt} \cdot y_{cand_{i}}$ 로 스코어링 계산

-  1 batch = $[(ctxt_1, cand_1), (ctxt_2, cand_2), …, (ctxt_n, cand_n)]$​​에서 각 $i$번째 context에 해당하는 $i$번째 candidate(=$cand_i$)만을 positive, 나머지는 negative하여 cross entropy를 계산한다. 

> 자세한 내용은 [코드](https://github.com/chijames/Poly-Encoder/blob/701354372c66396d6b6678b664e82416f65f3a84/encoder.py#L29-L35)를 참고하길 바란다.

<br/>

#### Inference Speed

Bi-encoder 방식은 inference 속도가 매우 빠르다. Candidate sequences에 대한 DB를 미리 확보해 놓고 임베딩을 미리 해놓을 수 있기 때문이다. 새로 들어오는 query에 대한 context만 인코딩하여 스코어를 계산하고, 준비된 embedding weights에 대한 inner product만 계산하면 되기 때문이다. 게다가 FAISS와 같은 효과적인 유사도 계산 방식의 사용으로 확장할 수 있기에 inference 측면에서는 매우 효과적인 방식이다. 

<br/>

### Cross-encoder

![image](https://user-images.githubusercontent.com/38639633/129478765-5ba8fd3e-d581-42b7-a8d6-f3a331e2ffcf.png){:.width="80%"}{:.center}

Cross-encoder 방식은 일반적인 BERT의 학습 방식과 유사한 형태를 띈다. 위 그림과 같이 context $(In_x1, In_x2, \dots, In_xN_x)$​​​​​​​과 candidates $(In_y1, In_y2, \dots, In_yN_y)$​​​​​​​​​를 이어 붙인 뒤 학습시킨다. BERT 인코더 내부에서 context와 candidate 간의 self-attention 계산이 수행되기 때문에 두 시퀀스의 관계를 더 잘 학습할 수 있다. 



<br/>

#### Score

출력된 $y_{ctxt,cand}$를 스칼라 값으로 만들어 주기 위한 $W$를 곱해줌으로써 스코어를 구한다. 


$$
s(ctxt, cand_i)=y_{ctxt,cand_i}W
$$




Bi-encoder와 유사하게 cross-entropy를 최소화하기 위한 negative 방식을 사용하지만, 응답 candidates를 리사이클하기 어렵기에 외부 레이블을 학습 과정에서 negative 값으로 사용한다. 

<br/>

#### Inference Speed

앞서 언급했듯이 들어온 하나의 context sequence를 모든 candidate sequence와 연결 후 스코어링하여 비교하는 방식이므로 많은 파라미터가 소모되고 비효율적이다. 

<br/>

### Poly-encoder

![image](https://user-images.githubusercontent.com/38639633/129481047-23d1bdc0-f8b3-401c-93a7-41939334dda5.png){:.width="80%"}{:.center}

논문에서는 Bi-encoder의 장점인 inference 시 계산 효율성과 Cross-encoder의 성능이라는 장점을 취하는 Poly-encoder를 제안하였다. 

위 그림에서 볼 수 있듯이 Bi-encoder처럼 context와 candidate를 따로 인코딩한다. 하지만, Bi-encoder와의 차이점은 context encoder의 output을 aggregation하는 부분에 있다. 

Bi-encoder에서는 $red(\cdot)$을 통해 하나의 벡터로 합친 반면, code vector와의 attention 연산을 통해 여러 개의 embedding 벡터를 만든다. 


$$
y_{ctxt}^i = \sum_j w_j^{c_i} h_j, \quad \text{where} \quad (w_1^{c_i}, ..., w_N^{c_i}) = \text{softmax}(c_i \cdot h_1, ..., c_i \cdot h_N)
$$


여기서 code vector는 일종의 latent variable이며 학습 과정에서 learnable한 파라미터로 사용된다. 이렇게 얻어진(그림에서 Emb 1, ..., Emb N)는 미리 구해놓은 candidate embedding과 attention 연산하게 된다. 그렇게 context embedding vector를 최종적으로 구하게 되며, 식은 아래와 같다. 


$$
y_{ctxt} = \sum_i w_i y_{ctxt}^i, \quad \text{where} \quad (w_1, ..., w_m) = \text{softmax}(y_{cand_i} \cdot y_{ctxt}^1, ..., y_{cand_i} \cdot y_{ctxt}^m)
$$


마지막으로 구한 context embedding과 candidate embedding의 스코어링을 통해 최종적으로 유사도를 계산한다. 

Bi-encoder의 단점은 context와 candidate가 만나는 지점이 마지막 scoring 하는 부분에서만 발생한다는 점에 있다. Context와 candidate 두 문장의 attention 연산이 불가능하기 때문에 서로의 정보를 조금 더 깊게 파악하기 힘든 거이다. 

제안한 poly-encoder는 스코어링이 진행되기 이전에 attention 연산을 수행함으로써 두 문장의 더 깊은 관계파악을 할 수 있을 뿐만 아니라 candidate의 embedding을 미리 진행할 수 있어서 더 빠른 inference가 가능하다. 



<br/>

## Experiments

논문에서는 두 가지 task에 대한 실험을 진행하였다. 

- Sentence selection in dialogue : 주어진 대화 context 다음에 올 말로 적절한 것 찾기(객관식 10~20개)
	- ConvAI2
	- DSTC7 Chaalenge Track 1
	- Ubuntu V2 corpus
- Article search in IR : 주어진 문장이 등장한 article 찾기(객관식 10000개)
	- Wikipedia Article Search

![image](https://user-images.githubusercontent.com/38639633/129482757-e45bf1ee-dc8f-450e-82f3-6c9a24d22140.png){:.width="80%"}{:.center}



실험한 결과는 아래와 같다. 

![image](https://user-images.githubusercontent.com/38639633/129482774-c146b071-98fe-48cb-b355-421937b35bbf.png){:.width="80%"}{:.center}

- 구조적으로 쉽게 예상할 수 있듯이, Poly-encoder는 Bi-encoder보다는 좋고 Cross-encoder보다는 좋지 못한 성능을 기록했다. 
- 또한, batch 내의 sampling을 negative sampling 하여 학습한다고 앞서 언급했는데, 이러한 이유로 batch size가 클 수록 성능이 향상되었다고 한다. 



![image](https://user-images.githubusercontent.com/38639633/129482902-af7322d8-28aa-4907-8c7c-5aa9c298fbf0.png){:.width="80%"}{:.center}

- 위 표는 각 환경에서 inference 속도를 측정한 결과이다. 
- Bi-encoder에는 못 미치지만 충분히 빠른 속도를 보이는 것을 확인할 수 있다. (~~Cross-encoder는 답도없다~~)

