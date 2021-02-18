---
layout: post
title: 부스트캠프 AI Tech - Transformer I
subtitle: 그야말로 Attention is all you need
thumbnail-img : https://user-images.githubusercontent.com/38639633/108290231-5307f280-71d3-11eb-9576-f3cf9eca37a0.png
gh-repo: ydy8989/ydy8989.github.io
gh-badge: [follow]
categories: [BOOSTCAMP]
tags: [boostcamp,transformer, machine translation]
comments: true
---

이번 강의에서는 현재 NLP 연구 분야에서 가장 많이 활용되고 있는 Transformer(Self-Attention)에 대해 자세히 알아봅니다. Self-Attention은 RNN 기반 번역 모델의 단점을 해결하기 위해 처음 등장했습니다. RNN과 Attention을 함께 사용했던 기존과는 달리 Attention 연산만을 이용해 입력 문장/단어의 representation을 학습을 하며 좀 더 parallel한 연산이 가능한 동시에 학습 속도가 빠르다는 장점을 보였습니다

**Further Reading**

- [Attention is all you need, NeurIPS'17](https://arxiv.org/abs/1706.03762)
- [Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)



# Transformer I

> 과거 Attention is all you need 논문을 [포스팅](https://ydy8989.github.io/2021-01-10-transformer/)했던 적이 있지만, naver boostcamp 과정을 수강하면서 다시 한 번 등장한 transformer에 대해 포스팅 하려고 한다. 지난번엔 논문의 흐름에 따라 설명을 진행했다면 이번 포스팅에서는 조금 더 실질적이고 사용적 측면에서 바라보며 포스팅할 예정이다. 



## RNN: Long-term dependency

![image](https://user-images.githubusercontent.com/38639633/108290420-b4c85c80-71d3-11eb-8d2d-dcbe3e1a4d69.png)

- "I go home"이라는 문장을 받았을 때 매 time step $t$마다 $x_t, h_{t-1}$을 받아서 $h_t$를 만들어낸다. 
- 그림에서 왼쪽에서 오른쪽 방향으로 가며 계산되는 hidden state를 encoding하게 된다. 
- Attention 연산을 한다해도, 뒤로 갈수록 먼저 입력된 단어 "I"는 희석되게 된다.  



## Bi-Directional RNNs

![image](https://user-images.githubusercontent.com/38639633/108291960-81d39800-71d6-11eb-945c-96fb9f0bd052.png)

- Vanilla RNN의 단점을 보완하기 위한 방식으로 제안된 Bi-directional RNN은 역방향으로도 한번 더 진행해오면서 양방향에서의 encoding 벡터를 학습한다. 
- 양방향으로 진행되는 Forward RNN과 Backward RNN 모듈을 병렬적으로 만들고 특정한 timestep에서의 hidden state vector를 concatenate함으로써 두 배의 사이즈로 만들어진 encoding vector를 만든다. 



## Transformer: Long-Term Dependency

![image](https://user-images.githubusercontent.com/38639633/108292428-6b7a0c00-71d7-11eb-80d8-66673d3e3cc7.png){:width="60%"}{:.center}

- Transformer의 attention 연산은 self-attention으로써, 기존 attention에서 encoder와 decoder의 입력 벡터가 달랐던 것과 달리 같은 hidden state vector를 사용한다고 생각하면 된다. 
- 즉, 그림에서 $x_1$은 decoder hidden state vector임과 동시에 encoder hidden state vector set인 $[x_1, x_2, x_3]$중 하나라고 생각할 수 있다. 
- 그러면 첫 번째 timestep을 기준으로 $x_1$은 $[x_1, x_2, x_3]$ 세 encoder hidden states 들과 내적을 통해 attention score를 계산하게 되고, 이는 $h_1$으로 계산될 수 있을 것이다. 
- 이러한 방식으로 $h_2, h_3$를 구하게 되는 큰 틀에서의 방식을 `Self-Attention`이라고 부른다.
- 하지만, 일반적인 방식으로 계산하게 된다면 당연하게도 **자기 자신과의 내적**이 큰 비중으로 할당되게 되고, self-attention module의 output인 $h_{1,2,3}$는 자기 자신에 대한 가중 평균이 높게 잡히게 될 것이다. 
- 따라서 이러한 문제를 개선하고자 Transformer에서는 확장된 방식의 attention module을 사용한다.



### Query, Key, Value Vectors

- **`Query vector`** : encoder-decoder 구조에서 decoder hidden state vector에 해당하는 vector를 의미한다. 즉, 현재 timestep $t$에서 계산할 주체가 되는 vector.
- **`Key vector`** : query vector와 내적을 하게 될 각각의 재료 벡터를 의미한다. 즉, encoder-decoder 구조에서 encoder hidden states인 $h_{1,2,3}^{(e)}$를 의미한다. 
- **`Value vector`** : 계산된 가중치(attention score)를 가중 평균해서 그 비중을 가중해주기 위해 곱해주는 원래 벡터 

![image](https://user-images.githubusercontent.com/38639633/108337579-9afd3880-7218-11eb-8130-b582c472370e.png)

> - $q_1\cdot k_1$, $q_1\cdot k_2$, $q_1\cdot k_3$를 통해 [3.8, -0.2, 5.9]의 vector를 얻게된다.   
> - 이는 softmax를 통과하여 [0.2, 0.1, 0.7]이 된다.   
> - 이렇게 나온 결과는 [$v_1, v_2, v_3$]과 pairwise product 연산을 진행하게된다.   
> - 결과적으로 $h_1=0.2v_1+0.1v_2+0.7v_3$가 된다. 

이러한 방식으로 연산되기 때문에, 자기 자신에 대한 self-attention 연산을 하여도 그 크기가 높지 않게 된다. 

### Operation process in self-attention

![image](https://user-images.githubusercontent.com/38639633/108339298-9afe3800-721a-11eb-90e5-31f24e7d278f.png){:width="80%"}{:.center}

- 실제 작동은 위와 같은 행렬 연산에 의해서 진행된다. 
- Embedding된 input $X$는 $W^{Q,K,V}$와의 행렬곱을 통해 $Q,K,V$로 구성된다. 
- $Q,K,V$의 각 행은 $X$의 각 행, 즉 각 토큰에 해당하는 vector가 된다. 

이 같은 방식을 통해 먼 단어간의 관계 및 유사도를 이전 모델과는 달리 손쉽게 파악할 수 있다. 



## Transformer: Scaled Dot-Product Attention

- **Inputs** : a query $q$ and a set of key-value $(k, v)$ pairs to an output  
- Query, key, value, and output is all vectors  
- **Output** is weighted sum of values  
- Weight of each value is computed by an inner product of query and corresponding key  
- Queries and keys have same dimensionality $d_k$, and dimensionality of value is $d_v$
	- Value vector는 마지막에 계산된 가중평균을 곱하는 역할만을 하기 때문에 차원의 크기가 Query, Key vector들과는 달라도 상관이 없다. 

$$
A(q, K, V)=\sum_i\frac{exp(q\cdot k_i)}{\sum_j exp(q\cdot k_j)}v_i
$$

- 즉, input은 하나짜리 query 벡터 $q$와 $K, V$가 된다.   

- ans it becomes : $A(Q, K, V) = softmax(QK^T)V$.

	![image](https://user-images.githubusercontent.com/38639633/108349953-f1717380-7226-11eb-95b1-544cc34ed8c0.png)

	> 논문에서의 Transformer 구현 상으로는 동일한 shape으로 mapping된 Q, K, V가 사용되어 각 matrix의 shape은 모두 동일하다. 



### Problem

- As $d_k$ gets large, the variance of $q^Tk$ increases
	- query와 key vector의 차원이 커질수록 해당 내적에 참여하는 dimension 역시 커지게 되고 이때의 분산은 점점 커지게 된다. 
- Some values inside the softmax get large
- The softmax gets very peaked
- Hence, its gradient gets smaller

### Solution

- Scaled by the length of query / key vectors:
	- $$A(Q, K, V) = softmax(\frac{QK^T}{\sqrt{d_k}})V$$
	- $\sqrt{d_k}$로 나눠줌으로써 scaling을 시켜준다. 

![image](https://user-images.githubusercontent.com/38639633/108353903-07356780-722c-11eb-9926-69f1500536ac.png){:width="30%"}{:.center}

# Transformer II (cont’d)

Transformer(Self-Attention)에 대해 이어서 자세히 알아봅니다.

**Further Reading**

- [Attention is all you need, NeurIPS'17](https://arxiv.org/abs/1706.03762)
- [Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)
- [Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html)
- [Group Normalization](https://openaccess.thecvf.com/content_ECCV_2018/papers/Yuxin_Wu_Group_Normalization_ECCV_2018_paper.pdf)

**Further Question**

- Attention은 이름 그대로 어떤 단어의 정보를 얼마나 가져올 지 알려주는 직관적인 방법처럼 보입니다. Attention을 모델의 Output을 설명하는 데에 활용할 수 있을까요?
	- 참고: [Attention is not explanation](https://arxiv.org/pdf/1902.10186.pdf)
	- 참고: [Attention is not not explanation](https://www.aclweb.org/anthology/D19-1002.pdf)

## Transformer: Multi-Head Attention

- The input word vectors are the queries, keys and values
- In other words, the word vectors themselves select each other
- **Problem** of single attention
	- Only one way for words to interact with one another
- **Solution**
	- Multi-head attention maps $𝑄, 𝐾, 𝑉$ into the $ℎ$ number of lower-dimensional spaces via $𝑊$ matrices
- Then apply attention, then concatenate outputs and pipe through linear layer

