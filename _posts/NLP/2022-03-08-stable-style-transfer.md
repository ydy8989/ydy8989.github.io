---
layout: post
title: REVIEW) Stable Style Transformer/Delete and Generate Approach with Encoder-Decoder for Text Style Transfer
gh-repo: ydy8989/ydy8989.github.io
gh-badge: [follow]
categories: [NLP]
tags: [style transfer, nlp]
comments: true
---

본 논문은 Text Style Transfer task에서 SOTA를 달성하진 않았지만, Yelp, Amazon 데이터 셋으로 동일 task를 수행한 여러 모델들과 비교하였을 때, 다양한 metric에서 안정적인 성능을 보여준다고 주장한 논문이다. 실제로 본 논문의 Experiments를 확인해보면, 여러 metric에서 우수한 성능을 내지만 하나씩은 치명적인 단점을 보유한 다른 모델들과는 달리 대부분의 성능평가 지표에서 준수한 모습을 보여준다. 

**Paper** : [https://arxiv.org/pdf/2005.12086.pdf](https://arxiv.org/pdf/2005.12086.pdf)

> 이 글은 저자인 [이주성](https://github.com/rungjoo)님의 [유튜브](https://www.youtube.com/watch?v=4D8uVWdNeLI)를 참고하여 작성했습니다. 

<br/>



# motivation & Background

- 챗봇 등에서 유저의 characteristics에 맞는 응답을 내놓는데에 사용 가능하다 
- 도메인에 상관 없이 안정적인 응답을 내놓는 것은 중요하다. 

<br/>

# problem statements : text style transfer

- 속성을 유지하고 스타일을 바꾸기 위함
	- 단순히 sentiment만의 문제가 아니라 나이대별 언어나 말투에도 적용 가능
	- 굳이 sentiment일 필요는 없다. 
	- 논문에서는 모델의 성능을 다른 벤치마크 데이터와 비교하기 위해 sentiment를 사용
- example 
	- The food is tasteless -> The food is delicious

<br/>



# Research lines

- find the disentangled latent representation
	- disentangled representation이란 feature를 분리된 공간으로 표현하여, 분리된 차원에서 직관적인 의미를 내포하게 하는 representation 방법
	- 적대적 학습(ex. GAN) 등 사용
	- disentangled latent representation 사용 
- do not find the disentangled latent representation
	- 그냥 오토 인코더 방식 사용
	- **벡터화 시키기 전에 discretet한 공간에서 없애는 방식**
		- <u>*논문에서 사용한 방식*</u>

<br/>



# Model overview

![image](https://user-images.githubusercontent.com/38639633/156958707-1b867096-5222-4cb1-89e5-5b18cc1fcf9d.png)

1. **Delete process**

	1. `스타일`에 해당하는 부분을 discrete한 공간에서 토큰을 하나하나씩 먼저 없애는 과정
		1. classifier의 확률이 많이 변하면 그 영향이 크다고 판단하여 해당 토큰을 삭제(조건부 확률)
		2. 위 예시에서 tasteless를 없애면 남는 것은 `The food is`. 이 문장은 긍정인지 부정인지 구별하기 힘듬. 따라서 tasteless가 긍/부정을 판단하는 중요 토큰이라고 판단
	2. 적절한 threshold를 설정하여 ($\alpha, \beta$) 삭제

2. **Generate process**

	1. 위에서 sentiment style을 삭제한 남은 문장을 인-디코더에 통과시킨다. 

	2. Decoder는 GPT-2를 사용하였음

	3. 여기서 Loss는 두 가지가 사용됨

		1. Reconstruction loss
			$$
			\mathcal{L}_{rec} = -log P_{\theta_{E}, \theta_{G}}(x|x^, s)
			$$

			- 원래 source style을 넣어서(긍->부 task에서 긍정 스타일) 제대로 긍정으로 generate가 되는지를 확인

		2. Style loss
			$$
			\mathcal{L}_{rec} = -log P_{\theta_{C}}(\hat{x}=\hat{s}|x^c, \hat{s})
			$$

			- 컨텐츠 토큰(Delete process를 통과한 토큰들)에 부정 스타일을 넣고, 이를 Classifier를 또 태웠을 때 제대로 분류가 되는지를 확인하는 loss

<br/>



# Experiments

- Yelp, Amazon 데이터 셋을 사용
- metric
	- contents : self-BLEU, human-BLEU
	- attribute : classification acc
	- Fluency : data-PPL, general-PPL
	- semantic : BERTScore
- 추가 실험 결과는 논문 참고 