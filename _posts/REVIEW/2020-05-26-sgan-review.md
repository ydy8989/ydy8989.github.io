---
layout: post
title: REVIEW / Semi-supervised learning with Generative Adversarial networks 
subtitle: 준지도 학습, 그리고 분류
gh-repo: ydy8989/ydy8989.github.io
gh-badge: [follow]
categories: [REVIEW]
tags: [paper, review, semi-supervised learning, gan]
comments: true
---



이전 직장인 삼성전자 협력사에서 근무할 당시 반도체 스마트 인터락 관련 프로젝트를 진행했었다. 처음 프로젝트를 진행할 당시에 많은 양의 데이터를 기대했지만, 얻을 수 없었다. 이유는 몇 가지가 있다 

1. **사용한 데이터는 저장하지 않고 흘린다** : 아마도 하루에 발생하는 많은 양의 데이터를 저장하기란 비용적으로 만만치 않은 일이니깐.
2. **기존의 구축되어있는 시스템을 건드리는 일이었기 때문에** : 건드리지 말래서 건드리지 않았다.
3. **윗선에서 귀찮아해서** : 아무래도 외주를 받는 입장인 협력사의 일개 직원이 데이터베이스에 접근하는 것은 프로젝트 매니저 입장에서도 부담스러운 일이다. 이는 일일히 csv로 저장 후 나에게 오는 방식으로 전해졌고, 매우 귀찮은 작업이 되었다(사실 귀찮다기보다는 다른 업무에 지장이 있을 정도였기 때문에)
4. **공정의 종류가 매우 많아서** : 독립적으로 작동하는 공정 센서의 종류는 수천 수만가지지만, 인터락 발생 시 분류되는 Class의 수는 10종류 미만이었다. 정말 연관없는 서로 다른 두 센서를 같은 클래스로 묶어서 레이블링하는 것은 또 다른 프로젝트였고, 불가능했다. 삼성전자 측에서도 무슨 변수든 상관없고 어찌됐든 분류만 되면 되는 모델을 만들어 주길 원했다. 
5. **레이블링이 되어있지 않아서 :** 이 부분이 데이터 부족의 가장 유효한 이유이기도 했다. 처음 프로젝트 진행 당시에는 레이블링이 되어있다고 했지만, 사실은 '레이블링을 할 수는 있지만 되어있지 않다'였다. 학습에 사용할 데이터 100개 가량을 제공받는 데에도 많은 협의가 오가야했다. 

아무튼 이러한 이유로 데이터를 확보할 수 없었고, 7개 클래스로 1500건의 인터락 이미지를 받았다... 심지어 클래스별 비율도 균일하지 않았기에 자연스레 준지도학습에 대하여 서칭을 시작하였고, 일단 시작해보자고 마음먹었던 모델이 **<u>SGAN</u>**이다. (개방 당시 모델의 부족한 부분은 시계열 데이터를 사용한 오토인코더 방식과 자잘한 머신러닝 기법을 사용한 앙상블로 해결하였다.) 

또한, 기본적인 구조는 CatGAN과 비슷하지만, 딱 한가지! objective function이 다른데 그 이유는 개념의 출발이 다르기 때문이다. CatGAN역시 추후에 다뤄볼 것이다.

<br/>

### Paper 내용

**[SGAN paper link](https://arxiv.org/abs/1606.01583)**

SGAN 논문 자체는 매우 짧고 objective function에 대한 내용이 없어서 자세한 내용은 생략하고, original GAN과의 차이점 및 특징을 먼저 적어보자면 :

1. DCGAN을 기반으로 작성하였다. : 이 부분에 대한 정확한 언급은 없지만 아마 성능이 더 좋아서 아닐까 싶다.
2. original GAN과는 달리 sigmoid가 아닌 softmax를 사용한다. 이유는 당연히 fake/true를 분류하는 것이아니라 N+1개의 클래스를 분류해야하기 때문이다.
3. Discriminator의 성능은 sample의 수가 적을 때 CNN보다 높다고 한다. 샘플의 수가 많아질 수록 CNN과 비슷해진다. 
4. 생성하는 이미지의 품질이 original GAN보다 좋다. 

로 표현할 수 있다. 



<br/>

### Cost function or objective function or loss function

cost function에 대한 내용은 논문에는 나와있지 않기 때문에 [source code(github)](https://github.com/nejlag/Semi-Supervised-Learning-GAN)가 있는 사이트의 readme를 참조하였다.

**1. Discriminator loss**

우선 k개의 class를 구별할 수 있는 D를 만들고 싶다면, G의 가짜이미지를 포함하는 K+1개의 class를 D가 분류할 수 있어야한다. 따라서 x가 fake라고 가정하면 x의 class가 될 확률은 다음과 같다. (아래 식에서 ***y = k + 1***이 의미하는 것은 fake를 k+1번째 클래스라고 가정했을 경우이다.)


$$
P_{model}(y=k+1|x) = \frac{exp(l_{k+1})}{\sum_{j=1}^{k+1}exp(l_j)}
$$


반대로 x가 label이 있는 데이터라면 k개의 class중 하나로 분류되어야 하기 때문에 그 확률은 다음과 같다.


$$
P_{model}(y=i|x,i<k+1) = \frac{exp(l_{i})}{\sum_{j=1}^{k+1}exp(l_j)}
$$



다음으로 D의 loss는 supervised loss + unsupervised loss이며, supervised loss는 class를 알고있기 때문에 negative log likelihood는 다음과 같이 표현할 수 있다. 



$$
L_{D_{supervised}} = -\mathbb{E}_{x,y~\sim~p_{data}}log\left[p_{model}(y=i|x,i<k+1)\right]
$$



반대로 D의 unsupervised loss의 경우에는 가짜가 아니라고 분류하는 것과, G에서 생성한 데이터를 가짜라고 분류해야하기 때문에 아래와 같이 cost function을 정의할 수 있다. 


$$
L_{D_{unsupervised}} = -\mathbb{E}_{x~\sim~p_{data}}log\left[1-p_{model}(y=k+1|x)\right]-\mathbb{E}_{x~\sim~G}log\left[p_{model}(y=k+1|x)\right]
$$

<br/>

**2. Generator loss**

다음으로 G의 loss는 다음 두 가지 파트로 이루어진다.

**Feature matching loss :** 쉽게 말하자면 진짜와 같은 feature를 G가 생성한 데이터가 가지고 있는지를 나타내는 loss라고 할 수 있다. 이를 위해서 D의 중간 층 activation 함수
$$
f(x)
$$
를 사용하는데 그 식은 다음과 같다:


$$
L_{G_{feature~matching}}=||\mathbb{E}_{x~\sim~p_{data}}f(x)-\mathbb{E}_{x~\sim~G}f(x)||^2_2
$$


즉, 학습 중인 현재 D의 중간 activation의 output(
$$
f(x)
$$
)이 생성에 필요한 하나의 feature가 되는데, G가 생성한 데이터의 분포가 학습된 데이터의 분포와 얼마나 비슷한지를 matching하는 loss라 할 수 있다. 결국 위의 loss function이 낮아지게끔 학습하는 과정에서 G가 생성하는 데이터는 "진짜" 데이터의  분포와 점점 같아지는 방향으로 학습이 진행되는 것이다. 



**Cross-entropy :** 다음으로 G는 D를 잘 속이는 방향으로 학습을 진행해야 하기 때문에, 아래와 같은 cross entropy loss를 추가해준다.


$$
L_{G_{cross-entropy}}=-\mathbb{E}_{x~\sim~G}log\left[1-p_{model}(y=k+1|x)\right]
$$


따라서 최종적인 G의 loss function은 


$$
L_G = L_{G_{feature~matching}}+L_{G_{cross-entropy}}
$$


로 정의할 수 있다. 

































