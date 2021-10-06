# Embedding Graph Auto-Encoder for Graph Clustering



Problem definition ; 기존 auto-encoder based architecture 은 graph clustering task 에 적합하지 않았음. 그리하여 본 논문에서는 auto-encoder architecture 을 tunning 함 . 좀 더 구체적으로는 encoder - decoder 로 이루어지는 one by one auto-encoder architecture 에서 one by dual auto-encoder ( two decoder ) 로 설정하여 task specific 하게 만듬.



![image](https://user-images.githubusercontent.com/52625664/136183617-595ad940-6643-4b57-9884-b118d75f771a.png)


architecture of EGAE

## Encoder

인코더는 raw data 를 multiple graph convolution layer를 토대로 embedding 하는 것이 목적임.

representation learning 토대로 발생한 값을 H_L로 명시하여 formulation 을 도출하면.
$$
H_i = \varphi_i(\hat{\mathcal{L}}H_{i-1}W_i )
$$
에서 \varphi_i(\dot)은 i 번째 layer의 activation function을 의미함. 또한 \mathcal{L}은 encoder 내의 총 layer를 대표함. Assumption1 에서의 가정 때문에 learned embedding Z 는 ZZ^T \geq 0 이여야 함. 이를 위해 setting 을 다음과 같이 함.
$$
\varphi_L(\cdot) = ReLU(\cdot).
$$
(그놈의 Assumption1...)

In other words, the condition of Theorem 3 will hold only if the similarity of two nodes in the same partition exceeds 0.5 in the category-balanced case.

또한 본 논문에서 제안한 방법론의 key 는 assumption 2 를 어떻게 cater 한것인가에 대해 많은 leverage를 두고 있음. (eigenvalue)

## Decoder

Decoder 는 embedding 그리고 reconstruction error 로 부터 최적화 된 값들을 토대로 adjacency A 를 reconstruction하는것이 목적임.

이 때 decoder 에서의 sigmoid function 을 활용하여 probability space 에 mapping 을 해줌.  이는 곧 0 , 1 즉 adjacency 여부를 의미함.

## Embed Clustering Model as Another Decoder

clustering 과 embedding learning 을 separating 하는 대신 EGAE는 joint learning ? 을 활용하여 해결하고자 하는데 그 부분이 여기에서 구현됨.
$$
\mathcal{J}_c = tr(ZZ^T) - tr(P^TZZ^TP),
$$

$$
min_{W_{i},P^TP=I}\mathcal{J} = \mathcal{J_r}+\alpha\mathcal{J_c}
$$

\alpha 는 tradeoff hyper-parameter을 의미함. 그것은 J_c 의 tunning point 로 활용. 이 때 포인트는 J_c 인데 이는 neural network 를 unsupervisedly eqaivalent 하다고 저자는 주장하였는데 자세히 내막을 들여다보면 J_c는 representation을 생성하는데 이 때의 기준은 relaxed k-means 가 preferable하게끔이 standard로 등장...! 다른 말로 하자면 with fixed P 그리고 F(논문에서는 F라고하였으나 위 equation 에서의 Z로 생각하여도 됨.) 상황에서 J_c의 목표는 ZP^T를 생성하는 것임 그러므로 그것은 decoder 라고 볼수있다고 한다...?

back propagation을 활용하기 위해서 다음 알고리즘 활용함.(closed-form solution ...! )

![image](https://user-images.githubusercontent.com/52625664/136183684-67ed8508-e14c-4a7d-b73d-c686820fc09b.png)


**Background**

Notation

Convolution on Graph

**Graph Auto-Encoder**

* Weak supervision ; **Weak supervision** is a branch of [machine learning](https://en.wikipedia.org/wiki/Machine_learning) where noisy, limited, or imprecise sources are used to provide supervision signal for labeling large amounts of [training data](https://en.wikipedia.org/wiki/Training,_validation,_and_test_sets) in a [supervised learning](https://en.wikipedia.org/wiki/Supervised_learning) setting.[[1\]](https://en.wikipedia.org/wiki/Weak_supervision#cite_note-:0-1) This approach alleviates the burden of obtaining hand-labeled data sets, which can be costly or impractical. Instead, inexpensive weak labels are employed with the understanding that they are imperfect, but can nonetheless be used to create a strong predictive model.[[2\]](https://en.wikipedia.org/wiki/Weak_supervision#cite_note-:10-2)[[3\]](https://en.wikipedia.org/wiki/Weak_supervision#cite_note-:2-3)[[4\]](https://en.wikipedia.org/wiki/Weak_supervision#cite_note-4)[[5\]](https://en.wikipedia.org/wiki/Weak_supervision#cite_note-5). 



기존 auto-encoder 구조로부터 inspired 하여 graph auto-encoder 는 다양한 GCN layer를 통해 노드의 embedding 을 employ 함. deep representations(GNN embedding variant) 을 통해 만들어진 inputted feature 보다 GAE는 reconstruction error 를 토대로 graph 내부의 node connection을 reconstruction 하는것에 초점을 두었음 이는 weakly supervised information 이라고도 regarded 할 수 있음.. 특히 decoder 에서는 node pair ( any two nodes ) 간의 inner-product 를 토대로 발생한 값을 probability space 에 mapping 해줌으로써 similarities 를 calculation함 . 이때 sigmoid function을 사용한다고 논문에서 밝혔으나 타 function도 가능함..! GAE variant architecture 도 존재하는데 ,그 중 Adversarial Regularized Graph Auto-Encoder , GAE with Adaptive Graph Convolution 두 가지 variant 를 본 논문에서는 언급함. 간단히 말하면 전자는 robustness enhancement 를 목적으로 adversarial learning 를 활용하였고 후자는 high-order convolution operator 를 활용하여 GAE 의 capa를 높였다고는 하는데 아마 layer를 쌓으며 발생하는 receptive field 의 efficacy 을 말하지 않았나 싶음 . (앞선 논문은 차후에 리뷰토록 하겠습니다!)  



**Problem Revisit**

network embedding 에서는 노드간의 divergence 를 단순히 inner-product 만으로 측정을 하고 있으며 기존 GAE-base model 에서는 Euclidean space 보다 inner-product space 에서의 embedding scatter을 경시 여기는 풍조가 있음. 이는 Euclidean distance 를 기초로 하는 clustering model 에서는 만족스러운 결과를 얻기에 부족한 감이 있음.



**Motivation**

clustering 에 관한 논문이다보니 K-means 알고리즘을 잠깐 언급하는 섹터.
$$
min_{f_j ,\ {g_ij}}\sum^n_{i=1}\sum^c_{j=1}g_{ij}||x_i-f_j||^2_2
$$

$$
s.t. g_{ij} \in (0,1),  \sum^c_{j=1}g_{ij}=1,
$$

{f_j}^c_j=1 은 centroids of c 즉 클러스터의 c 중심부를 의미 , g_ij 는 그것의 indicator 임. g_ij = 1 일 때 i-th 포인트는 j-th 클러스터에 소속함을 의미함.  아니면 g_ij = 0 ! k-means 는 Euclidean distance 에서 적절히 data point divergences 를 depict 할 수 있다는 가정을 가진 알고리즘임. 

... skip

Algorithm 1 ; Algorithm to optimize problem

input ; Data x 

Calculate c leading left singular vectors, P, of X

Normalize rows of P

Perform k-means on normalized P.

output ; Clustering assignments and P.

이후 본 논문에서의 알고리즘을 implementing 하기 위한 2가지 assumption 을 언급

> Assumption 1 . Non-Negative Property . For any two data points x_i and x_j , x_i^Tx_j \geq 0 always hold
>
> 간단히 말하면 어느 두 데이터 포인트들이나 내적을 하면 항상 0 이상의 값이라는 전제.



> Assumption 2. Let \lambda_i^(m) be the i-th largest eigenvalue of Q^(m). For any a and b, \lambda_1^(a) > \lambda_2^(b) always hold. 
>
> 간단히 말하면 eigenvalue sequence order == volume .

본 논문에서도 2번째 전제는 too strong and impractical 이라고 함.  (이는 공부를 좀 더 해서 보완설명토록 하겠습니다.. 혹시나 아시는 분 피드백 부탁드리겠습니다 \_.._\) 



> Theorem 1 . Assumption 1 과 2 가 hold 하다고 가정할 시 주어진 x_i^Tx_j = 0 은 만족하며 그리고 만약 x_i , x_j 가 각각 다른 cluster에 배정되었을때 이는 이상적인 partition이라 함.



> Theorem 2. Problem (14) is equivalent to the spectral clustering with normalized cut if and only if the mean vector µ is perpendicular to the centralized data. Or equivalently, µ is perpendicular to the affine space that all data points lie in.



Experiment

짜잔...

![image](https://user-images.githubusercontent.com/52625664/136183750-d0116fa5-4cfd-44e0-8559-b8be6c3dbd03.png)


![image](https://user-images.githubusercontent.com/52625664/136183780-887404b1-4478-4bcf-b0aa-c2aaeda2b205.png)


최종적으로 clustering evaluation measurement 인 ACC NMI ARI 에서 좋은 성능을 보였다고 합니다...! 
