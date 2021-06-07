# Review _ A Heterogeneous Information Network based cross Domain Insurance Recommendation System for Cold Start Users



[paper](https://arxiv.org/pdf/2007.15293v1.pdf)

## Preview

- Cold start problem 을 target domain 에서 non-financial user prediction으로 selection 했다는 것이 너무 기발했음.
- 2.2 Observations in Real Data 에서 각각 Q1 ~ Q4까지의 question을 토대로 online insurance 필드에서 왜 cold-start problem 이 중요한건지에 대한 논리전개가 아름다웠음. 
  - Q1 ; Is online insurance the tendency ? 
  - Q2 ; Do users' behaviours in nonfinancial domain have influence on their behaviors in insurance domain? 
  - Q3 ; Are insurance policies really complex ? 
  - Q4 ; Are agents affecting users' buying behaviors in insurance domain? 
- 





Abstract



Methodology



Observations in Real Data



Problem Formulation

- 최종적으로 유저와 아이템의 latent feature을 효율적으로 learning 하기 위함이 목적임. 또한 cold start 문제를 다루는 것에 도움이 되고자 mapping function을 활용하여 non-financial domain 과 insurance domain 의 특성을 잘 embedding 하는것도 부가적인 목표로 잡음.

HCDIR

![image](https://user-images.githubusercontent.com/52625664/120995006-f9e2bf80-c7bf-11eb-8c36-0a1b053f8fd8.png)

- 모델은 크게 3가지 파트로 나눠져 있음.

![image](https://user-images.githubusercontent.com/52625664/120994929-ed5e6700-c7bf-11eb-82ae-45fe1d82ec73.png)


  - Latent Feature in Insurance Domain

    - relational attention 하기 위해 one-hop heterogeneous neighbors 을 meta-path aggregation 함.

    - semantic attention 하기 위해 neighbor-set 에 기반하여 meta-path aggregation함.

    - relational attention + semantic attention 의 결과를 aggregation 하여 본래 node embedding 을 update 시킴.

      1. Insurance Heterogeneous information network construction.

      2. Relation Neighbor aggregation - Figure 5(a) 와 같이 I_1 그리고 A_1 은 모두 U_1의 이웃이긴하나 각각 다른 정보를 함축하고 있다. 우리는 이 정보를 활용하기 위해 one-hop neighbors relation attention aggregation을 하였고 모두 동일한 aggregation function을 주는것이 아닌 edge-specific aggregation function 을 활용하여 가중치를 두었음. 
         $$
         a_{ew} = \frac{exp(f_r(h^0_e,P_rh^0_w))}{\sum_{j\in N_1(e)}exp(f_r(h^0_e,P_rh^0_j))}, \forall w \in N_1(e)
         $$
         h^0_e는 현재 node e의 임베딩을 의미하며 각각 다른 종류의 neighbor 들의 node type들이 있다. 그래서 먼저 같은 node space 에서 projection 한 후에 그리고 윗 식과 같이 attention score 를 산출함. f_r(\cdot, \cdot)은 MLP 를 의미하며 relational attention을 계산하고자 고안됨, 그리고 a_ew 는 노드 e_w의 influence level을 의미한다. 또한 N_1 은 노드 e의 one-hop neighbors를 의미함. 만일 N_2 라고 하면 2-hop neighbor을 의미함.  최종적으로 아래 식을 통해 update  됨.
         $$
         h^1_e = \sigma(\sum_{w\in N_1(e)}a_{ew}h^0_w)
         $$
         

      3. Meta-path based Aggregation

         2개의 노드 또한 meta-path에 의해 연결될 수 있다, 다른 meta-path 로 부터 연결된 neighbor는 각각 다른 information을 함축하기에 ! 예를 들자면 윗 figure 에서 나타난 I_2 와 I_1 은 (I_2 - U_1 - I_1) 의 관계를 가지고 있다 허나 I_3 는 (I_2 - P_2 - I_3) 와 같이 같은 유저 내에서 존재하나 다른 정보들이 함축되어있다고한다. 그리하여 효율적으로 meta-path 를 활용하여 neighbor 을 선정하기 위해 attention score를 활용함.

         윗 과정과 별 차이는 없음. 추가적인 것은 더욱 정확한 node embedding 을 학습하기 위하여 fuse multiple node embedding 이란 방법을 시도하였는데 간단히 말하면 eta-path 의 importance 을 계산하여 얻어진 weight를 final node embedding 에 fuse 해주는 장치를 추가하였다.

         

      4. Node Updation 

         HIN node embedding 이후 우리는 유저 그리고 insurance product embedding 을 갖는데 , 얻어진 임베딩들을 각각 u^t , v^t 로 명명하겠다. target domain 에 대한 objective function은 다음과 같음.
         $$
         L_t = \sum_{(u,v)\in Y^t}-(y_{uv}log_{\hat y_{uv}}+(1-y_{uv})log(1-\hat y_{uv}))
         $$
         이 때 f는 ranking function이라고 논문에 나와있는데 윗식에는 존재하지 않음... 아마 생략되었나싶다.

  

  

  - Latent Feature in Non-financial Domain

    - word2vec 으로 representation 하였으며 embedding 간 length 를 맞춰주기 위해 maxpooling 을 활용함. 

  - Mapping Function Between Two Domains

    - MLP 를 활용하여 source domain ( insurance ) , target domain(non-financial) 간 latent space matching 을 하기 위하여 다음과 같은 loss function 을 설정함. 
      $$
      L_{cross} = \sum_{u\in u}||f_{mlp}(u^s)-u^t||_2
      $$

  - Recommendation for Cold start Users

    - 본 연구에선느 cold start users는 non-financial domain 에서 interaction이 있으나 insurance domain 에서는 interaction이 없는 유저들을 대상으로 하였음. 최종적으로 앞선 mapping feature 과정을 통해 나타난 
      $$
      \hat u^t = f_{mlp}(u^s)
      $$
       를 토대로 recommendation을 해보고자 함.

Experiment

- Single-domain models 와 Cross-domain Models 로 나누어 cross-domain 의 usefulness를 확인하였음. 



Online A/B Testing for Cold-Start Recommendation

- LightGBM 에 앞선 도출된 값을 feature로 적용한 결과 online evaluation에서 유의미한 향상을 보임. 



Conclusion and Future work

- Source domain 에서는 GRU 모듈을 활용하여 users dynamic interest 를 capture 하였음. 
- Target domain 에서는 IHIN based on data를 활용하여 3가지 level (relational, node, and semantic)을 attention aggregations 하여 유저 그리고 insurance product representation을 capture하였고 latent feature에서 overlapping users의 latent feature을 MLP 활용하여 obtaining 하였다. 
