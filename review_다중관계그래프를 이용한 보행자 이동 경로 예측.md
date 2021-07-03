# [EIRIC 세미나] 다중 관계 그래프를 이용한 보행자 이동 경로 예측 (전해곤)





https://www.youtube.com/watch?v=rA9Ir8TVgrg



- edge는 user hyperparameter
- 어떻게 적절하게 edge를 연결하냐 
- 새로운 시각으로 연결하여 inference , generation 하는가?
- 



- Representation of Graph

  - node feature matrix
    - Undirect
    - Direct
  - Adjacency matrix
  - Node feature update rule
  - 

- Model Architecture

  - Spatial Edge GCN
    - 사람과 보행자간의 거리
    - 사람과 사람간의 변이 
  - Temporal Edge CNN
  - Temporal Extrapolation
  - Global Temporal Aggregation
  - Predicted Trajectory Distributions

- Social Relation Modeling

  - Various Pedestrian behavior pattern
    - Group Waling
    - Joining group
    - Intention
    - Collision avoidance
  - Multi-Relational GCN
  - Multi-Relational STGCN on pedestrian graph
    - Spatial Edge ; Neighborhood aggregation
    - **Temporal Edge ;  Consecutive frame aggregation**

  

  ## Oversmooting problem

  1. 

  ![image-20210408182328192](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20210408182328192.png)

  



![image-20210408182337609](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20210408182337609.png)

**Multi-Scale Aggregation -> Disentangled Multi-Scale Aggregation**

2. 

**DropEdge on weighted graph**

from - DropEdge ; Towards deep graph convolutional networks on node classification.



## Estimate Future Trajectory

Prior works with recurrent approach

our solution ; Time prediction CNN

![image-20210408182723003](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20210408182723003.png)



# Ablation Results



![image-20210408182930269](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20210408182930269.png)



## Conclusion

- Multi-relational
- Disentangled multi-scale aggregation
- DropEdge on weighted graph
- Time prediction CNN
- Global Temporal Aggregation

**stochastic ; 20개의 예측가능 경로 샘플링 가장 정확한 애랑 ground-truth** 

**determinstic ; 1개 추론해서 1 groundtruth랑 얼마나 비슷한지**



- 적절한 레이어 배치와 학습기준
- 자율주행의 경우, 카메라가 전방을 향하기 때문에, 객체가 사라질 경우가 많음 , 중간에 객체가 사라지는 프레임이 있을 때는 어떻게 되나요 ? -> 일정 프레임 내에서는 사라질수 없다는 가정을 함.  전방을 주시했을때 ; first-person view 

- 변환 ; scene 안에 각각의 보행자에 대한 좌표 정보가 있음.



