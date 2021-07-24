# COMPGCN

**knowledge graph embedding**



핵심 contribution

compositional operation at multi relational graph



1. GCN 에 multi-relational information 적용한 novel case. 특히 node 와 relations 를 공통적으로 embed 하는 technique에 초점을 두었음.
2. scability in number of relations in the graph.
3. extensive experiments at graph generalization task ( node classification , link prediction , graph classification )





- Comparison of our proposed method.

list ; GCN , Directed-GCN , Weighted-GCN , Relational-GCN

-> only using Relation Embeddings 

윗 임베딩 방법들과의 차별점인 **parameter efficient** .



2가지 key composition

1. Relation-based Composition

기존 방법론들과 다르게 COMPGCN은 d-dimensional represtion 을 representation learning 함. 또한 over-parameterization( 상대적으로 relation이 많은 것들 위주로 relation이 편승되는 것을 완화해주는 기능이 존재.)
$$
e_0 = \phi(e_s,e_r)
$$
\phi 는 composition operator 을 의미하며 e의 밑 문자인 s , r, 그리고 o 는 각각 subject, relation, 그리고 object를 의미함. 

**composition operation을 정하는 것이 embedding 측면에서 중요함.** -> ? 

1. Update Equation

$$
h_v = f(\sum_{(u,r)\in N(v)}W_{\lambda_r}\phi(x_u,z_r))
$$

기존 R-GCN 은 over-parameterziation의 발생이 문제였으나 본 논문의 COMPGCN은 composition (\phi)를  각각의 node u and relation r 적용함으로 써  relation aware 하게끔 적용하여 feature dimension에 linear하게 시간복잡도를 발생하게끔 장치를 만들어 주었음. (그냥 이전에는 모두 relation 들을 self-loop로 받아들였으나 본 논문에서는 operation을 통해 연산 효율적 장치를 적용했다고 생각하면 될듯싶다.)



**On Comparision with relational-GCN**

왜 COMPGCN 이 Relation-GCN보다 더 효율적인가 ? 

COMPGCN은 matrcies를 통해 임베딩을 하는것이 아니라 대신에 벡터를 사용하여 임베딩하며 그리고 basis vector (기저벡터?) 를 첫 레이어에 정의한다. 마지막 레이어는 Equation 4 를 통해 relation을 공유하게 되는데 이게 R-GCN에 비해 효율적이라 함.
$$
h_r = W_{rel}z_r (equation \ 4 )
$$
W_rel 은 learnable transformation matrix 를 의미함.



Knowledge graph embedding evaluation ; score function 은 TransE , DistMult , ConvE를 사용함.

이 때 구체적으로 어떻게 실험이 진행되었는가 알아봐야할 듯 싶다 . 단순 Graph Embedding 을 한 후에 TransE와 같은 임베딩을 왜 사용하는지? 



recap

1. efficient computing cost 를 위해 operation을 도입하였는데 이 때 operation은 각 노드와 엣지 를 jointly 하게 학습을 해주며 시간복잡도를 선형적으로 하게끔 만들어 줌 . 
2. knowledge graph 는 over-parameterization 이 문제였음. 여기서 over-parameterization은 많은 relation 때문에 생긴 문제라고 이해하고 있는데 이를 basic decomposition 을 적용하여 완화했다고 저자는 주장함. 아마 첫번째 , 마지막 레이어 에 장치를 해두었던걸로 기억함. 그 장치들이 이 문제에 기여하지 않았을까 추측함.