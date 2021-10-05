Graphormer



summarize ; 트랜스포머를 gnn에 적용하면 성능향상이 적용될까?

트랜스 포머와 GNN의 차이점 ; 

GNNs with **multi-head attention** as the neighbourhood aggregation function. Wheras standard GNNs aggregate features from their local neighbourhood nodes 
$$
j \in N(i)
$$
Transformers for NLP treat the entire sentence S as the local neighbourhood, aggregating features from each word j \in S at each layer.



**Key insight ; To utilizing Transformer in the graph is the necessity of effectively encoding the structural information of a graph into the model.**

-> 즉 , 그래프의 structural information 을 어떻게 잘 보존하여 embedding 할것인가를 problem defintion 으로 targeting 하였고 다음의 3가지 요소들로 해결해보고자 함.



![img](https://github.com/microsoft/Graphormer/raw/main/docs/graphformer.png)



- Centrality Encoding

그래프 내에서의 노드 중요도를 측정하는 encoding tool

본 논문에서 주장하기를 기존 self-attention module 에서는 노드의 sematnic feature 을 반영하지 않았다는 점을 필두로 이를 해결하고자 node 중심성을 encoding 한 Centrality encoding 을 제안하였다. 좀 더 면밀히 체크해보자면 'degree centrality'를 이용하였다. 예를 들 자면 , 해당 노드의 degree를 측정하여 learnable vector 를 부여해주고 또한 input layer에 node feature을 추가해준다. 이 간단한 trick 이 trnasformer modeling 에 효율적임을 실증적으로 확인함.

**formulation view**
$$
h^{(0)}_i = x_i + z^-_{deg^-(v_i)}+z^+_{deg^+(v_i)}
$$
z^-, z^+ \in R^d 는 learnable embedding vectors 를 의미하며 indegree deg^-(v_i) 그리고 outdegree deg^+(v_i) 으로부터 각각 만들어짐. 



**code view**

```python
  # node feauture + graph token
        node_feature = self.atom_encoder(x).sum(dim=-2)           # [n_graph, n_node, n_hidden]
        if self.flag and perturb is not None:
            node_feature += perturb

            
            ## 이 때 node feature 에 node_Feature + degree encoder 를 넣어 줌.
        node_feature = node_feature + \
            self.in_degree_encoder(in_degree) + \
            self.out_degree_encoder(out_degree)
        graph_token_feature = self.graph_token.weight.unsqueeze(
            0).repeat(n_graph, 1, 1)
        graph_node_feature = torch.cat(
            [graph_token_feature, node_feature], dim=1)

        # in_degree_encoder , out_degree_encoder 
        self.in_degree_Encoder = nn.Embedding(64, hidden_dim, padding_idx=0)
        self.out_degree_encoder = nn.Embedding(64, hidden_dim, padding_idx=0)
        
```



- Spatial Encoding

노드 pair 간의 structural relaton 을 capture 하기 위해 제안한 방법. 기존 비정형 데이터들(image, language)는 canonical grid 하게 그래프형식으로 존재하지 않음을 시작으로 graph 는 사실 non-Euclidean space에 존재하며 edge에 의해 이어진다는 점을 언급. 즉 , 동일 비정형 데이터이긴하나 non-Euclidean vs. Euclidean 의 형태로 이야기 하고싶은 듯 함. 최종적으로 non-Euclidean space에 존재하는 데이터들의 특성을 임베딩하기 위해서는 spatial relation 에 의거한 learnable embedding 이 있으면 좋을거라는 가설을 정립하였음. 이 spatial realtion 은 두 노드간의 shortest path distance 를 활용하였으며 softmax attention 에 bias 형태로 인코딩 되었으며 이는 그래프 내부의 spatial dependency 를 잘 반영하였다고한다... 추가적으로 spatial information 이 edge feature에 존재하는 경우도 발생함 . molecular graph 를 예로 들자면 두 원자 간의 이어진 edge 가 이를 대표한다고 볼 수 있음. 좀 더 정확히 말해보자면 각각의 node pair 를 위해 edge feature 와 learnable embedding (shortest path)를 토대로 얻어진 값 의 dot-products 의 평균값을 활용하여 attention module 의 input으로 추가적으로 넣어줌. 

Transformer 는 global receptive field 의 feature을 모두 반영할 수 있다는 장점이 있었다. 이것은 positional dependency한 점이 있다. 허나 graph 는 not-positional dependency 즉 , non-sequence이기에 이를 어떻게 접목할지에 대해 고민을 한 저자는 spatial space 에서의 relation을 측정하여 접목하고자 하였다. 정확히는 그래프 G 내에서의 v_i, v_j 의 값을 측정하고자
$$
\phi(v_i,v_j) : V \times V \rightarrow \mathbb{R}
$$
의 function을 활용하였음. 이를 토대로 뒤의 formulation 의 bias term ? 형식으로 들어감. 계속

**formulation view**


$$
A_{ij} = \frac{(h_iW_Q)(h_jW_K)^T}{\sqrt{d}}+b_{\phi(v_i,v_j)}, \qquad Eq(6)
$$
b_{\phi(v_i,_v_j)}는 learnable scalar 이며 \phi(v_i,v_j)에 의해 index되는 값들임. 

최종적으로 A_ij 는 Query-Key product matrix A 이며 이를 토대로 attention module 의 input 으로 들어감.



윗 formulation 을 토대로 얻는 benefits은 다음 두 가지 이다.

1. 기존 conventional GNN 의 receptive field는 근처의 이웃들에 한정되어 있었으나 Eq(6) 을 활용하여 global information 을 활용하게 되었음.
2. b_{phi(v_i,v_j)}을 활용함으로써 node 들의 structural information 을 adaptively 하게 적용할 수 있음. 예를 들자면 b_{\phi(v_i,v_j)} 을 tunning 하는 function 인 \phi 구문에서 적은 값이 도출될 경우 각각의 모델 또한 less attention 을 부여함.



**code view**

```python
# cython 으로 코딩 되어있었음.

import cython
from cython.parallel cimport prange, parallel
cimport numpy
import numpy

def floyd_warshall(adjacency_matrix):

    (nrows, ncols) = adjacency_matrix.shape
    assert nrows == ncols
    cdef unsigned int n = nrows

    adj_mat_copy = adjacency_matrix.astype(long, order='C', casting='safe', copy=True)
    assert adj_mat_copy.flags['C_CONTIGUOUS']
    cdef numpy.ndarray[long, ndim=2, mode='c'] M = adj_mat_copy
    cdef numpy.ndarray[long, ndim=2, mode='c'] path = numpy.zeros([n, n], dtype=numpy.int64)

    cdef unsigned int i, j, k
    cdef long M_ij, M_ik, cost_ikkj
    cdef long* M_ptr = &M[0,0]
    cdef long* M_i_ptr
    cdef long* M_k_ptr

    # set unreachable nodes distance to 510
    for i in range(n):
        for j in range(n):
            if i == j:
                M[i][j] = 0
            elif M[i][j] == 0:
                M[i][j] = 510

    # floyed algo
    for k in range(n):
        M_k_ptr = M_ptr + n*k
        for i in range(n):
            M_i_ptr = M_ptr + n*i
            M_ik = M_i_ptr[k]
            for j in range(n):
                cost_ikkj = M_ik + M_k_ptr[j]
                M_ij = M_i_ptr[j]
                if M_ij > cost_ikkj:
                    M_i_ptr[j] = cost_ikkj
                    path[i][j] = k

    # set unreachable path to 510
    for i in range(n):
        for j in range(n):
            if M[i][j] >= 510:
                path[i][j] = 510
                M[i][j] = 510

    return M, path


def get_all_edges(path, i, j):
    cdef unsigned int k = path[i][j]
    if k == 0:
        return []
    else:
        return get_all_edges(path, i, k) + [k] + get_all_edges(path, k, j)
    
def gen_edge_input(max_dist, path, edge_feat):

    (nrows, ncols) = path.shape
    assert nrows == ncols
    cdef unsigned int n = nrows
    cdef unsigned int max_dist_copy = max_dist

    path_copy = path.astype(long, order='C', casting='safe', copy=True)
    edge_feat_copy = edge_feat.astype(long, order='C', casting='safe', copy=True)
    assert path_copy.flags['C_CONTIGUOUS']
    assert edge_feat_copy.flags['C_CONTIGUOUS']

    cdef numpy.ndarray[long, ndim=4, mode='c'] edge_fea_all = -1 * numpy.ones([n, n, max_dist_copy, edge_feat.shape[-1]], dtype=numpy.int64)
    cdef unsigned int i, j, k, num_path, cur

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if path_copy[i][j] == 510:
                continue
            path = [i] + get_all_edges(path_copy, i, j) + [j]
            num_path = len(path) - 1
            for k in range(num_path):
                edge_fea_all[i, j, k, :] = edge_feat_copy[path[k], path[k+1], :]

    return edge_fea_all
```





- Edge Encoding

node - node 의 관계를 성립하게 만들어주는 link 즉 , edge 또한 structural information을 반영할 경우가 존재함. 예를 들면 molecular graph 의 atom 원자 쌍에 존재하는 edge의 경우는 그것에 대한 유대 유형에 대한것을 describing 하는 feature을 가지고 있다. (기존 타 graph data와는 다르게) 이러한 features 들은 graph representation learning 에 중요한 요소로 작용하며 그것들을 또한 embedding 시에 반영해주는 것은 필수적이라 말할만큼 중요한 요소라 말 할 수 있음. 주로 edge encoding 방법에는 2가지가 쓰이는데.

1. edge feature를 associated nodes' feature로 간주하여 추가해줌.
2. edge feature을 node feature aggregation stage에 같이 input으로 넣어줌. 

하지만 앞선 방법들은 오직 그것과 연관된 node들로부터 propagate 가 이뤄지므로 associated node가 아닌 경우에는 누락이 된다. 이것은 즉슨 , 중요한 edge bond를 가지고 있을 node를 missed 하게 되는 오류를 초래하게 되며 당초 섹터의 목적인 edge information leverage에 어긋나는 이야기가 됨. 

최종적으로 edge feature을 효율적으로 attention layer에 넣기위해 (본 논문의 목적은 transformer 의 구조를 gnn 에 접목하여 유용함을 확인하는 것이기에 어쩔수 없이 모든것이 attention layer에 어떻게 효용적으로 넣을 것인가에 대해 초점을 두고 있음.) 이 node간의  correlation 을 추정하여 input으로 넣어주고자 함. 이 때 앞서 언급한 SPD(shortest path) 가 다시 활용됨. 
$$
SP_{ij} = (e_1,e_2,\ldots ,e_N)\quad from \quad v_i,v_j
$$
**formulation view**
$$
A_{ij}=\frac{(h_iW_Q)(h_jW_K)^T}{\sqrt{d}}+b_{\phi(v_i,v_j)}+c_{ij},\quad where \quad c_{ij} = \frac{1}{N}\sum^N_{n=1}x_{e_n}(w^E_n)^T
$$
이 때 x_{e_n} 은 n번째 edge의 feature을 의미함 ( SP_ij) 즉 SPD 에서 도출된 값으로 부터!

w^E_n 은 n번째 weight embedding 

d_E는 edge feature 의 dimensionality 를 의미함.



**code view**

```python
spatial_pos_bias = self.spatial_pos_encoder(spatial_pos).permute(0, 3, 1, 2)
graph_attn_bias[:, :, 1:, 1:] = graph_attn_bias[:,
                                                :, 1:, 1:] + spatial_pos_bias
# reset spatial pos here
t = self.graph_token_virtual_distance.weight.view(1, self.num_heads, 1)
graph_attn_bias[:, :, 1:, 0] = graph_attn_bias[:, :, 1:, 0] + t
graph_attn_bias[:, :, 0, :] = graph_attn_bias[:, :, 0, :] + t

# edge feature
if self.edge_type == 'multi_hop':
    spatial_pos_ = spatial_pos.clone()
    spatial_pos_[spatial_pos_ == 0] = 1  # set pad to 1
    # set 1 to 1, x > 1 to x - 1
    spatial_pos_ = torch.where(spatial_pos_ > 1, spatial_pos_ - 1, spatial_pos_)
    if self.multi_hop_max_dist > 0:
        spatial_pos_ = spatial_pos_.clamp(0, self.multi_hop_max_dist)
        edge_input = edge_input[:, :, :, :self.multi_hop_max_dist, :]
        # [n_graph, n_node, n_node, max_dist, n_head]
        edge_input = self.edge_encoder(edge_input).mean(-2)
        max_dist = edge_input.size(-2)
        edge_input_flat = edge_input.permute(
            3, 0, 1, 2, 4).reshape(max_dist, -1, self.num_heads)
        edge_input_flat = torch.bmm(edge_input_flat, self.edge_dis_encoder.weight.reshape(
            -1, self.num_heads, self.num_heads)[:max_dist, :, :])
        edge_input = edge_input_flat.reshape(
            max_dist, n_graph, n_node, n_node, self.num_heads).permute(1, 2, 3, 0, 4)
        edge_input = (edge_input.sum(-2) /
                      (spatial_pos_.float().unsqueeze(-1))).permute(0, 3, 1, 2)
        else:
            # [n_graph, n_node, n_node, n_head] -> [n_graph, n_head, n_node, n_node]
            edge_input = self.edge_encoder(
                attn_edge_type).mean(-2).permute(0, 3, 1, 2)

            graph_attn_bias[:, :, 1:, 1:] = graph_attn_bias[:,:, 1:, 1:] + edge_input
            graph_attn_bias = graph_attn_bias + attn_bias.unsqueeze(1)  # reset
```



이외의 trick

* Connection between Self-attention and Virtual Node.

  -> virtual node heuristic 과 self-attention 사이의 흥미로운 사실. virtual node trick 을 활용하여 추가적인 supernodes를 임의로 add해 주었을때  기존 GNN 의 performance 에 향상효과를 보임. 이 때 super node라 함은 degree centrality 가 타 노드 대비 월등하게 높은 node를 의미함. 허나 naive addition of a supernode to a graph 는 잠재적으로 over-smoothing 문제를 발생시킴 ( information propagation 과정에서 ) 우리는 이를 대신하고자 graph-level aggregation 그리고 propagation operation을 vanilla self-attention 에 활용하여 성능 향상을 꾀하였음.

Fact 2. By choosing proper weights, every node representation of the output of a Graphormer layer without additional encodings can represent MEAN READOUT functions. This fact takes the advantage of self-attention that each node can attend to all other nodes. Thus it can simulate graph-level READOUT operation to aggregate information from the whole graph. Besides the theoretical justification**, we empirically find that Graphormer does not encounter the problem of over-smoothing, which makes the improvement scalable. The fact also inspires us to introduce a special node for graph readout (see the previous subsection).**

**Experiment**

![image-20211005173928817](C:\Users\bitnine\AppData\Roaming\Typora\typora-user-images\image-20211005173928817.png)



