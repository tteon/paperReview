# Learning to Hash with Graph Neural Networks

[PAPER LINK]](https://arxiv.org/pdf/2003.01917.pdf)





![image](https://user-images.githubusercontent.com/52625664/129466017-55b70f50-248b-4628-935d-8f09866b16ff.png)


3줄 요약

1. Hash Code embedding

   ㄱ. Sign function 으로 binary code 화
   $$
   z_i = \sigma(W^Tu_i + b)
   $$
   ㄴ. above equation 을 토대로 발생한 z_i 를 resort
   $$
   h_i = sign(z_i)=\begin{cases} +1, if \ z_i\geq0\\-1, otherwise \end{cases}
   $$
   ㄷ. sign function 을 활용하여 +1 , -1 의 값을 도출해냄. 

   
   $$
   P(A_{ij}|h_i,h_j)) = \begin{cases} \sigma(dist(h_i,h_j)), A_{ij}=1 \\ 1 - \sigma(dist(h_i,h_j)), A_{ij} = 0\end{cases}
   $$
   ㄹ. 각각 발생한 노드 를 pairwise 하게 logistic function 화 하여 생각함. unsupervised representation learning 을 시작으로 similiarity 측정하여 link connection 을 정의함.

   ** keyword ; likelihood function , hamming distance , inner product

2. Straight-through estimator (STE)

   - 연속형 변수들의 gradient 를 copying 하여 이산형 변수를 optimization 함.
   - 4.1 Learning with Guidance 의 Lemma4.1  그리고 증명까지 포함되어 있음.

3. End-to-End fashoin , joint loss learning 



- Dataset
  - Bipartite graph







- Experiement
  - Baseline

  

  - ![image-20210815121931187](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20210815121931187.png)

  

  - PTE [46], BiNE [11], MF [35] and Graphsage [16]: They are commonly used embedding methods for recommendation. Here we build our proposed hashing model based on GraphSage for its scalability and deep expressive ability.

  - LSH [13]: This is a state-of-the-art unsupervised hashing method. For fair comparison, the input feature is the output of Graphsage. 
  - HashNet[6]: This is the state-of-the-art deep hashing method based on continuation methods [1]. We adapt it for graph data by replacing the original AlexNet [25] architecture with graph neural networks.
  - Hash_gumb: This is the implementation of learning to hashing with Gumbel-softmax [21]. The basic idea is to treat each bit element as a one-hot vector of size two. In the top of hash layer, we use Gumbel-softmax trick instead of siдn function to generate hash codes. Note that by gradually decreasing the temperature parameter towards zero, it can obtain exactly binary hash codes in theory.
  - Hash_ste: This is the implementation of state-of-the-art end-to-end hash learning method based on straight through estimator [2]. It is also a special case of our model.
  - HashGNN_sp: This is a variant of the proposed method by separating the graph embedding generation process with hash function learning. It first trains Graphsage to obtain node embeddings. The learned embeddings are then used as input feature to train the hash layer.
  - HashGNN_nr: This is a variant of the proposed method by excluding ranking loss. We introduce this variant to help investigate the benefit of using triplet loss.



![image-20210815120403281](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20210815120403281.png)



Research Question

1. **Hamming Space Retrieval**

   - First, LSH separates the encoding of feature representation from hashing and it achieves poor performance on three datasets. This indicates the importance of jointly learning feature representation and hash projections for high quality hash code generation.

   - Second, HashNet and Hash_gumb are two widely used continuous relaxation based approaches for hash learning. But HashNet outperforms Hash_gumb in several cases. A possible reason is HashNet adopts continuation methods to train the model, which approximates the original hard optimization problem with sign function with a sequence of soft optimization problems, making it easier to learn.

   - Third, Compared to HashNet, although Hash_ste is a truly end-to-end hashing method, it is still outperformed by HashNet in most cases. It makes sense since Hash_ste will magnify the gradients, which makes it deviated from standard continuous embedding learning problem.

   - Fourth, In general, HashGNN consistently yields the best performance on all the datasets. Increasing the number of n has positive influence on the performance of all methods. By using continuous embedding guidance, HashGNN is capable of mimicking the learning process of continuous embedding optimization, while ste deviates from continuous embedding learning problem by magnifying gradients.

     -> effectiveness of continuous embedding 를 입증함! 

     -> hamming space 에서의 relation capture 를 활용하면 추천시스템 task 에서 효율적이라는 것을 보임.

2. Performance Comparison on Hierarchical Search

3. Study of HashGNN

4. Parameter Analysis



Review ;

- STE 를 활용하여 gradient 유지하며 forwarding 하는 점이 신선했음.
- Hash function 을 활용하여 retrieval 접목한 점이 되게 신선했음.



