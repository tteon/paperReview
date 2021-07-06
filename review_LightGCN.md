# review_ LightGCN



## Problem settings

- Existing work that adapts GCN to recommendation lacks thorough ablation analyses on GCN, which is originally desinged for graph classification tasks and equipped with many neural network operations.
- feature transformation and nonlinear activation ; contribution little to the performance of collaborative filtering.
  - Even worse, including them adds to the difficulty of training and degrades recommendation performance.

solution ; adding only the most essential component in GCN - **neighborhood aggregation** - for collaborative filtering.

- **learns user and item embeddings by linearly propagating them on the user-item interaction graph, and uses the weighted sum of the embeddings learned at all layers as the final embedding.**

Such simple, linear, and neat model is much easier to implement and train, exhibiting substantial improvements over Neural Graph Collaborative Filtering.



why useless these functions at recommendation tasks?

- Feature transformation
- nonlinear activation

but useful 'neighborhood aggregation' at Collaborative filtering.



## NGCF Brief


$$
e^{(k+1)}_u = \sigma(W_1e_u^{k}+\sum_{i\in\N_u}\frac{1}{\sqrt{|N_u||N_i|}}(W_1e_i^{(k)}+W_2(e^{(k)}_i\odot e^{(k)}_u))
$$

$$
e^{(k+1)}_i = \sigma(W_1e_i^{k}+\sum_{i\in\N_u}\frac{1}{\sqrt{|N_u||N_i|}}(W_1e_u^{(k)}+W_2(e^{(k)}_u\odot e^{(k)}_i))
$$

- e_u^{(k)} and e_i^{(k)} is respectively denote the refined embedding of user u and item i after k layers propagation
- **\sigma is the nonlinear activation function.**
- N_u denotes the set of items that are interacted by user u
- N_i denotes the set of users that interact with item i
- **W_1 and W_2 are trainable weight matrix to perform feature transformation** 

**Conclusion , concatenates these L + 1 embeddings to obtain the final user embedding and item embedding, using inner product to generate the prediction score.**

In semi-supervised node classification, each node has rich semantic features as input, such as the title and abstract words of a paper. Thus performing multiple layers of nonlinear transformation is beneficial to feature learning. **NEVERTHELESS, in collaborative filtering, each node of user-item interaction graph only has an ID as input which has no concrete semantics. In this case, performing multiple nonlinear transformation will not contribute to learn better features ;** 

we change the way of obtaining final embedding from concatenation to sum.

![image-20210331202524414](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20210331202524414.png)` 

- NGCF-f, which removes the feature transformation matrices W_1 and W_2.
- NGCF-n, which removes the non-linear activation function \sigma.
- NGCF-fn, which removes both the feature transformation matrices and non-linear activation function.

Conclusion

1. Adding feature transformation imposes negative effect on NGCF, since removing it in both models of NGCF and NGCF-n improves the performance significantly;
2. Adding nonlinear activation affects slightly when feature transformation is included, but it imposes negative effect when feature transformation is disabled.
3. As a whole, feature transformation and nonlinear activation impose rather negative effect on NGCF, since by removing them simultaneously, NGCF-fn demonstrates large improvements over NGCF.

## LightGCN

Although they perform well on node or graph classification tasks that have semantic input features, they could be burdensome for collaborative filtering.

![image-20210331205034047](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20210331205034047.png)

LighGCN subsumes the effect of self-connection thus there is no need for LightGCN to add self-connection in adjacency matrix. we discuss the rleation with the Approximate Personalized Propagation of Neural Predictions (APPNP) which is recent GCN variant that addresses oversmoothing by inspiring from Personalized PageRank;

1. *Relation with SGCN*

![image-20210331210206823](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20210331210206823.png)

1. *Relation with APPNP*

![image-20210331210234275](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20210331210234275.png)

![image-20210331210246445](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20210331210246445.png)



1.  *Second-Order Embedding Smoothness.*

![image-20210331210410594](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20210331210410594.png)



## Model Training

BPR (*Bayesian Personalized Ranking*) loss ; A pairwise loss that encourages the prediction of an obeserved entry to be higher than its unobserved counterparts
$$
L_{BPR} = -\sum^M_{u=1}\sum_{i\in{N_u}}{\sum_{j\notin N_u}ln\sigma(\hat y _{ui}-\hat y_{ui})}+\lambda||E^{(0)}||^2
$$
it is technically viable to also learn the layer combination coefficients {a_k}^K_{k=0}, or parameterize them with an attention network.



## Experiments

Compared Methods.

- Multi-VAE .
  - This is an item-based CF method based on the variational autoencoder. IT assumes the data is generated from a multinomial distribution and using variational inference for parameter estimation. 

source; tensorflow - https://github.com/dawenl/vae_cf pytorch - https://github.com/younggyoseo/vae-cf-pytorch



# Contribution

raw feature is best at recommender system?

- We empirically show that two common designs in GCN, feature transformation and nonlinear activation, have no positive effect on the effectiveness of collaborative filtering.
- We propose LightGCN, which largely simplifies the model design by including only the most essential components in GCN for recommendation
- We empirically compare LightGCN with NGCF by following the same setting and demonstrate substantial improvements. In-depth analyses are provided towards the rationality of LightGCN from both technical and empirical perspectives.





