# GraphVAE

# Paper

GraphVAE: Towards Generation of Small Graphs Using Variational Autoencoders.





# Tutorial

## Tutorial on Variational Graph Auto-Encoders

source ; https://towardsdatascience.com/tutorial-on-variational-graph-auto-encoders-da9333281129



- if the Embedding captures more information from the input, the output will have a better performance
- Autoencoder measures the information lost during the reconstruction. 
- Storing images' low-dimensional embeddings could save storage space compared with storing their pixel intensities. 



### Main idea

The main idea of a variational autoencoder is that it embeds the input X to a distribution rather than a point. And then a random sample Z is taken from the distribution rather than generated from encoder directly.



## The architecture of the Encoder ad Decoder

- The encoder for a VAE is often written as q\phi(z|x), which takes a data point X and produces a distribution. The distribution is usually parameterized as a multivariate Gaussian. Therefore, the encoder predicts the means and standard deviation of the Gaussian distribution. The lower-dimensional embedding Z is sampled from this distribution. The decoder is a variation approximation, p\theta(x|z), which takes an embedding Z and produces the output X-hat.

## Loss Function

1. variational lower bound, which measures how well the network reconstructs the data.
2. A regularizer. It is the KL-divergence of the approximate from the true posterior (p(z)), which measures how closely the output distribution (q\phi(z|x)) match to p(z).

$$
l_i(\theta,\phi) = -E_{z~q_\phi(z|x_i)}[log_{p_\theta}(x_i|z)]+KL(q_{\phi}(z|x_i)||p(z))
$$



## Variational Graph Autoencoders

- we can't just straightforwardly apply the idea of VAE because graph-=structured data are irregular. Each graph has a variable size of unordered nodes and each node in a graph has a different number of neighbors, so we can't just use convolution directly anymore. 

### Adjacency Matrix

![image-20210401103522128](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20210401103522128.png)

left ; direct , right ; undirect

### Feature Matrix

![image-20210401103550282](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20210401103550282.png)

Summary

- Encoder

$$
q(z_i|X,A) = N(z_i|u_i,diag(\sigma^2_i)
$$



- Decoder

$$
p(A_{ij}=1|z_i,z_j)=\sigma(z^T_iz_j)
$$


$$
Z = \mu+\sigma*\epsilon
$$
The decoder (generative model) is defined by an inner product between latent variable Z. The output of our decoder is a reconstructed adjacency matrix A-hat, which is defined as
$$
\hat{A} = \sigma(zz^T)
$$


### Loss Function

The loss function for variational graph autoencoder is pretty much the same as before.


$$
L = E_{q(Z|X,A)}[logp(A|Z)]-KL[q(Z|X,A)||p(Z)]
$$

### The advantage of using inner product decoder

- After we get the latent variable Z, we want to find a way to learn the similarity of each row in the latent variable to generate the output adjacency matrix. 
- Inner product could calculate the cosine similarity of two vectors, which is useful when we want a distance measure that is invariant to the magnitude of the vectors. Therefore, by applying the inner product on the latent variable Z and Z^T, we can learn the similarity of each node inside Z to predict our adjacency matrix.



Main use ; Link prediction

