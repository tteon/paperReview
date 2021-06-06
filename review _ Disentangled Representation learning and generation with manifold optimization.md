Disentangled Representation learning and generation with manifold optimization

[paper](https://arxiv.org/abs/2006.07046)

Abstract

Disentanglement concept ; increasing the interpretability of generative models such as Variational Auto-Encoders, Generative Adversarial Models, and their many variants. An increase in space models, this work presents a representation learning framework that explicitly promotes disentanglement by encouraging orthogonal directions of variations.



Our analysis shows that such a construction promotes disentanglement by matching the principal directions in latent space with the directions of orthogonal variaiton in data space.

The training algorithm involves a stochastic optimization method on the Stiefel manifold, which increases only marginally the computing time compared to an analogous VAE.

![image-20210606150925096](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20210606150925096.png)



Introduction

Disentanglement. 

In latent variable modeling, one is often interested in modeling the data in terms of 'uncorrelated' or 'independent' components, yielding a so-called 'disentangled' representation which is often studied in the context of VAEs. In representation learning, disentanglement corresponds to a decoupling of generating factors.

![image-20210606151008678](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20210606151008678.png)

at figure 2,  We observe that moving along the first principal component generates images where only floor color is varying while all other features such as shape, scale, wall color, object color, etc. are constant . In the second row, the object scale is only changing. -> **An advantage of such a representation is that the different latent units impart more interpretability to the model.** Disentangled models are useful for the generation of plausible pseudo-data with certain desirable properties. e.g. generating new car designs with a predefined color or height. One popular variant achieving disentanglement but at the expense of generation quality is $$\beta$$VAE





2.1 Training jointly an auto-encoder and Principla Component Analysis in latent space

The idea consists of learning an auto-encoder along with finding an 'optimal' linear subspace of the latent space so that the variance of the training set in latent space is maximized within this space.

Both encoder and decoder are neural networks parameterized by vectors of 'reals \theta' , '?! unknown greek' However, it is unclear how to define a parameterization or an architecture of these neural networks so that the learned representation is disentangled. Therefore, in addition to these trained parameters, we also jointly find an m-dimensional linear subspace range(U) of the latent space R^l, so that the encoded training points mostly lie within this subspace.

Stiefel manifold St(l,m). supplement at later ! 

- an AE loss which promotes parameters such that equation O
- and, a PCA loss which aims to yield equation O

The basic idea is to combine in (St-RKM) different AE losses with a regularization term which penealizes the feature map in the orthogonal subspace U^\orthgonal 

our objective assumes that the neural network defining the encoder provides a better embedding if we impose that it maps training points on a linear subspace of dimension m < l in the l-dimensional latent space. In other words, the optimization of the parameters in the last layer of the encoder does not play a redundant role, since the second term in (st-RKM) cleary also depends on O.

3.2 Disentanglement

we argue that the principla directions in latent space match orthogonal directions of variation in the data space. Therefore, the disentanglement of our representation is due to the optimization over U \ in St(l, m) and is promoted by the stochastic AE loss. 

A disentangled representation would satisfy ; equation O . In other words, as the latent point moves along u_k or along u_k', the decoder output varies in a significantly different manner.

Conclusion

the interconnection matrix U is restricted to be a rectangular matrix with orthonormal columns, i.e., valued on a Stiefel manifold. Then , for the training, we use the Cayley Adam algorithm of for stochastic optimization on the Stiefel manifold. 

We propose several Auto-Encoder objectives and discuss that the combination of a stochastic AE loss with an explicit optimization on the Stiefel manifold promotes disentanglement.

we establish connections with probabilistic models, formulate an Evidence Lower Bound (ELBO), and discuss the independence of latent factors.

* [Stiefel manifold](https://en.wikipedia.org/wiki/Stiefel_manifold) 

Likewise one can define the complex Stiefel manifold V_k(C^n) of orthonormal k-frames in C^n and the quaternionic Stiefel manifold V_k(H^n) of orthonormal k-frames in H^n. More generally, the construction applies to any real, complex, or quaternionic inner product space.



Summary

- Stiefel manifold
- Training jointly an auto-encoder and Principal Component Analysis in latent space.

-> 잠재적 공간인 latent space 에서 PCA를 활용하여 variance handling 하는 training 과정을 토대로 optimal space를 찾는것이 이 논문의 아이디어의 핵심임.

- Proposition of two AE losses.

$$
L^{(\sigma)}_{\xi,U} = E_{\in}||x-\psi_{\xi}(P_Uz+\sigma U\varepsilon)||_2^2
$$

-> noise term {\sigma U \vareplison} 이 decoder network 을 smoother 해주게 하는 역할을 함. (additional term을 통하여 효율적인 임베딩을 하고자 제시함.)

-문제정의 

generative model 인 AE 에서의 reconstruction error 는 각각의 encoder / decoder 의 standard normal distribution 으로 파생 된다. 즉 , encoder 에서부터 compressed embedding of latent space 에서로부터 다시 decoder generating 되는것인데 이 때의 파생은 Evidence Lower Bound(ELBO)로부터 나타난다. 
$$
E_{z~q(z|x)}[log(p(x|z))]-\beta KL(q(z|x),p(z)) \leq logp(x)
$$
이 때 ,위 식의 하이퍼파라미터인  \beta > 1 일 수록 more disentanglement의 성질이 존재하는데 이를 optimization problem 으로 잡음.

**minimization of the 'regularized' AE** , 즉 regularized term 을 주어 해결해보고자 함 . 

-문제해결

1. joint training of Auto Encoder , PCA

$$
\min_{U\in S(l,m)}\lambda\frac{1}{n}\sum^n_{i=1}L_{\xi,U}(x_i,\phi_\theta(x_i))+\frac{1}{n}\sum^n_{i=1}||P_U\perp\phi_\theta(x_i)||^2_2
$$

left-hand ; auto-encoder objective , right-hand ; PCA objective (St-RKM)



Review 

비전공자인 나에겐 다소 어려운 수학적 지식이 많이 담겨있어 해석이 어려웠으나 논문의 저자가 대략 어떤것을 문제정의하고 그 문제에 대한 해결을 어떤 방식으로 했는지에 대한 맥락은 이해할 수 가 있었다.



















