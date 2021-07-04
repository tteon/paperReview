# Four Deep Learning Papes to Read in April 2021

## From Meta-Gradients to Clockwork VAEs, a Global Workspace Theory for Neural Networks and the Edge of Training Stability

source ; https://towardsdatascience.com/four-deep-learning-papers-to-read-in-april-2021-77f6b0e42b9b

## “Discovery of Options via Meta-Learned Subgoals“





## **“Clockwork Variational Autoencoders“**

-  At their core CW-VAEs scale these latent dynamics models by introducing a hierarchy of latents, which change at different fixed clock speeds.
- The top-level adapts at a slower rate and modulates the generative process of the lower levels. The ‚tick‘ speed increases as one goes down in the hierarchy.
- **The entire recurrent VAE architecture is trained end-to-end using the evidence lower bound (ELBO) objective**
- I especially enjoyed the cool ablation study, that aim to extract the content information stored at the different levels.
- In summary, a hierarchy over mechanisms, which act on different time-scales is not only highly useful for Reinforcement Learning but also generative modelling.

![img](https://miro.medium.com/max/3667/1*Xp-ZAMVLGnaBMXCXH2DvAQ.png)

## **“Coordination Among Neural Modules Through a Shared Global Workspace“**





## **“Gradient Descent on Neural Networks Typically Occurs at the Edge of Stability“**

- still not fully explained observations in Deep Learning is that we seem to be able to effectively optimise billions of parameters using only a simple algorithm such as stochastic gradient descent.
-  take a step back to investigate the special case of gradient descent when the batch consists of the entire dataset. The authors show that this full-batch gradient descent version operates in a very special regime.

![img](https://miro.medium.com/max/3672/1*sSAk09bH8e4d1jG6PDilRQ.png)