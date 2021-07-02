# Zero-shot Ingredient Recognition by Multi-Relational Graph Convolutional Network



## Abstract

- target the problem of ingredient recognition with zero training samples.

- multi-relational GCN that integrates ingredient hierarchy, attribute as well as co-occurrence for zero-shot ingredient recognition.
- 



## Intro



![image-20210421113745498](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20210421113745498.png)

### Problem Definition

- Exisiting works on ingredient recognition mainly focus on recognizing a relatively narrow set of ingredients, ranging from 93 to 1,276 categories
- The major obstacle for large-scale ingredient recognition is the lack of sufficient training samples. To overcome this data sparsity issue, a more practical way is to endow the model with the ability to recognize ingredient which has zero training sample.



## Key idea

- transfer the knowledge obtained from familiar categories to unfamiliar ones. 

- The knowledge transfer is either based on implicit knowledge representations, i.e. semantic embedding , or explicit knowledge bases which represents the knowledge as rules or relationships between objects. 
- We utilize both implicit and explicit knowledge for zero-shot ingredient recognition.
- Single-relational knowledge graph cannot differentiate the different effects towards predicting unseen ingredients brought by different relations.
- To address this, we define **a multiple-relational graph to capture multiple types of relations among ingredients, such as ingredient attribute ( i.e., color or shape), ingredient hierarchy, and ingredient co-occurrence**.

## Contribution

- construct a multi-relational knowledge graph that models three types of relations between ingredients that we find are crucial for this problem.
- efficiently exploits three types of relations among ingredients, for zero-shot ingredient recognition.
- explore other ways of coupling different types of relations.



## Related work

### Multi-label Zero-shot Learning

- multi-label zero-shot learning usually relies on the relationships between seen labels and unseen labels. 
  1. early model (COSTA) which estimates the classifiers for new labels as the weighted combination of seen classes, leveraging their co-occurrence statistics.
  2. performs multi-label zero-shot learning on a structured knowledge graph with three types of relations among labels, namely 'super-subordinate' compiled from WordNet and 'positive relation' as well as 'negative relation' obtained from label similarities.
  3. GSNN is applied to propagate the probabilities from seen labels to unseen labels.

differs from the aforementioned works in two aspects.

1. apart from co-occurrence and ISA relation.
2. consider interaction among different ingredient relations.

Baseline ; 

- Multi-relational graph

separate representations for nodes and edges are learnt via jointly optimizing on two tasks.

1. Link structure prediction.
2. Node attributes preservation.

local graphs are first sampled from multi-hop neighboring entities and relations of a given entity. Afterwards, localized graph convolutions are employted to generate node and relation embeddings.



- **ImageGCN for multi-relational image modeling**

https://github.com/mocherson/ImageGCN

ImageGCN for multi-relational image modeling, which **models the image-level relations to generate more informative image representations**. However, different relation edges are equally-weighted during propagation in above methods, which may fail to capture various interactions between different relations. To address this, **our model introduces the attention mechanism into GCN, such that various relations can contribute differently during graph propagation.** Although a similar idea was proposed in (Shang et al. 2018), in which they jointly learn attention weights and node features in graph convolution, their model are specially designed for **chemical datasets**, which have different attributes and relations with the food domain.





![image-20210421115332186](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20210421115332186.png)

Framework overview



## Multi-Label CNN

multi-label DCNN is built upon ResNet-50 by replacing the softmax loss with sigmoid cross-entropy loss since ingredient recognition is a multi-label learning problem. The convolutional layers can be considered as the feature learning layers, while the last fully connected layer as the classifier layers.

We denote the learned classifier weight as W, W \in \R^m\timesd, where m is the number of ingredient categories and d is the dimension of the image features. 

For the i^th ingredient in M, the learned classifier weight W_i \in R^d can be considered as a binary classifier, predicting whether the image contains the i^th ingredient or not. Then the learned binary classifiers W will be used as ground-truth classifiers supervising the learning of multi-relational Graph convolutional Network.



### Multi-relational graph convolutional network

relation selection

- Ingredient Hierarchy.

introduce 'isa' relations among ingredients, suggesting the knowledge between parents and children nodes. The ingredient hierarchy is manaually constructed according to retail websites and recipe websites.

- Ingredient Attribute.

We link the ingredients that share the same attributes, such as color, shape or cooking methods, exhibiting the attribute-aware knowledge among ingredients.

in this work, we consider 19 attributes in total, including 5 Common color, 8 shape attributes and 6 cooking methods that will have significant affects on the ingredient apperances.

- Ingredient Co-occurrence.

certain groups of ingredients co-occur more often while some ingredients are likely exclusive of each other.













