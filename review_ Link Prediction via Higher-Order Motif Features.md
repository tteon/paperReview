[Link Prediction via Higher-Order Motif Features](https://arxiv.org/abs/1902.06679)



3줄 요약

- Motif hyperparameter (k-motif) 에 따른 성능 중 3,4,5 motif 의 성능이 잘 나옴. 이 때 3,4,5 모두 합친 feature은 오히려 성능을 저하시키는 점 관찰할 수 있었음.

![image](https://user-images.githubusercontent.com/52625664/144146643-2ba1bb2f-4028-40a0-92b3-bf3c1b5a191d.png)



| ![image](https://user-images.githubusercontent.com/52625664/144146650-86a52712-ffe0-423c-9787-cf1e1d7b6a6e.png)| The first graph represents the co-purchase network of products on Amazon. It is the graph upon which the “customers who bought this also bought that” feature is built. The nodes are products, and an edge between any two nodes shows that the two products have been frequently bought together. In this application domain, link prediction tries to model the creation of an association between two products. In other words, it can help identify hidden relationships between products, which can then be used in a customer-facing recommendation system |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| ![image](https://user-images.githubusercontent.com/52625664/144146673-c87aedce-af79-4dc4-91f5-1803a39b934c.png) | [CondMat dataset descrpition](http://www.casos.cs.cmu.edu/tools/datasets/external/cond-mat/SNA/Cond_Mat.html)CondMat, is a graph which represents a subset of authorship relations between authors and publications in the arXiv condensed matter physics section. Nodes are authors (first set) or papers (second set). An edge represents that an author has written a given paper. Link prediction can identify whether an author is a likely contributor to a research paper, for instance, to identify missing records. |
| ![image](https://user-images.githubusercontent.com/52625664/144146678-5d995807-bcf0-4423-b2bb-34c01d1deaf9.png) | [AstroPh dataset description](https://snap.stanford.edu/data/ca-AstroPh.html)AstroPh, is a collaboration graph. In particular, it contains data about scientific collaboration between authors in the arXiv astrophysics section. Each node in the graph represents an author of a paper, and an edge between two authors represents a common publication. |





- Link prediction 시 positive , negative 각각에서의 motif generation 은 fair experiment 에서 중요한 과제임. 이를 train / test datasets 에서 어떻게 selection 해줄지에 대해 “RMV (Remove positive edges)” , “INS (Inserted into the graph)” 2가지 측면으로 접근함. case by case 이긴 하나 대체적으로 INS 가 더 좋은 성능을 보임. (± 0.3 )

![image](https://user-images.githubusercontent.com/52625664/144146689-96af5e0a-1e77-4750-a9d8-89124d23cf3f.png)

Earth Mover’s Distance (EMD) and Kullback– Leibler Divergence (KLD) between the distribution of motifs in the original graph and the one obtained by each feature extraction method, RMV and INS. A smaller distance indicates that the given feature extraction method is more faithful to the original graph. ( 본래 그래프의 graph distribution 과 RMV , INS 각각의 extraction 기법으로 추출된 graph distribution 의 비교 값. 값이 낮을수록 original graph의 특성을 온전히 보존했다고 볼 수 있음.)





| ![image](https://user-images.githubusercontent.com/52625664/144146699-4a3d83eb-e6b9-463f-8bef-0ff1ac60fd77.png) | ![image](https://user-images.githubusercontent.com/52625664/144146707-7397838c-6a65-40f5-b487-d0445e944c19.png)                                                 |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
|                                                                                                                             |
Negative sampling 측면에서 출발 노드부터 도착 노드까지 몇 path(hop) 이 후 연결되어 있지 않은 노드 간의 link 를 negative 라 지정해주는지에 대해 성능 차이가 있음을 보임.
      


Summary

 

Link prediction requires predicting which new links are likely to appear in a graph. Being able to predict unseen links with good accuracy has important applications in several domains such as social media, security, transportation, and recommendation systems. A common approach is to use features based on the common neighbors of an unconnected pair of nodes to predict whether the pair will form a link in the future. In this paper, we present an approach for link prediction that relies on higher-order analysis of the graph topology, well beyond common neighbors. We treat the link prediction problem as a supervised classification problem, and we propose a set of features that depend on the patterns or motifs that a pair of nodes occurs in. By using motifs of sizes 3, 4, and 5, our approach captures a high level of detail about the graph topology within the neighborhood of the pair of nodes, which leads to a higher classification accuracy. In addition to proposing the use of motif-based features, we also propose two optimizations related to constructing the classification dataset from the graph. First, to ensure that positive and negative examples are treated equally when extracting features, we propose adding the negative examples to the graph as an alternative to the common approach of removing the positive ones. Second, we show that it is important to control for the shortest-path distance when sampling pairs of nodes to form negative examples, since the difficulty of prediction varies with the shortest-path distance. We experimentally demonstrate that using off-the-shelf classifiers with a well constructed classification dataset results in up to 10 percentage points increase in accuracy over prior topology-based and feature learning methods.

