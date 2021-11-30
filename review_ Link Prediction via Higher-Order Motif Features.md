[Link Prediction via Higher-Order Motif Features](https://arxiv.org/abs/1902.06679)



3줄 요약

- Motif hyperparameter (k-motif) 에 따른 성능 중 3,4,5 motif 의 성능이 잘 나옴. 이 때 3,4,5 모두 합친 feature은 오히려 성능을 저하시키는 점 관찰할 수 있었음.

| ![image-20211201084907223](C:\Users\Win10\AppData\Roaming\Typora\typora-user-images\image-20211201084907223.png) | The first graph represents the co-purchase network of products on Amazon. It is the graph upon which the “customers who bought this also bought that” feature is built. The nodes are products, and an edge between any two nodes shows that the two products have been frequently bought together. In this application domain, link prediction tries to model the creation of an association between two products. In other words, it can help identify hidden relationships between products, which can then be used in a customer-facing recommendation system |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| ![image-20211201084753185](C:\Users\Win10\AppData\Roaming\Typora\typora-user-images\image-20211201084753185.png) | [CondMat dataset descrpition](http://www.casos.cs.cmu.edu/tools/datasets/external/cond-mat/SNA/Cond_Mat.html)CondMat, is a graph which represents a subset of authorship relations between authors and publications in the arXiv condensed matter physics section. Nodes are authors (first set) or papers (second set). An edge represents that an author has written a given paper. Link prediction can identify whether an author is a likely contributor to a research paper, for instance, to identify missing records. |
| ![image-20211201084805826](C:\Users\Win10\AppData\Roaming\Typora\typora-user-images\image-20211201084805826.png) | [AstroPh dataset description](https://snap.stanford.edu/data/ca-AstroPh.html)AstroPh, is a collaboration graph. In particular, it contains data about scientific collaboration between authors in the arXiv astrophysics section. Each node in the graph represents an author of a paper, and an edge between two authors represents a common publication. |





- Link prediction 시 positive , negative 각각에서의 motif generation 은 fair experiment 에서 중요한 과제임. 이를 train / test datasets 에서 어떻게 selection 해줄지에 대해 “RMV (Remove positive edges)” , “INS (Inserted into the graph)” 2가지 측면으로 접근함. case by case 이긴 하나 대체적으로 INS 가 더 좋은 성능을 보임. (± 0.3 )

![image-20211201084828491](C:\Users\Win10\AppData\Roaming\Typora\typora-user-images\image-20211201084828491.png)

Earth Mover’s Distance (EMD) and Kullback– Leibler Divergence (KLD) between the distribution of motifs in the original graph and the one obtained by each feature extraction method, RMV and INS. A smaller distance indicates that the given feature extraction method is more faithful to the original graph. ( 본래 그래프의 graph distribution 과 RMV , INS 각각의 extraction 기법으로 추출된 graph distribution 의 비교 값. 값이 낮을수록 original graph의 특성을 온전히 보존했다고 볼 수 있음.)



Negative sampling 측면에서 출발 노드부터 도착 노드까지 몇 path(hop) 이 후 연결되어 있지 않은 노드 간의 link 를 negative 라 지정해주는지에 대해 성능 차이가 있음을 보임. 

| ![image-20211201084900318](C:\Users\Win10\AppData\Roaming\Typora\typora-user-images\image-20211201084900318.png) | ![image-20211201084907223](C:\Users\Win10\AppData\Roaming\Typora\typora-user-images\image-20211201084907223.png) |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
|                                                              |                                                              |


![image](https://user-images.githubusercontent.com/52625664/144146495-23cbe680-6b4d-4cb9-a33d-4db739085f3f.png)



