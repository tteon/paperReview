# Adaptive Filters and Aggregator Fusion for Efficient Graph Convolutions



## Abstract

Training and deploying graph neural networks remains difficult due to their high memory consumption and inference latency.

**state-of-the-art performance with lower memory consumption and latency, along with characteristics suited to accelerator implementation.**

Our proposal uses memory proportional to the number of vertices in the graph, in contrast to competing methods which requires memory proportional to the number of edges; 

*adaptive filtering inspired by signal processing* ; it can be interpreted as enabling each vertex to have its own weight matrix, and is not related to attention.

Aggregator fusion, a technique to enable GNNs to significantly boost their representational power.



## Contribution

- new GNN architecture, Efficient Graph Convolution, which does not require trading accuracy for runtime memory or latency reductions. -> memory consumption is linear in the number of vertices in the graph . Our architecture is a *drop-in replacement* on a wide variety of tasks.
- hardware considerations a core aspect of our architecture design. Our architecture is well suited to existing accelerator designs, while offering substantially  better accuracy than existing approaches.
- rigorous evaluation ; from citation graphs to molecular property prediction.



## Our Architecture ; Efficient Graph convolution

EGC-S ; single aggregator.

EGC-M ; generalizes our approach by incorporating multiple aggregator functions, and which can be accelerated by our *aggregator fusion* approach.





Architecture Description
$$
y^{(i)} = \sum^B_{b=1}w_b^{(i)}\sum_{j\in N(i)}\alpha(i,j)\theta_bx^{(j)}
$$
**layer output for node is above equation.**

For a layer with in-dimension of F and out-dimension of F' we use B basis weights \theta_i \in \R^{F' \times F}.

- \alpha(i,j) is some function of nodes i and j,
- \N(i) denotes the in-neighbours of i.

**Adding Heads**

- **apply different weighting coefficients per head;** 

$$
y^{(i)} = ||_{h=1}^H\sum_{b=1}^Bw_{h,b}^{(i)}\sum_{j\in  N(i)\cup(i)}\frac{1}{\sqrt{deg(i)deg(j)}}\theta_bx^{(j)}
$$





**Boosting Representational Capacity**



It is better to combine several different aggregators.

To improve performance, we propose applying different aggregators to the represnetations calculated by \theta_bx^{(j)}. The choice of aggregators could include different variants of summation aggregators.



![image-20210412174742723](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20210412174742723.png)



**Aggregator Fusion**

The naive approach of performing each aggregation sequentially would cause this linear increase-but there is a better way to order our computation. The key observation to note is that we are *memory-bound*, and not compute-bound;

the loop over our aggregation functions should be the inner-most loop; performing the aggregations sequentially would correspond to having this loop over aggregators at the outer-most level.

![image-20210412180222075](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20210412180222075.png)





## Interpretation and Benefits



**Spatial Interpretation**





**Localised Spectral Filtering with Multiple Kernels**





**Interaction with Hardware**

- SpMM, it is also worth nothing for small or dense graphs, it may be faster to implement messgae propagation using dense matrix multiplication; this is not possible for architectures relying on matreialization.
- Our approach is also beneficial for data center and mobile workloads. 







