# Learning to Embed Categorical Features without Embedding Tables for Recommendation



DHE (Deep Hash Embedding)

1줄 요약

- large-vocab 혹은 dynamic setting 을 대체하고자 제안한 임베딩 방법.



**문제 제기**

NLP 에서는 unseen , out of vocabulary 그리고 recommendation 에서는 dynamic feature들이 빈번히 발생하는데 현 임베딩 방법들은 이 들을 capture하기에 한계들이 명확히 존재하고 있었음.



preliminary knowledge

**Hashing trick**

; is a representative hashing method for reducing the dimension for one-hot encoding for large vocabularies.

동일하게 Encoding function 이 존재하나 다른 cardinality 를 feature value 에 부여해줌.

위와 비슷하게 Decoding function 도 H(s)-th row 에서 가져와서 value 를 return 해줌. 

요약하자면 hashing trick 은 hashing을 feature value mapping 하는데 쓰이며 구체적으로는 m-dim one-hot vector로 transformation 해준다. 그리고 그때 1-layer network 를 활용함.



**Identity Encoding**

High Shannon Entropy ; measures the information carried in a dimension. 



## Dense Hash Encoding

; Equal Similarity and High dimensionality 측면을 모두 고려하고자 하였으나 모두 실패하였음. 본 섹션은 앞선 문제점을 해결해보고자 Dense Hash Encoding 을 제안하였음. 

* 참고로 이상적인 encoding의 분포를 살펴보면 고루게 분포되어있다. 

허나 NN 에 input으로 주기에 앞선 hash value는 적절치 않으므로 normalized 등 여러 전처리를 해줘야 한다. 그래서 우리는 다음과 같은 2가지 분포를 적용해주고자 함 . 

- Uniform Distribution.

간단하게 Encoding 을 [-1,1] range 로 normalized 해줌. 

- Gaussian Distribution.

Box-Muller transform을 통한 강제적으로 distributed sample을 해줌.



## Deep Embedding Network

Deep neural network 의 complex transformation 의 강점을 이용한 구조. 최근 연구에 따르면 deeper networks 가 shallow network 보다 훨씬 parameter 측면에서 효율적이다 라는 연구가 있음...?[Paper link](https://proceedings.neurips.cc/paper/2017/file/32cbf687880eb1674a07bf717761dd3a-Paper.pdf)

결과적으로 transformation task 는 매우 도전적인 과제이며 특히나 본 연구에서는 Deep Neural Network 를 활용하여 weight를 저장해야하기에 효율적인 DNN을 사용하고자 함.

feed forward network 를 decoding function 으로 활용하고자 함. ( embedding table을 활용하지않고 오직 hidden layer 의 weight만을 활용하고자 함 . 이러면 pre-trained 방식도 적용이 가능하지 않을까 ?? )

NN의 고질적인 문제인 오버피팅을 완화해보고자 여러 적용을 해보았는데 그 중 Batch normalization 과 Mish activation function 이 more benefit 했다는 결과를 보임.



**Summary**

2가지 요소로 구성되어 있음.

1. dense hash encoding
2. Deep embedding network.

- DHE의 타 임베딩 대비 큰 차별점은 embedding table의 부재이며 모든 embedding 정보를 weight 로 저장한다는 점이다.

최종적으로 앞선 장점들을 토대로 online learning setting (production level)에서 유용하게 활용될것이며 또한 멱법칙 측면에서도 유용할거라 생각이됨.(categorical value 들은 tail distribution 측면에서 disadvantage가 있었던 걸로 기억함.)