# Recsys2020 _ Takeaways and Notable papers

https://towardsdatascience.com/recsys2020-777590f51b44

https://eugeneyan.com/writing/recsys2020/#industry-applications-context-unexpectedness-interesting-use-cases



Day 1's keynote - 4 Reasons Why Social Media Make Us Vulnerable to Manipulation

As social media become major channels for the diffusion of news and information, it becomes critical to understand how the complex interplay between cognitive, social, and algorithmic biases triggered by our reliance on online social networks makes us vulnerable to manipulation and disinformation. This talk overviews ongoing network analytics, modeling, and machine learning efforts to study the viral spread of misinformation and to develop tools for countering the online manipulation of opinions.

Day 2's keynote - Bias in Search and Recommender Systems.

We cover all biases, to the best of our knowledge, that affect search and recommender systems. They include biases on the data, on the algorithms involved (and their evaluation), and on the user interaction, particularly the ones related to feedback loops (e.g., ranking and personalization). In each case, we cover the main concepts and when known, the techniques to mitigate them. We give special emphasis to exposure bias, which we believe is the main bias that impacts, both, users and systems. This presentation is partially based on Bias on the Web, Communications of the ACM, June 2018.

## Notable: Offline evaluation, MF > MLP, applications

Several papers on offline evaluation highlighted the [nuances and complexities](https://eugeneyan.com/writing/recsys2020/#towards-more-robust-offline-evaluation-and-study-reproducibility) of assessing recommender systems offline and suggested process improvements. Also, Netflix gave a great talk sharing their [findings from a comprehensive user study](https://eugeneyan.com/writing/recsys2020/#user-research-on-the-nuances-of-recommendations).

There was also a (controversial?) talk by Google [refuting the findings of a previous paper](https://eugeneyan.com/writing/recsys2020/#comparing-the-simple-dot-product-to-learned-similarities) where learned similarities via multi-layer perceptrons beat the simple dot product.

Of course, I also enjoyed the many papers sharing how organizations built and deploy [recommender systems in the wild](https://eugeneyan.com/writing/recsys2020/#industry-applications-context-unexpectedness-interesting-use-cases) (more [here](https://github.com/eugeneyan/applied-ml#recommendation)).

Netflix found that users have different expectations across different recommendation placements. For example, users have higher expectations of similarity when it’s a 1:1 recommendation (e.g., after completing a show, Netflix would recommend a *single* next title). Such recommendations are risky as there are no backups, and there are no other recommendations to help the user understand similarity.

In contrast, users have lower expectations in 1:many recommendations (e.g., a slate of recommendations), such as when the user is browsing. In the example below, “Queer Eye” might seem far removed from “Million Dollar Beach House”. But with the other recommendations in the slate, it makes sense within the overall theme of reality shows.

## User research on the nuances of recommendations





## Towards more robust offline evaluation and study reproducibility





## Comparing the simple dot-product to learned **similarities**



![Image for post](https://miro.medium.com/max/880/0*bvprGzQsoekdtYIL.jpg)





![Image for post](https://miro.medium.com/max/1400/0*ur30h_6XH1WXmMlD.jpg)



-  They found the matrix factorization parameters in the original NCF paper under regularized. Another possible reason could be the addition of explicit biases that have been empirically shown to improve model performance.





## Industry applications: context, unexpectedness, interesting use cases

![Image for post](https://miro.medium.com/max/878/0*qjr3_GIXUBskUXEV.jpg)

- they extracted top N queries associated with each listing and then trained embeddings for items and queries. With these embeddings, candidates were generated via approximate nearest neighbours. However, this did not work as well as the second, simpler approach.

  ![Image for post](https://miro.medium.com/max/33/0*wGBGj-OgQG_3jjKo.jpg?q=20)





![Image for post](https://miro.medium.com/max/879/0*wGBGj-OgQG_3jjKo.jpg)

-  For each search *query*, a set of items would be shown to the user (i.e., search results) — these are the *target* items. Then, for each target item, other items the user interacted with (in the same search session) become *candidate* items. Thus, for each query-target pair, they would have a set of candidates.

![Image for post](https://miro.medium.com/max/880/0*uQJIWrbDwX-lY-QG.jpg)









![Image for post](https://miro.medium.com/max/880/0*QxTcTdi3N43vvKLY.jpg)





![Image for post](https://miro.medium.com/max/880/0*iS-GnDiClSdwa_uZ.jpg)

- They used an RNN-based architecture to jointly learn from historical sequences and context. The key was to *fuse* the context-dependent user embeddings and long-term user embeddings using attention weights.

### Unexpectedness



Pan Li from New York University shared how Alibaba’s Youku [**introduces freshness and unexpectedness**](https://dl.acm.org/doi/10.1145/3383313.3412238) **into video recommendations**. He distinguished between two kinds of unexpectedness:

- Personalized: Some users are variety seekers and thus more open to new videos
- Session-based: If a user finishes the first episode of a series, it’s better to recommend the next episode. If the user binged on multiple episodes, it’s better to recommend something different.

![Image for post](https://miro.medium.com/max/880/0*Okl66ODhxgFe3FPr.jpg)





