# 	Network Embedding 

## 	Paper Review

`[2017]struc2vec: Learning Node Representations from Structural Identity`

>Structural identity is a concept of symmetry in which network nodes are identied according to the network structure and their relationship to other nodes. Structural identity has been studied in theory and practice over the past decades, but only recently has it been addressed with representational learning techniques. is work presents struc2vec, a novel and exible framework for learning latent representations for the structural identity of nodes. struc2vec uses a hierarchy to measure node similarity at different scales, and constructs a multilayer graph to encode structural similarities and generate structural context for nodes. Numerical experiments indicate that state-of-the-art techniques for learning node representations fail in capturing stronger notions of structural identity, while struc2vec exhibits much superior performance in this task, as it overcomes limitations of prior approaches. As a consequence, numerical experiments indicate that struc2vec improves performance on classication tasks that depend more on **structural identity.**

`[2016]node2vec: Scalable Feature Learning for Networks`

> Prediction tasks over nodes and edges in networks require careful effort in engineering features used by learning algorithms. Recent research in the broader ﬁeld of representation learning has led to signiﬁcant progress in automating prediction by learning the features themselves. However, present feature learning approaches are not expressive enough to capture **the diversity of connectivity patterns observed in networks.**
>
>  Here we propose node2vec, an algorithmic framework for learning continuous feature representations for nodes in networks. In node2vec, we learn a mapping of nodes to a low-dimensional space of features that **maximizes the likelihood of preserving network neighborhoods of nodes. **We deﬁne a ﬂexible notion of a node’s network neighborhood and design a biased random walk procedure, which efﬁciently explores diverse neighborhoods. Our algorithm generalizes prior work which is based on rigid notions of network neighborhoods, and we argue that the **added ﬂexibility in exploring neighborhoods is the key to learning richer representations.**
>
>  We design a ﬂexible neigh- borhood sampling strategy which allows us to smoothly interpolate between BFS and DFS. **We achieve this by developing a ﬂexible biased random walk procedure that can explore neighborhoods in a BFS as well as DFS fashion.**

`[2015]LINE: Large-scale Information Network Embedding`

> This paper studies the problem of embedding very large information networks into low-dimensional vector spaces, which is useful in many tasks such as visualization, node classiﬁcation, and link prediction. Most existing graph em- bedding methods do not scale for real world information networks which usually contain millions of nodes.In this paper, we propose a novel network embedding method called the “LINE,” which is **suitable for arbitrary types of information networks: undirected, directed, and/or weighted**. The method optimizes a carefully designed objective function that **preserves both the local and global network structures.** An edge-sampling algorithm is proposed that addresses the limitation of the classical stochastic gradient descent and improves both the eﬀectiveness and the eﬃciency of the in- ference. Empirical experiments prove the eﬀectiveness of the LINE on a variety of real-world information networks, including language networks, social networks, and citation networks. The algorithm is very eﬃcient, which is able to learn the embedding of a network with millions of vertices and billions of edges in a few hours on a typical single machine. The source code of the LINE is available online. 



