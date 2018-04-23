Bonus -- 10 points

Recall our use of the Jaccard index to quantify the similarity between two nodes in a social network. In this assignment, you will implement a modified version of Jaccard as follows:
$$
jaccard_{wt}(A, B) = \frac{\sum_{i \in (A \cap B)} \frac{1}{deg(i)}}{\frac{1}{\sum_{i \in A} deg(i)} + \frac{1}{\sum_{j \in B} deg(j)}}
$$
where $A$ and $B$ are sets of neighbors of two nodes to be scored, and $deg(i)$ is the degree of node $i$.

This method attempts to account for the fact that if we share a neighbor
that has low degree, we should have a higher recommendation score than if we share a neighbor with high degree. (E.g., two people that like the relatively obscure band Gang of Four are probably more similar than two people that like Justin Bieber).
