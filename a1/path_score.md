I've copied the path_score equation here for better readability. View it on Github to see the math rendered correctly.

One limitation of Jaccard is that it only has non-zero values for nodes two hops away.

Implement a new link prediction function that computes the similarity between two nodes $x$ and $y$  as follows:

$$
s(x,y) = \beta^i n_{x,y,i}
$$

where
- $\beta \in [0,1]$ and $m \in [2,\infty]$ are user-provided parameters
- $i$ is the length of the shortest path from $x$ to $y$
- $n_{x,y,i}$ is the number of shortest paths between $x$ and $y$ with length $i$
